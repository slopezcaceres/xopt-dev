import os
import importlib.util
from transformers import pipeline
import time
from xopt.generators.bayesian.llm_adaptive_bo import LLMAdaptiveBOGenerator
import torch
import pandas as pd
from pydantic import Field
from xopt.generator import Generator
from collections import deque
from typing import Dict, Any


def load_plugin(plugin_type: str, section: str):
    plugin_path = os.path.expanduser(
        f"~/Library/Application Support/Badger/plugins/{plugin_type}/{section}/__init__.py"
    )
    module_name = f"{plugin_type}_{section}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def initialize_llm(model_path):
    return pipeline(
        "text-generation",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1
    )

def init_section_generator(vocs, logbook_path, llm_path, good_tune, beta=1.0):
    gen = LLMAdaptiveBOGenerator(
        vocs=vocs,
        logbook_path=logbook_path,
        llm_model_path=llm_path,
        good_tune=good_tune,
        beta=beta
    )
    return gen

class LLMOrchestratorGenerator(Generator):
    name = "llm_orchestrator_generator"

    gp_constructor: Dict[str, Any] = Field(default_factory=lambda: {"use_low_noise_prior": False}, description="Dummy field to satisfy GUI expectations")
    llm_model_path: str = Field("path/to/llama2-local", description="Path to LLM")
    section_names: list[str] = Field([], description="Ordered section names")
    logbook_paths: dict = Field(default_factory=dict, description="Optional section-wise logbook paths")
    good_tunes: dict = Field(default_factory=dict, description="Optional section-wise good tunes")
    archive_dir: str = Field("LLMarchive", description="Path to archive directory")
    plateau_window: int = Field(5, description="Number of steps to consider for plateau detection")
    plateau_epsilon: float = Field(10, description="Minimum improvement to avoid plateau")
    initial_beta: float = Field(2, description="Initial beta for BO exploration")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_section_idx = 0
        self.generator = None
        self._llm_pipeline = initialize_llm(self.llm_model_path)
        self._rate_history = deque(maxlen=self.plateau_window)
        self._load_section(self.current_section_idx)

    def _log_transition(self, section):
        os.makedirs(self.archive_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        with open(f"{self.archive_dir}/LLMBadgerOpt-{timestamp}.yaml", "w") as f:
            f.write(f"# Transitioned to section: {section}\n")

    def _load_section(self, idx):
        section = self.section_names[idx]
        os.environ["BADGER_PLUGIN_ENVIRONMENT"] = section
        os.environ["BADGER_PLUGIN_INTERFACE"] = section
        env_module = load_plugin("environments", section)
        load_plugin("interfaces", section)
        vocs = env_module.VOCS
        self.generator = init_section_generator(
            vocs=vocs,
            logbook_path=self.logbook_paths.get(section, "logbook.csv"),
            llm_path=self.llm_model_path,
            good_tune=self.good_tunes.get(section, {}),
            beta=self.initial_beta
        )
        self._rate_history.clear()
        self._log_transition(section)

    def _query_llm_for_next_section(self) -> int:
        history_path = self.logbook_paths.get(self.section_names[self.current_section_idx], "logbook.csv")
        try:
            history_df = pd.read_csv(history_path)
        except FileNotFoundError:
            print("[LLM-Orchestrator] No logbook found. Not switching sections.")
            return self.current_section_idx

        sorted_df = history_df.sort_values(by=self.generator.vocs.output_names[0], ascending=False)
        top_configs = sorted_df.head(3)

        prompt = """
You are optimizing beamline sections sequentially to maximize beam flux.

Here are the top 3 historical configurations:
"""
        for _, row in top_configs.iterrows():
            volts = {k: row[k] for k in self.generator.vocs.variable_names}
            rate = row[self.generator.vocs.output_names[0]]
            prompt += f"\nSection: {self.section_names[self.current_section_idx]} | Voltages: {volts} | Beam rate: {rate}"

        prompt += "\n\nRecent beam rates (most recent last):\n"
        prompt += ", ".join(f"{r:.2f}" for r in self._rate_history)

        prompt += """

If the recent rates have plateaued and are significantly lower than the best historical rates (e.g., <500 when historical bests are ~1000), consider switching.
But if both recent and historical rates are low (e.g., all under 500), do NOT switch — it may be premature.

Is it time to move to the next beamline section? Respond with YES or NO.
"""
        response = self.generator._llm_pipeline(prompt, max_new_tokens=10)[0]["generated_text"].strip().upper()
        print("[LLM-Orchestrator] LLM response for section switch:", response)

        if "YES" in response and self.current_section_idx < len(self.section_names) - 1:
            return self.current_section_idx + 1
        return self.current_section_idx

    def _is_plateaued(self):
        if len(self._rate_history) < self.plateau_window:
            return False
        diffs = [j - i for i, j in zip(self._rate_history, list(self._rate_history)[1:])]
        return all(abs(d) < self.plateau_epsilon for d in diffs)

    def generate(self, n_candidates: int = 1) -> list[dict]:
        suggestions = self.generator.generate(n_candidates)

        latest_rate = self.generator.data[self.generator.vocs.output_names[0]].iloc[-1]
        self._rate_history.append(latest_rate)

        if self._is_plateaued():
            history_path = self.logbook_paths.get(self.section_names[self.current_section_idx], "logbook.csv")
            try:
                history_df = pd.read_csv(history_path)
                global_best = history_df[self.generator.vocs.output_names[0]].max()
            except FileNotFoundError:
                global_best = 0

            if latest_rate < 500 and global_best >= 1000:
                print("[LLM-Orchestrator] Plateau at suboptimal rate. Increasing exploration (beta).")
                self.generator.beta = min(self.generator.beta * 2.0, 10.0)
                print(f"[LLM-Orchestrator] New beta: {self.generator.beta}")
                
            next_idx = self._query_llm_for_next_section()
            if next_idx != self.current_section_idx:
                print("[LLM-Orchestrator] Restoring best configuration before switching.")
                self.generator._reset_to_good_tune()
                self.current_section_idx = next_idx
                self._load_section(next_idx)

        # Reset beta if we see a significant improvement over recent average,
        # even if not technically plateaued — to switch back to exploitation.        
        if len(self._rate_history) == self.plateau_window:
            recent_avg = sum(self._rate_history) / self.plateau_window
            if latest_rate > recent_avg + 100:
                if self.generator.beta != self.initial_beta:
                    print("[LLM-Orchestrator] Found better region after exploration. Resetting beta.")
                    self.generator.beta = self.initial_beta

        return suggestions

    def add_data(self, new_data):
        self.generator.add_data(new_data)

    @property
    def vocs(self):
        return self.generator.vocs
