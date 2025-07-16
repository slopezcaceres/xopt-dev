# FILE: xopt/generators/bayesian/llm_adaptive_bo.py

import torch
import pandas as pd
from transformers import pipeline
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.models.llm_gp import LLMGPModelConstructor
from xopt.generators.bayesian.utils import set_botorch_weights
from xopt.generators.bayesian.objectives import CustomXoptObjective

from botorch.acquisition import (
    UpperConfidenceBound,
    qUpperConfidenceBound,
    ScalarizedPosteriorTransform,
)


class LLMAdaptiveBOGenerator(BayesianGenerator):
    name = "llm_adaptive_bo"
    supports_batch_generation: bool = False
    supports_multi_objective: bool = False
    supports_constraints: bool = False

    # Custom parameters
    loss_threshold: float = Field(default=1e-3, description="Absolute rate threshold for beam loss (optional).")
    beam_loss_patience: int = Field(default=3, description="How many consecutive low-rate steps to consider as beam loss.")
    relative_loss_threshold: float = Field(default=0.2, description="Fraction of best rate below which recent points must fall to trigger beam loss.")
    logbook_path: str = Field(default="logbook.csv", description="Path to CSV logbook of past runs.")
    llm_model_path: str = Field(default="path/to/llama2-local", description="Path to local LLM model.")
    good_tune: dict = Field(default_factory=dict, description="Fallback parameter set if no logbook or LLM fails.", exclude=True)
    custom_kernel_params: dict = Field(default_factory=dict, exclude=True)
    
    beta: float = Field(default=2.0, description="Beta parameter for UCB acquisition function.")

    _llm_pipeline = None
    _good_tune_triggered = False


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_llm()

        # Trigger warm start if no good tune and no logbook
        if not self.good_tune:
            try:
                pd.read_csv(self.logbook_path)
            except FileNotFoundError:
                print("[LLM-BO] No logbook and no good_tune found. Using zero-shot warm start.")
                self.good_tune, lengthscale = self._query_llm_for_good_tune()
                self.custom_kernel_params["lengthscale"] = lengthscale
                self.data = pd.DataFrame([self.good_tune])
    
    def __getstate__(self):
        """
        Limit which fields are serialized by the GUI or yaml.dump.
        This avoids dumping unserializable objects like HuggingFace pipelines or DataFrames.
        """
        state = super().__getstate__() if hasattr(super(), '__getstate__') else self.__dict__.copy()
        state.pop("_llm_pipeline", None)
        state.pop("data", None)
        state.pop("_good_tune_triggered", None)
        return state

    def dict(self):
        return self.__getstate__()

    def _initialize_llm(self):
        if self._llm_pipeline is None:
            self._llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model_path,
                device=0 if torch.cuda.is_available() else -1
            )

    def _detect_beam_loss(self) -> bool:
        if self.data is None or len(self.data) < self.beam_loss_patience + 2:
            return False

        y = self.data[self.vocs.output_names[0]]
        best_rate = y.max()

        if best_rate == 0:
            return False

        recent_rates = y.iloc[-self.beam_loss_patience:]
        below_fraction = recent_rates < (self.relative_loss_threshold * best_rate)

        return below_fraction.all()

    def _query_llm_for_good_tune(self) -> tuple[dict, float]:
        """
        Query the LLM for a good tune and GP kernel lengthscale.
        Uses few-shot in-context learning if logbook exists, otherwise falls back to zero-shot prompting.
        """
        try:
            history_df = pd.read_csv(self.logbook_path)

            # Use the most recent 100 entries
            recent_df = history_df.tail(100)

            # Sort by beam rate (descending)
            sorted_df = recent_df.sort_values(by=self.vocs.output_names[0], ascending=False)

            # Select top-k examples for few-shot prompt (adjust k if needed)
            k = min(5, len(sorted_df))
            few_shot_df = sorted_df.head(k)

            # Build few-shot prompt
            prompt = "You are optimizing beamline voltages. Use past examples to suggest a good configuration.\n\n"
            for i, row in few_shot_df.iterrows():
                voltages = {key: row[key] for key in self.vocs.variable_names}
                rate = row[self.vocs.output_names[0]]
                prompt += f"Example {i+1}:\nInput voltages: {voltages}\nBeam rate: {rate}\n\n"

            prompt += "Based on these examples, suggest a promising voltage configuration and a GP kernel lengthscale.\n"
            prompt += "Return format:\nvoltages: {...}, lengthscale: ..."

        except FileNotFoundError:
            print("[LLM-BO] No logbook found. Using zero-shot prompt.")
            param_space_str = ", ".join(
                f"{k} in {self.vocs.variables[k]['bounds']}" for k in self.vocs.variable_names
            )

            prompt = (
                f"You are optimizing control voltages to maximize beam flux.\n"
                f"The parameters are: {param_space_str}.\n"
                f"Too low values result in no beam. Extremely high voltages may degrade components.\n\n"
                f"Return format:\nvoltages: {{...}}, lengthscale: ..."
            )

        # Query the LLM
        response = self._llm_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
        print("[LLM-BO] LLM response:", response)

        # Parse response
        try:
            volts_str = response.split("voltages:")[1].split("lengthscale:")[0].strip()
            ls_str = response.split("lengthscale:")[1].strip().split()[0]

            voltages = eval(volts_str)
            lengthscale = float(ls_str)
            return voltages, lengthscale

        except Exception as e:
            print(f"[LLM-BO] Failed to parse LLM response: {e}")
            return self.good_tune or {}, self.custom_kernel_params.get("lengthscale", 0.1)

    def _reset_to_good_tune(self):
        print("[LLM-BO] Beam loss detected. Querying LLM for good tune from history.")
        good_tune_dict, lengthscale = self._query_llm_for_good_tune()
        self.good_tune = good_tune_dict
        self.custom_kernel_params["lengthscale"] = lengthscale
        self.data = pd.DataFrame([self.good_tune])
        self._good_tune_triggered = True

    def generate(self, n_candidates: int = 1) -> list[dict]:
        if self._detect_beam_loss():
            self._reset_to_good_tune()
        return super().generate(n_candidates)

    def train_model(self):
        if self.data is None or len(self.data) < 2:
            raise RuntimeError("Insufficient data to train model.")

        lengthscale = self.custom_kernel_params.get("lengthscale", 0.1)
        constructor = LLMGPModelConstructor(
            self.data, self.vocs, self.tkwargs, lengthscale=lengthscale
        )
        model, mll = constructor.build()
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll)
        return model

    def _get_acquisition(self, model):
        objective = self._get_objective()
        if self.n_candidates > 1 or isinstance(objective, CustomXoptObjective):
            sampler = self._get_sampler(model)
            return qUpperConfidenceBound(
                model,
                beta=self.beta,
                sampler=sampler,
                objective=objective
            )
        else:
            weights = set_botorch_weights(self.vocs).to(**self.tkwargs)
            return UpperConfidenceBound(
                model,
                beta=self.beta,
                posterior_transform=ScalarizedPosteriorTransform(weights)
            )

