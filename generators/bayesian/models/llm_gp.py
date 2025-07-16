# FILE: xopt/generators/bayesian/models/llm_gp.py

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class LLMGPModelConstructor:
    def __init__(self, data, vocs, tkwargs, lengthscale=0.1):
        self.data = data
        self.vocs = vocs
        self.tkwargs = tkwargs
        self.lengthscale = lengthscale

    def build(self):
        X = torch.tensor(
            self.data[self.vocs.variable_names].to_numpy(), **self.tkwargs
        )
        Y = torch.tensor(
            self.data[self.vocs.output_names[0]].to_numpy().reshape(-1, 1),
            **self.tkwargs
        )

        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            outcome_transform=Standardize(m=1)
        )

        # Inject LLM-suggested lengthscale into the RBF kernel
        model.covar_module.base_kernel.lengthscale = self.lengthscale
        model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return model, mll
