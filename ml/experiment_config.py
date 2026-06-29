from dataclasses import dataclass
import pandas as pd

@dataclass
class ExperimentConfig:
    models: list
    X_columns: pd.Index
    api_profile: dict
    api_name: str

    split_mode: str = "within_api"

    n_models: int = 5

    # mode:
    #     - exploitation: prioritize high predicted EE
    #     - balanced: balance predicted EE and uncertainty
    #     - exploration: balance predicted EE, uncertainty, and novelty
    
    acquisition_mode: str = "balanced"

    beta: float = None
    gamma: float = None

    n_candidates: int = 5000
    n_formulation_trials: int = 500 # Increase this number for more thorough optimization
    n_suggestions: int = 5

    def __post_init__(self):

        settings = {
            "exploitation": (0.2, 0.1),
            "balanced": (0.8, 0.4),
            "exploration": (1.5, 1.0),
        }

        if self.acquisition_mode not in settings:
            raise ValueError(
                f"Unknown acquisition mode: {self.acquisition_mode}. "
                f"Choose from {list(settings.keys())}"
            )

        if self.beta is None:
            self.beta = settings[self.acquisition_mode][0]

        if self.gamma is None:
            self.gamma = settings[self.acquisition_mode][1]