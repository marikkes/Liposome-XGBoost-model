from dataclasses import dataclass

from pathlib import Path
from classes.pca_model import PCAModel
import pandas as pd

from get_api_profile import get_api_profile, load_api_properties

@dataclass
class ExperimentConfig:
    models: list
    X_columns: pd.Index
    api_db_path: Path
    api_name: str = "Micrococcin P1" #Change this to the API you want to optimize for
    api_profile: dict = None

    split_mode: str = "within_api"

    n_models: int = 5

    # mode:
    #     - exploitation: prioritize high predicted EE
    #     - balanced: balance predicted EE and uncertainty
    #     - exploration: balance predicted EE, uncertainty, and novelty
    
    acquisition_mode: str = "balanced"

    lipid_selection_mode: str = "PCA"  # or "RANDOM"
    pca_model: PCAModel = None
    n_pca_components: int = 3

    beta: float = None
    gamma: float = None

    n_candidates: int = 5000
    n_formulation_trials: int = 500 # Increase this number for more thorough optimization
    n_suggestions: int = 5

    def __post_init__(self):

        # -----------------------
        # Load API profile
        # -----------------------

        if self.api_profile is None:

            if self.api_db_path is None:
                raise ValueError(
                    "api_db_path must be provided when api_profile is not supplied"
                )

            api_df = load_api_properties(
                self.api_db_path
            )

            self.api_profile = get_api_profile(
                self.api_name,
                api_df
            )

        # -----------------------
        # Acquisition settings
        # -----------------------

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