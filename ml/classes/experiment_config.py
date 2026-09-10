from dataclasses import dataclass

from pathlib import Path
from classes.pca_model import PCAModel
import pandas as pd
import numpy as np

from get_api_profile import get_api_profile, load_api_properties

@dataclass
class ExperimentConfig:
    models: list
    X_columns: pd.Index
    api_db_path: Path
    api_name: str = "Azithromycin dihydrate" #Change this to the API you want to optimize for
    api_profile: dict = None

    # API-to-lipid ratio constraints in log space
    api_ratio_min: float = 1e-3
    api_ratio_max: float = 1.0

    # split_mode:
    #     - within_api: split data within the same API, each API is split into train and test sets
    #     - api: split data across different APIs, whole APIs are either in train or test set
    #     - random: random split of the entire dataset
    split_mode: str = "within_api"

    n_models: int = 5

    # acquisition_mode:
    #     - exploitation: mostly prioritize high predicted EE
    #     - balanced: balance predicted EE, uncertainty, and novelty
    #     - exploration: strongly prioritize uncertainty and novelty
    
    acquisition_mode: str = "balanced"

    beta: float = None # Weight of uncertainty
    gamma: float = None # Weight of novelty

    # Diversity selection:
    #     - score_weight: importance of acquisition score
    #     - diversity_weight: importance of distance between selected formulations

    score_weight: float = 0.7
    diversity_weight: float = 0.3

    lipid_selection_mode: str = "PCA"  # or "RANDOM"
    pca_model: PCAModel = None
    n_pca_components: int = 3

    n_candidates: int = 5000
    n_formulation_trials: int = 1000 # Increase this value for more thorough optimization
    n_suggestions: int = 5 #Change this value to switch between batch BO (n>1) and single BO (n=1)

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

        if self.api_ratio_min <= 0:
            raise ValueError("api_ratio_min must be greater than 0")

        if self.api_ratio_max <= self.api_ratio_min:
            raise ValueError("api_ratio_max must be greater than api_ratio_min")

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

        # -----------------------
        # Diversity selection
        # -----------------------

        if not np.isclose(
            self.score_weight + self.diversity_weight,
            1.0
        ):
            raise ValueError(
                "score_weight and diversity_weight must sum to 1."
            )