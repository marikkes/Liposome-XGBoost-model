import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def create_split(X, y, groups, split_mode="within_api"):

    if split_mode == "random":

        train_idx, test_idx = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    elif split_mode == "api":
        splitter = GroupShuffleSplit(
        test_size=0.2,
        random_state=42
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

    elif split_mode == "within_api":
        rng = np.random.RandomState(42)

        test_idx = []

        # Minimum number of samples an API needs to contribute to test set
        min_samples = 4
        # Fraction of samples to put in test set for each API that meets the minimum requirement
        test_fraction = 0.3

        unique_apis = groups.unique()

        for api in unique_apis:

            api_idx = np.where(groups == api)[0]

            if len(api_idx) < min_samples:
                continue

            n_test = max(
                1,
                int(np.ceil(test_fraction * len(api_idx)))
            )

            api_test_idx = rng.choice(
                api_idx,
                size=n_test,
                replace=False
            )

            test_idx.extend(api_test_idx)


        test_idx = np.array(test_idx)

        train_idx = np.setdiff1d(
            np.arange(len(X)),
            test_idx
        )

    else:
        raise ValueError(
            f"Unknown split mode: {split_mode}"
        )


    return train_idx, test_idx