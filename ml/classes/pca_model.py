from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import numpy as np

@dataclass
class PCAModel:
    lipid_names: np.ndarray
    lipid_scores: np.ndarray
    pca: PCA
    nn: NearestNeighbors