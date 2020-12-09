"""Methods for transforming/decomposing reflectance data (e.g., using PCA)
"""
from sklearn.decomposition import PCA as _PCA
from . import read as _read


def pca(features, n_pcs=100):
    """PCA transformation function

    Args:
        features - the input feature data to transform
        n_pcs    - the number of components to keep after transformation

    Returns:
        an array of PCA-transformed features
    """
    reducer = _PCA(n_components=n_pcs, whiten=True)
    return reducer.fit_transform(features)


def from_path(path, features, n_features=None):
    """Transformation using a saved decomposition object

    Args:
        path       - the path to the saved decomposition object
        features   - the input feature data to transform
        n_features - the number of features to keep after transformation
    """
    # read the object and perform the transformation
    reducer = _read.pck(path)
    transformed = reducer.fit_transform(features)

    # ship the transformed data
    if n_features is None:
        return reducer, transformed
    else:
        return reducer, transformed[:, 0:n_features]
