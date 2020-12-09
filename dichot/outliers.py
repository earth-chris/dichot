"""Methods for outlier identification in refl data
"""
import numpy as _np
from sklearn.decomposition import PCA as _PCA


def with_pca(features, n_pcs=20, thresh=3):
    """PCA-based outlier removal function

    Args:
        features - the input feature data for finding outliers
        n_pcs    - the number of principal components to look for outliers in
        thresh   - the standard-deviation multiplier for outlier id
                   (e.g. thresh = 3 means values > 3 stdv from the mean will
                   be flagged as outliers)

    Returns:
        mask     - a boolean array with True for good values, False for outliers
    """
    # create the bool mask where outlier samples will be flagged as False
    mask = _np.repeat(True, features.shape[0])

    # set up the pca reducer, then transform the data
    reducer = _PCA(n_components=n_pcs, whiten=True)
    transformed = reducer.fit_transform(features)

    # loop through the number of pcs set and flag values outside the threshold
    for i in range(n_pcs):
        outliers = abs(transformed[:, i]) > thresh
        mask[outliers] = False

    return mask
