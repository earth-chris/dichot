"""Methods for resampling feature data prior to classification
"""
import numpy as _np


def uniform(features, crown_labels, n_per_class=400, other_array=None):
    """Performs a random uniform resampling of each class to a fixed number of samples

    Args:
        features     - the feature data to evenly resample
        crown_labels - labels that correspond to each sample in the feature
                       data and define the unique IDs to resample from
        n_per_class  - the number of samples to select per class

    Returns:
        list of [resample_x, resample_y]
        resample_x   - the feature data resampled with shape (n_lables * n_per_class, n_features)
        resample_y   - the class labels from 0 to n_unique_labels with shape (n_lables * n_per_class)
    """
    # get the unique species labels for balanced-class resampling
    unique_labels = _np.unique(crown_labels)
    n_labels = len(unique_labels)

    # set up the x and y variables for storing outputs
    resample_x = _np.zeros((n_labels * n_per_class, features.shape[1]))
    resample_y = _np.zeros(n_labels * n_per_class, dtype=_np.uint8)

    if other_array is not None:
        if other_array.ndim == 1:
            resample_o = _np.zeros(n_labels * n_per_class)
        else:
            resample_o = _np.zeros((n_labels * n_per_class, other_array.shape[1]))

    # loop through and randomly sample each species
    for i in range(n_labels):
        ind_class = _np.where(crown_labels == unique_labels[i])
        ind_randm = _np.random.randint(0, high=ind_class[0].shape[0], size=n_per_class)

        # assign the random samples to the balanced class outputs
        resample_x[i * n_per_class : (i + 1) * n_per_class] = features[
            ind_class[0][ind_randm]
        ]
        resample_y[i * n_per_class : (i + 1) * n_per_class] = i

        if other_array is not None:
            resample_o[i * n_per_class : (i + 1) * n_per_class] = other_array[
                ind_class[0][ind_randm]
            ]

    if other_array is None:
        return [resample_x, resample_y]
    else:
        return [resample_x, resample_y, resample_o]
