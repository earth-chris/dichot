"""Methods for ensembling pixel-wise classifications to the crown scale
"""
import numpy as _np


# a function to average probabilities by crown id
def average(predictions, id_labels, sp_labels):
    """Sets the output labels for prediction probabilities by id (e.g., by crown) and by species.

    Args:
        id_labels - the labels (usually, crown labels) that probabilities are aggregated to
        sp_labels - the species labels

    Returns:
        output_pr - the averaged prediction probabilities
    """
    # create the output array to store the results
    id_unique = _np.unique(id_labels)
    sp_unique = _np.unique(sp_labels)
    n_id = len(id_unique)
    n_sp = len(sp_unique)
    output_pr = _np.zeros(n_id * n_sp)

    # loop through each crown, calculate the average probability per crown, and write it to the array
    for i in range(n_id):
        id_index = id_labels == id_unique[i]
        output_pr[i * n_sp : (i + 1) * n_sp] = predictions[id_index].mean(axis=0)

    return output_pr


# a function to reconcile the crown and species labels for csv output
def get_csv_labels(id_labels, sp_labels):
    """Sets the output labels for prediction probabilities by id (e.g., by crown) and by species.

    Args:
        id_labels - the labels (usually, crown labels) that probabilities are aggregated to
        sp_labels - the species labels

    Returns:
        id_rows, sp_rows - the csv ordered id and species labels
    """
    # get the unique id and species labels
    id_unique = _np.unique(id_labels)
    sp_unique = _np.unique(sp_labels)
    n_id = len(id_unique)
    n_sp = len(sp_unique)

    id_rows = _np.repeat(id_unique, n_sp)
    sp_rows = _np.repeat(sp_unique, n_id).reshape(n_sp, n_id).flatten(order="F")

    return id_rows, sp_rows
