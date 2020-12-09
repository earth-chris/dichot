"""Helper functions to write CCB-ID output data
"""
import pickle as _pickle


class predictions:
    def __init__(self):
        pass

    @staticmethod
    def to_csv(path, predictions, crown_ids, species_ids):
        pass

    @staticmethod
    def to_raster(path, predictions, gdal_params):
        pass


def pck(path, variable):
    """Writes a python/pickle format data file

    Args:
       path     - the path to the output pickle file
       variable - the python variable to write

    Returns:
       None
    """
    with open(path, "wb") as f:
        _pickle.dump(variable, f)
