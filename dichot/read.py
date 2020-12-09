"""A series of functions for reading CCB-ID formatted data
"""
import os as _os
import pickle as _pickle
from osgeo import gdal as _gdal
import numpy as _np
import pandas as _pd

_gdal.UseExceptions()


def bands(path):
    """Reads the wavelengths and good data bands from a csv file
    (based on ccb-id/suport_files/neon-bands.csv')

    Args:
        path        - the path to the wavelength file

    Returns:
        list of [wavelengths, good_bands]
        wavelengths - an array with the wavelengths associated with each band
        good_bands  - a boolean array for bands to include in analysis (True = good)
    """
    df = _pd.read_csv(path)
    wavelengths = _np.array(df["Wavelength"])
    flag = _np.array(df["Flag"])
    good_bands = flag == 1

    return [wavelengths, good_bands]


def species_id(path):
    """Reads the species identities from a csv file
    (based on ccb-id/suport_files/species_id.csv)

    Args:
        path         - the path to the species id file

    Returns:
        list of [crown_id, species_id, species_name]
        crown_id     - an array with the unique crown identities
        species_id   - an array with the species ID codes per crown
        species_name - an array with the species names per crown
    """
    df = _pd.read_csv(path)
    cr_id = _np.array(df["crown_id"])
    sp_id = _np.array(df["species_id"])
    sp_name = _np.array(df["species"])

    return [cr_id, sp_id, sp_name]


def genus_id(path):
    """Reads the genus identities from a csv file
    (based on ccb-id/suport_files/species_id.csv)

    Args:
        path       - the path to the genus id file

    Returns:
        list of [crown_id, genus_id, genus_name]
        crown_id   - an array with the unique crown identities
        genus_id   - an array with the genus ID codes per crown
        genus_name - an array with the genus names per crown
    """
    df = _pd.read_csv(path)
    cr_id = _np.array(df["crown_id"])
    ge_id = _np.array(df["genus_id"])
    ge_name = _np.array(df["genus"])

    return [cr_id, ge_id, ge_name]


def training_data(path):
    """Reads the input training data from a csv file
    (based on ccb-id/support_files/training.csv)

    Args:
        path     - the path to the training data csv file

    Returns:
        list of [crown_id, features]
        crown_id - an array of per-sample crown IDs
        features - an array of input feature data with shape (n_samples, n_features)
    """
    df = _pd.read_csv(path)
    crown_id = _np.array(df.iloc[:, 0])
    features = _np.array(df.iloc[:, 1:])

    return [crown_id, features]


def is_raster(path):
    """Tests if a file is a raster (i.e., gdal readable)

    Args:
        path     - the path to the file to check

    Returns:
        True if it is a raster, False if not.
    """
    # its a dang raster if GDAL can open it
    try:
        _gdal.Open(path)
        return True
    except RuntimeError:
        return False


def is_csv(path):
    """Tests if a file is a CSV (i.e., pandas readable)

    Args:
        path     - the path to the file to check

    Returns:
        True if it is a csv, False if not.
    """
    # check file ending
    ext = _os.path.splitext(path)[1]
    if ext.lower() in [".csv", ".tsv"]:
        return True
    else:
        return False


def pck(path):
    """Reads a python/pickle format data file

    Args:
        path - the path to the input pickle file

    Returns:
        the object stored in the pickle file
    """
    with open(path, "r") as f:
        return _pickle.load(f)


class raster:
    def __init__(self, input_file):
        """Reads metadata from a raster file and stores it in an object

        Args:
            input_file: a path to a raster file to read

        Returns:
            An object with the raster metadata as object variables (e.g.,
            ras = ccbid.read.raster('file.tif') # file with dims x = 30, y = 50
            ras.nx will be 30, ras.ny will be 50, etc.)
        """

        # read the gdal reference as read-only
        ref = _gdal.Open(input_file, 0)
        self.file_name = input_file

        # get file dimensions
        self.nx = ref.RasterXSize
        self.ny = ref.RasterYSize
        self.nb = ref.RasterCount

        # get georeferencing info
        self.prj = ref.GetProjection()
        geo = ref.GetGeoTransform()
        self.xmin = geo[0]
        self.xps = geo[1]
        self.xoff = geo[2]
        self.ymax = geo[3]
        self.yoff = geo[4]
        self.yps = geo[5]
        self.xmax = self.xmin + self.xoff + (self.nx * self.xps)
        self.ymin = self.ymax + self.yoff + (self.ny * self.yps)

        # get no-data info
        band = ref.GetRasterBand(1)
        self.no_data = band.GetNoDataValue()

        # get data type
        self.dt = band.DataType

        # create an empty 'data' variable to read into later
        self.data = None

        # get driver info
        self.driver_name = ref.GetDriver().ShortName

        # kill the gdal references
        ref = None
        band = None

    # a function to read raster data from a single band
    def read_band(self, band):
        """Reads the raster data from a user-specified band into the self.data variable

        Args:
            band: the 1-based index for the band to read

        Returns:
            the aei.Raster object with the object.data variable updated with a
            numpy array of raster values
        """
        ref = _gdal.Open(self.file_name, 0)
        band = ref.GetRasterBand(band)
        self.data = band.ReadAsArray()

    # a function to read raster data from all bands
    def read_all(self):
        """Reads all bands of raster data

        Args:

        Returns:
            the aei.raster object with the object.data variable updated with a
            numpy array of raster values
        """
        ref = _gdal.Open(self.file_name, 0)
        self.data = ref.ReadAsArray()

    # a function to write raster data to a single band
    def write_band(self, band, data):
        """Writes new raster data to a user-specified band

        Args:
            band: a 1-based integer with the band to write to
            data: a numpy array with raster data to write

        Returns:
            None.
        """
        ref = _gdal.Open(self.file_name, 1)
        band = ref.GetRasterBand(band)
        band.WriteArray(data)

    # a function to write raster data to all bands
    def write_all(self, data=None):
        """Writes new raster data to all bands

        Args:
            data: a numpy array with raster data to write. if not set, writes self.data

        Returns:
            None.
        """
        ref = _gdal.Open(self.file_name, 1)

        # see which data to write
        if data:
            ref.WriteRaster(data)
        else:
            ref.WriteRaster(self.data)

    def write_metadata(self, ref=None):
        """Updates the metadata of a file if changed by the user

        Args:
            ref: the gdal file reference object (a la ref = gdal.Open('some_file.tif', 1)

        Returns:
            None
        """
        if ref is None:
            ref = _gdal.Open(self.file_name, 1)

        # update projection and geotransform at file-level
        ref.SetProjection(self.prj)
        ref.SetGeoTransform(
            [self.xmin, self.xps, self.xoff, self.ymax, self.yoff, self.yps]
        )

        # update no-data value band by band
        if self.no_data is not None:
            for band in range(1, self.nb + 1):
                b = ref.GetRasterBand(band)
                b.SetNoDataValue(self.no_data)

    def copy(self, file_name, nb=None, driver=None, dt=None, options=None):
        """Creates a new raster file and object with a copy of the raster metadata that can be
        used to write a new derived data product.

        Args:
            file_name: the name of the new file to create as a new reference
            nb       : the number of bands for the new file
            driver   : the name of the driver to use. default is the driver of the input reference
            dt       : the output data type
            options  : gdal raster creation options, passed as a list

        Returns:
            a new aei.raster object with updated properties, and a new gdal file
            with those properties written to its header. no raster data is written to
            this file, only metadata.
        """
        import copy as cp

        # create a copy of the input object to manipulate
        new_obj = cp.copy(self)

        # update with the new parameters
        new_obj.file_name = file_name

        if nb:
            new_obj.nb = nb

        if driver:
            new_obj.driver_name = driver

        if dt:
            new_obj.dt = dt

        # create a new raster file with these parameters, but don't write any data
        ref = _gdal.GetDriverByName(new_obj.driver_name).Create(
            new_obj.file_name,
            new_obj.nx,
            new_obj.ny,
            new_obj.nb,
            new_obj.dt,
            options=options,
        )

        # set the projection and geotransform parameters
        new_obj.write_metadata(ref=ref)

        # kill the reference and return the new object
        ref = None

        return new_obj
