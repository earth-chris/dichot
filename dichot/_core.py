import copy as _copy
import os as _os

import numpy as _np
from sklearn import calibration as _calibration
from sklearn import ensemble as _ensemble
from sklearn import utils as _utils

_path = _os.path.realpath(__file__)


def match_species_ids(crown_id, label_id, labels):
    """Matches the crown IDs from the training data with the labels associated with species IDs

    Args:
        crown_id - an array of crown IDs from the training data
        label_id - an array of crown IDs associated with the labeled data
        labels   - an array of labels (e.g., species names, species codes)
                   label_id and labels should be of the same size

    Returns:
        [unique_labels, unique_crowns, crown_labels]
        unique_labels - a list of the unique entities from the input labels variable
        unique_crowns - a list of the unique crown entities from the crown_id variable
        crown_labels  - an array with the labels aligned with the original shape of crown_id
    """
    # get the unique labels and crown id's
    unique_labels = _np.unique(labels)
    unique_crowns = _np.unique(crown_id)
    n_crowns = len(unique_crowns)

    # set up the output array
    nchar = _np.max([len(label) for label in unique_labels])
    crown_labels = _np.chararray(len(crown_id), itemsize=nchar)

    for i in range(n_crowns):
        index_crown = crown_id == unique_crowns[i]
        index_label = label_id == unique_crowns[i]
        crown_labels[index_crown] = labels[index_label]

    return [unique_labels, unique_crowns, crown_labels]


def get_sample_weights(y):
    """Calculates the balanced sample weights for a set of unique classes

    Args:
        y - the input class labels

    Returns:
        weights_sample - an array of length (y) with the per-class weights per sample
    """
    # get the unique classes in the array
    classes = _np.unique(y)
    n_classes = len(classes)

    # calculate the per-class weights
    weights_class = _utils.class_weight.compute_class_weight("balanced", classes, y)

    # create and return an array the same dimensions as the input y vector
    weights_sample = _np.zeros(len(y))
    for i in range(n_classes):
        ind_y = y == classes[i]
        weights_sample[ind_y] = weights_class[i]

    return weights_sample


# -----
# functions to handle the CCB-ID classification models
# -----
class model:
    def __init__(
        self,
        models=None,
        params=None,
        calibrator=None,
        run_calibration=None,
        average_proba=True,
        labels=None,
        good_bands=None,
        reducer=None,
    ):
        """Creates an object to build the CCB-ID models. Should approximate the functionality
        of the sklearn classifier modules, though not perfectly.

        Args:
            models          - a list containing the sklearn models for classification
                              (defaults to using gradient boosting and random forest classifiers)
            params          - a list of parameter values used for each model. This should be a list of length
                              n_models, with each item containing a dictionary with model-specific parameters
            calibrator      - an sklearn CalibratedClassifier object (or other calibration object)
            run_calibration - a boolean array with True values for models you want to calibrate,
                              and False values for models that do not require calibration
            average_proba   - flag to report the output probabilities as the average across models
            labels          - the species labels for each class
            good_bands      - a boolean array of good band values to store (but not used by this object)
            reducer         - the data reducer/transformer to apply to input data

        Returns:
            a CCB-ID model object with totally cool functions and attributes.
        """
        # set the base attributes for the model object
        if models is None:
            gbc = _ensemble.GradientBoostingClassifier()
            rfc = _ensemble.RandomForestClassifier()
            self.models_ = [gbc, rfc]
        else:
            # if a single model is passed, convert to a list so it is iterable
            if type(models) is not list:
                models = list(models)
            self.models_ = models

        # set an attribute with the number of models
        self.n_models_ = len(self.models_)

        # set the model parameters if specified
        if params is not None:
            for i in range(self.n_models_):
                self.models_[i].set_params(**params[i])

        # set the model calibration function
        if calibrator is None:
            self.calibrator = _calibration.CalibratedClassifierCV(
                method="sigmoid", cv=3
            )
        else:
            self.calibrator = calibrator

        # set the attribute determining whether to perform calibration on a per-model basis
        # if run_calibration is None:
        #    self.run_calibration_ = _np.repeat(True, self.n_models_)
        # else:
        #    self.run_calibration_ = run_calibration

        # set an attribute to hold the final calibrated models
        self.calibrated_models_ = _np.repeat(None, self.n_models_)

        # set the flag to average the probability outputs
        self.average_proba_ = average_proba

        # and set some properties that will be referenced later
        #  like species labels and a list of good bands
        if labels is None:
            self.labels_ = None
        else:
            self.labels_ = labels

        if good_bands is None:
            self.good_bands_ = None
        else:
            self.good_bands_ = good_bands

        if reducer is None:
            self.reducer = None
        else:
            self.reducer = reducer

        self.n_features_ = None
        self.is_calibrated_ = False

    def fit(self, x, y, sample_weight=None):
        """Fits each classification model

        Args:
            x             - the training features
            y             - the training labels
            sample_weight - the per-sample training weights

        Returns:
            None. Updates each item in self.models_
        """
        for i in range(self.n_models_):
            self.models_[i].fit(x, y, sample_weight=sample_weight)

        # have this function update the species labels if not already set
        if self.labels_ is None:
            labels = []
            y_unique = _np.unique(y)
            for unique in y_unique:
                labels.append("SP-{}".format(unique))
            self.labels_ = labels

    def calibrate(self, x, y, run_calibration=None):
        """Calibrates the probabilities for each classification model

        Args:
            x               - the probability calibration features
            y               - the probability calibration labels
            run_calibration - a boolean array with length n_models specifying
                              True for each model to calibrate

        Returns:
            None. Updates each item in self.calibrated_models_
        """
        for i in range(self.n_models_):
            # if self.run_calibration_[i] or run_calibration[i]:
            self.calibrator.set_params(base_estimator=self.models_[i])
            self.calibrator.fit(x, y)
            self.calibrated_models_[i] = _copy.copy(self.calibrator)
            # else:
            #    self.calibrated_models_[i] = _copy.copy(self.models_[i])

        self.is_calibrated_ = True

    def tune(self, x, y, param_grids, criterion):
        pass

    def predict(self, x, use_calibrated=False):
        """Predict the class labels for given feature data

        Args:
            x              - the input features
            use_calibrated - boolean for whether to use the calibrated model for
                             calculating predictions

        Returns:
            output         - an array with the predicted class labels
        """
        for i in range(self.n_models_):
            if use_calibrated:
                predicted = self.calibrated_models_[i].predict(x)
            else:
                predicted = self.models_[i].predict(x)

            if i == 0:
                output = _np.expand_dims(predicted, 1)
            else:
                output = _np.append(output, _np.expand_dims(predicted, 1), axis=1)

        return output

    def predict_proba(self, x, use_calibrated=False, average_proba=None):
        """Predict the probabilities for each class label

        Args:
            x              - the input features
            use_calibrated - boolean for whether to use the calibrated model for
                             calculating predictions
            average_proba  - flag to report the output probabilities as the average across models
        """
        if average_proba:
            self.average_proba_ = True

        for i in range(self.n_models_):
            if use_calibrated:
                predicted = self.calibrated_models_[i].predict_proba(x)
            else:
                predicted = self.models_[i].predict_proba(x)

            if i == 0:
                output = _np.expand_dims(predicted, 2)
            else:
                output = _np.append(output, _np.expand_dims(predicted, 2), axis=2)

        # average the final probabilities, if set
        if self.average_proba_:
            return output.mean(axis=2)
        else:
            return output

    def set_params(self, params):
        """Sets the parameters for each model

        Args:
            params - a list of parameter dictionaries to set for each model

        Returns:
            None. Updates each item in self.models_
        """
        for i in range(self.n_models_):
            self.models_[i].set_params(**params[i])
