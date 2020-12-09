"""Methods for formatted printing
"""
from sklearn import metrics as _metrics


def status(msg):
    print("[ STATUS ] {}".format(msg))


def error(msg):
    print("[ ERROR! ] {}".format(msg))


def line_break():
    print("[ ------ ]")


def model_report(ytrue, ypred, yprob):
    status("Mean accuracy score: {}".format(_metrics.accuracy_score(ytrue, ypred)))
    status("Mean log loss score: {}".format(_metrics.log_loss(ytrue, yprob)))
