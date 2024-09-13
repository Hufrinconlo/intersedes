from sklearn.metrics import f1_score
import numpy as np


def compute_dice(y_pred, y_true, T=0.5):
    Dice_score = 0
    y_pred = y_pred.reshape(
        y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3], 1)
    y_true = y_true.reshape(
        y_true.shape[0]*y_true.shape[1]*y_true.shape[2]*y_true.shape[3], 1)
    y_pred = np.where(y_pred > T, 1., 0)
    y_true = np.where(y_true > 0.5, 1., 0)
    # In binary case F1 is equall to Dice Score (https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
    Dice_score = f1_score(y_true, y_pred, labels=None,
                          average='binary', sample_weight=None)

    return Dice_score
