from tensorflow import keras
from keras.models import load_model as ld
from convRFF.models.layers import ConvRFF
from gcpds.loss.dice import DiceCoefficient
from gcpds.metrics.jaccard import Jaccard 
from gcpds.metrics.sensitivity import Sensitivity
from gcpds.metrics.specificity import Specificity
from gcpds.metrics.dice import DiceCoefficientMetric

def load_model(path='model.h5'):
    model = ld(path, custom_objects={'ConvRFF':ConvRFF, 
                                     'DiceCoefficient':DiceCoefficient,
                                     'DiceCoeficiente':DiceCoefficient,
                                     'Jaccard':Jaccard, 
                                     'Sensitivity':Sensitivity,
                                     'Specificity':Specificity,
                                     'DiceCoefficientMetric':DiceCoefficientMetric,
                                     'DiceCoeficienteMetric':DiceCoefficientMetric
                                     }
                                     )
    return model 

if __name__ == "__main__":
    import os 
    print(os.getcwd())
    model = load_model('./model.h5')
    model.summary()