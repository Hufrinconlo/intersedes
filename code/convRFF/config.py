import tensorflow as tf
from gcpds.loss.dice import DiceCoefficient
from gcpds.metrics.jaccard import Jaccard 
from gcpds.metrics.sensitivity import Sensitivity
from gcpds.metrics.specificity import Specificity
from gcpds.metrics.dice import DiceCoefficientMetric

from gcpds.models.baseline_fcn import fcn_baseline
from gcpds.models.baseline_unet import unet_baseline
from gcpds.models.baseline_res_unet import res_unet_baseline
from convRFF.models.fcn.b_skips import get_model as fcn_b_skips
from convRFF.models.fcn.rff_skips import get_model as fcn_rff_skips
from convRFF.models.res_unet.b_skips import get_model as res_unet_b_skips
from convRFF.models.res_unet.rff_skips import get_model as res_unet_rff_skips
from convRFF.models.unet.b_skips import get_model as unet_b_skips
from convRFF.models.unet.rff_skips import get_model as unet_rff_skips

from convRFF.data import get_data
from wandb.keras import WandbCallback

def compile_parameters():
  return {'loss':DiceCoefficient(),
          'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3),
          'metrics':[Jaccard(),
                     Jaccard(name='Jaccard_0',target_class=0),
                     Jaccard(name='Jaccard_1',target_class=1),
                     Jaccard(name='Jaccard_2',target_class=2),
                     Sensitivity(),
                     Sensitivity(name='Sensitivity_0',target_class=0),
                     Sensitivity(name='Sensitivity_1',target_class=1),
                     Sensitivity(name='Sensitivity_2',target_class=2),
                     Specificity(),
                     Specificity(name='Specificity_0',target_class=0),
                     Specificity(name='Specificity_1',target_class=1),
                     Specificity(name='Specificity_2',target_class=2),
                     DiceCoefficientMetric(),
                     DiceCoefficientMetric(name='DiceCoeficienteMetric_0',target_class=0),
                     DiceCoefficientMetric(name='DiceCoeficienteMetric_1',target_class=1),
                     DiceCoefficientMetric(name='DiceCoeficienteMetric_2',target_class=2),
                     ]
  }


def get_model(model, input_shape, out_channels, kernel_regularizer):
    MODELS = {'u_b': unet_baseline(input_shape=input_shape, out_channels=out_channels),
              'u_b_s': unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'u_r_s': unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'u_r_s_m1': unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1),
              'u_b_s_m3': unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'f_b': fcn_baseline(input_shape=input_shape, out_channels=out_channels),
              'r_b': res_unet_baseline(input_shape=input_shape, out_channels=out_channels),
              'f_b_s': fcn_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'f_r_s': fcn_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'f_r_s_m1': fcn_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1),
              'f_b_s_m3': fcn_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'r_b_s': res_unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'r_b_s_m3': res_unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'r_r_s': res_unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'r_r_s_m1': res_unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1)
              }
    return MODELS[model]

def gcp():
  return {'loss':DiceCoefficient(),
          'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3),
          'metrics':[Jaccard(),
                     Sensitivity(),
                     Specificity(),
                     DiceCoefficientMetric()
                     ]
  }

def gtp(dataset_class, data_augmentation=True,validation=True, **kwargs_data_augmentation):
    train_data, val_data, test_data = get_data(dataset_class, data_augmentation=data_augmentation, **kwargs_data_augmentation)
    params = {'x':train_data,
            'validation_data':val_data,
            'epochs':200,
            'callbacks':[WandbCallback(save_model=True)],}
    if not validation:
      params.pop('validation_data')
    return params

def get_t_p(*args,**kwargs):
    f = gtp(*args,**kwargs)
    f.update({'epochs':100})
    return f

def get_c_p(*args,**kwargs):
    f = gcp(*args,**kwargs)
    f.update({'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3)})
    return f

