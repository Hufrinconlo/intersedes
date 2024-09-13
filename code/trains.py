import wandb
from convRFF.config import get_model, get_c_p, get_t_p
from convRFF.train import train as train_model
from gcpds.DataSet.infrared_thermal_feet import InfraredThermalFeet

kwargs_data_augmentation = dict(repeat=7,
                                   flip_left_right=True,
                                   flip_up_down=False,
                                   range_rotate=(-15,15),
                                   translation_h_w=(0.10,0.10),
                                   zoom_h_w=(0.15,0.15),
                                   batch_size=16,
                                   shape=224,
                                    split=[0.1,0.1]
                                   )
kernel_regularizer =None

get_compile_parameters = get_c_p
get_train_parameters = get_t_p
out_channels = 1
input_shape = 224,224,1

sweep_config = {
    'method':'grid'
}

parameters_dict = {
    'model': {
        'values':['f_b']#,'r_b','u_b','f_b_s_m3','r_b_s_m3','u_b_s_m3', 'f_b_s', 'r_b_s','u_b_s']
        },
    'dataset':{
        'values':['infrared_thermal_feet']
    }
    }
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='weak_supervised_learning')

def train(config=None):
    with wandb.init(config=config) as run:
        config= dict(run.config)
        model = config['model']
        name_dataset = config['dataset']
        data_augmentation = False

        dataset_class = InfraredThermalFeet
        model = get_model(model, input_shape, out_channels, kernel_regularizer)
        train_model(model, dataset_class, run,  data_augmentation=data_augmentation,
                                                get_compile_parameters=get_compile_parameters,
                                                get_train_parameters=get_train_parameters,
                                                **kwargs_data_augmentation)
import os
os.environ["WANDB_API_KEY"] = "97361c7ad33016f07e43eabc7bc22555c00299c8"
SWEEP_ID = sweep_id
wandb.agent(SWEEP_ID, train, count=200, project='weak_supervised_learning')