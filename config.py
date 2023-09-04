from munch import Munch
import numpy as np

NB_VARIABLES = 11
TIMESTEPS = 500
NUM_CLASSES = 2
MALSTM_DATA_DIR = "dataset/swdd-7k_npz_500_{}".format(TIMESTEPS)
SKTIME_DATA_DIR = "dataset/swdd-7k_ts_origin_500_{}".format(TIMESTEPS)
SAVE_DIR = "results/swdd-7k_model_500_{}".format(TIMESTEPS)

lab_config = Munch(
    {
        "model_save_directory": SAVE_DIR,
        "malstm": {
            "data_dir": MALSTM_DATA_DIR,
            "model_config": {
                "alstm_units": 8,
                "dropout": 0.8,
                "filters": [128, 256, 128],
                "kernel_sizes": [8, 5, 3],
                "padding": "same",
                "kernel_initializer": "he_uniform",
                "activation": "relu",
                "num_classes": NUM_CLASSES,
                "input_shape": (NB_VARIABLES, TIMESTEPS),  # TIMESTEPS
            },
            "train_config": {
                "batch_size": 128,
                "epochs": 100,
                "learning_rate": 1e-3,
                "callback_config": {
                    "reduce_lr": {
                        "monitor": "val_loss",  # "loss",
                        "patience": 100,
                        "mode": "auto",
                        "factor": 1.0 / np.cbrt(2),
                        "min_lr": 1e-4,
                    },
                },
            },
            "model_file": "malstm_fcn_7k.keras",
        },
        "sktime_dl": {
            "data_dir": SKTIME_DATA_DIR,
            "network_config": {
                "mcnn": {
                    "kernel_size": 7,
                    "avg_pool_size": 3,
                    "nb_conv_layers": 2,
                    "filter_sizes": [6, 12],
                },
                "fcn": {},
                "mcdcnn": {
                    "kernel_size": 5,
                    "pool_size": 2,
                    "filter_sizes": [8, 8],
                    "dense_units": 732,
                },
                "twiesn": {},
                "inception": {
                    "nb_filters": 32,
                    "use_residual": True,
                    "use_bottleneck": True,
                    "bottleneck_size": 32,
                    "depth": 6,
                    "kernel_size": 41 - 1,
                },
            },
            "train_config": {
                "mcnn": {
                    "batch_size": 32,  # 16,
                    "nb_epochs": 200,  # 200
                    "verbose": True,
                    "random_state": 0,  # 在comparison实验中需统一
                    "model_save_directory": SAVE_DIR,
                    "model_name": "cnn-7k",
                },
                "fcn": {
                    "nb_epochs": 200,  # 200, # 2000,
                    "batch_size": 64,  # 16,
                    "verbose": True,
                    "random_state": 0,
                    "model_name": "fcn-7k",
                    "model_save_directory": SAVE_DIR,
                },
                "inception": {
                    "nb_epochs": 500,  # 500 # 1500,
                    "batch_size": 64,
                    "verbose": True,
                    "random_state": 0,
                    "model_name": "inception-7k",
                    "model_save_directory": SAVE_DIR,
                },
                "mcdcnn": {
                    "nb_epochs": 120,  # 120
                    "batch_size": 16,
                    "verbose": True,
                    "random_state": 0,
                    "model_name": "mcdcnn-7k",
                    "model_save_directory": SAVE_DIR,
                },
                "twiesn": {
                    "rho_s": [0.55, 0.9, 2.0, 5.0],
                    "alpha": 0.1,  # leaky rate
                    "verbose": True,
                    "random_state": 0,
                    "model_name": "twiesn-7k",
                    "model_save_directory": SAVE_DIR,
                },
            },
        },
    }
)
