import copy
from utils.analysis import calc_metrics_binary

# from types import new_class
import tensorflow as tf
import keras
import gc
from config import lab_config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = lab_config.sktime_dl


def setNetwork(cls, network_config=None, train_config=None):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif.
    :param cls: String indicating which classifier you want
    :return: A classifier
    """
    from sktime_dl.deeplearning import FCNClassifier
    from sktime_dl.deeplearning import MCDCNNClassifier
    from sktime_dl.deeplearning import CNNClassifier
    from sktime_dl.deeplearning import TWIESNClassifier
    from sktime_dl.deeplearning import InceptionTimeClassifier
    from pathlib import Path
    import os

    # fold = train_config["random_state"]
    model_save_dir = train_config.get("model_save_directory", "")
    # model_name = cls + "_" + str(fold)
    # train_config["model_name"] = model_name

    if model_save_dir:
        try:
            os.makedirs(model_save_dir)
        except os.error:
            pass

    # fold = int(fold)
    cls = cls.lower()
    if cls == "mcnn":
        return CNNClassifier(**network_config, **train_config)
    elif cls == "fcn":
        return FCNClassifier(**network_config, **train_config)
    elif cls == "mcdcnn":
        return MCDCNNClassifier(**network_config, **train_config)
    elif cls == "twiesn":
        train_cfg_copy = {
            k: v for k, v in train_config.items() if k != "model_save_directory"
        }
        return TWIESNClassifier(**network_config, **train_cfg_copy)
    elif cls == "inception":
        return InceptionTimeClassifier(**network_config, **train_config)
    else:
        raise Exception("UNKNOWN CLASSIFIER: " + cls)


def read_dataset(data_dir):
    import os
    import numpy as np
    from sktime.utils.data_io import load_from_tsfile_to_dataframe

    X_train, y_train = load_from_tsfile_to_dataframe(os.path.join(data_dir, "train.ts"))
    X_test, y_test = load_from_tsfile_to_dataframe(os.path.join(data_dir, "test.ts"))

    from sklearn.model_selection import train_test_split

    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
        random_state=2022,
        stratify=y_test,
    )
    print(
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
        X_val.shape,
        y_val.shape,
    )
    print(np.unique(y_train))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


model_list = [
    "fcn",
    "mcnn",
    "mcdcnn",
    "twiesn",  # sklearn Ridge object has no attribute save. 明明是个sklearn，却要用keras.save。。
    "inception",  # 作者居然说这是strongest??
]


if __name__ == "__main__":
    flag = 1
    for r in range(10):
        data_dir = "dataset/swdd-7k_ts_origin_500_0"
        # load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = read_dataset(
            data_dir=data_dir
        )
        for i in range(50, 501, 50):
            # TODO:
            # if flag and i < 500:
            #     continue
            # data_dir = "swdd-7k_ts_500_{}".format(i)
            save_dir = "results/swdd-7k_model_500_{}_simple".format(i)

            print(data_dir, save_dir)

            for cls in model_list:
                # TODO:
                # if flag and cls != "tlenet":
                #     continue
                if flag:
                    flag = 0
                import os

                train_cfg = config["train_config"][cls]
                network_cfg = config["network_config"][cls]
                train_cfg["model_save_directory"] = save_dir

                # build network
                network = setNetwork(
                    cls, network_config=network_cfg, train_config=train_cfg
                )

                # train
                network.fit(
                    X_train[:i],
                    y_train[:i],
                    validation_X=X_val[:i],
                    validation_y=y_val[:i],
                )

                # # analyze model
                report = calc_metrics_binary(network, X_test[:i], y_test[:i])

                res_save_path = os.path.join(save_dir, cls + ".txt")
                with open(res_save_path, "a+") as f:
                    f.write(report)
                    f.write("\n" + "*" * 15 + "\n")
                del network
                gc.collect()
                keras.backend.clear_session()
            tf.keras.backend.clear_session()
