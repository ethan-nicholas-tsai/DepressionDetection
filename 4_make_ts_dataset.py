import os
import jsonlines
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import namedtuple


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Dataset = namedtuple("Dataset", ["X", "y"])


def preprocess(dataset, normalize=True, unify_dims=False, to_categorical=True):
    """
    ## Prepare the data
    """
    X, y = dataset.X, dataset.y

    if normalize:
        X_mean = X.mean()
        X_std = X.std()
        X = (X - X_mean) / (X_std + 1e-8)

    if unify_dims:
        # cutoff or expand timestamps
        pass

    if to_categorical:
        # 将标签独热编码
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(y)

    return Dataset(X, y)


def make_dataset_ts(
    data_dir="swdd-7k",
    feat_dir="swdd-7k_embedding_500_50",
    save_dir="swdd-7k_ts_500_50",
    samp_cnt=7000,
):
    """
    Note: 1. 在x中继续嵌套元组，将索引编进去，这样最后导出来就是带编号的sample，可以进行错误案例分析
          2. 时间序列Z-Score预处理不在此处进行
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    X = []
    y = []

    for i in range(samp_cnt):
        file_id = i
        time_series = np.load(os.path.join(feat_dir, "%04d.npy" % file_id))
        X.append(time_series)
        with open(
            os.path.join(data_dir, "%04d.jsonl" % file_id), "r", encoding="utf8"
        ) as f:
            for item in jsonlines.Reader(f):
                datum = item
                y.append(datum["label"])
    X = np.array(X)
    y = np.array(y)
    Dataset.X = X
    Dataset.y = y
    # Dataset_pre = preprocess(dataset=Dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        Dataset.X,
        Dataset.y,
        test_size=0.4,
        random_state=2022,
        stratify=Dataset.y,
    )
    # val_x, X_test, val_y, y_test = train_test_split(
    #     X_test, y_test, test_size=0.5, random_state=2022, stratify=y_test
    # )
    # print(len(y_train), len(y_test), len(val_y), np.sum(y_test == 1))
    print(len(y_train), len(y_test), np.sum(y_test == 1))

    with open(os.path.join(save_dir, "train.ts"), "w") as f:
        for idx, v in enumerate(X_train):
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    f.write(str(v[i][j]))
                    if j < v.shape[1] - 1:
                        f.write(",")
                f.write(":")
            f.write(str(y_train[idx]) + "\n")

    with open(os.path.join(save_dir, "test.ts"), "w") as f:
        for idx, v in enumerate(X_test):
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    f.write(str(v[i][j]))
                    if j < v.shape[1] - 1:
                        f.write(",")
                f.write(":")
            f.write(str(y_test[idx]) + "\n")


def dataset_7k_origin_main():

    make_dataset_ts(
        data_dir="dataset/swdd-7k",
        feat_dir="dataset/swdd-7k_embedding_origin_500_0",
        save_dir="dataset/swdd-7k_ts_origin_500_0",
    )


def dataset_4k_prop_origin_main():
    for i in range(10, 100, 10):
        data_dir = "dataset/swdd-4k_{}".format(i)

        make_dataset_ts(
            data_dir=data_dir,
            feat_dir="{}_embedding_origin_500_0".format(data_dir),
            save_dir="{}_ts_origin_500_0".format(data_dir),
            samp_cnt=4000,
        )


if __name__ == "__main__":
    # dataset_7k_origin_main()
    dataset_4k_prop_origin_main()
