from utils.clog import count_time
import os
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "swdd"
dep_file = "depressed.jsonl"
con_file = "control.jsonl"
dep_path = os.path.join(data_dir, dep_file)
con_path = os.path.join(data_dir, con_file)


@count_time
def load_swdd_all(data_dir):
    """ """

    dep_file = "depressed.jsonl"
    con_file = "control.jsonl"
    dep_path = os.path.join(data_dir, dep_file)
    con_path = os.path.join(data_dir, con_file)

    data = []
    for filename in [dep_path, con_path]:
        print("Loading {}".format(filename))
        cnt = 0
        with open(filename, "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                datum = {
                    "label": 1 if item["label"]["depressed"] else 0,
                    **item["user"],
                    "tweets": pd.DataFrame(item["tweets"]),
                }
                data.append(datum)
                cnt += 1
        print("Sample Num: {}".format(cnt))
    df = pd.DataFrame(data)
    return df


def get_quantile_upper_outliers(df, column_name, quantile=0.75):
    s = df[column_name]

    df_ = df.copy()
    # 这里将大于上四分位数(Q3)的设定为异常值
    # df_['isOutlier'] = s > s.quantile(0.75)
    df_.loc[:, "isOutlier"] = s > s.quantile(quantile)
    df_rst = df_[df_["isOutlier"] == True]
    return df_rst


def get_quantile_lower_outliers(df, column_name, quantile=0.25):
    s = df[column_name]

    df_ = df.copy()
    # 这里将小于下四分位数(Q1)的设定为异常值
    df_.loc[:, "isOutlier"] = s < s.quantile(quantile)
    df_rst = df_[df_["isOutlier"] == True]
    return df_rst


def get_box_plot_outliers(df, column_name):
    s = df[column_name]

    df_ = df.copy()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_.loc[:, "isOutlier"] = s.mask((s < low) | (s > up))
    df_rst = df_[df_["isOutlier"] == True]
    return df_rst


@count_time
def gen_swdd_7k(data_dir):
    """ """

    import os
    import numpy as np
    import json
    import jsonlines

    # 加载数据集
    df = load_swdd_all(data_dir=data_dir)

    # 删除不要字段
    cols = [
        i
        for i in df.columns
        if i
        not in ["avatar_url", "cover_image_url", "verified_reason", "verified_type"]
    ]
    df = df[cols]

    # 删除离异点
    dep_follow_outliers = get_quantile_upper_outliers(
        df[df["label"] == 1], column_name="follow_count", quantile=0.999
    )
    dep_follower_outliers = get_quantile_upper_outliers(
        df[df["label"] == 1], column_name="followers_count", quantile=0.99
    )
    dep_outliers = df.iloc[
        np.union1d(dep_follow_outliers.index.values, dep_follower_outliers.index.values)
    ]
    con_follow_outliers = get_box_plot_outliers(
        df[df["label"] == 0], column_name="follow_count"
    )
    con_follower_outliers = get_box_plot_outliers(
        df[df["label"] == 0], column_name="followers_count"
    )
    con_outliers = df.iloc[
        np.union1d(con_follow_outliers.index.values, con_follower_outliers.index.values)
    ]

    df_ = df.copy()
    df_ = df_.drop(dep_outliers.index.values)
    df_ = df_.drop(con_outliers.index.values).reset_index(drop=True)
    print(df_.describe())


    # 删除原创推文少于30的
    for i in range(len(df_)):
        if len(df_['tweets'][i][df_['tweets'][i]['is_origin']]) < 30:
            df_ = df_.drop(i)


    # 采样7k
    samp_cnt = 3500
    df_7k = (
        (
            pd.concat(
                [
                    df_[df_["label"] == 1].sample(n=samp_cnt),
                    df_[df_["label"] == 0].sample(n=samp_cnt),
                ]
            )
        )
        .sample(n=samp_cnt * 2)
        .reset_index(drop=True)
    )

    # 删除推文字段（剔除转发推文）
    cols = [
        i
        for i in df_7k.iloc[0]["tweets"].columns
        if i
        not in [
            "edit_at",
            "pics_url",
            "publish_place",
            "publish_tool",
            "video_url",
            "article_url",
            "topics",
            "at_users",
            "retweet",
        ]
    ]
    df_ = df_7k.copy()
    for i in range(len(df_)):
        df_["tweets"][i] = df_["tweets"][i][cols]

    df_7k = df_
    print(df_7k.describe())

    # return df_7k
    swdd_7k_dir = data_dir + "-7k"
    if not os.path.exists(swdd_7k_dir):
        os.mkdir(swdd_7k_dir)

    print("Writing to {}".format(swdd_7k_dir))

    for i in range(len(df_7k)):
        samp = json.loads(df_7k.iloc[i].to_json(orient="columns"))
        samp["tweets"] = json.loads(
            df_7k.iloc[i]["tweets"].to_json(orient="records")
        )  # fuck it !!!!

        with jsonlines.open(
            os.path.join(swdd_7k_dir, "%04d.jsonl" % (i)), mode="w"
        ) as writer:
            writer.write(samp)

    print("Done")

    return df_7k


@count_time
def gen_swdd_4k(data_dir):
    import os
    import numpy as np
    import json
    import jsonlines

    # data_dir = "swdd"

    # 加载数据集
    df = load_swdd_all(data_dir=data_dir)

    # 删除不要字段
    cols = [
        i
        for i in df.columns
        if i
        not in ["avatar_url", "cover_image_url", "verified_reason", "verified_type"]
    ]
    df = df[cols]

    # 删除离异点
    dep_follow_outliers = get_quantile_upper_outliers(
        df[df["label"] == 1], column_name="follow_count", quantile=0.999
    )
    dep_follower_outliers = get_quantile_upper_outliers(
        df[df["label"] == 1], column_name="followers_count", quantile=0.99
    )
    dep_outliers = df.iloc[
        np.union1d(dep_follow_outliers.index.values, dep_follower_outliers.index.values)
    ]
    con_follow_outliers = get_box_plot_outliers(
        df[df["label"] == 0], column_name="follow_count"
    )
    con_follower_outliers = get_box_plot_outliers(
        df[df["label"] == 0], column_name="followers_count"
    )
    con_outliers = df.iloc[
        np.union1d(con_follow_outliers.index.values, con_follower_outliers.index.values)
    ]

    import pandas as pd

    # 采样4k
    samp_cnt = 4000

    for dep_prop in range(10, 100, 10):
        df_ = df.copy()
        df_ = df_.drop(dep_outliers.index.values)
        df_ = df_.drop(con_outliers.index.values).reset_index(drop=True)
        print(df_.describe())

        # 删除原创推文少于20的
        for i in range(len(df_)):
            if len(df_['tweets'][i][df_['tweets'][i]['is_origin']]) < 20:
                df_ = df_.drop(i)

        dep_prop = dep_prop / 100
        dep_cnt = int(samp_cnt * dep_prop)
        # con_cnt = int(samp_cnt * (1 - dep_prop)) # 浮点数问题。。
        con_cnt = samp_cnt - dep_cnt
        if dep_cnt % 100:
            con_cnt = int(samp_cnt * (1 - dep_prop))
            dep_cnt = samp_cnt - con_cnt

        print(dep_cnt, con_cnt)

        df_4k = (
            (
                pd.concat(
                    [
                        df_[df_["label"] == 1].sample(n=dep_cnt),
                        df_[df_["label"] == 0].sample(n=con_cnt),
                    ]
                )
            )
            .sample(n=samp_cnt)
            .reset_index(drop=True)
        )

        # 删除推文字段（剔除转发推文）
        cols = [
            i
            for i in df_4k.iloc[0]["tweets"].columns
            if i
            not in [
                "edit_at",
                "pics_url",
                "publish_place",
                "publish_tool",
                "video_url",
                "article_url",
                "topics",
                "at_users",
                "retweet",
            ]
        ]
        df_ = df_4k.copy()
        for i in range(len(df_)):
            df_["tweets"][i] = df_["tweets"][i][cols]

        df_4k = df_
        print(df_4k.describe())

        swdd_4k_dir = data_dir + "-4k_{}".format(int(dep_prop * 100))
        if not os.path.exists(swdd_4k_dir):
            os.mkdir(swdd_4k_dir)

        print("Writing to {}".format(swdd_4k_dir))

        for i in range(len(df_4k)):
            samp = json.loads(df_4k.iloc[i].to_json(orient="columns"))
            samp["tweets"] = json.loads(
                df_4k.iloc[i]["tweets"].to_json(orient="records")
            )  # fuck it !!!!

            with jsonlines.open(
                os.path.join(swdd_4k_dir, "%04d.jsonl" % (i)), mode="w"
            ) as writer:
                writer.write(samp)

        print("Done")


@count_time
def load_swdd_xk(data_dir):
    data = []
    # 乱序。。
    # for _, _, files in os.walk(data_dir):
    #     for file in files:
    #         with open(os.path.join(data_dir, file), "r", encoding="utf8") as f:
    #             for item in jsonlines.Reader(f):
    #                 data.append(item)
    files = os.listdir(data_dir)
    files.sort()
    for file in files:
        with open(os.path.join(data_dir, file), "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                data.append(item)
    return data


@count_time
def load_swdd_xk_emb(data_dir):
    import numpy as np

    data = []

    files = os.listdir(data_dir)
    files.sort()
    for file in files:
        datum = np.load(os.path.join(data_dir, file), allow_pickle=True)
        df = pd.DataFrame({k: list(datum[k]) for k in datum.files})
        data.append(df)
    return data


@count_time
def load_swdd_xk_npz(data_dir):
    import os
    import numpy as np

    train_data = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    X_train = []
    id_train = []
    X_test = []
    id_test = []

    for datum in train_data["X"]:
        X_train.append(datum[0])
        id_train.append(datum[1])

    for datum in test_data["X"]:
        X_test.append(datum[0])
        id_test.append(datum[1])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    id_train = np.array(id_train)
    id_test = np.array(id_test)
    y_train = train_data["y"]
    y_test = test_data["y"]

    return X_train, X_test, y_train, y_test, id_train, id_test


def inspect_time_series(data_dir, dir_suffix, file_id):
    """查看某个用户的时间序列特征，封装成DataFrame返回
    # basic usage
    df.iloc[:144].plot(subplots=True, figsize=(10,12))
    df.loc['2020-05'].plot()
    df.resample('Q')['sui'].mean()
    df.resample('2W').mean().fillna(0).values.T[0]
    # plot month average bar
    df_month = df.resample("M").mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.bar(df_month['2020':].index, df_month.loc['2020':,
        "ene"], width=25, align='center')
    """
    import numpy as np
    import os
    from utils.symptom import symptoms_dsm_5 as symptoms

    time_series = np.load(
        os.path.join("{}_{}".format(data_dir, dir_suffix), "%04d.npy" % file_id)
    )
    tweet_meta = np.load(
        os.path.join(data_dir, "%04d.npz" % file_id), allow_pickle=True
    )
    df = pd.DataFrame({k: list(tweet_meta[k]) for k in tweet_meta.files})

    num = 0
    for k, v in symptoms.items():
        df_symp = pd.DataFrame({k[:3]: time_series[num]})
        df = pd.concat([df, df_symp], axis=1)
        num += 1

    # 按照时间索引
    df["time"] = df["time"].astype(np.string_)  # np.str_-> np.string_
    df["time"] = df["time"].apply(lambda x: str(x, encoding="utf-8"))  # bytes -> str
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    return df


def plot_ex_width(x, y, x_maxsize):

    plt.plot(x, y)
    # plt.ylim((0, 1000))
    # plt.title("Demo")
    plt.xlabel("x")
    plt.ylabel("y")

    # change x internal size
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()

    # set size
    maxsize = x_maxsize
    m = 0.2
    N = len(x)
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])


def plot_time_series(time_series, expand_width=2):
    from utils.symptom import symptoms_dsm_5 as symptoms

    label_list = [k[:3] for k, v in symptoms.items()]

    x_values = list(range(1, time_series.shape[1] + 1))
    for i in range(time_series.shape[0]):
        # plt.plot(x_values, time_series[i])
        plot_ex_width(x_values, time_series[i], expand_width)
    plt.legend(labels=label_list)
