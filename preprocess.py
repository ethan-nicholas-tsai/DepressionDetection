from utils.data import load_swdd_xk, load_swdd_xk_emb
from utils.extractor import WeiboText, get_post_time
from utils.symptom import symptoms_dsm_5 as symptoms
from sentence_transformers import SentenceTransformer, util
import torch
from torch.nn import ZeroPad2d
import pandas as pd
import numpy as np
import os


def extract_embedding(
    data_dir="swdd-7k",
    save_dir="swdd-7k_embedding",
    modelname="paraphrase-xlm-r-multilingual-v1"
):
    """提取推文向量
    model_list = [
        "distiluse-base-multilingual-cased-v1",  # 512
        "paraphrase-xlm-r-multilingual-v1",  # 768, best
        "stsb-xlm-r-multilingual",  # 768
    ]
    """
    data = load_swdd_xk(data_dir=data_dir)
    model = SentenceTransformer(modelname)
    weibo_cleaner = WeiboText()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cnt = 0
    for i in range(len(data)):
        cnt += 1
        if not cnt % 100:
            print(cnt, end=" ", flush=True)
        if not cnt % 1000:
            print()
        # extract embedding
        tweets = data[i]["tweets"]
        df_tweets = pd.DataFrame(
            [
                {
                    "is_origin": tweet["is_origin"],
                    "time": get_post_time(tweet["post_time"]),
                    "text": weibo_cleaner.get_cleaned_text(tweet["text"]),
                }
                for tweet in tweets
            ]
        )

        tweets_emb = model.encode(df_tweets["text"].tolist())
        df_emb = pd.DataFrame({"embedding": list(tweets_emb)})
        df = pd.concat([df_tweets, df_emb], axis=1)

        np.savez(
            os.path.join(save_dir, "%04d.npz" % (i)),
            is_origin=df["is_origin"].tolist(),
            time=df["time"].tolist(),
            text=df["text"].tolist(),
            embedding=df["embedding"].tolist(),
        )  # 不能用to_numpy!!


def extract_time_series_feature(
    data_dir="swdd-7k_embedding",
    modelname="paraphrase-xlm-r-multilingual-v1",
    pad_len=500,
    interval_spans=0,
    origin_only=False
):
    if origin_only:
        save_dir = "{}_origin_{}_{}".format(data_dir, pad_len, interval_spans)
    else:
        save_dir = "{}_{}_{}".format(data_dir, pad_len, interval_spans)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = SentenceTransformer(modelname)
    symp_text = [v for k, v in symptoms.items()]
    symp_emb = model.encode(symp_text)
    symp_emb = torch.from_numpy(symp_emb)
    print("Symptom Embed: ({}, {})".format(symp_emb.shape[0], symp_emb.shape[1]))

    # load_emb
    data = load_swdd_xk_emb(data_dir=data_dir)
    for i in range(len(data)):
        # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
        # tweet_emb = torch.Tensor(data[i]["embedding"].tolist())
        df = data[i]
        if origin_only:  # NOTE: 可能出现origin为0的情况，仅这样不妥，需要保证origin数量充分多
            # print('origin only')
            df = df[df['is_origin']]
            df = df.reset_index(drop=True)
        # extract embedding
        # tweet_emb = torch.Tensor(np.array(data[i]["embedding"].tolist())) # NOTE:
        tweet_emb = torch.Tensor(np.array(df["embedding"].tolist()))
        # Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(symp_emb, tweet_emb)
        # print(cosine_scores.shape)
        pad = ZeroPad2d(padding=(0, pad_len - tweet_emb.shape[0], 0, 0))
        cosine_scores = pad(cosine_scores)
        # print(cosine_scores.shape)
        # np.save(os.path.join(save_dir, "%04d" % i), cosine_scores)
        # yield cosine_scores
        # 如何将pad和interval_day结合起来？
        # df = data[i] # NOTE:
        time_series = cosine_scores
        if interval_spans:
            num = 0
            for k, v in symptoms.items():
                df_symp = pd.DataFrame({k[:3]: time_series[num]})
                df = pd.concat([df, df_symp], axis=1)
                num += 1

            # 按照时间索引
            df['time'] = df['time'].astype(np.string_)  # np.str_-> np.string_
            df['time'] = df['time'].apply(lambda x: str(x, encoding='utf-8'))  # bytes -> str
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            # time_series = df.resample('{}D'.format(interval_days)).mean().fillna(0).values.T
            idx = df.index
            time_series = df.resample((np.max(idx)-np.min(idx))/(interval_spans-1)).mean().fillna(0).values.T
        # print(time_series.shape)
        np.save(os.path.join(save_dir, "%04d" % i), time_series)
        # yield time_series       
    return save_dir