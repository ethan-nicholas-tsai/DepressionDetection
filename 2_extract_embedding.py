from preprocess import extract_embedding

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # data_dir = "dataset/swdd-7k"
    # emb_dir = data_dir + "_embedding"
    # extract_embedding(
    #     data_dir=data_dir,
    #     save_dir=emb_dir,
    #     modelname="paraphrase-xlm-r-multilingual-v1",
    # )

    for i in range(10, 100, 10):
        data_dir = "dataset/swdd-4k_{}".format(i)
        emb_dir = data_dir + "_embedding"
        extract_embedding(
            data_dir=data_dir,
            save_dir=emb_dir,
            modelname="paraphrase-xlm-r-multilingual-v1",
        )
