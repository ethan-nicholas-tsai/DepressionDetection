from preprocess import extract_time_series_feature
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # extract_time_series_feature(data_dir="dataset/swdd-7k_embedding", origin_only=True)
    for i in range(10, 100, 10):
        extract_time_series_feature(
            data_dir="dataset/swdd-4k_{}_embedding".format(i), origin_only=True
        )
