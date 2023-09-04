from utils.data import gen_swdd_4k, gen_swdd_7k

if __name__ == "__main__":
    data_dir = "dataset/swdd"
    # gen_swdd_7k(data_dir=data_dir)
    gen_swdd_4k(data_dir=data_dir)
