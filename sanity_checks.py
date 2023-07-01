import pandas as pd

DATA_PATH_PPI = "./NBFNet/data/lnctard/train1-ppi.txt"
DATA_PATH_TEST = "./NBFNet/data/lnctard/test.txt"
PATH_ENTITY_TYPES = "/Users/lisa/Documents/Uni/ML4RG/linkpred/NBFNet/data/lnctard/entity_types_lnctardppi.txt"
def check_node_overlap(path_ppi, path_file2check):
    df_ppi = pd.read_csv(path_ppi,sep="\t", encoding="latin-1",names = ["head", "tail"], usecols=[0,2]).drop_duplicates(ignore_index=True)
    df_2check = pd.read_csv(path_file2check, sep="\t", names = ["head", "tail"], usecols=[0,2])
    df_ppi["index"], df_2check["index"] = df_ppi.index, df_2check.index
    df_ppi['key'] = df_ppi[["head", "tail"]].apply(lambda row: tuple(sorted(row)), axis=1)
    df_2check['key'] = df_2check[["head", "tail"]].apply(lambda row: tuple(sorted(row)), axis=1)
    overlap = pd.merge(df_ppi, df_2check, how='inner', on="key")
    return overlap

def check_rel_distribution(path):
    df = pd.read_csv(path, names=["head", "relation", "tail"], sep="\t")

    #checks how many different nodepairs exist for each relation
    rel_types = df.groupby(by="relation").nunique()

    #checks how many samples exist for each relation
    rel_types_occurance = df.groupby(by='relation').size()

    #checks heads and tails distribution
    heads = df.groupby(by="head").size()
    tails = df.groupby(by="tail").size()

    return

def count_nan(path, col_index):
    df = pd.read_csv(path, sep="\t", usecols=[col_index])
    return df.isna().sum()

if __name__ == "__main__":
    #overlap = check_node_overlap(DATA_PATH_PPI, DATA_PATH_TEST)
    #count_nan(PATH_ENTITY_TYPES, 1)
    check_rel_distribution(DATA_PATH_TEST)

