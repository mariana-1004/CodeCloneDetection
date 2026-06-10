from datasets import load_dataset


def load_bigclonebench(dataset_name):
    dataset = load_dataset(dataset_name)

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def preprocess(df):
    excluded_cols = ["id", "id1", "id2"]
    df = df.drop(columns=excluded_cols)

    func1_list = df["func1"].astype(str).tolist()
    func2_list = df["func2"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return func1_list, func2_list, labels


def sample_dataframe(df, sample_size, random_state=42):
    if sample_size is None or sample_size >= len(df):
        return df

    return df.sample(sample_size, random_state=random_state)
