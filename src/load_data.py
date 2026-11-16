from datasets import load_dataset


DATASET_URL = (
    "https://huggingface.co/datasets/Hello-SimpleAI/HC3/"
    "resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet"
)


def load_hc3():
    dataset = load_dataset(
        "parquet",
        data_files=DATASET_URL
    )

    return dataset["train"]


if __name__ == "__main__":
    data = load_hc3()

    print("Dataset loaded successfully")
    print("Number of questions:", len(data))
    print("Columns:", data.column_names)