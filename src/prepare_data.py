import pandas as pd

from load_data import load_hc3

import os

os.makedirs("data/raw", exist_ok=True)



def prepare_data():
    dataset = load_hc3()

    rows = []

    for record in dataset:
        human_answers = record["human_answers"]
        chatgpt_answers = record["chatgpt_answers"]

        for answer in human_answers:
            if answer and answer.strip():
                rows.append({
                    "text": answer.strip(),
                     "label": 0,
                     "source": record["source"],
                     "group": record["question"]
                })

        for answer in chatgpt_answers:
            if answer and answer.strip():
                rows.append({
                    "text": answer.strip(),
                     "label": 0,
                     "source": record["source"],
                     "group": record["question"]
                })

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset="text")
    df = df.dropna(subset=["text"])

    df["text_length"] = df["text"].str.len()

    df = df[df["text_length"] >= 20]

    df = df[["text", "label", "source","group"]]

    return df


if __name__ == "__main__":
    df = prepare_data()

    print("\nPrepared dataset")
    print("----------------")
    print("Total samples:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nFirst sample:")
    print(df.iloc[0])

    output_path = "data/raw/hc3_processed.csv"

    df.to_csv(output_path, index=False) 
    print(f"\nSaved dataset to: {output_path}")