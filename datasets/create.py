import argparse
import pandas as pd
from typing import Literal

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()


def harmless(
    size,
    name: Literal["harmless_train", "harmless_val"],
):
    df = pd.read_parquet(
        "hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
    )
    series = df["instruction"]
    series = series.sample(n=size)
    series.to_csv(f"datasets/{name}", index=False)


def harmful(size, name: Literal["harmful_train", "harmful_val"]):
    df = pd.read_csv("llm-attacks/data/advbench/harmful_behaviors.csv")
    series = df["goal"]
    series = series.sample(n=size)
    series.to_csv(f"datasets/{name}", index=False)


if args.dataset == "harmless_val":
    harmless(32, args.dataset)
elif args.dataset == "harmful_val":
    harmful(32, args.dataset)
elif args.dataset == "harmful_train":
    harmful(32, args.dataset)
elif args.dataset == "harmless_train":
    harmless(32, args.dataset)
else:
    raise Exception("unrecognized dataset")
