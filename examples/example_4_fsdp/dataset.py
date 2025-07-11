from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ajibawa-2023/WikiHow", split="all")  # or "sep" for paragraph-level
print("dataset from wikihow: ")
print(dataset)
#print("Train size:", dataset["train"].shape)
#print("Validation size:", dataset["validation"].shape)

import tensorflow_datasets as tfds

# Load the "all" version (full articles)
ds = tfds.load("wikihow/all", split="train")