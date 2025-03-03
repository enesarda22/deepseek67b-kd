import os
import random
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    NUM_SAMPLES = 100_000
    MAX_LENGTH = 512

    TOKENIZER_PATH = "gpt2"
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(42)
    redpajama_ds = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "default",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    count = 0

    for example in tqdm(redpajama_ds):
        samples.append(example)
        count += 1
        if count >= NUM_SAMPLES:
            break

    raw_dataset = Dataset.from_list(samples)
    print(f"Final raw subset size: {len(raw_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )


    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]  # drop original text column
    )
    print("Tokenization complete.")
    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "tokenized_ds"))
    print(f"Saved tokenized under: {OUTPUT_DIR}")
