import os
import random
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    NUM_SAMPLES = 100_000

    TOKENIZER_PATH = "gpt2"
    OUTPUT_DIR = "data"
    ds_path = "HuggingFaceFW/fineweb-edu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(42)
    ds = load_dataset(ds_path, name="sample-10BT", split="train", streaming=True)

    samples = []
    count = 0

    for example in tqdm(ds):
        samples.append({"text": example["text"]})
        count += 1
        if count >= NUM_SAMPLES:
            break

    raw_dataset = Dataset.from_list(samples)
    print(f"Final raw subset size: {len(raw_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


    def tokenize_fn(samples):
        tokens = tokenizer(samples["text"])

        tokens["input_ids"] = [[tokenizer.eos_token_id] + input_ids + [tokenizer.eos_token_id] for input_ids in tokens["input_ids"]]
        tokens["attention_mask"] = [[1] + attention_mask + [1] for attention_mask in tokens["attention_mask"]]
        return tokens


    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenization complete.")
    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(os.path.join(OUTPUT_DIR, f"{TOKENIZER_PATH}_tokenized_fineweb-edu"))
    print(f"Saved tokenized under: {OUTPUT_DIR}")
