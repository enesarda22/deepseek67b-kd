import itertools
import os

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmarks import PubMedQA, MedMAQA

if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_TOKENS = 2**25  # 33_554_432 tokens
    OUTPUT_DIR = "./data"

    student_model_name = "meta-llama/Llama-3.2-1B"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    bench = MedMAQA(split="train")
    ds1 = bench.dataset.map(lambda x: {"text": bench.get_train_prompt(x)}, batched=False)

    bench = PubMedQA(split="train")
    ds2 = bench.dataset.map(lambda x: {"text": bench.get_train_prompt(x)}, batched=False)

    texts = ds1["text"] + ds2["text"]

    input_ids_list = []
    count = 0

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        tokens = student_tokenizer(batch)

        input_ids = list(itertools.chain.from_iterable(tokens["input_ids"]))
        input_ids_list.extend(input_ids)

    input_ids_np = np.array(input_ids_list, dtype=np.uint32)
    input_ids_np = input_ids_np[:NUM_TOKENS]

    fn = os.path.join(OUTPUT_DIR, "tokenized_finetune_data")
    np.save(fn, input_ids_np)
    print(f"Saved tokenized data under: {OUTPUT_DIR}")
