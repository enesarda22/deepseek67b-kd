import itertools
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, pipeline


if __name__ == "__main__":
    NUM_TOKENS = 1_000_000
    MAX_NEW_TOKENS = 1024
    BATCH_SIZE = 32

    TEACHER_MODEL = "gpt2"
    STUDENT_MODEL = "TinyLlama/TinyLlama_v1.1"
    DS = "HuggingFaceFW/fineweb-edu"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_seed(42)
    ds = load_dataset(DS, name="sample-10BT", split="train", streaming=True)
    generator = pipeline("text-generation", model=TEACHER_MODEL)
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

    input_ids_np = np.empty(NUM_TOKENS, dtype=np.uint32)
    count = 0
    for batch in ds.iter(batch_size=BATCH_SIZE):
        generated_texts = generator(batch["text"], max_new_tokens=MAX_NEW_TOKENS, temperature=1.1, top_p=0.9, do_sample=True)
        generated_texts = [b[0]["generated_text"] for b in generated_texts]

        tokens = student_tokenizer(generated_texts)
        input_ids = list(itertools.chain.from_iterable(tokens["input_ids"]))

        if count+len(input_ids) > NUM_TOKENS:
            input_ids_np[count:] = input_ids[:NUM_TOKENS-count]
            break
        else:
            input_ids_np[count:count+len(input_ids)] = input_ids
            count += len(input_ids)

    fn = os.path.join(OUTPUT_DIR, "tokenized_fineweb-edu")
    np.save(fn, input_ids_np)
    print(f"Saved tokenized under: {OUTPUT_DIR}")
