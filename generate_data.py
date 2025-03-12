import os
from tqdm import tqdm

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from llama_cpp import Llama


def generate_text(text):
    output = model(text, max_tokens=MAX_NEW_TOKENS, echo=True, temperature=1.1, top_p=0.9)
    return output["choices"][0]["text"]


if __name__ == "__main__":
    NUM_TOKENS = 1_000_000
    MAX_NEW_TOKENS = 512
    BATCH_SIZE = 8

    TEACHER_MODEL = "deepseek-ai/deepseek-llm-67b-base"
    STUDENT_MODEL = "TinyLlama/TinyLlama_v1.1"
    DS = "HuggingFaceFW/fineweb-edu"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_seed(42)

    ds = load_dataset(DS, name="sample-10BT", split="train", streaming=True)
    print("Dataset loaded!")

    model = Llama(model_path=TEACHER_MODEL, n_ctx=2048)
    print("Teacher model loaded!")

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

    input_ids_np = np.empty(NUM_TOKENS, dtype=np.uint32)
    count = 0
    for sample in tqdm(ds):
        generated_text = generate_text(sample["text"])

        tokens = student_tokenizer(generated_text)
        input_ids = tokens["input_ids"]

        if count+len(input_ids) > NUM_TOKENS:
            input_ids_np[count:] = input_ids[:NUM_TOKENS-count]
            break
        else:
            input_ids_np[count:count+len(input_ids)] = input_ids
            count += len(input_ids)
        print(f"Token Count: {count}")

    fn = os.path.join(OUTPUT_DIR, "tokenized_fineweb-edu")
    np.save(fn, input_ids_np)
    print(f"Saved tokenized under: {OUTPUT_DIR}")
