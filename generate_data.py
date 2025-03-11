import itertools
import os
from tqdm import tqdm

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def generate_text(batch):
    inputs = teacher_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=1.1,
        top_p=0.9,
        pad_token_id=teacher_tokenizer.eos_token_id
    )

    generated_texts = teacher_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts


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

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
    )
    print(f"Teacher model loaded to {model.hf_device_map}")

    model = torch.compile(model)
    model.eval()
    print("Teacher model compiled!")

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

    input_ids_np = np.empty(NUM_TOKENS, dtype=np.uint32)
    count = 0
    for batch in tqdm(ds.iter(batch_size=BATCH_SIZE)):
        generated_texts = generate_text(batch["text"])

        tokens = student_tokenizer(generated_texts)
        input_ids = list(itertools.chain.from_iterable(tokens["input_ids"]))

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
