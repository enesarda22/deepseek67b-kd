import itertools
import os
from tqdm import tqdm

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from vllm import LLM, SamplingParams

from benchmarks import SuperGLUE, GLUE, SQuAD


def generate_text(batch):
    outputs = teacher_model.generate(batch, sampling_params)
    generated_texts = [output.prompt + output.outputs[0].text for output in outputs]
    return generated_texts


if __name__ == "__main__":
    NUM_TOKENS = 3 * 2**25  # 100_663_296 tokens
    MAX_NEW_TOKENS = 128  # hf: 222 tok/s, llama.cpp: 500 tok/s, vllm: 544 tok/s
    BATCH_SIZE = 32
    sampling_params = SamplingParams(
        temperature=1.1,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    TEACHER_MODEL = "deepseek-ai/deepseek-llm-67b-base"
    STUDENT_MODEL = "meta-llama/Llama-3.2-1B"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_seed(42)

    # load models
    teacher_model = LLM(
        model=TEACHER_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=2,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    print("Teacher model loaded!")

    # SuperGLUE --- Binary Question-Answer
    bench = SuperGLUE(task="boolq", split="train")
    ds1 = bench.dataset.map(lambda x: {"text": bench.get_train_prompt(x)}, batched=False)
    ds1 = ds1.select(np.random.randint(0, len(ds1), 8192))

    # GLUE --- Sentiment Analysis
    bench = GLUE(task="sst2", split="train")
    ds2 = bench.dataset.map(lambda x: {"text": bench.get_train_prompt(x)}, batched=False)
    ds2 = ds2.select(np.random.randint(0, len(ds2), 8192))

    # SQuAD --- Question-Answer
    bench = SQuAD(split="train")
    ds3 = bench.dataset.map(lambda x: {"text": bench.get_train_prompt(x)}, batched=False)
    ds3 = ds3.select(np.random.randint(0, len(ds3), 8192))

    texts = ds1["text"] + ds2["text"] + ds3["text"]
    del ds1, ds2, ds3

    # initialize np array
    input_ids_np = np.empty(NUM_TOKENS, dtype=np.uint32)
    count = 0

    # generate and tokenize
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        generated_texts = generate_text(batch)

        tokens = student_tokenizer(generated_texts)
        input_ids = list(itertools.chain.from_iterable(tokens["input_ids"]))

        input_ids_np[count:count + len(input_ids)] = input_ids
        count += len(input_ids)
        print(f"Token Count: {count}")

    del texts

    # fineweb-edu --- General Language Understanding
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, trust_remote_code=True)
    for batch in tqdm(ds.iter(BATCH_SIZE)):
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

    fn = os.path.join(OUTPUT_DIR, "tokenized_data")
    np.save(fn, input_ids_np)
    print(f"Saved tokenized data under: {OUTPUT_DIR}")
