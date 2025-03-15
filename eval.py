from tqdm import tqdm

import benchmarks
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    BATCH_SIZE = 32
    MODEL_NAME = "enesarda22/Med-Llama-3.1-8B-DeepSeek67B-Distilled"
    BENCH_NAMES = ["MedMAQA", "PubMedQA"]  # SuperGLUE, GLUE, SQuAD, XTREME, MedMAQA, PubMedQA

    model = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=2,
    )
    print("Model loaded!")
    results = {}
    for bench_name in BENCH_NAMES:
        sampling_params = SamplingParams(
            temperature=0,  # no sampling
            max_tokens=10 if bench_name in ["SQuAD", "XTREME"] else 1,
            stop=["\n", model.get_tokenizer().eos_token]
        )

        bench = getattr(benchmarks, bench_name)(split="validation")

        predictions = []
        references = []
        answers = []

        for i in tqdm(range(0, len(bench.dataset), BATCH_SIZE)):
            batch = bench.dataset.select(range(i, min(i+BATCH_SIZE, len(bench.dataset))))
            prompts = [bench.get_validation_prompt(s) for s in batch]

            b_outputs = model.generate(prompts, sampling_params)

            b_answers = [output.outputs[0].text for output in b_outputs]
            b_predictions = [bench.answer_to_prediction(a, s) for a, s in zip(b_answers, batch)]
            b_references = [bench.sample_to_reference(s) for s in batch]

            answers.extend(b_answers)
            predictions.extend(b_predictions)
            references.extend(b_references)

        results[bench_name] = bench.get_metric(predictions, references)

    print(results)
