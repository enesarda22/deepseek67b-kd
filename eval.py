from tqdm import tqdm

import benchmarks
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    BATCH_SIZE = 5
    MODEL_NAME = "./models/DistLlama-3.2-1B"
    BENCH_NAME = "SuperGLUE"  # SuperGLUE, GLUE, SQuAD, MedMAQA, PubMedQA

    sampling_params = SamplingParams(
        temperature=0,  # no sampling
        max_tokens=10 if BENCH_NAME == "SQuAD" else 1,
    )

    model = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=4,
    )
    print("Model loaded!")

    bench = getattr(benchmarks, BENCH_NAME)(split="validation")

    predictions = []
    references = []
    answers = []

    for i in tqdm(range(0, len(bench.dataset), BATCH_SIZE)):
        batch = bench.dataset[i:i+BATCH_SIZE]
        prompts = [bench.get_validation_prompt(s) for s in batch]

        b_outputs = model.generate(prompts, sampling_params)

        b_answers = [output.outputs[0].text for output in b_outputs]
        b_predictions = [bench.answer_to_prediction(a, s) for a, s in zip(b_answers, batch)]
        b_references = [bench.sample_to_reference(s) for s in batch]

        answers.extend(b_answers)
        predictions.extend(b_predictions)
        references.extend(b_references)

    print(bench.get_metric(predictions, references))
