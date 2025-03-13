import evaluate
from datasets import load_dataset


class SuperGLUE:
    def __init__(self, task="boolq", split="train"):
        self.dataset = load_dataset("super_glue", task, split=split, trust_remote_code=True)
        self.metric = evaluate.load("super_glue", task)

    @staticmethod
    def get_train_prompt(sample):
        return f"""Passage: {sample["passage"]}
        Question: {sample["question"]}
        Answer:"""

    @staticmethod
    def get_validation_prompt(sample):
        return f"""Passage: The sky is blue.
        Question: Is the sky green?
        Answer: No
        
        Passage: A strawberry is red.
        Question: Is a strawberry red?
        Answer: Yes
        
        Passage: {sample["passage"]}
        Question: {sample["question"]}
        Answer:"""

    @staticmethod
    def answer_to_prediction(answer):
        return int(answer == "yes")

    def get_metric(self, predictions, references):
        results = self.metric.compute(predictions=predictions, references=references)
        return results


class GLUE:
    def __init__(self, task="sst2", split="train"):
        self.dataset = load_dataset("glue", task, split=split, trust_remote_code=True)
        self.metric = evaluate.load("glue", task)

    @staticmethod
    def get_train_prompt(sample):
        return f"""Sentence: {sample["sentence"]}
        Sentiment:"""

    @staticmethod
    def get_validation_prompt(sample):
        return f"""Sentence: I absolutely loved this film—it's a delightful, exhilarating experience from start to finish.
        Sentiment: positive
        
        Sentence: This was the most boring movie I've seen in a long time; I couldn't wait for it to end.
        Sentiment: negative

        Sentence: {sample["sentence"]}
        Sentiment:"""

    @staticmethod
    def answer_to_prediction(answer):
        return int(answer == "positive")

    def get_metric(self, predictions, references):
        results = self.metric.compute(predictions=predictions, references=references)
        return results


class SQuAD:
    def __init__(self, split="train"):
        self.dataset = load_dataset("squad", split=split, trust_remote_code=True)
        self.metric = evaluate.load("squad")

    @staticmethod
    def get_train_prompt(sample):
        return f"""Context: {sample["context"]}
        Question: {sample["question"]}
        Answer:"""

    @staticmethod
    def get_validation_prompt(sample):
        return f"""Context: The Statue of Liberty was originally a gift from the people of France to the people of the United States. The statue is located in New York Harbor and is recognized as a universal symbol of freedom and democracy.
        Question: Who gave the Statue of Liberty to the people of the United States?
        Answer: The people of France
        
        Context: Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different scientific fields.
        Question: Which country was Marie Curie originally from?
        Answer: Poland
        
        Context: {sample["context"]}
        Question: {sample["question"]}
        Answer:"""

    @staticmethod
    def answer_to_prediction(answer, sample):
        return {"id": sample["id"], "prediction_text": answer}

    @staticmethod
    def sample_to_reference(sample):
        return {"id": sample["id"], "answers": sample["answers"]}

    def get_metric(self, predictions, references):
        results = self.metric.compute(references=references, predictions=predictions)
        return results


class CoNLL:
    def __init__(self, split="train"):
        self.dataset = load_dataset("conll2003", split=split)
        self.metric = evaluate.load("conll2003")