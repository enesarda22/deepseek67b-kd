from abc import ABC

import evaluate
from datasets import load_dataset


class Benchmark(ABC):

    def __init__(self):
        self.dataset = None
        self.metric = None

    @staticmethod
    def get_train_prompt(sample):
        pass

    @staticmethod
    def get_validation_prompt(sample):
        pass

    @staticmethod
    def answer_to_prediction(answer, sample):
        pass

    @staticmethod
    def sample_to_reference(sample):
        pass

    def get_metric(self, predictions, references):
        results = self.metric.compute(references=references, predictions=predictions)
        return results


class SuperGLUE(Benchmark):
    def __init__(self, task="boolq", split="train"):
        super().__init__()
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
    def answer_to_prediction(answer, sample):
        return int(answer == "yes")

    @staticmethod
    def sample_to_reference(sample):
        return sample["label"]


class GLUE(Benchmark):
    def __init__(self, task="sst2", split="train"):
        super().__init__()
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
    def answer_to_prediction(answer, sample):
        return int(answer == "positive")

    @staticmethod
    def sample_to_reference(sample):
        return sample["label"]

    def get_metric(self, predictions, references):
        results = self.metric.compute(predictions=predictions, references=references)
        return results


class SQuAD(Benchmark):
    def __init__(self, split="train"):
        super().__init__()
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


class MedMAQA(Benchmark):
    def __init__(self, split="train"):
        super().__init__()
        self.dataset = load_dataset("bigbio/med_qa", split=split, trust_remote_code=True)
        self.metric = evaluate.load("accuracy")

    @staticmethod
    def get_train_prompt(sample):
        return f"""Question: {sample["question"]}
        Possible Answers:
        {sample["options"][0]["key"]}) {sample["options"][0]["value"]}
        {sample["options"][1]["key"]}) {sample["options"][1]["value"]}
        {sample["options"][2]["key"]}) {sample["options"][2]["value"]}
        {sample["options"][3]["key"]}) {sample["options"][3]["value"]}
        {sample["options"][4]["key"]}) {sample["options"][4]["value"]}
        Correct Answer: {sample["answer_idx"]}"""

    @staticmethod
    def get_validation_prompt(sample):
        return f"""Question: A 28-year-old woman presents with a 3-month history of intermittent abdominal cramping, watery diarrhea, and bloating. She notices her symptoms worsen after meals containing dairy products. Physical examination is unremarkable. Which of the following is the most likely diagnosis?
        Possible Answers:
        A) Irritable bowel syndrome (IBS)
        B) Celiac disease
        C) Lactose intolerance
        D) Crohn’s disease
        E) Giardiasis
        Correct answer: C

        Question: A 65-year-old man with a 40-year smoking history presents with a chronic cough, recent weight loss, and occasional hemoptysis. A chest X-ray shows a cavitary lesion in the right upper lobe. Which of the following is the most likely diagnosis?
        Possible answers:
        A) Tuberculosis
        B) Squamous cell carcinoma of the lung
        C) Adenocarcinoma of the lung
        D) Lung abscess
        E) Chronic bronchitis
        Correct answer: B

        Question: {sample["question"]}
        Possible Answers:
        {sample["options"][0]["key"]}) {sample["options"][0]["value"]}
        {sample["options"][1]["key"]}) {sample["options"][1]["value"]}
        {sample["options"][2]["key"]}) {sample["options"][2]["value"]}
        {sample["options"][3]["key"]}) {sample["options"][3]["value"]}
        {sample["options"][4]["key"]}) {sample["options"][4]["value"]}
        Correct Answer:"""

    @staticmethod
    def answer_to_prediction(answer, sample):
        return answer.strip().lower()

    @staticmethod
    def sample_to_reference(sample):
        return sample["answer_idx"].strip().lower()


class PubMedQA(Benchmark):
    def __init__(self, split="train"):
        super().__init__()
        self.dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split, trust_remote_code=True)
        self.metric = evaluate.load("accuracy")

    @staticmethod
    def get_train_prompt(sample):
        context_str = ""
        for i, c in enumerate(sample["context"]["contexts"], start=1):
            context_str += f"Context {i}: {c}\n"

        return f"""{context_str}
        Question: {sample["question"]}
        Answer: {sample["final_decision"]}"""

    @staticmethod
    def get_validation_prompt(sample):
        context_str = ""
        for i, c in enumerate(sample["context"]["contexts"], start=1):
            context_str += f"Context {i}: {c}\n"

        return f"""Context 1: Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.\nContext 2: The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.
        Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
        Answer: yes

        Context 1: Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with the Snellen E was compared to the Landolt C acuity.\nContext 2: 100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In the group of patients, the eyes with the lower visual acuity, and the right eyes of the healthy subjects, were evaluated.\nContext 3: Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of the other groups were similar with only small differences between LR and SE.
        Question: Landolt C and snellen e acuity: differences in strabismus amblyopia?
        Correct Answer: no
        
        {context_str}
        Question: {sample["question"]}
        Answer:"""

    @staticmethod
    def answer_to_prediction(answer, sample):
        return answer.strip().lower()

    @staticmethod
    def sample_to_reference(sample):
        return sample["final_decision"].strip().lower()
