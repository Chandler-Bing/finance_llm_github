import os
import pandas as pd
from datasets import Dataset
from .common import CommonExam

class Mmlu(CommonExam):
    def __init__(self,shot,data_path='./exams/mmlu'):
        categories = {
            "abstract_algebra": ["math"],
            "anatomy": ["health"],
            "astronomy": ["physics"],
            "business_ethics": ["business"],
            "clinical_knowledge": ["health"],
            "college_biology": ["biology"],
            "college_chemistry": ["chemistry"],
            "college_computer_science": ["computer science"],
            "college_mathematics": ["math"],
            "college_medicine": ["health"],
            "college_physics": ["physics"],
            "computer_security": ["computer science"],
            "conceptual_physics": ["physics"],
            "econometrics": ["economics"],
            "electrical_engineering": ["engineering"],
            "elementary_mathematics": ["math"],
            "formal_logic": ["philosophy"],
            "global_facts": ["other"],
            "high_school_biology": ["biology"],
            "high_school_chemistry": ["chemistry"],
            "high_school_computer_science": ["computer science"],
            "high_school_european_history": ["history"],
            "high_school_geography": ["geography"],
            "high_school_government_and_politics": ["politics"],
            "high_school_macroeconomics": ["economics"],
            "high_school_mathematics": ["math"],
            "high_school_microeconomics": ["economics"],
            "high_school_physics": ["physics"],
            "high_school_psychology": ["psychology"],
            "high_school_statistics": ["math"],
            "high_school_us_history": ["history"],
            "high_school_world_history": ["history"],
            "human_aging": ["health"],
            "human_sexuality": ["culture"],
            "international_law": ["law"],
            "jurisprudence": ["law"],
            "logical_fallacies": ["philosophy"],
            "machine_learning": ["computer science"],
            "management": ["business"],
            "marketing": ["business"],
            "medical_genetics": ["health"],
            "miscellaneous": ["other"],
            "moral_disputes": ["philosophy"],
            "moral_scenarios": ["philosophy"],
            "nutrition": ["health"],
            "philosophy": ["philosophy"],
            "prehistory": ["history"],
            "professional_accounting": ["other"],
            "professional_law": ["law"],
            "professional_medicine": ["health"],
            "professional_psychology": ["psychology"],
            "public_relations": ["politics"],
            "security_studies": ["politics"],
            "sociology": ["culture"],
            "us_foreign_policy": ["politics"],
            "virology": ["health"],
            "world_religions": ["philosophy"],
        }
        categories = {item:item for item in categories.keys()}
        file_cols=['question', 'A', 'B', 'C', 'D', 'answer']
        prompt = "The following are multiple choice questions (with answers) about {}."
        super(Mmlu, self).__init__(data_path=data_path,shot=shot,test_split='test',categories=categories,file_cols=file_cols,prompt=prompt)

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s[1:]

    def read(self,task,split):
        filename = os.path.join(self.data_path,split,f'{task}_{split}.csv')
        #print(filename)
        df = pd.read_csv(filename,names=self.file_cols,header=None,skiprows=0)
        df = df.fillna('None')
        ds = Dataset.from_pandas(df)
        return ds
    def get_few_shot(self,task,choices):
        prompt = self.prompt.format(self.format_subject(task))
        ds = self.read(task=task,split=self.few_shot_split)
        if self.shot != 0:
            for i in range(min(self.shot, len(ds))):
                prompt += "\n\n" + self.build_example(ds[i], with_answer=True,choices = choices)
        return prompt

    def build_example(self, data, choices,with_answer=True,):
        question = data['question']
        if choices[0].isascii():
            choice = "\n".join(
                [
                    f'{item}. {data[item]}' for item in choices
                ]
            )
        else:
            choice = ''
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\nAnswer: {answer}"




if __name__ == '__main__':
    from tqdm import tqdm
    import time
    a = Mmlu(shot=5)
    for data in tqdm(a,total=len(a)):
        print(data)
        time.sleep(1)
