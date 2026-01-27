import os
import pandas as pd
from datasets import Dataset
from .common import CommonExam

class Cmmlu(CommonExam):
    def __init__(self,shot,data_path='./exams/cmmlu'):
        categories = [
            'agronomy',
            'anatomy',
            'ancient_chinese',
            'arts',
            'astronomy',
            'business_ethics',
            'chinese_civil_service_exam',
            'chinese_driving_rule',
            'chinese_food_culture',
            'chinese_foreign_policy',
            'chinese_history',
            'chinese_literature',
            'chinese_teacher_qualification',
            'clinical_knowledge',
            'college_actuarial_science',
            'college_education',
            'college_engineering_hydrology',
            'college_law',
            'college_mathematics',
            'college_medical_statistics',
            'college_medicine',
            'computer_science',
            'computer_security',
            'conceptual_physics',
            'construction_project_management',
            'economics',
            'education',
            'electrical_engineering',
            'elementary_chinese',
            'elementary_commonsense',
            'elementary_information_and_technology',
            'elementary_mathematics',
            'ethnology',
            'food_science',
            'genetics',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_geography',
            'high_school_mathematics',
            'high_school_physics',
            'high_school_politics',
            'human_sexuality',
            'international_law',
            'journalism',
            'jurisprudence',
            'legal_and_moral_basis',
            'logical',
            'machine_learning',
            'management',
            'marketing',
            'marxist_theory',
            'modern_chinese',
            'nutrition',
            'philosophy',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_study',
            'sociology',
            'sports_science',
            'traditional_chinese_medicine',
            'virology',
            'world_history',
            'world_religions',
        ]
        categories = {item:item for item in categories}
        prompt = '以下是单项选择题，请选出其中的正确答案。\n'
        super(Cmmlu, self).__init__(data_path=data_path,shot=shot,test_split='test',categories=categories,prompt=prompt)

    def read(self,task,split):
        filename = os.path.join(self.data_path,split,f'{task}.csv')
        #print(filename)
        df = pd.read_csv(filename,names=self.file_cols,header=None,skiprows=1)
        ds = Dataset.from_pandas(df)
        return ds





if __name__ == '__main__':
    from tqdm import tqdm
    import time
    import json
    a = Cmmlu(shot=0)
    #print(a.cot)
    for data in tqdm(a,total=len(a)):
        text = data['prompt']
        answer = data['answer']
        print(json.dumps({'text':text,'answer':answer,'tag':f'cmmlu-{data["subject"]}'},ensure_ascii=False))
        #time.sleep(1)