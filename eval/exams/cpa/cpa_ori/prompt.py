TASK2DESC = {
        'cpa_company':'注册会计师《公司战略与风险管理》科目',
        'cpa_tax':'注册会计师《税法》科目',
        'cpa_financial_management':'注册会计师《财管》科目',
        'cpa_accounting':'注册会计师《会计》科目',
        'cpa_economic_law':'注册会计师《经济法》科目',
        'cpa_auditing':'注册会计师《审计》科目',
    }

#prompt = f"以下是中国关于{TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"

def build_example(data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
                ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"

import os
import json
import pandas as pd

for filename in os.listdir('./'):
    if filename.endswith('.csv') and 'train' in filename:
        basename = filename.split('.csv')[0]
        df = pd.read_csv(filename)
        for index,row in df.iterrows():
            prompt = f"以下是中国关于{TASK2DESC[basename]}考试的单项选择题，请选出其中的正确答案。\n"
            prompt += '\n' + build_example(row)
            dic = {}
            dic['tag'] = 'cpa'
            dic['text'] = prompt
            print(json.dumps(dic,ensure_ascii=False))
