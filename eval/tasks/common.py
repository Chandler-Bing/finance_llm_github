import os
import pandas as pd
import loguru
from datasets import Dataset


class CommonExam():
    def __init__(
            self,
            data_path,
            shot,
            categories,
            prompt='以下是关于{}的单项选择题，请直接给出正确答案的选项。\n',
            file_cols=None,
            cot = False,
            few_shot_split='dev',
            test_split='val',
    ):
        if file_cols is None:
            file_cols = ['id', 'question', 'A', 'B', 'C', 'D', 'answer']
        self.data_path = data_path
        self.shot = shot
        self.cot = cot
        self.file_cols = file_cols
        self.prompt = prompt
        self.few_shot_split = few_shot_split
        self.test_split = test_split
        self.categories = categories
        self.data = []
        loguru.logger.info(f'building dataset {self.data_path}...... please wait for about 1 min')
        self._build()

    def _build(self):
        for task in self.categories.keys():
            choices = ['A','B','C','D']
            for data in self.read(task,split=self.test_split):
                prompt = self.get_few_shot(task=task,choices = choices) + '\n' + self.build_example(data=data,choices = choices,with_answer=False)
                self.data.append({
                    'subject':task,
                    'prompt':prompt,
                    'choices':choices,
                    'answer':data['answer']
                })

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
        return f"题目：{question}\n{choice}\n答案是：{answer}"

    def read(self,task,split):
        filename = os.path.join(self.data_path,split,f'{task}_{split}.csv')
        #print(filename)
        if self.cot:
            df = pd.read_csv(filename,names=self.file_cols+['explanation'],header=None,skiprows=1)
        else:
            df = pd.read_csv(filename,names=self.file_cols,header=None,skiprows=1)
        ds = Dataset.from_pandas(df)
        return ds

    def get_few_shot(self,task,choices):
        prompt = self.prompt.format(self.categories[task])
        ds = self.read(task=task,split=self.few_shot_split,)
        if self.shot != 0:
            for i in range(min(self.shot, len(ds))):
                prompt += "\n" + self.build_example(ds[i], with_answer=True,choices = choices)
        return prompt

    def __iter__(self):
        for item in self.data:
            yield item
        # for task in self.categories.keys():
        #     choices = ['A','B','C','D']
        #     for data in self.read(task,split=self.test_split):
        #         prompt = self.get_few_shot(task=task,choices = choices) + '\n' + self.build_example(data=data,choices = choices,with_answer=False)
        #         yield {
        #             'subject':task,
        #             'prompt':prompt,
        #             'choices':choices,
        #             'answer':data['answer']
        #         }
    def __len__(self):
        return len(self.data)
        # total = 0
        # for task in self.categories.keys():
        #     total += len(self.read(task,split=self.test_split))
        # return total

    def submit(self,datas,output_dir):
        pass



