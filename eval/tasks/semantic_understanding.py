import os
import json
from .common import CommonExam

class SemanticUnderstanding(CommonExam):
    def __init__(self,shot,data_path='./exams/semantic_understanding'):
        super(SemanticUnderstanding, self).__init__(data_path=data_path,shot=shot,categories={})


    def _build(self):
        choices = ['A','B','C','D']
        tasks = ['llsrc','slpwc','slrfc','slsrc']
        for task in tasks:
            json_file = os.path.join(self.data_path,f'{task}.jsonl')
            with open(json_file,'r',encoding='utf-8') as f:
                for line in f.readlines():
                    e = json.loads(line.strip('\n'))
                    self.data.append(
                        {
                            'subject':task,
                            'prompt':e['rounds'][0]['prompt'],
                            'choices':choices,
                            'answer':e['rounds'][0]['response'],
                        }
                    )



if __name__ == '__main__':
    a = SemanticUnderstanding(shot=5)
    from tqdm import tqdm
    import time
    import json
    #print(a.cot)
    for data in tqdm(a,total=len(a)):
        print(data)
        # text = data['prompt']
        # answer = data['answer']
        # print(json.dumps({'text':text,'answer':answer,'tag':f'cmmlu-{data["subject"]}'},ensure_ascii=False))
        #time.sleep(1)







