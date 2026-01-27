from .common import CommonExam

class FinIQ(CommonExam):
    def __init__(self,shot,data_path='./exams/fin_IQ'):
        categories = [
            '注册会计师（CPA）',
            '银行从业资格',
            '证券从业资格',
            '基金从业资格',
            '保险从业资格CICE',
            '经济师',
            '税务师',
            '期货从业资格',
            '理财规划师',
            '精算师-金融数学',
        ]
        categories = {item:item for item in categories}
        super(FinIQ, self).__init__(data_path=data_path,shot=shot,categories = categories)






if __name__ == '__main__':
    from tqdm import tqdm
    import time
    import json
    a = FinIQ(shot=5)
    for data in tqdm(a,total=len(a)):
        text = data['prompt']
        answer = data['answer']
        print(json.dumps({'text':text,'answer':answer,'tag':f'fin_IQ-{data["subject"]}'},ensure_ascii=False))
        #time.sleep(1)