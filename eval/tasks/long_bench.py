from datasets import load_dataset
from .common import CommonExam
import json
import os
from .long_bench_metric import  (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

class LongBench(CommonExam):
    def __init__(self,shot=5,data_path='./exams/long_bench'):
        categories = ["multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh",]
        categories = {item:item for item in categories}

        self.dataset2metric = {
            "narrativeqa": qa_f1_score,
            "qasper": qa_f1_score,
            "multifieldqa_en": qa_f1_score,
            "multifieldqa_zh": qa_f1_zh_score,
            "hotpotqa": qa_f1_score,
            "2wikimqa": qa_f1_score,
            "musique": qa_f1_score,
            "dureader": rouge_zh_score,
            "gov_report": rouge_score,
            "qmsum": rouge_score,
            "multi_news": rouge_score,
            "vcsum": rouge_zh_score,
            "trec": classification_score,
            "triviaqa": qa_f1_score,
            "samsum": rouge_score,
            "lsht": classification_score,
            "passage_retrieval_en": retrieval_score,
            "passage_count": count_score,
            "passage_retrieval_zh": retrieval_zh_score,
            "lcc": code_sim_score,
            "repobench-p": code_sim_score,
        }
        self.dataset2prompt = {
            "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
            "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
            "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
            "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
            "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
            "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
            "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
            "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题，尽量从原文中提取文字回答。\n\n问题：{input}\n回答：",
            "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
            "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
            "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
            "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
            "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
            "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
            "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
            "lsht": "你是一个严格的新闻类别分类器，请根据以下新闻内容，从下面24个分类中选取一个作为新闻的类别:\n农业、农村\n军事\n文学、艺术\n体育\n传媒业\n电子信息产业\n文化、休闲娱乐\n社会、劳动\n经济\n"
                    "服务业、旅游业\n环境、气象\n能源、水务、水利\n财政、金融\n教育\n科学技术\n对外关系、国际关系\n矿业、工业\n政治\n交通运输、邮政、物流\n灾难、事故\n基本建设、建筑业、房地产\n医药、卫生\n法律、司法\n商业、外贸、海关\n"
                    "不要输出其他文字，以下是一些例子。\n\n{context}\n{input}",
            "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
            "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
            "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请给出摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案：",
            "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
            "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
        }
        self.dataset2maxlen={
            "narrativeqa": 128,
            "qasper": 128,
            "multifieldqa_en": 64,
            "multifieldqa_zh": 64,
            "hotpotqa": 32,
            "2wikimqa": 32,
            "musique": 32,
            "dureader": 128,
            "gov_report": 512,
            "qmsum": 512,
            "multi_news": 512,
            "vcsum": 512,
            "trec": 64,
            "triviaqa": 32,
            "samsum": 128,
            "lsht": 64,
            "passage_count": 32,
            "passage_retrieval_en": 32,
            "passage_retrieval_zh": 32,
            "lcc": 64,
            "repobench-p": 64
        }
        self.data = []
        super(LongBench, self).__init__(data_path=data_path,shot=shot,cot=True,categories=categories)

    def _build(self):
        for category in self.categories.keys():
            filename = os.path.join(self.data_path,'val',f'{category}.jsonl')
            with open(filename,'r',encoding='utf-8') as f:
                for line in f.readlines():
                    e = json.loads(line.strip('\n'))
                    self.data.append({
                        'subject': category,
                        'prompt': self.dataset2prompt[category].format(**e),
                        'answer': e['answers'],
                        'length': e['length'],
                        'language':e['language'],
                        'all_classes': e['all_classes'],
                        'metric': e.get('metric',''),
                        '_id':e['_id'],
                        'ori_answers':e.get('ori_answers',e['answers'])
                    })



if __name__ == '__main__':
    from tqdm import tqdm
    import time
    a = LongBench(shot=5)
    print(a.cot)
    for data in tqdm(a,total=len(a)):
        print(data)
        time.sleep(1)

