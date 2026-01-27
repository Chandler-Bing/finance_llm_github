import argparse
import os
import re
import jieba
import string
from rouge import Rouge


def extract_choice(response: str) -> str:
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    if response == '':
        return ""
    choices = ["A", "B", "C", "D", "E"]
    if response == '':
        return ""
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCDE])', 3),
        (r'答案(是|为)选项 ?([ABCDE])', 2),
        (r'故?选择?：? ?([ABCDE])',1),
        (r'([ABCDE]) ?选?项(是|为)?正确',1),
        (r'正确的?选项(是|为) ?([ABCDE])',2),
        (r'答案(应该)?(是|为)([ABCDE])',3),
        (r'选项 ?([ABCDE]) ?(是|为)?正确',1),
        (r'选择答案 ?([ABCDE])',1),
        (r'答案?：?([ABCDE])',1),
        (r'([ABCDE])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCDE])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCDE])', 3), # chatgpt
        (r'选项([ABCDE])是最恰当的', 1),
        (r'选项([ABCDE]).*最恰当', 1),
        (r'选项([ABCDE]).*最能恰当', 1),
        (r'选项([ABCDE]).*最能', 1),
        (r'最恰当.*是选项([ABCDE])', 1),
        (r'correct answer is.*([ABCDE])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCDE])(.*?)当选', 1),
        (r'([ABCDE])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCDE])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioned choices
    pattern = r'^[^ABCDE]*([ABCDE])[^ABCDE]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    # 5. Check the only mentioned choices in the start of the sentence
    m = re.match(pattern, response[:4])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    m = re.match(pattern, response[:2])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    return ""


def extract_yn(response: str) -> str:
    choices = ["是", "否", "对", "错"]

    if response == '':
        return ""

    # Single match
    patterns = [
        (r'([是对])[ ？]*正确', 1),
        (r'([否错])[ ？]*错误', 1),
        (r'([是对])', 1),
        (r'([否错])', 1),
    ]

    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            if answer in choices:
                return answer

    return ""


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def fin_alipay_get_score(data):
    check_ture_false_list = ["安全合规+金融合规性", "安全合规+金融问题识别", "安全合规+信息安全合规", "安全合规+金融事实性"]
    if data['subject'] in check_ture_false_list:
        data['extract_answer'] = extract_yn(data['pred_answer'])
    else:
        data['extract_answer'] = extract_choice(data['pred_answer'])
    data['correct'] = 1 if data['extract_answer'] == data['answer'] else 0

    return data


def exact_match(prediction, ground_truth, **kwargs):
    return 1 if prediction==ground_truth else 0

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def acc_match(prediction, ground_truth, **kwargs):
    if ground_truth in prediction:
        return len(ground_truth) / len(prediction)
    return 0

if __name__ == '__main__':
    d = fin_alipay_get_score({'subject':'安全合规+金融合规性','pred_answer':'是','answer':'是'})
    print(d)