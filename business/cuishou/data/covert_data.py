import sys
import re
import json

prompt = '''你是360借条的一个催收话术分析专家。接下来我会发送一段贷款催收的对话文本给你，对话文本格式为：[句子ID]->角色:通话文本，示例：
[1]->客服：您好，我是三六零借条的
请你根据对话找到客服所说的内容与这14个要素匹配的文本片段，下面是14个要素：
1、自报身份
2、客户称呼确认
3、逾期金额确认
4、逾期天数确认
5、确认借款用途
6、通过征信施压
7、通过案件升级施压
8、通过取消客户分期施压
9、通过上平台黑名单施压
10、通过给客户发催收函施压
11、确认还款时间
12、确认还款金额
13、确认还款方式
14、通过联系紧急联系人施压
只对客服所说的话判断是否匹配上述14个要素分类，并按照
{'isProblem':'是否包含要素(回复yes|no)','problemSnippet':[{'sentenceId':'句子id，是字符串类型','text':'含有要素的句子','role':'角色','modelName':'要素名称','wordsList':'具体有要素的字'}]}
json格式返回结果。不要输出任何解释
'''

def process_conversation(text):
    role_pattern = re.compile(r'(客服|用户)(:)(\[\d+\])')
    text = re.sub(role_pattern,r'\n\3->\1：',text)
    return text

def process_answer(text):
    response = eval(text)
    for problem in response['problemSnippet']:
        yield f'<{problem["sentenceId"]}\t{problem["text"]}\t{problem["modelName"]}\t{problem["wordsList"]}>'


def main(which):
    fp = open(f'brp_{which}','w',encoding='utf8')
    with open(f'{which}','r',encoding='utf8') as f:
        for line in f.readlines():
            e = json.loads(line.strip('\n'))
            rounds = [
                {'prompt':prompt,'response':'好的，请提供通话文本，我会根据您的要求进行分析并只返回JSON格式的结果，不做文字解释。'},
                {'prompt':process_conversation(e['rounds'][-1]['prompt']),'response':e['rounds'][-1]['response']}
            ]
            e['rounds'] = rounds
            fp.write(json.dumps(e,ensure_ascii=False))
            fp.write('\n')
    fp.close()


if __name__ == '__main__':
    text = "客服:[1]哎，您好，您这边是那个钟磊宗乡是吧？用户:[2]好嘞。用户:[3]就是。客服:[4]哦，我这边是那个三六零借条的，呃，您这个平台上面这个当前的两千多块钱是欠款，你这边处理了吗？用户:[5]没有呢？"
    main('finetune_train_data_qifu_format.jsonl')
    main('data_file.txt')