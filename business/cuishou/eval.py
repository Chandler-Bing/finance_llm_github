import json
import os

def fun(file_path):

    with open('marked.json', 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)

    examples = []
    results = []
    for j in os.listdir(file_path):
        with open(file_path+'/'+j, 'r', encoding='utf-8') as f:
            data = json.load(f)

            examples.extend(data['examples'])
            results.extend(data['results'])

    #json_data = json.dumps(data_list, ensure_ascii=False, indent=4)

    ## 将JSON数据写入文件
    #with open('1.json', 'w', encoding='utf-8') as json_file:
    #    json_file.write(json_data)
    #exit()


    #print(data_list[1])

    def find_outer_brackets(s):
        start = s.find('{')  # 查找第一个 '{' 的位置
        if start == -1:
            return None  # 如果没有找到，返回 None

        count = 0  # 用于跟踪当前开放的花括号数量
        for i in range(start, len(s)):
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
                if count == 0:
                    return s[start:i+1]  # 当计数器回到零时，返回子字符串

        return None  # 如果没有闭合的花括号对，返回 None


    #精度召回可解析

    total_num = 0
    evalable_num = 0

    acc_den = 0
    acc_nom = 0

    recall_den = 0
    recall_nom = 0


    for s, i in enumerate(examples):

        total_num += 1

        j = str(s)
        # k = examples[s][-1]['prompt']
        # m = results[s]['data']['content']
        k = examples[s]["rounds"][-1]['prompt']
        m = results[s]['Output']


        m = find_outer_brackets(m)

        #if s == 69:
        #    print(m)
        #    exit()


        try:
            if type(eval(m)) == dict:
                if 'problemSnippet' in eval(m).keys():
                    evalable_num += 1
                else:
                    print(m)
                    continue
            else:
                print(m)
                continue
        except Exception as e:
            print(m)
            continue


        # 精度
        for sentence_data in eval(m)['problemSnippet']:

            acc_den += 1

            # {'sentenceId': '1', 'text': '您好，刘桂莲刘先生，三六零借条的工作人员工号五九二八', 'role': '客服','modelName': '自报身份', 'wordsList': '三六零借条,工作人员,工号五九二八'}

            try:
                assert loaded_data[j][sentence_data['sentenceId']][0] in k


                marked = loaded_data[j][sentence_data['sentenceId']][1 :]
                if sentence_data['modelName'] in marked:
                    acc_nom += 1
                if sentence_data['modelName'] not in marked:
                    pass
                    # print(s)
                    #print(eval(m))
                    #print(eval(k))
                    #exit()
            except:
                #exit()
                continue


        # 召回

        for sentence_id, sentence_mark in loaded_data[j].items():
            #assert sentence_mark[0] in k
            for sentence_mark_single in sentence_mark[1 :]:
                if sentence_mark_single == 'nan':
                    continue
                recall_den += 1

                for sentence_data in eval(m)['problemSnippet']:
                    try:
                        if sentence_data['sentenceId'].strip() == sentence_id and sentence_data['modelName'].strip() == sentence_mark_single:
                            recall_nom += 1
                            break
                    except:
                        continue
    print("解析度", evalable_num / total_num)

    print("acc", acc_nom / acc_den)

    print("recall", recall_nom /  recall_den)



for file_path in [
    '/app/nfs_share_dir/5/boruipeng/finance_llm/sft_task/sft-llama3-qifu-8192-20k_merge/cuishou'
]:
    print(file_path)
    fun(file_path)
    print("\n\n===================")

