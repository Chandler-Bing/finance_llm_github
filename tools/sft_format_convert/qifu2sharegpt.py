import json
import copy
import os

"""Qifu sft通用数据结构
https://gysepbka8x.feishu.cn/docx/V0i5dEOgXodvRPxeZh5cyqZ1nKh
{
    "id": "",                       //数据id
    "instruction": "",              // SFT任务的meta instruction
    "rounds": [                     // 多轮对话交互
        {
            "prompt": "",            // user prompt/user input
            "response": ""           // model response
        },
        {
            "prompt": "",
            "response": ""
        }
    ],
    "src": "",                       // 数据来源，如【新闻/网页/内部数据/...】
    "dataset": "",                   // 所属数据集，如【FinEval...】
    "task": "",                      // SFT任务类型，如【自由对话/问答/信息抽取/征信解读/...】
    "tag": [""]                      // 其他标签说明  
}
"""
"""shareGPT format
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
"""


def format_qifu2shareGPT(input_data: dict, 
                       role_mapping: dict = {"user": "human", "bot": "gpt"}
                       ):
    """
    将Qifu的格式转换为shareGPT的格式。
    :param input_data: Qifu的格式数据。
    :return: 
        output_data: shareGPT的格式数据。
        meta_data: Qifu_format 中对话之外的字段
    """
    # 处理Qifu的格式数据，将其转换为shareGPT的格式数据
    output_data = dict()
    meta_data = copy.deepcopy(input_data)
    system_prompt = meta_data.pop("instruction", "")
    rounds = meta_data.pop("rounds")    
    conv = []
    for _round in rounds:
        user_prompt = _round.get("prompt")
        bot_response = _round.get("response")
        
        conv.append({"from": role_mapping["user"], "value": user_prompt})
        conv.append({"from": role_mapping["bot"], "value": bot_response})
    
    output_data["system"] = system_prompt
    # 现阶段没有 tools, 留空
    # output_data["tools"] = "xxx"
    output_data["conversations"] = conv

    return output_data, meta_data

def format_qifu2shareGPT_batch(input_data, role_mapping: dict = {"user": "human", "bot": "gpt"}):
    """
    将Qifu的格式数据批量转换为shareGPT的格式数据。
    :param input_data: Qifu的格式数据, 可逐个读取数据即可。
    :param role_mapping: 角色映射字典，用于将Qifu的角色映射为shareGPT的角色。
    :return: 转换后的shareGPT格式数据。
    """
    for data_line in input_data:
        yield format_qifu2shareGPT(data_line, role_mapping)

def format_qifu2shareGPT_file(input_file_path, output_file_path, role_mapping: dict = {"user": "human", "bot": "gpt"}):
    output_data = []
    with open(input_file_path, 'r') as f: # load from jsonl file 
        for l in f.readlines():
            example = json.loads(l.strip())
            st, _ = format_qifu2shareGPT(example)
            output_data.append(st)
    
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
        print("mkdir", os.path.dirname(output_file_path))

    with open(output_file_path, "w", encoding="utf-8",) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("saved to", output_file_path)
    
if __name__ == "__main__":
    # 示例数据
    qifu_data = {
        "id": "123",
        "instruction": "欢迎使用智能客服系统，请输入您的问题。",
        "rounds": [{"prompt": "甲肝和乙肝有什么区别", "response": "这个我不不好说，需要专业医生来给出"},
                    {"prompt": "没关系，把你知道的说出来就好", "response": ""}
                    ],
        "true_answer": "只是为了方便后续对比处理，留空也没关系"
    }
    
    # 转换为ChatML格式数据
    print(format_qifu2shareGPT(qifu_data))
    
    # 批量转换
    batch_data = [qifu_data, qifu_data]
    for data in format_qifu2shareGPT_batch(batch_data):
        print(data)

    # input_file = "/data/oceanus_ctr/j-xinzhimin-jk/share_sh/data/combine0329/kmeans_coreset_general_sample+task_50000/train/train.jsonl"
    # output_file = "/data/oceanus_ctr/j-xinzhimin-jk/share_sh/data/combine0329_shareGPT_format/kmeans_coreset_general_sample+task_50000/train/train.json"
    # format_qifu2shareGPT_file(input_file, output_file)

    # input_file = "/data/oceanus_ctr/j-xinzhimin-jk/share_sh/data/combine0329/kmeans_coreset_general_sample+task_20000/train/train.jsonl"
    # output_file = "/data/oceanus_ctr/j-xinzhimin-jk/share_sh/data/combine0329_shareGPT_format/kmeans_coreset_general_sample+task_20000/train/train.json"
    # format_qifu2shareGPT_file(input_file, output_file)

