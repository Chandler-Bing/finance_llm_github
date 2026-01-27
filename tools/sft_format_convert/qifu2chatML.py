import json
import copy


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

def format_qifu2chatML(input_data: dict, 
                       role_mapping: dict = {"system": "system", "user": "user", "bot": "assistant"}
                       ):
    """
    将Qifu的格式转换为ChatML的格式。
    :param input_data: Qifu的格式数据。
    :param output_data: ChatML的格式数据。
    :return: 
        output_data: ChatML的格式数据。
        meta_data: Qifu_format 中对话之外的字段
    """
    # 处理Qifu的格式数据，将其转换为ChatML的格式数据
    # ...
    # 将转换后的数据添加到output_data列表中
    output_data = []
    meta_data = copy.deepcopy(input_data)
    system_prompt = meta_data.pop("instruction", "")
    rounds = meta_data.pop("rounds")
    
    if system_prompt:
        output_data.append({"role": role_mapping["system"], "content": system_prompt})
    
    for _round in rounds:
        user_prompt = _round.get("prompt")
        bot_response = _round.get("response")
        
        output_data.append({"role": role_mapping["user"], "content": user_prompt})
        output_data.append({"role": role_mapping["bot"], "content": bot_response})
    
    return output_data, meta_data

def format_qifu2chatML_batch(input_data, role_mapping: dict = {"system": "system", "user": "user", "bot": "assistant"}):
    """
    将Qifu的格式数据批量转换为ChatML的格式数据。
    :param input_data: Qifu的格式数据, 可逐个读取数据即可。
    :param role_mapping: 角色映射字典，用于将Qifu的角色映射为ChatML的角色。
    :return: 转换后的ChatML格式数据。
    """
    for data_line in input_data:
        yield format_qifu2chatML(data_line, role_mapping)

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
    print(format_qifu2chatML(qifu_data))
    
    # 批量转换
    batch_data = [qifu_data, qifu_data]
    for data in format_qifu2chatML_batch(batch_data):
        print(data)