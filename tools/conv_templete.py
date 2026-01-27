from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Tuple

class SeparatorStyle:
    ADD_COLON_TWO = 0 # 轩辕 chat use
    QF = 10
    Naive = 11
    Detail = 13
    Detail_baichuan = 14
    Finma = 15
    Zhilu = 16
    Yayi = 17
    Baichuan = 18
    ChatGLM2 = 19
    Qwen = 20
    Llama3 = 21


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    instruction: str
    # Two roles
    roles: Sequence[str]
    # All messages. Each item is (role, message).
    rounds: List[Dict[str, str]]
    # Separators
    sep_style: int
    sep: str
    sep2: str = None
    # Stops generation if meeting any token in this list
    stop_tokens: List[str] = None

    def get_qa_list(self) -> List[Tuple[str]]:
        """Get the list of QA pairs with prompts and instruction for tokenize"""
        # [(q,a), (q,a)]
        # first turn, instruction and Q A  
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            round = self.rounds[0]
            ret = [ (f"{self.instruction}{self.sep}{self.roles[0]}{round['prompt']}{self.sep}{self.roles[1]}" 
                        if self.instruction else f"{self.roles[0]}{round['prompt']}{self.sep}{self.roles[1]}",
                    f"{round['response']}{self.sep2}")]
            for round in self.rounds[1:]:
                ret.append(( f"{self.roles[0]}{round['prompt']}{self.sep}{self.roles[1]}",
                    f"{round['response']}{self.sep2}"))
            return ret
        elif self.sep_style == SeparatorStyle.QF:
            round = self.rounds[0]
            ret = [ (f"<s>{self.instruction}{self.sep}{self.roles[0]}: {round['prompt']}{self.sep}{self.roles[1]}: ",
                    f"{round['response']}{self.sep2}")]
            for round in self.rounds[1:]:
                ret.append(( f"<s>{self.roles[0]}: {round['prompt']}{self.sep}{self.roles[1]}: ",
                    f"{round['response']}{self.sep2}"))
            return ret
        elif self.sep_style in [SeparatorStyle.Detail, SeparatorStyle.Detail_baichuan]:
            # 默认是基于 llama + 自定义特殊token 的文本
            round = self.rounds[0]
            instruction_prompt = f"<|instruction|>{self.instruction}</|instruction|>" if self.instruction else ""
            ret = [ (f"<s>{instruction_prompt}<|human|>{round['prompt']}</|human|><|assistant|>",
                    f"{round['response']}</|assistant|>")]
            for round in self.rounds[1:]:
                ret.append(( f"<|human|>{round['prompt']}</|human|><|assistant|>",
                    f"{round['response']}</|assistant|>"))
            ret[-1] = (ret[-1][0], ret[-1][1] + "</s>")
            
            # 如果是baichuan，就把 llama 中加入的特殊 token 替换成baichuan预留的字段
            if self.sep_style == SeparatorStyle.Detail_baichuan:
                replace_dict = {
                    "<|human|>": "<reserved_21>", "</|human|>": "<reserved_22>",
                    "<|assistant|>": "<reserved_23>", "</|assistant|>": "<reserved_24>",

                    "<|system|>": "<reserved_25>", "</|system|>": "<reserved_26>",
                    "<|instruction|>": "<reserved_27>", "</|instruction|>": "<reserved_28>",
                    "<|inthought|>": "<reserved_29>", "</|inthought|>": "<reserved_30>",
                    "<|api|>": "<reserved_31>", "</|api|>": "<reserved_32>",
                    "<|code|>": "<reserved_33>", "</|code|>": "<reserved_34>",
                    "<|md|>": "<reserved_35>", "</|md|>": "<reserved_36>",
                    
                    "<|math|>": "<reserved_37>", "</|math|>": "<reserved_38>",
                    "<|file|>": "<reserved_39>", "</|file|>": "<reserved_40>",
                    
                    "<|extra_1|>": "<reserved_41>", "</|extra_1|>": "<reserved_42>",
                    "<|extra_2|>": "<reserved_43>", "</|extra_2|>": "<reserved_44>",
                }
                for i in range(len(ret)):
                    for k, v in replace_dict.items():
                        ret[i] = (ret[i][0].replace(k, v), ret[i][1].replace(k, v))
            return ret
        elif self.sep_style == SeparatorStyle.Finma:
            round = self.rounds[0]
            ret = [ (f"<s>{self.roles[0]}: \n{self.instruction} {round['prompt']}\n\n{self.roles[1]}: \n",
                    f"{round['response']}{self.sep2}")]
            for round in self.rounds[1:]:
                ret.append(( f"<s>{self.roles[0]}: {round['prompt']}\n\n{self.roles[1]}: \n",
                    f"{round['response']}{self.sep2}"))
            return ret
        elif self.sep_style == SeparatorStyle.Naive:
            round = self.rounds[0]
            ret = [ (f"{self.instruction}{round['prompt']}",
                    f"{round['response']}")]
            for round in self.rounds[1:]:
                ret.append(( f"{round['prompt']}",
                    f"{round['response']}"))
            return ret
        elif self.sep_style == SeparatorStyle.Zhilu:
            round = self.rounds[0]
            ret = [ (f"<s>[INST] <<SYS>>\n{self.instruction}\n<</SYS>>\n\n{round['prompt']} [/INST]",
                    f"{round['response']}</s>")]
            for round in self.rounds[1:]:
                ret.append(( f"[INST] {round['prompt']} [/INST]",
                    f"{round['response']}</s>"))
            return ret
        elif self.sep_style == SeparatorStyle.Yayi:
            round = self.rounds[0]
            ret = [ (f"<|System|>:\n{self.instruction}\n\n\n<|Human|>:\n{round['prompt']}\n\n\n<|YaYi|>:\n",
                    f"{round['response']}")]
            for round in self.rounds[1:]:
                ret.append(( f"\n\n\n<|Human|>:\n{round['prompt']}\n\n\n<|YaYi|>:\n",
                    f"{round['response']}"))
            ret[-1] = (ret[-1][0], ret[-1][1] + "<|End|>")
            return ret
        elif self.sep_style == SeparatorStyle.Baichuan:
            round = self.rounds[0]
            ret = [ (f"{self.instruction}{self.roles[0]}{round['prompt']}{self.roles[1]}",
                    f"{round['response']}")]
            for round in self.rounds[1:]:
                ret.append(( f"{self.roles[0]}{round['prompt']}{self.roles[1]}",
                    f"{round['response']}"))
            ret[-1] = (ret[-1][0], ret[-1][1] + "</s>")
            return ret
        elif self.sep_style == SeparatorStyle.ChatGLM2:
            round = self.rounds[0]
            ret = [ (f"{self.instruction}\n\n{self.roles[0]}：{round['prompt']}\n\n{self.roles[1]}：",
                    f"{round['response']}")]
            for round in self.rounds[1:]:
                ret.append(( f"\n\n{self.roles[0]}：{round['prompt']}\n\n{self.roles[1]}：",
                    f"{round['response']}"))
            return ret
        elif self.sep_style == SeparatorStyle.Qwen:
            # prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
            round = self.rounds[0]
            instruction_prompt = f"<|im_start|>system\n{self.instruction}<|im_end|>\n" if self.instruction else ""
            ret = [ (f"{instruction_prompt}<|im_start|>user\n{round['prompt']}<|im_end|>\n<|im_start|>assistant\n",
                    f"{round['response']}<|im_end|>")]
            for round in self.rounds[1:]:
                ret.append(( f"\n<|im_start|>user\n{round['prompt']}<|im_end|>\n<|im_start|>assistant\n",
                    f"{round['response']}<|im_end|>"))
            ret[-1] = (ret[-1][0], ret[-1][1] + "<|endoftext|>")
            return ret
        elif self.sep_style == SeparatorStyle.Llama3:
            # prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
            round = self.rounds[0]
            instruction_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{self.instruction}<|eot_id|>" if self.instruction else ""
            ret = [ (f"<|begin_of_text|>{instruction_prompt}<|start_header_id|>user<|end_header_id|>\n\n{round['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{round['response']}<|eot_id|>")]
            for round in self.rounds[1:]:
                ret.append((f"<|start_header_id|>user<|end_header_id|>\n\n{round['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    f"{round['response']}<|eot_id|>"))
            ret[-1] = (ret[-1][0], ret[-1][1] + "<|end_of_text|>")
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt(self) -> str:
        """Get the prompt for generation. 最后一个qa_pair仅构造到q,不构造a"""
        qa_pairs = self.get_qa_list()
        return "".join(["".join(qa_pair) for qa_pair in qa_pairs[:-1]]) + qa_pairs[-1][0]

    def append_round(self, round):
        """Append a new message round."""
        self.rounds.append(round)

    def copy(self):
        return Conversation(
            name=self.name,
            instruction=self.instruction,
            roles=self.roles,
            rounds=self.rounds.copy(),
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_tokens=self.stop_tokens,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "instruction": self.instruction,
            "roles": self.roles,
            "rounds": self.rounds,
        }



# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
                template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

# xuanyuan
register_conv_template(
    Conversation(
        name="XuanYuan-Chat",
        instruction="以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n",
        roles=("Human: ", "Assistant:"),
        rounds=[],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
        stop_tokens=["</s>"],
    )
)

# atom-chat
register_conv_template(
    Conversation(
        name="atom",
        instruction="<s>",
        roles=("Human: ", "Assistant:"),
        rounds=[],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n</s><s>",
        sep2="</s>",
        stop_tokens=["</s>"],
    )
)

"""Qwen template
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat  https://huggingface.co/jxy/Tongyi-Finance-14B-Chat
chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
"""
register_conv_template(
    Conversation(
        name="qwen",
        instruction="You are a helpful assistant.",
        roles=("user", "assistant"),
        rounds=[],
        sep_style=SeparatorStyle.Qwen,
        sep="",
        stop_tokens=["<|im_end|>",],
    )
)

# finma templete
register_conv_template(
    Conversation(
        name="finma",
        instruction="",
        # f'Human: \n{ctx}\n\nAssistant: \n'
        roles=("Human", "Assistant"),
        rounds=[],
        sep_style=SeparatorStyle.Finma,
        sep="",
        sep2="</s>",
        stop_tokens=["</s>"],
    )
)

# old qifu tec templete
register_conv_template(
    Conversation(
        name="old-qifu-chat",
        instruction="",
        roles=("<|Human|>", "<|Assistant|>"),
        rounds=[],
        sep_style=SeparatorStyle.QF,
        sep="</s><s>",
        sep2="</s>",
        stop_tokens=["</s>"],
    )
)

# 问 答 query templete
register_conv_template(
    Conversation(
        name="qaq",
        instruction="",
        roles=("问", "答"),
        rounds=[],
        sep_style=SeparatorStyle.QF,
        sep="</s><s>",
        sep2="</s>",
        stop_tokens=["</s>"],
    )
)

# detail: 全部决策都设置为 special_token
register_conv_template(
    Conversation(
        name="detail",
        instruction="",
        roles=("human", "bot"),
        rounds=[],
        sep_style=SeparatorStyle.Detail,
        sep="",
        sep2="",
        stop_tokens=["</s>", "</|assistant|>", ],
    )
)

# detail-baichuan: 全部决策都设置为 special_token
register_conv_template(
    Conversation(
        name="detail-baichuan",
        instruction="",
        roles=("human", "bot"),
        rounds=[],
        sep_style=SeparatorStyle.Detail_baichuan,
        sep="",
        sep2="",
        stop_tokens=["</s>", "<reserved_24>", ], 
    )
)


register_conv_template(
    Conversation(
        name="zhilu",
        instruction="You are a helpful assistant. 你是一个乐于助人的助手。",
        roles=("user", "ZhiLu"),
        rounds=[],
        sep_style=SeparatorStyle.Zhilu,
        sep="",
        sep2="",
        stop_tokens=["</s>"],
    )
)

register_conv_template(
    Conversation(
        name="yayi",
        instruction="You are a helpful, respectful and honest assistant named YaYi developed by Beijing Wenge Technology Co.,Ltd. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        roles=("human", "YaYi"),
        rounds=[],
        sep_style=SeparatorStyle.Yayi,
        sep="",
        sep2="",
        stop_tokens=["</s>", "<|Human|>", "<|YaYi|>", "<|End|>"],  # </s> <|Human|> <|YaYi|> <|End|>
    )
)

"""Baichuan template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py#L31
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan",
        instruction="",
        roles=("<reserved_102>", "<reserved_103>"),
        rounds=[],
        sep_style=SeparatorStyle.Baichuan,
        sep="",
        sep2="",
        stop_tokens=["</s>"],
    )
)

"""Baichuan2 template
Support: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
         https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan2",
        instruction="",
        roles=("<reserved_106>", "<reserved_107>"),
        rounds=[],
        sep_style=SeparatorStyle.Baichuan,
        sep="",
        sep2="",
        stop_tokens=["</s>"],
    )
)

"""ChatGLM2 template
Support: https://huggingface.co/THUDM/chatglm2-6b
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    Conversation(
        name="chatglm2",
        instruction="",
        roles=("问", "答"),
        rounds=[],
        sep_style=SeparatorStyle.ChatGLM2,
        sep="\n\n",
        stop_tokens=["</s>"],
    )
)

"""ChatGLM3 template
Support: https://huggingface.co/THUDM/chatglm3-6b
source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
"""
register_conv_template(
    Conversation(
        name="chatglm3",
        instruction="",
        roles=("<|user|>\n", "<|assistant|>"),
        rounds=[],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="",
        sep2="\n",
        stop_tokens=["<|user|>"],  # <|user|>
    )
)

"""
Llama3
"""
register_conv_template(
    Conversation(
        name="llama3",
        instruction="You are a helpful assistant.",
        roles=("user", "bot"),
        rounds=[],
        sep_style=SeparatorStyle.Llama3,
        sep="",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
    )
)

"""
Llama3-empty
"""
register_conv_template(
    Conversation(
        name="llama3-empty",
        instruction="",
        roles=("user", "bot"),
        rounds=[],
        sep_style=SeparatorStyle.Llama3,
        sep="",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
    )
)

# build yourself, check everything
register_conv_template(
    Conversation(
        name="naive",
        instruction="",
        roles=("", ""),
        rounds=[],
        sep_style=SeparatorStyle.Naive,
        sep="",
        stop_tokens=["</s>"],
    )
)

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()
