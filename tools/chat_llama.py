"""
qifu.tec style Llama model for chat

some simple demo with Chat Model, run 
# CUDA_VISIBLE_DEVICES=0,1 python tools/chat_llama.py --model_path=/sft/model/path
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from attr import dataclass
from threading import Thread
import argparse

import torch
from transformers import TextIteratorStreamer

try:
    # 从本地导入，防止因为版本不同导致的问题
    from modeling_llama import LlamaForCausalLM, LlamaConfig, logger
    from tokenization_llama import LlamaTokenizer
except ImportError:
    # 也可以直接从 transformers 库导入，问题不大
    from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
    from transformers.utils import logging
    logger = logging.get_logger(__name__)


@dataclass
class ChatTemplete:
    system_instruction_templete = "<s>{system_name}{system_instruction}</|instruction|>"
    q_templete = "{user}{input}</|human|>{bot}"
    a_templete = "{output}</|assistant|>"
    system_name = "<|instruction|>"
    user_name = "<|human|>"
    bot_name = "<|assistant|>"

    def build_system_instruction(self, system_instruction):
        return self.system_instruction_templete.format_map({"system_name":self.system_name, "system_instruction":system_instruction})

    def build_q(self, q):
        return self.q_templete.format_map({"user":self.user_name, "input":q, "bot":self.bot_name})

    def build_a(self, a):
        return self.a_templete.format_map({"output":a})
    
    def build_qa(self, q, a):
        return self.build_q(q) + self.build_a(a)


class LLamaTokenizerForChat(LlamaTokenizer):
    """
没有修改 LlamaTokenizer 原有的方法和属性，只是增加了 构造chat形式文本的 方法 build_chat_input
    """
    def __init__(
        self,
        vocab_file,
        chat_templete: ChatTemplete = ChatTemplete(),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=True,
        spaces_between_special_tokens=False,
        legacy=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sp_model_kwargs=sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            **kwargs,)
        
        self._set_chat_templete(chat_templete)

    def _set_chat_templete(self, chat_template: ChatTemplete):
        self.chat_templete = chat_template

    def build_chat_input(self, new_human_input: str, history: List[Dict[str, str]]=None, system_instruction="") -> str:
        """
- history :: [{"prompt": str, "response": str  }, ...]  
        """
        ct = self.chat_templete
        ret = [ct.build_system_instruction(system_instruction)]
        if history is not None:
            for qa_round in history:
                ret.append(ct.build_qa(qa_round["prompt"], qa_round["response"]))
        ret.append(ct.build_q(new_human_input))
        ret = "".join(ret)
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ):
        ret = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        cls._set_chat_templete(cls, ChatTemplete())
        return ret


class LlamaForChat(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, tokenizer_path: Union[str, os.PathLike],):
        super().__init__(config)
        self._set_tokenizer(tokenizer_path)
    
    def _set_tokenizer(self, tokenizer_path: Union[str, os.PathLike]):
        self.tokenizer = LLamaTokenizerForChat.from_pretrained(tokenizer_path)

    @torch.inference_mode()
    def chat(
        self,
        text: str,
        history=None,
        system_instruction="",
        max_new_tokens=512,
        temperature=0.4,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
        context_len=4096,
        eos_token_id_list=[2, 65001, 65003, 65005, ],  # </s> </|human|> </|assistant|> </|instruction|>
    ) -> str:
        torch.cuda.empty_cache()
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id_list if eos_token_id_list else None,
        )

        max_src_len = context_len - max_new_tokens - 2 # -2 to make sure the length is enough
        chat_input = self.tokenizer.build_chat_input(text, history, system_instruction)
        input_ids = self.tokenizer(chat_input).input_ids
        if len(input_ids) > max_src_len:
            logger.warning(f"Panic! Input len {len(input_ids)} is too long for {max_new_tokens} new tokens! Cut it !")
            input_ids = input_ids[-max_src_len:]
        output_ids = self.generate(
            input_ids=torch.as_tensor([input_ids]).to(self.device),
            **generation_config,
        )[0]

        # decode and split response
        while output_ids[-1] in eos_token_id_list:
            output_ids = output_ids[:-1]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        l_prompt = len(self.tokenizer.decode(input_ids, skip_special_tokens=False))
        output = output[l_prompt:]
        return output

    @torch.inference_mode()
    def stream_chat(
        self,
        text: str,
        history=None,
        system_instruction="",
        max_new_tokens=512,
        temperature=0.4,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
        context_len=4096,
        eos_token_id_list=[2, 65001, 65003, 65005, ],  # </s> </|human|> </|assistant|> </|instruction|>
        timeout=10.0
    ):
        
        torch.cuda.empty_cache()
        streamer = TextIteratorStreamer(self.tokenizer, timeout=timeout, skip_prompt=True, skip_special_tokens=True)
        
        max_src_len = context_len - max_new_tokens - 2 # -2 to make sure the length is enough
        chat_input = self.tokenizer.build_chat_input(text, history, system_instruction)
        input_ids = self.tokenizer(chat_input).input_ids
        if len(input_ids) > max_src_len:
            logger.warning(f"Panic! Input len {len(input_ids)} is too long for {max_new_tokens} new tokens! Cut it !")
            input_ids = input_ids[-max_src_len:]
        
        generation_kws = dict(
            input_ids=torch.as_tensor([input_ids]).to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id_list if eos_token_id_list else None,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kws)
        thread.start()

        stop_str_list = [self.tokenizer.decode(eos_token_id) for eos_token_id in eos_token_id_list]
        for new_text in streamer:
            yield new_text
        return 

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        pretrained_tokenizer_name_or_path: Union[str, os.PathLike] = None,
        **kwargs,
    ):
        tokenizer_path = pretrained_tokenizer_name_or_path if pretrained_tokenizer_name_or_path is not None else pretrained_model_name_or_path
        ret = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, tokenizer_path=tokenizer_path, **kwargs)
        return ret


if __name__=="__main__":
    # CUDA_VISIBLE_DEVICES=0,1 python tools/chat_llama.py --model_path=/sft/model/path
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_model_path', default=None, type=str, required=True)
    args = parser.parse_args()

    model_path = args.sft_model_path
    tokenizer_path = model_path

    #########  test tokenizer
    tokenizer = LLamaTokenizerForChat.from_pretrained(tokenizer_path)
    chat_info = tokenizer.build_chat_input(
        new_human_input="请详细介绍一下你自己",
        history=[
            {
                "prompt": "你好",
                "response": "你好啊，我是QifuGPT，有什么可以帮助你的"
            },
        ],
        system_instruction="你是QifuGPT，一个人工智能"
    )
    print([chat_info])
    
    print("loading model ...")
    model = LlamaForChat.from_pretrained(model_path, pretrained_tokenizer_name_or_path=tokenizer_path, device_map="auto")

    #########  basic chat demo  ##########
    history = [
        {
            "prompt": "你好",
            "response": "你好啊，我是QifuGPT，有什么可以帮助你的"
        },
    ]
    input_text = "介绍一下你自己"
    response = model.chat( text=input_text,
            history=history,
            system_instruction="你是QifuGPT，一个人工智能"
        )
    history.append({"prompt": input_text, "response": response})
    print("normal chat response:", response)

    #########  stream_chat_demo  ##########
    chat_info = tokenizer.build_chat_input(
        new_human_input="请写一篇关于酸奶的营销文案",
        history=history,
        system_instruction="你是QifuGPT，一个人工智能"
    )
    print([chat_info])
    print("**** stream chat response:")
    response = ""
    response_stream = model.stream_chat(
        text="请写一篇关于酸奶的营销文案",
        history=history,
        system_instruction="你是QifuGPT，一个人工智能"
    )
    for new_text in response_stream:
        print(new_text, end="", flush=True)
        response += new_text
    print()
    history.append({"prompt": "请写一篇关于酸奶的营销文案", "response": response})


    ########    interactive stream chat demo  ##########
    print("\n**** clear all history!  new chat start ...")
    system_instruction = ""
    history = []
    while True:
        input_text = input("Human: ")
        if input_text == "!!exit":
            print("Bot: bye ...")
            break
        
        response = ""
        response_stream = model.stream_chat(
            text=input_text,
            history=history,
            system_instruction=system_instruction
        )
        print("Bot: ", end="")
        for new_text in response_stream:
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history.append({"prompt": input_text, "response": response})
