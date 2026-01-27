# -*- coding: utf-8 -*-
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation.utils import GenerationConfig


from tools.sft_format_convert.qifu2chatML import format_qifu2chatML


MODEL_CLASSES = {
    "chatglm3": (AutoModel, AutoTokenizer, {}),
    "zhinao": (AutoModelForCausalLM, AutoTokenizer, {"use_flash_attn": False}),
    "internLM": (AutoModelForCausalLM, AutoTokenizer, {}),
    "Yi": (AutoModelForCausalLM, AutoTokenizer, {}),
}

def chatglm3_model_chat(model, tokenizer, qifu_format_data, generation_config):
    chatML_data, meta_data = format_qifu2chatML(qifu_format_data)
    history, prompt, _ = chatML_data[:-2], chatML_data[-2], chatML_data[-1]
    response, new_history = model.chat(
        tokenizer, prompt, history=history, 
        **generation_config,)
    return response

def zhinao_model_chat(model, tokenizer, qifu_format_data, generation_config):
    chatML_data, meta_data = format_qifu2chatML(qifu_format_data)
    
    generation_config = GenerationConfig(**generation_config)
    
    messages = chatML_data[:-1]

    response = model.chat(
            tokenizer=tokenizer, 
            messages=messages,
            system=None,
            stream=False,
            use_pot=True,
            generation_config=generation_config,)

    return response

def internLM_model_chat(model, tokenizer, qifu_format_data, generation_config):
    rounds =  qifu_format_data["rounds"]            
    prompt = rounds[-1]['prompt']
    rounds = rounds[:-1]

    history = []
    for _ in rounds:
        history.append((_['prompt'], _['response']))

    response, new_history = model.chat(
        tokenizer, prompt, history=history, 
        **generation_config,)
    
    # 很奇怪，绝大多数回复都有一个 不应出现的 [UNUSED_TOKEN_145] 
    if response.endswith("[UNUSED_TOKEN_145]"):
        response = response[:-len("[UNUSED_TOKEN_145]")]
    return response

def Yi_model_chat(model, tokenizer, qifu_format_data, generation_config):
    chatML_data, meta_data = format_qifu2chatML(qifu_format_data)
    messages = chatML_data[:-1]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    
    output_ids = model.generate(input_ids.to(model.device),
                                **generation_config)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

class SimpleChatIO:
    def __init__(self, save_path) -> None:
        self.save_path = save_path

    def prompt_for_input(self, role) -> str:
        text =  input(f"{role}: ")
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(f"{role}: {text}\n")
        return text

    def prompt_for_output(self, role: str):
        self.print2save_and_stdout(f"{role}: ", end="")

    def get_output(self, output_stream):
        self.print2save_and_stdout(output_stream)
        return output_stream

    def print2save_and_stdout(self, text: str, end="\n"):
        with open(self.save_path, "a", encoding="utf-8") as f:
            print(text, file=f, end=end)
        print(text, flush=True, end=end)

class SimpleConv:
    rounds = []
    instruction = ""

    def to_dict(self):
        return {
            "rounds": self.rounds,
            "instruction": self.instruction,
        }

    def clear(self):
        self.rounds = []
        self.instruction = ""

MODEL_CHAT_FUN = {
    "chatglm3": chatglm3_model_chat,
    "zhinao": zhinao_model_chat,
    "internLM": internLM_model_chat,
    "Yi": Yi_model_chat,
}

@torch.inference_mode()
def generate_answer(
        model_type,
        model,
        tokenizer,
        input_data,
        max_new_tokens=100,
        temperature=0.1,
        top_k=5,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
):
    torch.cuda.empty_cache()
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
    )

    chat_fun = MODEL_CHAT_FUN[model_type]
    response = chat_fun(model, tokenizer, input_data, generation_config)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit (for lora)')
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="alpaca", type=str, help="Prompt template name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--sample_gen_config', default=None, type=str, 
                        help="A jsonl file that contains the generation configuration for each sample in --data_file (max_new_tokens, temperature, repetition_penalty, )")
    parser.add_argument('--gradio', action='store_true', help="run in the web mode supported by gradio")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    args = parser.parse_args()
    print(args)
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class, model_add_kwgs = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, encode_special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
        **model_add_kwgs,
    )

    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    print(tokenizer)
    # test data
    if args.data_file is None:
        examples = [
            {
                "rounds": [{"prompt": "介绍一下北京", "response": ""}],
                "instruction": "",
                "true_answer": "开放性答案"
            },
            {
                "rounds": [{"prompt": "甲肝和乙肝有什么区别", "response": "这个我不不好说，需要专业医生来给出"},
                           {"prompt": "没关系，把你知道的说出来就好", "response": ""}
                          ],
                "instruction": "",
                "true_answer": "只是为了方便后续对比处理，留空也没关系"
            }
        ]
    else:
        with open(args.data_file, 'r') as f: # load from jsonl file 
            examples = [json.loads(l.strip()) for l in f.readlines()]

        if args.sample_gen_config: # 为每个样例单独配置 chat 的参数
            with open(args.sample_gen_config, "r") as f:
                examples_gen_config = [json.loads(l.strip()) for l in f.readlines()]
            assert len(examples) == len(examples_gen_config)
        else:
            examples_gen_config = [{} for _ in examples]
        
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)



    if args.gradio:
        pass
    elif args.interactive:
        HELP_INFO = """进入交互式多轮对话
特殊命令：
!!ins:xxxx      --- 将xxxx设置为当前对话的instruction
!!clear         --- 重置所有对话历史 和 instruction, 开始新对话
!!exit          --- 结束并退出
\"\"\"xxx\"\"\"       --- 输入特殊字符 如 \\t \\r \\n 等, 可以使用 前后三个" 传入一个符合json格式要求的 str      
!!set.xx yyy    --- 将xx参数设置为 yyy , 目前支持 tt -> temperature  
                                                rp -> repetition_penalty 
                                                nt -> max_new_tokens
!!help          --- 输出帮助文本
"""
        print("Start inference with interactive mode.")
        conv = SimpleConv()
        chatio = SimpleChatIO(save_path=args.predictions_file)
        chatio.print2save_and_stdout(f"args: {args}\n=========\n{HELP_INFO}\n=========\n\n")
        
        while True:
            try:
                inp = chatio.prompt_for_input("user")
            except EOFError:
                inp = ""
            except UnicodeDecodeError:
                chatio.print2save_and_stdout("UnicodeDecodeError, please try again.")
                continue
            if inp == "!!exit":
                chatio.print2save_and_stdout("exit...")
                break
            if inp == "!!clear":
                chatio.print2save_and_stdout("clearing history...\n\n==========  new chat  ==========\n")
                conv.clear()
                continue
            if inp == ("!!help"):
                chatio.print2save_and_stdout(HELP_INFO)
                chatio.print2save_and_stdout(str(args))
                continue
            if inp.startswith("!!set"):
                code_kws_dict = {"tt": ("temperature", float), 
                    "rp": ("repetition_penalty", float), 
                    "nt": ("max_new_tokens", int)}
                try:
                    kw = code_kws_dict[inp[6:8]][0]
                    value = code_kws_dict[inp[6:8]][1](inp[9:])
                    args.__setattr__(kw, value)
                    chatio.print2save_and_stdout(f"set {kw} -> {value} sucess")
                except Exception as e:
                    chatio.print2save_and_stdout(f"set {inp[6:8]} -> {inp[9:]} failed\n{e}")
                continue
            if inp.startswith("!!ins:"):
                inp = inp[6:]
                if inp.startswith('"""') and inp.endswith('"""') and len(inp) > 6:
                    inp = json.loads(inp[2:-2])
                conv.instruction = inp
                chatio.print2save_and_stdout(f"Instruction: {inp}")
                continue
            
            if inp.startswith('"""') and inp.endswith('"""') and len(inp) > 6:
                inp = json.loads(inp[2:-2])

            if inp:
                conv.rounds.append({
                    "prompt": inp,
                    "response": "",
                })
            else:
                # 键入空，提示
                chatio.print2save_and_stdout("Warning: 不要输入空字符串...请重新输入")
                continue
            

            chatio.prompt_for_output("bot")
            output = generate_answer(
                model_type=args.model_type,
                model=model,
                tokenizer=tokenizer,
                input_data=conv.to_dict(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
            response = chatio.get_output(output.strip())
            # NOTE: strip is important to align with the training data.
            conv.rounds[-1]["response"] = response.strip()
            # print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    else:
        print("Start inference.")
        results = []
        for index, (example, example_gen_config) in enumerate(zip(examples, examples_gen_config)):
            if example.get('true_answer', None) is None:
                example["true_answer"], example["rounds"][-1]["response"] = example["rounds"][-1]["response"], ""
            
            response = generate_answer(
                model_type=args.model_type,
                model=model,
                tokenizer=tokenizer,
                input_data=example,
                max_new_tokens=example_gen_config.get("max_new_tokens", args.max_new_tokens),
                temperature=example_gen_config.get("temperature", args.temperature),
                repetition_penalty=example_gen_config.get("repetition_penalty", args.repetition_penalty)
            )
            response = response.strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"True: {example['true_answer']}\n")
            print(f"Output: {response}\n")

            example["predict"] = response
            results.append({"Output": response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname, exist_ok=True)
        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            json.dump({
                "examples": examples,
                "results": results
            }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
