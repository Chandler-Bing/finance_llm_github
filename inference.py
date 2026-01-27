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
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)

from tools.conv_templete import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


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


@torch.inference_mode()
def generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=100,
        temperature=0.1,
        top_k=5,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
        context_len=4096,
        eos_token_list:list =None
):
    eos_token_id_list = []
    if eos_token_list:
        for token in eos_token_list:
            token_id = tokenizer.encode(token, add_special_tokens=False)
            assert len(token_id)==1
            eos_token_id_list.append(token_id[0])
        eos_token_id_list = list(set(eos_token_id_list)) 

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
    #print("\n-------   input   ---------\n",prompt,"\n---------\n")
    input_ids = tokenizer(prompt).input_ids
    #print("\n-------   input_ids   ---------\n",input_ids,"\n---------\n")
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_output = model.generate(
        input_ids=torch.as_tensor([input_ids]).to(device),
        **generation_config,
    )
    output_ids = generation_output[0]
    while output_ids[-1] in eos_token_id_list:
        output_ids = output_ids[:-1]
    output = tokenizer.decode(output_ids, skip_special_tokens=False)
    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))
    output = output[l_prompt:]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit (for lora)')
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--use_fast_tokenizer', default=False, type=bool, help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument('--template_name', default="alpaca", type=str, help="Prompt template name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_context_length", type=int, default=4096)
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

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, encode_special_tokens=True, use_fast=args.use_fast_tokenizer)
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
        examples_gen_config = [{} for _ in examples]
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


    # Chat
    def new_chat():
        return get_conv_template(args.template_name)

    if args.gradio:
        import gradio as gr
        from gradio.data_classes import InterfaceTypes
        conv = new_chat()
        def mini_func_for_ui(inp):
            conv.rounds.append({
                    "prompt": inp,
                    "response": "",
                })
            prompt = conv.get_prompt()
            output = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                context_len=args.max_context_length,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                eos_token_list=conv.stop_tokens,
            )
            response = output.strip()
            # NOTE: strip is important to align with the training data.
            conv.rounds[-1]["response"] = response.strip()
            return json.dumps({"instruction": conv.instruction, "rounds":conv.rounds}, ensure_ascii=False, indent=2)

        class Interface(gr.Interface):
            def attach_clear_events(self,
                clear_btn,
                input_component_column,
            ):
                clear_btn.add(self.input_components + self.output_components)
                clear_btn.click(
                    lambda: conv.rounds.clear(),
                    [],
                    (
                        [input_component_column] if input_component_column else []
                    ),  # type: ignore
                    js=f"""() => {json.dumps(
                        
                            [{'variant': None, 'visible': True, '__type__': 'update'}]
                            if self.interface_type
                            in [
                                InterfaceTypes.STANDARD,
                                InterfaceTypes.INPUT_ONLY,
                                InterfaceTypes.UNIFIED,
                            ]
                            else []
                        
                    )}
                    """,
                )

        with gr.Blocks() as demo:
            interface = Interface(
                title="Llama2问答对话",
                description="它可以连续对话",
                article=f"基于{args.base_model}",
                fn = mini_func_for_ui,
                inputs = gr.Textbox(lines=5, placeholder="请输入。。。", label="用户输入", show_label=True),
                outputs = gr.Textbox(lines=5, max_lines=15, placeholder="", label="对话历史", autoscroll=True, show_copy_button=True),
            )
            demo.launch(share=True)


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
直接键入换行     --- 更底层的【继续】，不认为是新轮次，而让模型在上一轮对话的基础上继续生成回复
                    可近似理解为  用多个短回复，拼凑一个长回复
"""
        print("Start inference with interactive mode.")
        chatio = SimpleChatIO(save_path=args.predictions_file)
        chatio.print2save_and_stdout(f"args: {args}\n=========\n{HELP_INFO}\n=========\n\n")
        conv = new_chat()
        while True:
            try:
                inp = chatio.prompt_for_input(conv.roles[0])
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
                conv = new_chat()
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
                prompt = conv.get_prompt()
            else:
                # 键入空，则在上文基础上继续生成
                prompt = conv.get_prompt() + conv.rounds[-1]['response']
            

            chatio.prompt_for_output(conv.roles[1])
            output = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                context_len=args.max_context_length,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                eos_token_list=conv.stop_tokens,
            )
            response = chatio.get_output(output.strip())
            # NOTE: strip is important to align with the training data.
            conv.rounds[-1]["response"] = response.strip()
            # print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    else:
        print("Start inference.")
        results = []
        for index, (example, example_gen_config) in enumerate(zip(examples, examples_gen_config)):
            conv = new_chat()
            if example.get('true_answer', None) is None:
                example["true_answer"], example["rounds"][-1]["response"] = example["rounds"][-1]["response"], ""
            conv.rounds = example["rounds"]
            conv.instruction = example["instruction"]

            prompt = conv.get_prompt()
            response = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=example_gen_config.get("max_new_tokens", args.max_new_tokens),
                context_len=example_gen_config.get("max_context_length", args.max_context_length),
                temperature=example_gen_config.get("temperature", args.temperature),
                repetition_penalty=example_gen_config.get("repetition_penalty", args.repetition_penalty),
                eos_token_list=conv.stop_tokens
            )
            response = response.strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"True: {example['true_answer']}\n")
            print(f"Output: {response}\n")

            example["predict"] = response
            results.append({"Input": prompt, "Output": response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname, exist_ok=True)
        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            json.dump({
                "examples": examples,
                "results": results
            }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
