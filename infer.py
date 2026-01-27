import os
import sys
import json
import torch
import loguru
import argparse
from tqdm import tqdm
from functools import partial
from eval.niah.zh_niah import LLMNeedleHaystackTester
from tools.inference_container import InferenceContainer


logger = loguru.logger
logger.remove()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', default='llama3', type=str, required=True)
    parser.add_argument('--model_paths', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_paths', default=None, type=str, required=True)
    parser.add_argument('--torch_dtype', default='float16', type=str)
    parser.add_argument('--mode', default=None, required=True,choices=['gradio','sft-task','niah'], help="which mode to run")
    parser.add_argument('--max_new_tokens', default=20, type=int, required=False)
    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--predictions_file', default=None, type=str)
    args = parser.parse_args()
    model_paths = args.model_paths.split(',')
    tokenizer_paths = args.tokenizer_paths.split(',')
    model_types = args.model_types.split(',')
    base_model_names = [item.split('/')[-1] for item in model_paths]

    inference_obj = InferenceContainer(
        model_types = model_types,
        model_paths=model_paths,
        tokenizer_paths=tokenizer_paths,
        torch_dtype=getattr(torch,args.torch_dtype),
        gradio=True if args.mode == 'gradio' else False,
    )

    if args.mode == 'gradio':
        logger.add(sys.stdout,level='DEBUG')
        inference_obj.run_gradio(base_model_names)
    elif args.mode == 'sft-task':
        logger.add(sys.stdout,level='INFO')
        which = base_model_names[0]
        with open(args.data_file, 'r',encoding='utf-8') as f:
            examples = [json.loads(l.strip('\n')) for l in f.readlines()]
        results = []
        for example in tqdm(examples,desc=f'running {args.data_file}'):
            if example.get('true_answer', None) is None:
                example["true_answer"], example["rounds"][-1]["response"] = example["rounds"][-1]["response"], ""
            message,history = [],[]
            for r in example['rounds'][:-1]:
                history.append([r['prompt'],r['response']])
            message = example['rounds'][-1]['prompt']
            response = inference_obj.generate_fn(
                message = message,
                history = history,
                which = which,
                system_prompt = '你是一个严格遵从指令的中文人工智能助手',
                top_k = 5,
                top_p = 0.9,
                temperature = 0.01,
                repetition_penalty = 1.0,
                max_new_tokens = int(args.max_new_tokens),
                do_sample = True,
                num_beams = 1,
            )
            predict = list(response)[-1]
            example["predict"] = predict
            results.append({"Input": message, "Output": predict})
        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname,exist_ok=True)
        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            json.dump({"examples": examples,'results':results}, f, ensure_ascii=False, indent=4)
    elif args.mode == 'niah':
        logger.add(sys.stdout,level='INFO')
        model_name = model_paths[0].split('/')[-1]
        models = [
            partial(
                inference_obj.generate_fn,
                history=[],
                which=base_model_names[0],
                system_prompt = '',
                top_k = 5,
                top_p = 0.9,
                temperature = 0.1,
                repetition_penalty = 1.0,
                max_new_tokens = 128,
                do_sample = True,
                num_beams = 1,
            )
        ]
        haystack_dir = './eval/niah/haystack'
        needle = "\n中国最屌的民谣歌手是赵雷。\n"
        retrieval_question = "中国最屌的民谣歌手是谁？"
        needle = '\n王莽是一名勤奋的店员，他每天凌晨就起床，赶在第一缕阳光照亮大地之前到达店铺，为即将开始的一天做准备。他清扫店铺，整理货架，为顾客提供方便。他对五金的种类和用途了如指掌，无论顾客需要什么，他总能准确地找到。\n然而，他的老板刘秀却总是对他吹毛求疵。刘秀是个挑剔的人，他总能在王莽的工作中找出一些小错误，然后以此为由扣他的工资。他对王莽的工作要求非常严格，甚至有些过分。即使王莽做得再好，刘秀也总能找出一些小问题，让王莽感到非常沮丧。\n王莽虽然对此感到不满，但他并没有放弃。他知道，只有通过自己的努力，才能获得更好的生活。他坚持每天早起，尽管他知道那天可能会再次被刘秀扣工资。他始终保持微笑，尽管他知道刘秀可能会再次对他挑剔。\n'
        retrieval_question = "王莽在谁的手下工作？"

        further_instruct = "仅基于上述文档，不要给出上述文档以外的信息。也不要重复回答"
        # 英文大海捞针 further_instruct 可以试下：
        # further_instruct = "Don't give information outside the above text."
        ht = LLMNeedleHaystackTester(
            needle=needle,
            haystack_dir=haystack_dir,
            retrieval_question=retrieval_question,
            further_instruct=further_instruct,
            results_version = 1,
            context_lengths_min = 1024,
            context_lengths_max = 32768,
            context_lengths_num_intervals = 11,
            context_lengths = None,
            document_depth_percent_min = 0,
            document_depth_percent_max = 100,
            document_depth_percent_intervals = 6, # 11对应10等分
            document_depth_percents = None,
            document_depth_percent_interval_type = "linear",
            model_provider = "hf",
            tokenizer = args.tokenizer_paths,
            models = models,
            model_name=model_name,
            num_concurrent_requests = 1,
            save_results = True,
            save_contexts = False,
            final_context_length_buffer = 500,
            seconds_to_sleep_between_completions = None,
            print_ongoing_status = True,
            evaluation_criterion = 'f1_zh',
            pool_multiplier = 1,
            question_at_beginning = False
        )
        ht.start_test(mp=False)


if __name__ == '__main__':
    main()