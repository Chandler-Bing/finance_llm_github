import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from score_utils import fin_alipay_get_score,exact_match,acc_match,rouge_zh_score
from tasks.ceval import CEval
from tasks.fin_eval import FinEval,FinEval2
from tasks.fin_IQ import FinIQ
from tasks.cmmlu import Cmmlu
from tasks.mmlu import Mmlu
from tasks.semantic_understanding import SemanticUnderstanding
from tasks.fin_alipay import FinAlipay
from tasks.safety_bench import SafetyBench
from tasks.long_bench import LongBench
from evaluator import Evaluator

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="model name or path"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, required=True, help="tokenizer name or path"
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, required=True, help="input max length"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="which task to eval"
    )
    parser.add_argument(
        "--logit_from_choice", action='store_true',help="whether choose argmax from choices ['a','b','c','d']"
    )
    parser.add_argument(
        "--subject", type=str, default='all', help="which subject to eval"
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output2", help="output directory"
    )
    parser.add_argument(
        "--lora_weights", type=str, default="", help="lora weights path"
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--test",action='store_true',help="for submit task"
    )
    parser.add_argument(
        "--llama",action='store_true',help="use llama"
    )
    parser.add_argument(
        "--bbt",action='store_true',help="use bbt"
    )


    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(2024)
    args = parse_argument()
    model_name = os.path.basename(args.model_name_or_path)
    lora_name = os.path.basename(args.lora_weights)
    output_dir = os.path.join(args.output_dir, model_name+lora_name,args.task,f'{args.shot}shot')
    os.makedirs(output_dir,exist_ok=True)
    task2data = {
        'ceval':CEval,
        'fin_eval':FinEval2,
        'fin_IQ':FinIQ,
        'cmmlu':Cmmlu,
        'mmlu':Mmlu,
        'fin_alipay':FinAlipay,
        'safety_bench':SafetyBench,
        'long_bench':LongBench,
        'semantic_understanding':SemanticUnderstanding

    }
    task2score_util = {
        'fin_alipay':fin_alipay_get_score,
        'long_bench':LongBench().dataset2metric,
        'ceval':exact_match,
        'fin_eval':exact_match,
        'fin_eval2':exact_match,
        'fin_IQ':exact_match,
        'cmmlu':exact_match,
        'mmlu':exact_match,
        'safety_bench':None,
        'semantic_understanding':exact_match
    }

    dataset = task2data[args.task](shot=args.shot)
    evaluator = Evaluator(
        model_path=args.model_name_or_path,
        tokenizer_path=args.tokenizer_name_or_path,
        max_length=args.max_length,
        lora_weight=args.lora_weights,
        llama=args.llama,
        bbt=args.bbt,
        logit_from_choice=args.logit_from_choice
    )
    results = {}
    submits = []
    print('====' * 10 + f'running task:{args.task}' + '===='*10,)
    for data in tqdm(dataset,total=len(dataset)):
        subject = data['subject']
        if args.subject != 'all':
            subject_to_eval = args.subject.split(',')
            if subject not in subject_to_eval:
                continue
        metrics = task2score_util[args.task]

        if args.task == 'long_bench':
            evaluator.long_bench_answer(data)
            metrics = metrics[subject]
            m = data.get('metric',None)
            if m == 'rouge':
                metrics = rouge_zh_score
            elif m == 'acc':
                metrics = acc_match
            else:
                pass

        else:
            evaluator.logit_answer(data)
        submits.append(data)

        if metrics:
            score = 0
            #For Long Bench tasks https://github.com/THUDM/LongBench/blob/main/eval.py#L70C9-L71C64
            prediction = data['pred_answer']
            if subject in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = data['pred_answer'].lstrip('\n').split('\n')[0]
            if isinstance(data['answer'],list):
                for ground_truth in data['answer']:
                    score = max(score,metrics(prediction,ground_truth,all_classes = data['all_classes']))
            else:
                #score = metrics(prediction,data['answer'])
                score = max(
                    metrics(data['pred_id'],data['groudth_id']),
                    metrics(data['answer'],data['pred_answer']),
                )
            data['score'] = score



        if subject not in results.keys():
            results[subject] = []
        del data['subject']
        results[subject].append(data)

    dataset.submit(submits,output_dir)

    if not args.test:
        accs = {}
        correct = 0
        total = 0
        for subject,result in results.items():
            result_path = os.path.join(output_dir,f"{subject}.json")
            labels = list(map(lambda x:x['score'],result))
            acc = sum(labels) / len(labels)
            correct += sum(labels)
            total += len(labels)
            accs[subject] = acc
            with open(result_path, "w", encoding='utf8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            #print(f"save result to {result_path}")

        average_acc = correct / total
        loss = [item.get('loss',0) for item in dataset]
        average_loss = sum(loss) / len(loss)

        accs['avg'] = average_acc
        accs['loss_avg'] = average_loss
        acc_path = os.path.join(output_dir,"acc.json")
        with open(acc_path, "w", encoding='utf8') as f:
            json.dump(accs, f, indent=4, ensure_ascii=False)
            print(f'average acc: {average_acc}')
            print(f'average loss: {average_loss}')
    else:
        for subject,result in results.items():
            result_path = os.path.join(output_dir,f"{subject}.json")
            with open(result_path, "w", encoding='utf8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    main()






