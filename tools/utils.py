import os
import glob
import json
import opencc
import datetime
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from loguru import logger
from transformers import set_seed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from functools import partial
import matplotlib.pyplot as plt
from transformers.optimization import (
    _get_cosine_schedule_with_warmup_lr_lambda,
    _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
    _get_linear_schedule_with_warmup_lr_lambda,
    _get_polynomial_decay_schedule_with_warmup_lr_lambda,
    _get_inverse_sqrt_schedule_lr_lambda,
)
from datasets import Dataset, load_dataset, IterableDataset
from transformers import LlamaTokenizerFast, AutoTokenizer

def seed_everything(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_name_or_path):
    logger.info(f'using tokenizer in {tokenizer_name_or_path}')

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    return tokenizer


def traditional_to_simplified(traditional_text):
    converter = opencc.OpenCC('t2s')
    # 进行简繁转换
    simplified_text = converter.convert(traditional_text)
    return simplified_text


def muti_process_token_decode(
        json_file_path,
        tokenizer,
        result_file_path,
):
    logger.info(f'processing {json_file_path}~~~')
    # print(f'parquet_file_path is : {parquet_file_path}')
    basename = os.path.basename(json_file_path)
    train_result_path = os.path.join(result_file_path, 'train', basename)
    eval_result_path = os.path.join(result_file_path, 'eval', basename)
    data = []
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip('\n')))
    texts = [
        {
            'tag': 'baichuan_decode',
            'text': tokenizer.decode(sample).replace('<s>','<|im_start|>').replace('</s>','<|im_end|>')
        } for sample in tqdm(data, desc=json_file_path)
    ]

    with open(train_result_path,'w',encoding='utf-8') as f:
        for text in texts:
            f.write(json.dumps(text,ensure_ascii=False))
            f.write('\n')

    logger.info(f'saving {train_result_path} and {eval_result_path} done ')

def muti_process_token_generate_jsonline(
        json_file_path,
        tokenizer,
        result_file_path,
        eval_percent,
        parquet,
        block_size
):
    logger.info(f'processing {json_file_path}~~~')
    # print(f'parquet_file_path is : {parquet_file_path}')
    basename = os.path.basename(json_file_path)
    train_result_path = os.path.join(result_file_path, 'train', basename)
    eval_result_path = os.path.join(result_file_path, 'eval', basename)
    data = []
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip('\n')))

    tokens = [
        {
            'tag': sample.get('tag', 'fin_exam'),
            #'input_ids': tokenizer.encode(traditional_to_simplified(sample['text']))
            'input_ids': tokenizer.encode(traditional_to_simplified(sample['text']) + '<|endoftext|>')
            #'input_ids': tokenizer.encode('<|im_start|>' + traditional_to_simplified(sample['text']) + '<|im_end|>')
        } for sample in tqdm(data, desc=json_file_path)
    ]
    train = tokens[:int(len(tokens) * (1 - eval_percent))]
    eval = tokens[int(len(tokens) * (1 - eval_percent)):]

    train_df = pd.DataFrame(group(train, block_size=block_size))
    train_df = train_df[['input_ids']]

    if eval:
        eval_df = pd.DataFrame(group(eval, block_size=block_size))
        eval_df = eval_df[['input_ids']]
        logger.info(f'saving {train_result_path} and {eval_result_path}......')
    else:
        eval_df = pd.DataFrame()
    if parquet:
        train_df.to_parquet(train_result_path)
        eval_df.to_parquet(eval_result_path)
    else:
        train_df.to_csv(train_result_path, index=False, header=False, sep='\t')
        eval_df.to_csv(eval_result_path, index=False, header=False, sep='\t')
    logger.info(f'saving {train_result_path} and {eval_result_path} done ')


def group(all_data, block_size):
    out_t = []
    out_i, out_a = [], []
    for line in all_data:
        tag, input_ids = line["tag"], line["input_ids"]
        out_t.append(tag)
        out_i.extend(input_ids)
        if len(out_i) >= block_size:
            for _ in range(len(out_i) // block_size):
                yield {
                    "input_ids": out_i[:block_size],
                }
                out_i = out_i[block_size:]
                out_t = [tag] if len(out_i) else []


def read_parquet(train_file_dir, validation_file_dir):
    tic = datetime.datetime.now()
    train_files = glob.glob(os.path.join(train_file_dir, '*.parquet'))
    eval_files = glob.glob(os.path.join(validation_file_dir, '*.parquet'))
    iter_dataset_dict = load_dataset('parquet', data_files={'train': train_files, 'eval': eval_files}, num_proc=16)
    train_dataset, eval_dataset = iter_dataset_dict['train'], iter_dataset_dict['eval']
    # train_dataset = Dataset.from_parquet(train_file_dir+'/*.parquet',num_proc=10).to_iterable_dataset()
    # eval_dataset =  Dataset.from_parquet(validation_file_dir+'/*.parquet',num_proc=10).to_iterable_dataset()
    tac = datetime.datetime.now()
    print(f'load到dataset花费时间:{(tac - tic).seconds}s')
    return train_dataset, eval_dataset


def test_lr_scheduler():
    lr_scheduler_map = {
        'cosine': _get_cosine_schedule_with_warmup_lr_lambda,
        'linear': _get_linear_schedule_with_warmup_lr_lambda,
        'cosine_with_restarts': _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        'polynomial': _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        'inverse_sqrt': _get_inverse_sqrt_schedule_lr_lambda
    }

    func = partial(_get_cosine_schedule_with_warmup_lr_lambda,
                   num_warmup_steps=0,
                   num_training_steps=100,
                   num_cycles=0.5
                   )
    #
    x = [i for i in range(100)]
    y = [func(i) for i in x]
    print(x, y)
    plt.plot(x, y)
    plt.show()


def main(num_proc=32):
    #json_file_path = '/app/nfs_share_dir/1/archive/v2/part'
    json_file_path = '/app/nfs_share_dir/1/archive/v2/baichuan-decode/train'
    data_files = glob.glob(f"{json_file_path}/*")

    func = partial(
        muti_process_token_generate_jsonline,
        tokenizer = get_tokenizer('/app/nfs_share_dir/3/llm_model/Qwen1_5-0_5B'),
        result_file_path = '/app/nfs_share_dir/1/archive/v2/token-Qwen-4k-eot',
        eval_percent = 1e-5,
        parquet = False,
        block_size = 4096,
    )
    # func = partial(
    #     muti_process_token_decode,
    #     tokenizer = get_tokenizer('/app/nfs_share_dir/3/llm_model/Baichuan2-7B-Base'),
    #     result_file_path = '/app/nfs_share_dir/1/archive/v2/baichuan-decode',
    # )
    pool = multiprocessing.Pool(num_proc)
    pool.map(func, data_files)
    pool.close()
    pool.join()


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    tac = datetime.datetime.now()
    print(f'花费时间:{(tac - tic).seconds}s')
