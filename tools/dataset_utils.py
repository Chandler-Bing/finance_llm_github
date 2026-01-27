import os
import copy
import json
import torch
from glob import glob
import transformers
from tqdm import tqdm
from functools import reduce
from loguru import logger
from typing import Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset

from typing import Dict

IGNORE_INDEX = -100


def get_file_lines(file_path: str) -> int:
    total_lines = int(os.popen(f"wc -l {file_path}").readlines()[0].split()[0])
    return total_lines


class LazyFile(object):
    def __init__(self, file_path, save_index=False, map_func=None):
        self.file_path = file_path
        self.file_lines_number = get_file_lines(self.file_path)
        #file_index_path = self.file_path + ".index.json"
        file_index_path = os.path.join('index',self.file_path.split('/')[-1]+'.index.json')
        if not os.path.exists(file_index_path):
            self.load(self.file_path)
            if save_index:
                logger.warning("save index my cause error when using muti-process")
                self.write_index(file_index_path)
        else:
            self.load_with_index(file_index_path, self.file_path)
        self.map_func = map_func

    def __del__(self):
        if hasattr(self, "fin"):
            self.fin.close()

    def load(self, file_path: str):
        self.offset_mapping = self.get_offset_mapping(file_path)
        self.fin = open(file_path, "r", encoding="utf8")

    def get_offset_mapping(self, file_name: str) -> list:
        """
        iter a file,get a offset list like:
        [0,2,4,6,8...]
        which means:
        line 0 starts at offset(byte) 0,
        line 1 starts at offset(byte) 2,
        ps, \n take 1 byte
        """
        key_2_offset_map = []
        with open(file_name, "r", encoding="utf-8") as f:
            offset = 0
            for idx, line in enumerate(
                    tqdm(
                        iter(f.readline, ''),  # the callable is called until it returns the sentinel value,if set to '' will iter to end.
                        total=self.file_lines_number,
                        desc=f"generate index of {file_name}",
                    )
            ):
                key_2_offset_map.append(offset)
                offset = f.tell()
                if not line:
                    break
        return key_2_offset_map

    def write_index(self, output_file_path: str):
        json.dump(self.offset_mapping, open(output_file_path, "w"))
        logger.warning(f"save index to {output_file_path}")

    def load_with_index(self, index_file_path, file_path):
        self.offset_mapping = json.load(open(index_file_path, "r"))

        assert len(self.offset_mapping) == self.file_lines_number, (
            f"totol lines of {file_path} is {self.file_lines_number}, but max index of"
            f" {index_file_path} is {len(self.offset_mapping)}"
        )
        self.fin = open(file_path, "r", encoding="utf-8")
        logger.warning(f"load index of {file_path} from {index_file_path}")

    def __getitem__(self, key):
        """
        return specified line by the given key
        consider key is line num,when we get a large file 10000th line,
        first get 10000th line offset, use f.seek(offset) to set cursor to the line we want,
        then use f.readline() get the line.
        """
        offset = self.offset_mapping[key]
        self.fin.seek(offset)
        value = self.fin.readline().strip("\n")
        if self.map_func is not None:
            value = self.map_func(value)
        return self.file_path + '#' + str(key), value

    def __len__(self):

        return len(self.offset_mapping)


class LazyFiles(object):
    def __init__(self, file_path_list: list, save_index=False, map_func=None) -> None:
        self.file_path_list = sorted(
            [file_path for file_path in file_path_list if not file_path.endswith(".index.json")]
        )
        self.file_list = [
            LazyFile(file_path, save_index, map_func=map_func) for file_path in self.file_path_list
        ]
        self.idx_2_file = self.get_idx_2_file(self.file_list)
        self.file_length_interval = self.get_file_length_interval(self.file_list)

    def get_idx_2_file(self, file_list) -> list:
        '''
        return id to file map
        eg. [0,0,0,1,1,1,1,2,2,2,2,2]
        there are 3 files
        file A has 3 lines
        file B has 4 lines
        file C has 5 lines
        '''
        idx_2_file = []
        for file_idx, file in enumerate(file_list):
            idx_2_file.extend([file_idx] * len(file))
        return idx_2_file

    def get_file_length_interval(self, file_list) -> list:
        '''
        return file length
        eg. [0,3,7,12]
        file A begins at 0,ends at 3,because file A has 3 lines
        file B begins at 3,ends at 7,because file B has 4 lines
        file C begins at 7,ends at 12,because file C has 5 lines
        '''
        file_length_interval = [0]
        start = 0
        for file in file_list:
            start += len(file)
            file_length_interval.append(start)
        return file_length_interval

    def __len__(self):
        '''
        total line num
        '''
        return len(self.idx_2_file)

    def __getitem__(self, idx):
        '''
        when get an idx,
        1. first find target file
        2. then find offset
        eg. when get(11)
        1. target file idx = self.idx_2_file[idx] which is 2, means file C
        2. target file offset = 11 - len(A) - len(B) = 4

        '''
        file_idx = self.idx_2_file[idx]
        target_file = self.file_list[file_idx]
        idx_in_target_file = idx - self.file_length_interval[file_idx]
        return target_file[idx_in_target_file]


class QifuDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=1024, padding=False, lazy_mode=False,tag = True,**kwargs):
        if not os.path.exists(data_path):
            raise ValueError(f"{data_path} is not existed.")
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.padding = padding
        self.lazy_mode = lazy_mode
        self.max_seq_length = max_seq_length
        self.tag = tag


        self.data = self.load_data(data_path, lazy_mode)

    def load_data(self, data_path, lazy_mode):
        if os.path.isfile(data_path):
            data_path_list = [data_path]
        else:
            data_path_list = glob(os.path.join(data_path, "*"))
            data_path_list = sorted(data_path_list)
        assert len(data_path_list) > 0, f"data_path:{data_path} is empty"

        if lazy_mode:
            return LazyFiles(data_path_list, save_index=False, map_func=None)
        else:
            data = []
            for data_path in data_path_list:
                logger.info(f"Loading {data_path}")
                with open(data_path, "r", encoding="utf8") as fp:
                    for line in fp:
                        data.append(line)
            return data

    def pretrain_data_preprocess(self, data_item):
        tag,input_ids = data_item
        input_ids = json.loads(input_ids)
        if 'exams' not in self.data_path:
            attention_mask = [1] * len(input_ids)
            labels = copy.deepcopy(input_ids)
        else:
            # for exam eval dataset, only cal the loss of the answer
            if self.padding:
                non_pad_token_id_lenth = len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
                attention_mask = [1] * non_pad_token_id_lenth + [0] * (self.max_seq_length-non_pad_token_id_lenth)
                labels = (non_pad_token_id_lenth - 1) * [IGNORE_INDEX] + [input_ids[non_pad_token_id_lenth-1]] + (self.max_seq_length-non_pad_token_id_lenth) * [IGNORE_INDEX]
            else:
                attention_mask = [1] * len(input_ids)
                labels = (len(input_ids) - 1) * [IGNORE_INDEX] + [input_ids[-1]]
        if self.tag:
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "tag": tag}
        else:
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.data[i]
        feature_dict = self.pretrain_data_preprocess(data_item)
        for k,v in feature_dict.items():
            if not isinstance(v,str):
                feature_dict[k] = torch.tensor(v)
        return feature_dict

class QifuSftDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=4096,tag = True,sft_group=True,all_loss = True,split_multi_turn=True,system_prompt='',just_last_answer=True, **kwargs):
        self.data_path = data_path
        self.tokenizer = tokenizer #type:transformers.PreTrainedTokenizer
        self.max_seq_length = max_seq_length
        self.tag = tag
        self.sft_group = sft_group
        self.data = []
        self.just_answer_labels = []
        self.all_loss = all_loss
        self.just_last_answer = just_last_answer
        self.split_multi_turn = split_multi_turn
        self.system_prompt = system_prompt
        self.load_data()

    def read_file(self):
        with open(self.data_path,'r',encoding='utf-8') as f:
            for line in tqdm(f.readlines(),desc=f'parsing lines from {self.data_path}'):
                tmp= [{'role':'system','content':self.system_prompt}] if self.system_prompt else []
                e = json.loads(line.strip('\n'))
                for round in e['rounds']:
                    tmp.extend(
                        [
                            {'role':'user','content':round['prompt']},
                            {'role':'assistant','content':round['response']}
                        ]

                    )
                    if self.split_multi_turn:
                        yield tmp
                yield tmp

    def soft_single(self,input_ids):
        tmp = []
        res = []
        for per_line in input_ids:
            if len(tmp) + len(per_line) > self.max_seq_length:
                res.append(tmp)
                tmp = per_line
            else:
                tmp.extend(per_line)
        if tmp:
            res.append(tmp)
        return res


    def ignore_question(
            self,
            per_line,
            ignore_index,
            assistant_token_id,
            role_header_end_id,
            eot_id
    ):
        per_label,i,length = [ignore_index],1,len(per_line)
        while i < length:
            if per_line[i] == assistant_token_id:
                if role_header_end_id:
                    i += 2
                    per_label.extend([ignore_index,ignore_index])
                else:
                    i += 1
                    per_label.extend([ignore_index])
                while per_line[i] != eot_id:
                    per_label.append(per_line[i])
                    i += 1
                #<eot_id>也要学习
                per_label.append(per_line[i])
                i += 1
            else:
                per_label.append(ignore_index)
                i += 1
        if self.just_last_answer:
            #maeke sure [-100,...,-100,1,2,3,-100,...,-100,4,5,6,-100,...] to [-100,...,-100,-100,-100,-100,-100,...,-100,4,5,6,-100,...]
            tmp = per_label[::-1]
            for i in range(len(tmp)):
                if tmp[i] != ignore_index:
                    while tmp[i] != ignore_index:
                        i+=1
                    break
            per_label = [ignore_index] * (len(per_label)-i) + per_label[-i:]

        return per_label

    def load_data(self):
        input_ids = []
        labels = []
        for per_sample in self.read_file():
            per_line = self.tokenizer.apply_chat_template(
                per_sample,
                tokenize=True,
                add_generation_prompt=False,
                truncation=False
            )
            per_line.append(self.tokenizer.eos_token_id)
            per_label = self.ignore_question(
                per_line = per_line,
                ignore_index=-100,
                assistant_token_id=self.tokenizer.encode('assistant',add_special_tokens=False)[0],
                role_header_end_id=128007,
                eot_id=128009
            )

            if self.sft_group:
                input_ids.extend(per_line)
                labels.extend(per_label)
            else:
                input_ids.append(per_line)
                labels.append(per_label)

        if self.sft_group:
            for i in range(0,len(input_ids),self.max_seq_length):
                self.data.append(input_ids[i:i+self.max_seq_length])
                self.just_answer_labels.append(labels[i:i+self.max_seq_length])
            if len(self.data[-1]) < self.max_seq_length:
                del self.data[-1],self.just_answer_labels[-1]
        else:
            self.data = self.soft_single(input_ids)
            self.just_answer_labels = self.soft_single(labels)

    def pretrain_data_preprocess(self, data_item,label_item):
        assert len(data_item) == len(label_item), 'input_ids and labels must have the same length!'
        input_ids = data_item
        attention_mask = [1] * len(input_ids)
        if self.all_loss:
            labels = copy.deepcopy(input_ids)
        else:
            labels = label_item
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.data[i]
        label_item = self.just_answer_labels[i]
        feature_dict = self.pretrain_data_preprocess(data_item,label_item)
        for k,v in feature_dict.items():
            if not isinstance(v,str):
                feature_dict[k] = torch.tensor(v)
        return feature_dict

    def __len__(self):
        return len(self.data)




if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/app/nfs_share_dir/3/llm_model/Meta-Llama-3-8B-Instruct',trust_remote_code=True,use_fast=True)
    a = QifuSftDataset(
        '/app/nfs_share_dir/5/boruipeng/xzm/combine0329/kmeans_weight-coreset_general_sample+task_20000/train/tmp.jsonl',
        tokenizer=tokenizer,
        max_seq_length=4096,
        tag=True,
        sft_group=False,
        all_loss=True,
        just_last_answer=True,
        system_prompt='',
        split_multi_turn=True,
    )
    print(tokenizer.decode(a[0]['input_ids'].tolist()))
    print('----'*10)
    print(tokenizer.decode([item for item in a[0]['labels'].tolist() if item != -100]))
