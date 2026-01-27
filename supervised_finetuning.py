# -*- coding: utf-8 -*-
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from glob import glob
from typing import List, Sequence, Optional, Dict, Tuple
from collections import deque, defaultdict
from tqdm import tqdm
import random
from functools import partial

import torch
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from sklearn.metrics import accuracy_score
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training

from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    deepspeed,
    DataCollatorForSeq2Seq,
    trainer_utils,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother

#from modeling_flash_llama import LlamaForCausalLM as FlashLlamaForCausalLM
FlashLlamaForCausalLM = LlamaForCausalLM
from tools.conv_templete import get_conv_template

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "llama_flash": (AutoConfig, FlashLlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

IGNORE_INDEX = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_max_length: Optional[int] = field(default=4096, metadata={"help": "The maximum length of the model"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    encode_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether to encode special tokens."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    template_name: Optional[str] = field(default="alpaca", metadata={"help": "The template name."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    split_multi_turn: bool = field(
        default=False, metadata={"help": "True: 多轮对话数据，每一轮次对话(加上历史对话)，都拆分为一个新的数据样本"}
    )
    latest_turn_only: bool = field(default=False, metadata={"help": "与split_multi_turn互斥。多轮对话仅视为1个样本计算 loss。 推荐前期经过数据筛选后，再打开此配置"})
    full_qa_loss: bool = field(default=False, metadata={"help": "是否使用全部输入计算loss"})
    group: bool = field(default=False, metadata={"help": "是否将多个短样本拼接至 model_max_length"})
    group_method: Optional[str] = field(default="greedy", metadata={"help": "choice from [naive, greedy]"})

    loss_fun_name: Optional[str] = field(default="default", metadata={"help": "损失函数，default为model自带loss，可选：'focal-loss'"})
    focal_alpha: Optional[float] = field(default=1.0, metadata={"help": "focal-loss 权重"})
    focal_gamma: Optional[float] = field(default=3.0, metadata={"help": "focal-loss gamma"})

    da_group: bool = field(default=False, metadata={"help": "是否将多组短对话拼接成一组长多轮对话, 尽量拼接至 model_max_length"})
    da_group_repeat_times: Optional[str] = field(default="auto", metadata={"help": "拼接的重复次数, 仅限传入正整数, auto 代表持续拼接至与原始数据条目数一致"})
    da_group_method: Optional[str] = field(default="greedy", metadata={"help": "choice from [naive, greedy]"})

    def __post_init__(self):
        if self.da_group and self.group:
            raise ValueError("da_group and group can not be set at the same time.")
        if self.da_group and self.split_multi_turn:
            raise ValueError("da_group and split_multi_turn are not suggested to be set at the same time.")

        if self.split_multi_turn and self.latest_turn_only:
            raise ValueError("split_multi_turn and latest_turn_only can not be set at the same time.")
        if self.latest_turn_only and self.full_qa_loss:
            raise ValueError("latest_turn_only and full_qa_loss can not be set at the same time.")


@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    
    preds = logits.argmax(-1)

    shift_labels = labels[:, 1:].reshape(-1)
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fct(shift_logits, shift_labels)
    # Compute Perplexity
    perplexity = torch.exp(loss)

    return preds, loss, perplexity

def compute_metrics(eval_preds):
    (preds, loss, perplexity), labels = eval_preds
    
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    
    non_ignore_mask = labels != IGNORE_INDEX
    preds = preds[non_ignore_mask]
    labels = labels[non_ignore_mask]
    
    # Compute Acc
    acc = float(accuracy_score(y_true=labels, y_pred=preds, normalize=True, sample_weight=None))
    return {"accuracy": acc, 
            "perplexity": np.mean(perplexity),
            "loss_cal": np.mean(loss)}

class GroupSFTSolver():
    def __init__(self, M, group_method="greedy") -> None:
        """
        group_method:  
            - greedy - 贪心算法，排序后再拼接，保持尽量少的空余位置，但可能改变数据分布  
            - naive  - 朴素算法，按照输入顺序直接拼接，可能出现较多空余位置
        """
        self.dict = defaultdict(list)
        self.M = M
        self.group_mehod = group_method

    def _insert_one_num(self, n_index: Tuple[int, int]): # num, index
        """
        for greedy method
        return : 
            True  - 插入成功
            False - 插入失败
        """
        need_space, index = n_index
        if need_space > self.M:
            return False
        self.dict[self.M] = [[]] # 设置一个空余为M的空group
        for i in range(need_space, self.M+1, 1):
            if self.dict[i]:
                group = self.dict[i].pop()
                group.append(n_index)
                self.dict[i - need_space].append(group)
                self.dict[self.M] = [] # 删除空余为M的空group（空余为M必为空group）
                return True
        # 出现异常未正确添加时，同样删除空余为M的空group
        self.dict[self.M] = [] 
        return False
    
    @staticmethod
    def counting_sort(num_index_list):
        """
        使用 counting_sort 对 num_index_list 根据 num 降序排序
        """
        X = num_index_list
        max_val = max(i[0] for i in X)
        min_val = min(i[0] for i in X)
        range_val = max_val - min_val + 1 
        count = [0] * range_val 
        output = [0] * len(X) 

        for i in X:
            count[i[0] - min_val] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]
            
        for i in range(len(X) - 1, -1, -1):
            output[count[X[i][0] - min_val] - 1] = X[i]
            count[X[i][0] - min_val] -= 1

        return output[::-1]  # Reverse the order to get descending order

    def _insert_sorted_num_index_list(self, sorted_num_index_list):
        """
        for greedy method, 让每一个Group的长度和尽量接近M, 但缺点是会倾向性改变数据分布
        若M=10, 则长度为4,6的数据总会拼接在一起，而4,5/4,2,2这些拼接方式出现的概率很低  
        :param sorted_num_index_list: 降序的 (num, index) 列表
        :return: 
        """
        for num_index in sorted_num_index_list:
            if not self._insert_one_num(num_index):
                assert False, f"num:{num_index[0]}, index:{num_index[1]} 插入失败"
        return True

    def _insert_naive_num_index_list(self, sorted_num_index_list):
        """
        for naive method, 直接按照输入长度进行比对, 若拼到现有组中不超长就拼接, 否则另起一组
        :param sorted_num_index_list: 无序的 (num, index) 列表
        :return: 
        """
        # self.dict[self.M] = [[]] # 设置一个空余为M的空group
        now_group, now_left_space = [], self.M
        for num, index in sorted_num_index_list:
            if num <= now_left_space:
                now_group.append((num, index))
                now_left_space -= num
            else:
                self.dict[now_left_space].append(now_group)
                now_group, now_left_space = [], self.M
        if now_group:
            self.dict[now_left_space].append(now_group)
        return True

    def insert_num_list(self, num_list):
        """
        :param num_list: 无序的 num 列表
        :return: 
        """
        num_with_index_list = [(num, index) for index, num in enumerate(num_list) if num <= self.M]
        if self.group_mehod == "greedy":
            sorted_num_index_list = self.counting_sort(num_with_index_list)
            return self._insert_sorted_num_index_list(sorted_num_index_list)
        elif self.group_mehod == "naive":
            return self._insert_naive_num_index_list(num_with_index_list)

    def get_group_index_result(self):
        for i in range(self.M+1):
            for group in self.dict[i]:
                yield [i[1] for i in group]

class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """
    def __init__(self, *args, 
                 loss_fun_name="default", focal_alpha=1.0, focal_gamma=3,
                 **kwargs):
        super().__init__(*args, **kwargs)

        loss_dict = {
            "default": super().compute_loss,
            "focal-loss": partial(self.compute_loss_exp1, alpha=focal_alpha, gamma=focal_gamma),
            "focal-ce-loss": partial(self.compute_loss_exp2, gamma=focal_gamma),
        }
        self.compute_loss = loss_dict[loss_fun_name]

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """重写sampler，默认会调用 RandomSampler， 而我们希望顺序与输入是严格一致的"""
        if self.train_dataset is None or not trainer_utils.has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            return super()._get_train_sampler()

        else:
            return SequentialSampler(self.train_dataset)

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        step 后释放显存
        """
        ret = super().training_step(model, inputs)
        # logger.info(f"input_ids[0][:-20] in rank{self.args.local_rank} is {inputs['input_ids'][0][:-20]}")
        # torch.cuda.empty_cache()
        return ret

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        step 后释放显存
        """
        ret = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        torch.cuda.empty_cache()
        return ret

    def compute_loss_exp1(self, model, inputs, return_outputs=False, alpha=1.0, gamma=3):
        """
        focal-loss like method, cal by float32
        """
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if isinstance(outputs, dict) else outputs[0]
        loss = None
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.get_input_embeddings().weight.shape[0]) # (batch_size * seq_len) * vocab_size
            shift_labels = shift_labels.view(-1) # (batch_size * seq_len)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss_fun = torch.nn.CrossEntropyLoss(reduction="none")
            log_pt = - ce_loss_fun(shift_logits, shift_labels) # ce_loss = - log_p_true
            pt = torch.exp(log_pt)
            loss = - alpha * (1-pt)**gamma * log_pt
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss
    
    def compute_loss_exp2(self, model, inputs, return_outputs=False, gamma=3):
        """
        loss = p * focal_like_loss + (1 - p) * CE_loss
        观测 focal-loss 的 loss曲线，eval-loss epoch1 先下降又上升, epoch2继续下降，猜测focal的加权过大，因此结合起来看看。
        """
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if isinstance(outputs, dict) else outputs[0]
        loss = None
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.get_input_embeddings().weight.shape[0]) # (batch_size * seq_len) * vocab_size
            shift_labels = shift_labels.view(-1) # (batch_size * seq_len)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss_fun = torch.nn.CrossEntropyLoss(reduction="none")
            ce_loss = ce_loss_fun(shift_logits, shift_labels)
            log_pt = - ce_loss # ce_loss = - log_p_true
            pt = torch.exp(log_pt)
            fc_like_loss = - (1-pt)**gamma * log_pt
            loss = pt * fc_like_loss + (1 - pt) * ce_loss
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

def set_training_args_manually(training_args):
    # training_args = training_args.set_lr_scheduler(name="cosine")
    if training_args.deepspeed: # ds config file name
        import json
        with open(training_args.deepspeed, "r") as f:
            ds_args = json.load(f)
            training_args.deepspeed = ds_args
        # fix bf16 and fp16
        if training_args.bf16:
            training_args.deepspeed["bf16"] = {"enabled": True}
            training_args.deepspeed["fp16"] = {"enabled": False}
#             if training_args.gradient_accumulation_steps > 1:
#                 assert False, (f"""Warning: bf16 is not encouraged with gradient_accumulation_steps > 1
# https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/deepspeed#bf16
# please check your config 
# """)
        
        # force change the deepspeed output file (tensorboard/flops) to output_dir that we set 
        if training_args.deepspeed.get("tensorboard",{}).get("enabled",False):
            training_args.deepspeed["tensorboard"]["output_path"] = os.path.join(training_args.output_dir, "ds_log", "tensorboard/")
        if training_args.deepspeed.get("flops_profiler",{}).get("enabled",False):
            training_args.deepspeed["flops_profiler"]["output_file"] = os.path.join(training_args.output_dir, "ds_log", "flops.txt")

        if training_args.deepspeed.get("scheduler", {}).get("type","") == "OneCycle":
            assert training_args.warmup_steps != 0, training_args.warmup_steps
            training_args.deepspeed["scheduler"]["params"]["cycle_first_step_size"] = training_args.warmup_steps
            training_args.deepspeed["scheduler"]["params"]["cycle_first_stair_count"] = training_args.warmup_steps
            training_args.deepspeed["scheduler"]["params"]["cycle_second_step_size"] = training_args.warmup_steps * 3
            training_args.deepspeed["scheduler"]["params"]["cycle_second_stair_count"] = training_args.warmup_steps * 3
            training_args.deepspeed["scheduler"]["params"]["cycle_max_lr"] = training_args.learning_rate
            training_args.deepspeed["scheduler"]["params"]["cycle_min_lr"] = training_args.learning_rate * 0.1
        
    
    training_args.__post_init__()
    return training_args

def set_random_seed(seed: int = 42):
    """Set random seed for reproducability. According to   
         https://github.com/THUDM/GLM/blob/main/pretrain_glm.py  and 
         https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy  and  
         https://zhuanlan.zhihu.com/p/629526120
    """

    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB).
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args = set_training_args_manually(training_args)

    logger.warning(f"Model args: {model_args}")
    logger.warning(f"Data args: {data_args}")
    logger.warning(f"Training args: {training_args}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Set seed before initializing model.
    set_random_seed(training_args.seed)
    logger.warning("wait 10 seconds for every GPU's RNG state setting DONE  ...")
    time.sleep(10)

    # Load model
    if not model_args.model_type:
        raise ValueError("Please specify a model_type, e.g. llama, chatglm, bloom, etc.")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}
        if training_args.qlora and (len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=model_args.cache_dir
        )
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            load_in_8bit=model_args.load_in_8bit,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            ) if training_args.qlora else None,
        )
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "model_max_length": model_args.model_max_length,
        "trust_remote_code": model_args.trust_remote_code,
        "encode_special_tokens": model_args.encode_special_tokens,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # set as the <unk> token

    if training_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            target_modules = training_args.target_modules.split(',') if training_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {training_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    logger.debug(f"Tokenizer: {tokenizer}")
    logger.debug(f"Model: {model}")

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        # Loading a dataset from local files.
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    def DA_Group(train_dataset: Dataset, 
                 max_seq_len: int = tokenizer.model_max_length, 
                 repeat_times="auto",
                 group_method="greedy") -> Dataset:
        """
        把多个短的对话，拼成一个长的多轮对话
        输入样本是文本， 输出样本也是文本
        """
        conv = get_conv_template(data_args.template_name)
        data_templete_delete = conv.bos_eos_len
        max_seq_len -= data_templete_delete
        new_data_list = []

        # step1 : 计算每个样本的 input_id 长度。注意删除 system 字段，并减去 data_templete_delete带来的影响
        # 删除 system 字段
        train_dataset = train_dataset.map(lambda example: {"instruction": ""})
        # 计算每个样本的 input_id 长度
        index2len = dict()
        for index, example in tqdm(enumerate(train_dataset), desc="da-group tokenizing ..."):
            conv.rounds = example.get('rounds', [])
            conv.instruction = example.get('instruction', "")
            input_id = []
            for q_str, a_str in conv.get_qa_list():
                q_id, a_id = tokenizer([q_str, a_str], add_special_tokens=False).input_ids
                input_id = input_id + q_id + a_id
            # 减去 data_templete_delete带来的影响
            index2len[index] = len(input_id) - data_templete_delete
            # 超长样本视为最长样本
            if index2len[index] > max_seq_len:
                index2len[index] = max_seq_len 
        
        # step2 : 根据 input_id 长度，DA-Group 样本， 重复 repeat_times 次
        # 估算 repeat_time if is auto 
        if repeat_times == "auto":
            _repeat_times = 1 + int(max_seq_len // ( sum(index2len.values()) / float(len(index2len))))
        else:
            _repeat_times = int(repeat_times)
        
        items_list = list(index2len.items())
        # 重复 repeat_times
        for _ in range(_repeat_times):
            logger.info(f"da_group 设置重复次数: {repeat_times}, 预期重复次数: {_repeat_times}, 当前已重复次数: {_}")

            # 每次都随机打乱
            random.shuffle(items_list)
            solver = GroupSFTSolver(M=max_seq_len, group_method=group_method)
            solver.insert_num_list([item[1] for item in items_list])
            for group_result in solver.get_group_index_result():
                random.shuffle(group_result)
                new_data = {
                    "rounds": [],
                    "instruction": "",
                    "id": "",
                }
                for result_id in group_result:
                    index = items_list[result_id][0]
                    new_data["rounds"].extend(train_dataset[index].get("rounds", []))
                    new_data["id"] += train_dataset[index].get("id", "") + "\n"
                new_data_list.append(new_data)

        # if is auto , 保持输入长度的一致
        if repeat_times == "auto":
            new_data_list = new_data_list[:len(train_dataset)]

        # step3 : 返回最终数据集
        return Dataset.from_list(new_data_list)

    def preprocess_function(examples, 
            split_multi_turn=data_args.split_multi_turn,
            latest_turn_only=data_args.latest_turn_only,
            full_qa_loss=data_args.full_qa_loss):
        """
        Preprocessing the datasets.
        """
        conv = get_conv_template(data_args.template_name)
        max_length=tokenizer.model_max_length

        input_ids, labels, attention_mask= [], [], []
        # Apply prompt templates
        rounds_list = examples.get('rounds', [])
        instruction_list = examples.get('instruction', [])
        for i, (rounds, instruction) in enumerate(zip(rounds_list, instruction_list)):
            conv.rounds = rounds
            conv.instruction = instruction
            if split_multi_turn:
                input_q, input_a = "", ""
                for q_str, a_str in conv.get_qa_list():
                    input_q, input_a = input_q + input_a + q_str, a_str
                    q_id, a_id = tokenizer([input_q, input_a], add_special_tokens=False).input_ids
                    input_id = q_id + a_id
                    label = input_id if full_qa_loss else [IGNORE_INDEX] * len(q_id) + a_id
                    if len(label) > max_length: 
                        break # 过长的对话，截断
                    input_ids.append(input_id[:max_length])
                    labels.append(label[:max_length])
            else:
                input_id, label = [], []
                for q_str, a_str in conv.get_qa_list():
                    q_id, a_id = tokenizer([q_str, a_str], add_special_tokens=False).input_ids
                    input_id = input_id + q_id + a_id
                    label = input_id if full_qa_loss else label + [IGNORE_INDEX] * len(q_id) + a_id
                if latest_turn_only:
                    # 找到最后一个 IGNORE_INDEX 并把他前面的全替换成 IGNORE_INDEX
                    final_ignore_index = len(label) - label[::-1].index(IGNORE_INDEX)
                    if final_ignore_index >= max_length:
                        continue
                    label = [IGNORE_INDEX] * final_ignore_index + label[final_ignore_index:]
                
                input_ids.append(input_id[:max_length])
                labels.append(label[:max_length])

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def group_sft(train_dataset: Dataset, 
                  max_seq_len: int = tokenizer.model_max_length,
                  group_method: str = "greedy",) -> Dataset:
        """抛弃train_dataset中长度大于max_seq_len的元素，把其余的元素分成尽可能少的组，每组中元素之和小于max_seq_len。"""
        input_ids = train_dataset["input_ids"]
        labels = train_dataset["labels"]
        sample_len_list = [len(x) for x in input_ids]
        print("diag round num is", len(sample_len_list))
        print("sum token num is", sum(sample_len_list))
        
        def group_elements(X: List[int], m: int, group_method: str):
            """X: seq_len_list; m: max_len_num"""
            solver = GroupSFTSolver(M=m, group_method=group_method)
            solver.insert_num_list(X)
            return solver.get_group_index_result()

        group_input_ids = []
        group_labels = []
        for index_list in tqdm(group_elements(sample_len_list, max_seq_len, group_method), desc="build group dataset"):
            one_group_input_ids, one_group_labels = [], []
            random.shuffle(index_list) # 随机打乱，不然总是长样本在前面，短样本在后面
            for index in index_list:
                one_group_input_ids.extend(input_ids[index])
                one_group_labels.extend(labels[index])
            group_input_ids.append(one_group_input_ids)
            group_labels.append(one_group_labels)
        
        return Dataset.from_dict({
            "input_ids": group_input_ids,
            "labels": group_labels,
            })

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        group_temp_dataset_path = os.path.join(training_args.output_dir, "group_temp_dataset")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            if data_args.da_group:
                # 由于 Group 和 da_group 互斥，因此使用相同的 group_temp_dataset_path 作为中间文件夹
                print(f"da group train dataset to {tokenizer.model_max_length}")
                if os.path.exists(group_temp_dataset_path):
                    train_dataset = load_from_disk(group_temp_dataset_path)
                else:
                    train_dataset = DA_Group(train_dataset, max_seq_len=tokenizer.model_max_length, repeat_times=data_args.da_group_repeat_times, group_method=data_args.da_group_method)
                    train_dataset.save_to_disk(group_temp_dataset_path)
            train_dataset = train_dataset.shuffle(seed=training_args.seed).map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            if data_args.group:
                print(f"grouping train dataset to {tokenizer.model_max_length}")
                if os.path.exists(group_temp_dataset_path):
                    train_dataset = load_from_disk(group_temp_dataset_path)
                else:
                    train_dataset = group_sft(train_dataset, group_method=data_args.group_method).shuffle(seed=training_args.seed)
                    train_dataset.save_to_disk(group_temp_dataset_path)
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                partial(preprocess_function, split_multi_turn=False, full_qa_loss=False, latest_turn_only=False), # 保证不同配置的train, eval 时指标统一
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    try:
        model.enable_input_require_grads()
    except:
        logger.warning(f"Could not enable input require_grads on model, skipping.")
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4,  # prepare for shift short attention
    )
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
        loss_fun_name=data_args.loss_fun_name,
        focal_alpha=data_args.focal_alpha,
        focal_gamma=data_args.focal_gamma,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        logger.debug(f"Training metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        # save_model(training_args.output_dir, model, tokenizer, training_args)
        trainer.save_model(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        logger.debug(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 删除暂存文件夹
    # 在打开一个文件时，nfs 文件系统会在文件所在的目录生成一个 .nfs 文件，如果有文件描述符为关闭，这时去删除文件所在的目录，就会发生如上错误。
    del trainer
    with training_args.main_process_first(desc="Train dataset tokenization"):
        if os.path.exists(group_temp_dataset_path):
            shutil.rmtree(group_temp_dataset_path, ignore_errors=True)
    

if __name__ == "__main__":
    main()
