import os
import warnings
import transformers
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from tools.qifu_trainer import QifuTrainer,qifu_default_data_collator
from tools.dataset_utils import QifuDataset,IGNORE_INDEX,QifuSftDataset
from tools.utils import seed_everything
from transformers.utils.versions import require_version


######################
#default log level is warning
######################

#transformers.logging.set_verbosity_info()

mode2model = {
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

mode2dataset = {
    "pretrain":QifuDataset,
    "sft":QifuSftDataset,
}


@dataclass
class ModelArguments:
    finetune_mode: Optional[str] = field(
        default="pretrain",
        metadata={
            "help": ('pretrain or sft'),
            "choices": ["pretrain", "sft"],
        },
    )
    model_type: str = field(
        default=None,
        metadata={"help": "Model type"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help":"The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    mode2model.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")

@dataclass
class DataTrainingArguments:
    train_file_dir: Optional[str] = field(default=None,metadata={"help": "The train text data file folder."})
    validation_file_dir: Optional[str] = field(default=None,metadata={"help": "The evaluation data file folder."})
    block_size: Optional[int] = field(default=1024,metadata={"help": "max seq length"},)
    padding: bool = field(default = False,metadata={"help": ""})
    lazy_mode: bool = field(default=True,metadata={"help": "Whether to use lazy mode load dataset or not."})
    sft_group: bool = field(default=True,metadata={'help': "whether to group or not"})
    all_loss: bool = field(default=True,metadata={'help': "whether to use all input loss in sft training"})
    split_multi_turn: bool = field(default=True,metadata={'help': "whether to split multi-turn"})
    system_prompt: str = field(default='你是一个严格遵从指令的中文人工智能助手',metadata={'help':'system_prompt'})
    just_last_answer: bool = field(default=True,metadata={'help': "just use last answer loss in multi-round or not, only works when all_loss is False"})

    def __post_init__(self):
        pass


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }

def compute_metrics(eval_preds,metric_key_prefix):
    '''
    boruipeng: different eval dataset to different metric
    '''
    preds, labels = eval_preds
    labels = labels[:, 1:]#type:np.ndarray
    preds = preds[:, :-1]#type:np.ndarray
    if 'corpus' in metric_key_prefix:
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
    else:
        index = (labels != IGNORE_INDEX).nonzero()
        labels = labels[index]
        preds = preds[index]
    return accuracy(predictions=preds, references=labels)




def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


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
    try:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    except ZeroDivisionError:
        print(f"trainable params: {trainable_params} || all params: {all_param}")




def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    logger.info(f"training_args parameters:\n {training_args}")
    logger.info('-----------------------------' * 5)
    logger.info(f"model_args parameters:\n {model_args}")
    logger.info('-----------------------------' * 5)
    logger.info(f"data_args parameters:\n {data_args}")
    logger.info('-----------------------------' * 5)

    # Set seed before initializing model.
    seed_everything(int(training_args.seed))

    ################
    # Model & Tokenizer
    ################
    config_class, model_class, tokenizer_class = mode2model[model_args.model_type]
    torch_dtype = getattr(torch, model_args.torch_dtype)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype = (torch_dtype),
    )

    tokenizer_name_or_path = model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else  model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name_or_path,
        cache_dir = model_args.cache_dir,
        use_fast = True,
        trust_remote_code = model_args.trust_remote_code,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    #for Qwen1
    #tokenizer.pad_token_id = tokenizer.eod_id

    ################
    # Dataset
    ################

    dataset_class = mode2dataset[model_args.finetune_mode]
    logger.info("Full parameters training")
    print_trainable_parameters(model)

    logger.info('-' * 25 + f'loading dataset from {data_args.train_file_dir}' + '-' * 25)
    train_dataset = dataset_class(
        data_args.train_file_dir,
        tokenizer,
        lazy_mode=data_args.lazy_mode,
        max_seq_length=data_args.block_size,
        padding=data_args.padding,
        sft_group=data_args.sft_group,
        all_loss=data_args.all_loss,
        split_multi_turn=data_args.split_multi_turn,
        just_last_answer=data_args.just_last_answer,
        system_prompt=data_args.system_prompt
    )

    eval_dataset = {}
    if model_args.finetune_mode == 'pretrain':
        for shot in [0,5]:
            for exam in ['ceval','fin_eval','fin_IQ','cmmlu']:
                eval_dataset[f'{exam}_{shot}shot'] = QifuDataset(
                    f'/app/nfs_share_dir/1/archive/v2/exams/4-exams-token/llama3/exams-{exam}-{shot}shot-token',
                    tokenizer,
                    lazy_mode=data_args.lazy_mode,
                    max_seq_length=data_args.block_size,
                    padding=data_args.padding,
                    tag=False
                )
    eval_dataset['corpus'] = dataset_class(
        data_args.validation_file_dir,
        tokenizer,
        lazy_mode=data_args.lazy_mode,
        max_seq_length=data_args.block_size,
        sft_group=False,
        tag=False,
        all_loss=data_args.all_loss,
        just_last_answer=data_args.just_last_answer,
        split_multi_turn=data_args.split_multi_turn,
        system_prompt=data_args.system_prompt
    )

    logger.info(f"train datasets: {tokenizer.decode(train_dataset[0]['input_ids'])}")
    logger.info(f"eval datasets: {eval_dataset}")



    ################
    # Training
    ################

    trainer = QifuTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=qifu_default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )
    #trainer.add_callback(EvalOnEpochBeginCallback)

    #before training, evaluate
    # trainer.evaluate()

    # Training
    if training_args.do_train:
        logger.info("***" * 5 +  "Train" + 5 * "***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        # metrics["train_samples"] = max_train_samples
        logger.info(f"Training metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        logger.info(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
