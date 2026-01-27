nvidia-smi nvlink --status | grep inactive >1
if [ $? -ne 1 ];then
  echo "nvlink is inactive! check manually"
  exit -1
else
  echo "nvlink is ok~"
fi



EXPERIMENT_TAG='sft-llama3-qifu-8192-20k'

deepspeed --hostfile config/hostfile \
    --master_addr 11.7.48.230 \
    --ssh_port 2255 \
    pretraining.py \
    --model_type auto \
    --finetune_mode sft \
    --sft_group True \
    --all_loss False \
    --just_last_answer True \
    --split_multi_turn False \
    --system_prompt "你是一个严格遵从指令的中文人工智能助手" \
    --model_name_or_path  /data/oceanus_share/boruipeng/project/finance_llm/outputs/llama3-8b-v2-4k-eot \
    --tokenizer_name_or_path /data/oceanus_share/llm_model/Meta-Llama-3-8B-Instruct \
    --train_file_dir /data/oceanus_share/boruipeng/project/xzm/combine0329/kmeans_weight-coreset_general_sample+task_20000/train/train.jsonl \
    --validation_file_dir /data/oceanus_share/boruipeng/project/xzm/combine0329/kmeans_weight-coreset_general_sample+task_20000/test/test.jsonl \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --save_only_model True \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --seed 3407 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --weight_decay 1e-4 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_dir  /data/oceanus_share/boruipeng/project/tensorboard/new/${EXPERIMENT_TAG} \
    --save_strategy epoch \
    --save_total_limit 10 \
    --gradient_accumulation_steps 1 \
    --block_size 4096 \
    --torch_compile True \
    --output_dir ./outputs/${EXPERIMENT_TAG} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --log_on_each_node True \
    --torch_dtype bfloat16 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --deepspeed ./config/ds_2_config.json \
    --bf16 \
    --bf16_full_eval