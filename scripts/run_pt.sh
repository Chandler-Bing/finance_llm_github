nvidia-smi nvlink --status | grep inactive >1
if [ $? -ne 1 ];then
  echo "nvlink is inactive! check manually"
  exit -1
else
  echo "nvlink is ok~"
fi

#export HF_HOME=/app/nfs_share_dir/5/cache


EXPERIMENT_TAG=''

deepspeed --hostfile config/hostfile \
    --master_addr 11.7.48.230 \
    --ssh_port 2255 \
    pretraining.py \
    --model_type auto \
    --model_name_or_path  /data/oceanus_share/llm_model/Meta-Llama-3-8B \
    --train_file_dir /data/oceanus_share/boruipeng/archive/v2/token-llama3-4k/train \
    --validation_file_dir /data/oceanus_share/boruipeng/archive/v2/token-llama3-4k/eval \
    --lazy_mode True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 150 \
    --seed 3 \
    --warmup_ratio 0.01 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 1e-4 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_dir  /data/oceanus_share/boruipeng/project/tensorboard/new/${EXPERIMENT_TAG} \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 5 \
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


