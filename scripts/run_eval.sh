CUDA_VISIBLE_DEVICES=0 python eval/main.py \
  --model_name_or_path PATH_TO_MODEL \
  --tokenizer_name_or_path PATH_TO_TOKENIZER \
  --task ceval \
  --output_dir output \
  --shot 5 \
#  --test \
#  --llama \
#  --chat_mode