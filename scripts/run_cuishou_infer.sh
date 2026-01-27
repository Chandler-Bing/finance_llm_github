model_tag="sft-llama3-qifu-8192-20k_merge"
task_list=("brp_data_file_00" "brp_data_file_01" "brp_data_file_02" "brp_data_file_03" "brp_data_file_04" "brp_data_file_05" "brp_data_file_06" "brp_data_file_07")
data_path=("")
for i in ${!data_path[@]}; do
  type=${data_path[$i]}
  for j in ${!task_list[@]}; do
    task_name=${task_list[$j]}
    gpu=$(((0+j) % 8))
    echo ----gpu device $gpu running $type $task_name ----
    CUDA_VISIBLE_DEVICES=$gpu python infer.py \
      --model_type llama3 \
      --model_paths /data/oceanus_share/boruipeng/project/finance_llm/outputs/$model_tag \
      --tokenizer_paths /data/oceanus_share/llm_model/Meta-Llama-3-8B-Instruct \
      --data_file /data/oceanus_share/boruipeng/project/finance_llm/business/cuishou/data/$task_name \
      --predictions_file ./sft_task/$model_tag/cuishou$type/$task_name.json \
      --max_new_tokens 1000 \
      --task &
  done
  wait
done