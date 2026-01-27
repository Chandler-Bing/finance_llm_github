model_tag="sft-llama3-qifu-4096-100k-single"
task_list=(llsrc.jsonl slpwc.jsonl slrfc.jsonl slsrc.jsonl bbtfinfe.jsonl c3dialogue.jsonl cfbenchcompany.jsonl clueocnli.jsonl csldcp.jsonl  duee.jsonl iree.jsonl bq.jsonl c3text.jsonl cfbenchsentiment.jsonl  cluewcs.jsonl  cvalue.jsonl  eprstmt.jsonl  tnews.jsonl)
data_path=("" _5_shot _5_shot_rounds_mod)
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
      --data_file /data/oceanus_share/gpu04/llm_sft_data/3_train_dataset/review-eval-data/sft-format$type/$task_name \
      --predictions_file ./sft_task/$model_tag/sft-format$type/$task_name.json \
      --task &
  done
  wait
done


