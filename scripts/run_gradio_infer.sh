#--model_paths: split with ,
#--tokenizer_paths: split with ,
#--model_types: split with ,define your model type with stop token_ids in tools/inference_container.py
#--gradio: run infer in gradio mode

nohup python infer.py \
        --model_types llama3 \
        --model_paths  /data/oceanus_share/llm_model/Meta-Llama-3-8B-Instruct \
        --tokenizer_paths  /data/oceanus_share/llm_model/Meta-Llama-3-8B-Instruct \
        --gradio &