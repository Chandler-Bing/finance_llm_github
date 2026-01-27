import loguru
import torch
import json
import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

logger = loguru.logger

class Llama3StopOnTokens(StoppingCriteria):
    def __init__(self):
        self.stop_ids = [128009,128001]
        super(Llama3StopOnTokens).__init__()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class Qwen2StopOnTokens(StoppingCriteria):
    def __init__(self):
        self.stop_ids = [151643,151645]
        super(Qwen2StopOnTokens).__init__()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class Yi1_5StopOnTokens(StoppingCriteria):
    def __init__(self):
        self.stop_ids = [2,7]
        super(Yi1_5StopOnTokens).__init__()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class InferenceContainer:
    def __init__(self,model_types,model_paths,tokenizer_paths,torch_dtype,trust_remote_code=True,gradio=False):
        self.models = {}
        self.model2stop_ids = {
            'llama3':Llama3StopOnTokens(),
            'qwen2':Qwen2StopOnTokens(),
            'yi1.5':Yi1_5StopOnTokens(),
        }
        self.gradio = gradio
        #build model here...
        for device,(model_type,model_path,tokenizer_path) in enumerate(zip(model_types,model_paths,tokenizer_paths)):
            assert device < 8,'not enough gpus'
            base_model_name = model_path.split('/')[-1]
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=(torch_dtype),
                trust_remote_code=trust_remote_code
            ).to(torch_dtype).to(f'cuda:{device}')
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            self.models[base_model_name] = (model,tokenizer,device,model_type)

    @torch.inference_mode()
    def generate_fn(
            self,
            message,
            history,
            which,
            system_prompt = 'you are a helpful assistant',
            top_k = 5,
            top_p = 0.9,
            temperature = 0.1,
            repetition_penalty = 1.0,
            max_new_tokens = 512,
            do_sample = True,
            num_beams = 1,
    ):
        model,tokenizer,device,model_type = self.models[which]
        stop = self.model2stop_ids[model_type]

        if isinstance(message,list):
            conversation = message
        else:
            history_transformer_format = history + [[message, ""]]
            conversation = [{'role':'system','content':system_prompt}] if system_prompt else []
            for user_content,assistant_content in history_transformer_format:
                conversation.append({'role':'user','content':user_content})
                conversation.append({'role':'assistant','content':assistant_content})
            del conversation[-1]

        model_inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        model_inputs_text = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.debug(f'model input text: {json.dumps({"input_text":model_inputs_text},ensure_ascii=False)}',)

        model_inputs = model_inputs.to(f'cuda:{device}')
        streamer = TextIteratorStreamer(tokenizer, timeout=20., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=model_inputs,
            streamer=streamer if self.gradio else None,
            stopping_criteria=StoppingCriteriaList([stop]),
            #pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
        hyperparameter = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True if do_sample else False,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
        )
        generate_kwargs.update(hyperparameter)
        hyperparameter.update({'which_model':which,'system_prompt':system_prompt})
        logger.debug(f'hyperparameter: {json.dumps(hyperparameter,ensure_ascii=False,indent=4)},if system_prompt==False, will not use system prompt, here just for logging purposes')


        if self.gradio:
            t = Thread(target=model.generate, kwargs=generate_kwargs)
            t.start()
            partial_message = ''
            for new_token in streamer:
                partial_message += new_token
                if streamer.next_tokens_are_prompt:
                    logger.debug(f'partial_message:{json.dumps({"partial_message":partial_message},ensure_ascii=False,indent=4)}')
                yield partial_message
        else:
            generation_output = model.generate(**generate_kwargs)
            output_ids = generation_output[0]
            while output_ids[-1] in stop.stop_ids:
                output_ids = output_ids[:-1]
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            l_prompt = len(tokenizer.decode(model_inputs.tolist()[0], skip_special_tokens=False))
            output = output[l_prompt:]
            yield output


    def run_gradio(self,base_model_names):
        with gr.Blocks() as demo:
            with gr.Row(variant="panel",equal_height = False):
                with gr.Column(scale=2, variant="compact"):
                    which = gr.Radio(base_model_names,label="which",value=base_model_names[0],)
                    system_prompt = gr.Textbox("", label="System Prompt",scale=2)
                    top_k = gr.Slider(0,100, label="top_k",step=1,value=50,scale=2)
                    top_p = gr.Slider(0.0,1.0, label="top_p",step=0.1, value=0.9,scale=2)
                    temperature = gr.Slider(0.0,2.0, label="temperature", step=0.1,value=0.6,scale=2)
                    repetition_penalty = gr.Slider(0.0,10.0, label="repetition_penalty",step=0.1, value=1.0,scale=2)
                    max_new_tokens = gr.Slider(1,1024, label="max_new_tokens", value=512,scale=2)
                    do_sample = gr.Slider(0,1, label="do_sample", step=1, value=1,scale=2)
                    num_beams = gr.Slider(0,100, label="num_beams",step=1,value=1,scale=2)
                with gr.Column(scale=20, variant="compact"):
                    additional_inputs = [which,system_prompt,top_k,top_p,temperature,repetition_penalty,max_new_tokens,do_sample,num_beams]
                    chatbot = gr.ChatInterface(
                        self.generate_fn,
                        chatbot=gr.Chatbot(min_width=500,scale=200,height=700,render=False),
                        additional_inputs=additional_inputs,
                        title="a simple chat bot",
                        submit_btn="⬅ Send",
                        retry_btn="🔄 Regenerate Response",
                        undo_btn="↩ Delete Previous",
                        clear_btn="🗑️ Clear Chat",
                    )
            demo.launch(
                root_path='/MTEuNy40OC45MTo4MDcx/user/j-boruipeng-jk/vscode/proxy/7860',
                share=False,
                server_name='0.0.0.0',
                server_port=7860

            )