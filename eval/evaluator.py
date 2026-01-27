import torch
from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    GPT2LMHeadModel,
    T5Tokenizer,
    GPT2Tokenizer
)

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name.lower():
        prompt = tokenizer.build_chat_input(prompt)
    elif 'qifu' in model_name.lower():
        prompt = f'<s><reserved_21>{prompt}<reserved_22><reserved_23>'
    elif 'baichuan2' in model_name.lower():
        prompt = f'<reserved_106>{prompt}<reserved_107>'
    elif 'qwen1_5' in model_name.lower():
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt


class Evaluator():
    def __init__(self,model_path,tokenizer_path,max_length=4096,lora_weight='',llama = False,bbt = False,logit_from_choice=True,device = 'cuda'):
        self.max_length = max_length
        self.model_path = model_path
        self.logit_from_choice = logit_from_choice
        self.device = device
        if llama:
            self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=(
                    torch.bfloat16
                ),
            ).to(torch.bfloat16)
            if lora_weight:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_weight,
                    torch_dtype=torch.bfloat16,
                )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
        elif bbt:
            self.model = GPT2LMHeadModel.from_pretrained(model_path,device_map=device,)
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            #self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=(
                    torch.bfloat16
                ),
            ).to(torch.bfloat16)
            if lora_weight:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_weight,
                    torch_dtype=torch.bfloat16,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True,
            )
        self.model.config.use_cache = True
        self.model.eval()

    @torch.inference_mode()
    def logit_answer(self,data):
        prompt = data['prompt']
        with_answer = True if 'answer' in data else False

        # import pandas as pd
        # df = pd.read_csv('./fin_alipay_shot')
        # tmp = ''
        # for index,row in df.iterrows():
        #     tmp += f'{row["question"]}\nA.{row["A"]}\nB.{row["B"]}\nC.{row["C"]}\nD.{row["D"]}\n答案是：{row["answer"]}\n'
        # ins = prompt.split('\n')[0]
        # timu = '\n'.join(prompt.split('\n')[1:])
        # prompt = '\n'.join([ins,tmp,timu])
        # data['prompt'] = prompt

        text = prompt + data['answer'] if with_answer else prompt
        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        tmp = input_ids.flatten().tolist()
        attention_mask = [1] * len(tmp)
        labels = [-100] * (len(tmp) - 1) + [tmp[-1]]
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        labels = torch.tensor([labels]).to(self.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels = labels
        )
        logits = output.logits[:, -2].flatten() if with_answer else output.logits[:, -1].flatten()
        loss = output.loss.item()
        data['loss'] = loss

        if self.logit_from_choice:
            raise ValueError('Do not use logit_from_choice way')
            # candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in choices]
            # candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            # probs = (
            #     torch.nn.functional.softmax(
            #         candidate_logits,
            #         dim=0,
            #     )
            #         .detach()
            #         .cpu()
            #         .numpy()
            # )
            # pred_answer = {i: k for i, k in enumerate(choices)}[np.argmax(probs)]
        else:
            pred_id = logits.argmax(dim=-1)
            pred_answer = self.tokenizer.decode(pred_id)

        data['pred_answer'] = pred_answer
        data['pred_id'] = str(pred_id.item())
        data['groudth_id'] = str(input_ids.flatten().tolist()[-1])
        return data

    @torch.no_grad()
    def long_bench_answer(self,data):
        #max_gen = LongBench().dataset2maxlen[data['subject']]
        ans_len = list(map(lambda x:len(x),data['answer']))
        max_gen = max(ans_len) + 3
        prompt = data['prompt']

        tokenized_prompt  = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in self.model_path:
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > self.max_length:
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            half = int(self.max_length/2)
            prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        #TODO: build chat template for each model
        if data['subject'] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(self.tokenizer, prompt, self.model_path)


        if "chatglm3" in self.model_path:
            if data['subject'] in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input_ids = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.device)
            else:
                input_ids = prompt.to(self.device)
        else:
            input_ids = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.device)

        context_length = input_ids.input_ids.shape[-1]
        if data['subject'] == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = self.model.generate(
                **input_ids,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = self.model.generate(
                **input_ids,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = pred.replace('<reserved_24>','')
        data['pred_answer'] = pred

        return data

    @torch.no_grad()
    def ppl_on_window(self,text,stride=1,max_position_embeddings=4096 * 8):

        self.model.config.max_position_embeddings = max_position_embeddings

        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
        ).to(self.device)
        labels = input_ids.detach().clone()
        length = input_ids.size(-1)

        logits = self.model(input_ids).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        for i in range(2,length,stride):
            tmp_shift_logits = shift_logits[..., :i, :].view(-1, self.model.config.vocab_size)
            tmp_labels = shift_labels[..., :i].view(-1)
            loss = loss_fct(tmp_shift_logits, tmp_labels)
            ppl = torch.exp(loss)
            print(i,loss.item(),ppl.item(),sep = '==')

    @torch.no_grad()
    def get_ppl(self,text,max_position_embeddings=4096 * 8):

        self.model.config.max_position_embeddings = max_position_embeddings

        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
        ).to(self.device)
        labels = input_ids.detach().clone()

        loss = self.model(
            input_ids=input_ids,
            labels=labels,
        ).loss
        return torch.exp(loss).item()






if __name__ == "__main__":
    pass








