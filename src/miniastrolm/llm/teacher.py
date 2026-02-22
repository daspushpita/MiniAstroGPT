from __future__ import annotations
from dataclasses import dataclass
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from miniastrolm.utils.device import resolve_device



class Llama_Teacher:
    def __init__(self, my_config,
                device: str | None = None,
                torch_dtype: torch.dtype = torch.float16,):
        
        """Initializes the Llama Teacher model and tokenizer.
        Mac GPU path uses MPS.
        """
            
        self.my_config = my_config
        self.device = resolve_device(self.config.training.device)
        self.model_id = my_config.model_id
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        
        if self.my_config.four_bit_teacher:
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            )
            if self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map={"": "mps"},
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                )
        else:
            if self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map={"": "mps"},
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            
        self.model.eval()

    def generate_response(self,
                        prompt: str,
                        max_new_tokens: int,
                        temperature : float,
                        top_p: float,
                        repetition_penalty: float,
                        do_sample: bool) -> str:
        """Generates responses from a given prompt

        Args:
            tokenizer (_type_): The tokenizer instance
            model (_type_): The model instance
            prompt (str): The input prompt string
        """

        # 1) tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # keep this length so we can slice out only the new tokens later
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
        # 3. Decode the output tokens to text
        output_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()
    
    def generate_response_chat(self,
                        system_prompt: str,
                        user_prompt: str,
                        max_new_tokens: int,
                        temperature : float,
                        top_p: float,
                        repetition_penalty: float,
                        do_sample: bool) -> str:
        """Generates responses from a given prompt

        Args:
            tokenizer (_type_): The tokenizer instance
            model (_type_): The model instance
            prompt (str): The input prompt string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # 1) tokenize
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 2) move to device (MPS or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # keep this length so we can slice out only the new tokens later
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
        # 3. Decode the output tokens to text
        output_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return output_text.strip()
    
    
    def generate_response_chat_batch(self,
                        system_prompts: list[str],
                        user_prompts: list[str],
                        max_new_tokens: int,
                        temperature : float,
                        top_p: float,
                        repetition_penalty: float,
                        do_sample: bool) -> list[str]:
        
        self.system_prompts = system_prompts
        self.user_prompts = user_prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.prompts = []
        for s, u in zip(self.system_prompts, self.user_prompts):
            messages = [
            {"role": "system", "content": s},
            {"role": "user", "content": u},
        ]
            templated_prompts = self.tokenizer.apply_chat_template(messages,tokenize=False,
                                                                add_generation_prompt=True)
            self.prompts.append(templated_prompts)
            

        # Make batching safe
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        inputs = self.tokenizer(self.prompts, return_tensors="pt", 
                                padding=True, truncation=True).to(self.device)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # prompt lengths = number of tokens that are part of the prompt (not padding)
        # attention_mask is 1 for real tokens, 0 for padding
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )        
        
        texts: list[str] = []
        for i in range(outputs.shape[0]):
            start = int(prompt_lens[i])
            gen_ids = outputs[i][start:]
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return texts