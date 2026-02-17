from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import List
from llama_cpp import Llama, LlamaGrammar

JSON_SCHEMA_GBNF = r"""
root ::= obj
obj ::= "{" ws "\"id\"" ws ":" ws jstring ws "," ws "\"explanation\"" ws ":" ws jstring ws "}"
ws ::= [ \t]*
jstring ::= "\"" jchars "\""
jchars ::= jchar*
jchar ::= escaped | unescaped
escaped ::= "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
unescaped ::= [^"\\\n\r]
"""

class LlamaCppTeacher:
    def __init__(self, my_config, enable_grammer=1, device: str| None = None):
        self.cfg = my_config
        self.model_path = self.cfg.model_path
        self.enable_grammer = enable_grammer
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.cfg.n_gpu_layers,      # -1 means "put all layers on GPU"
            n_ctx=self.cfg.n_ctx,           # Good balance for M1 Pro memory
            n_batch=self.cfg.n_batch,          # Essential for GPU parallel processing
            n_threads=self.cfg.n_threads,          # M1 Pro has 8 Performance cores
            flash_attn=self.cfg.flash_attn,
            verbose=self.cfg.verbose,          # This will show the "BLAS = 1" log confirming GPU use
            seed=self.cfg.seed)
        
        if self.enable_grammer:
            self.json_grammar = LlamaGrammar.from_string(JSON_SCHEMA_GBNF)
        else:
            self.json_grammar = None
        self.device = device if device is not None else "cpu"
                
    def _complete(self, prompt: str, *, 
                max_tokens: int, 
                temperature: float, 
                top_p: float, 
                repetition_penalty: float) -> str:
        
        model_output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=[],
            grammar=self.json_grammar
        )
        return (model_output["choices"][0]["text"] or "").strip()
    
    def generate_response(self, 
                        prompt: str, 
                        max_new_tokens: int,
                        temperature: float,
                        top_p: float,
                        repetition_penalty: float,
                        do_sample: bool) -> str:
        temp = 0.0 if not do_sample else temperature
        outputs = self._complete(prompt=prompt,
                                max_tokens=max_new_tokens,
                                temperature=temp,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty)
        return outputs
    
    def generate_response_chat(self,
                        system_prompt: str,
                        user_prompt: str,
                        max_new_tokens: int,
                        temperature : float,
                        top_p: float,
                        repetition_penalty: float,
                        do_sample: bool) -> str:

        temp = 0.0 if not do_sample else temperature

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            grammar=self.json_grammar,
        )

        return (output["choices"][0]["message"]["content"] or "").strip()
        
    def generate_response_chat_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
    ) -> List[str]:
        # llama-cpp-python doesn’t natively batch prompts in one call.
        # Still useful: you can keep your pipeline interface unchanged.
        outputs: List[str] = []
        for sp, up in zip(system_prompts, user_prompts):
            outputs.append(
                self.generate_response_chat(
                    system_prompt=sp,
                    user_prompt=up,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                )
            )
        return outputs