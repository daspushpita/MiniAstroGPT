from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_casual_llm(model_dir_or_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir_or_name).to(device)
    model.eval()
    return tokenizer, model


def build_student_prompt(abstract: str, bos: str = "") -> str:
    return (
        f"{bos}Task: Explain the abstract in simple, non-technical language. "
        f"Stay strictly on-topic.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Explanation:\n"
    )


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str,
                max_new_tokens: int = 200,
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True,
                repetition_penalty: float = 1.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    output_ids = model.generate(**gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()

def run_baseline(eval_samples_path: str,
                out_path: str,
                device: str,
                baseline_name: str = "gpt2",
                build_prompt_fn=None):
    
    tokenizer, model = load_casual_llm(baseline_name, device)
    bos = tokenizer.bos_token or ""
    prompt = (
        build_prompt_fn
        if build_prompt_fn is not None
        else (lambda abstract, _id: build_student_prompt(abstract, bos=bos))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    count = 0
    with open(eval_samples_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            abstract = item.get("input")
            _id = item.get("id")
            if not abstract or not _id:
                print(f"Skipping malformed item with missing 'input' or 'id': {item}")
                continue
            prompt_text = prompt(abstract, _id)
            predictions = generate_one(model, tokenizer, prompt_text)
            output_item = {
                "id": _id,
                "input": abstract,
                "output": predictions,
            }
            print (f"Processed item {_id}: {predictions[:60]}...", count)
            results.append(output_item)
            count += 1
        
        with open(out_path, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved baseline predictions to {out_path}")
    

if __name__ == "__main__":
    run_baseline(
        eval_samples_path="/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/evals/eval_v1_10.jsonl",
        out_path="/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/evals/baseline_gpt2_v1_10.jsonl",
        device="mps",
        baseline_name="gpt2")
