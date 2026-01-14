# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import List

def load_disease_names(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_openai_api(prompt: str, model_name: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()

def call_local_llama2(prompt: str, model_path: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    out = pipe(prompt, max_new_tokens=256, temperature=0.7, do_sample=True, return_full_text=False)
    return out[0]["generated_text"]

def call_local_chatglm(prompt: str, model_path: str) -> str:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
    resp, _ = model.chat(tokenizer, prompt, history=[])
    return resp

def generate_questions_for_disease(disease: str, backend: str, model: str) -> List[str]:
    prompt = f"""
You are a medical dialogue assistant. I will give you a disease name. 
Generate exactly 5 different natural language questions in English, 
each requiring the answerer to consult relevant medical documents for accurate responses.
Cover different aspects such as symptoms, diagnostic methods, treatment plans, 
preventive measures, and risk factors. 

Disease name: {disease}
Output format: only return the list of questions, one per line, without numbering.
"""

    if backend == "openai_api":
        raw = call_openai_api(prompt, model)
    elif backend == "llama2_local":
        raw = call_local_llama2(prompt, model)
    elif backend == "chatglm_local":
        raw = call_local_chatglm(prompt, model)
    else:
        raise ValueError(f"Unknown backend {backend}")

    lines = [l.strip("- ").strip() for l in raw.split("\n") if l.strip()]
    questions = [l for l in lines if l.endswith("?")]
    
    return questions[:5]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="Information/Target_Disease.json")
    parser.add_argument("--output_file", type=str, default="Information/questions.jsonl")
    parser.add_argument("--backend", type=str, choices=["openai_api", "llama2_local", "chatglm_local"], default="openai_api")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name or local path")
    args = parser.parse_args()

    diseases = load_disease_names(args.input_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for d in diseases:
            questions = generate_questions_for_disease(d, args.backend, args.model)
            for q in questions:
                f.write(json.dumps({"disease": d, "question": q}, ensure_ascii=False) + "\n")

    print(f"[OK] Generated questions saved to {args.output_file}")

if __name__ == "__main__":
    main()

