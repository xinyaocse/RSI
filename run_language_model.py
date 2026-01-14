import fire
from llama.llama import Llama
import warnings
import json
import os
from openai import OpenAI
import requests
from typing import Optional


def load_prompts(path):
    prompts_path = f"./Inputs&Outputs/{path}/prompts.json"
    if not os.path.exists(prompts_path):
        prompts_path = f"./Inputs&Outputs/{path}/prompts.jsonl"

    all_prompts = []
    if prompts_path.endswith(".json") and os.path.exists(prompts_path):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)
    elif prompts_path.endswith(".jsonl") and os.path.exists(prompts_path):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            for ln in f:
                obj = json.loads(ln)
                if isinstance(obj, dict) and "prompt" in obj:
                    all_prompts.append(obj["prompt"])
                else:
                    all_prompts.append(obj)
    else:
        raise FileNotFoundError(f"No prompts.json or prompts.jsonl found in {path}")

    return all_prompts


def resolve_provider(model_name: str) -> str:
    name = (model_name or "").lower()
    if "deepseek" in name:
        return "zzz"
    if name.startswith("gpt") or name.startswith("gpt-"):
        return "zzz"
    return "openai"


def build_client_for_provider(provider: str) -> Optional[OpenAI]:
    provider = (provider or "").lower()

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
        return OpenAI(api_key=key)

    if provider == "zzz":
        return None

    raise ValueError(f"Unknown provider: {provider}")


def call_chat(
    client: Optional[OpenAI],
    model: str,
    prompt: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 256,
    system: str = "You are a helpful assistant",
    provider: str = "openai",
) -> str:

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    provider = (provider or "").lower()

    if provider == "zzz":
        key = os.environ.get("ZZZ_API_KEY")
        if not key:
            raise EnvironmentError("Missing ZZZ_API_KEY in environment variables.")

        url = "https://api.zhizengzeng.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"[zzz] Empty choices in response: {data}")

        msg = choices[0].get("message", {}) or {}
        content = msg.get("content", None)

        if content is None:
            content = choices[0].get("text", None)

        if content is None:
            raise RuntimeError(f"[zzz] Cannot find content in response: {data}")

        return content

    if client is None:
        raise RuntimeError(f"[{provider}] OpenAI client is None, but SDK call is required.")

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )

    if not getattr(resp, "choices", None):
        raw = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
        raise RuntimeError(f"[{provider}] Parsed response has no choices: {raw}")

    return resp.choices[0].message.content or ""


def main(
    ckpt_dir: str,
    path: str,
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 256,
    max_batch_size: int = 1,
):
    print(path)
    generator = None

    if os.path.exists(f'./Inputs&Outputs/{path}/set.json'):
        flag_llm = 'llama'
        print('summarizing now')

        with open(f'./Inputs&Outputs/{path}/set.json', "r") as file:
            settings = json.load(file)

        summary_model = settings['infor']
        para_flag = False
        if summary_model.find('-para') != -1:
            para_flag = True
            summary_model = summary_model.replace('-para', '')

        api_client_sum = None
        api_provider_sum = None
        if ('gpt' in summary_model) or ('deepseek' in summary_model):
            flag_llm = 'api'
            if summary_model == 'gpt':
                summary_model = 'gpt-4o-mini' 

            api_provider_sum = resolve_provider(summary_model)
            api_client_sum = build_client_for_provider(api_provider_sum)
        else:
            generator = Llama.build(
                ckpt_dir='Model/' + summary_model,
                tokenizer_path='Model/' + tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )

        suf = settings['suffix']
        adh_1 = settings['adhesive_con']
        adh_2 = settings['adhesive_prompt']

        with open(f"./Inputs&Outputs/{path}/question.json", 'r', encoding='utf-8') as f_que:
            questions = json.loads(f_que.read())
        with open(f"./Inputs&Outputs/{path}/context.txt", 'r', encoding='utf-8') as f_con:
            contexts = json.loads(f_con.read())

        su_1 = (
            "Given the following question and context, extract any part of the context"
            " *AS IS* that is relevant to answer the question. If none of the context is relevant"
            " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Question: "
        )
        if para_flag:
            su_1 = (
                "Given the following question and context, extract any part of the context"
                " *AS IS* that is relevant to answer the question. If none of the context is relevant"
                " return NO_OUTPUT.\n\n> Question: "
            )

        su_2 = "\n> Context:\n>>>\n"
        su_3 = "\n>>>\nExtracted relevant parts:"

        prompt_ge_contexts = []
        summarize_contexts = []

        for i in range(len(questions)):
            ques = questions[i]
            k_contexts = contexts[i]
            ge_contexts = []
            sum_contexts = []

            for j in range(len(k_contexts)):
                context = k_contexts[j]
                prompt_ge_context = su_1 + ques + su_2 + context + su_3
                ge_contexts.append(prompt_ge_context)

                if flag_llm == 'api':
                    ans = call_chat(
                        api_client_sum,
                        summary_model,
                        prompt_ge_context,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_gen_len,
                        provider=api_provider_sum,
                    )
                else:
                    results = generator.text_completion(
                        [prompt_ge_context],
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    ans = results[0]['generation']

                sum_contexts.append(ans)

            summarize_contexts.append(sum_contexts)
            prompt_ge_contexts.append(ge_contexts)

        with open(f"./Inputs&Outputs/{path}/summarize_contexts.json", 'w', encoding='utf-8') as f_c:
            f_c.write(json.dumps(summarize_contexts, ensure_ascii=False))
        with open(f"./Inputs&Outputs/{path}/generate_summarize_prompt.json", 'w', encoding='utf-8') as f_g:
            f_g.write(json.dumps(prompt_ge_contexts, ensure_ascii=False))

        prompts = []
        for i in range(len(questions)):
            con_u = adh_1.join(summarize_contexts[i])
            prompt = suf[0] + con_u + adh_2 + suf[1] + questions[i] + adh_2 + suf[2]
            prompts.append(prompt)

        with open(f"./Inputs&Outputs/{path}/prompts.json", 'w', encoding='utf-8') as f_p:
            json.dump(prompts, f_p, ensure_ascii=False, indent=2)

    flag_llm = 'llama'
    api_client = None
    api_model = None
    api_provider = None

    if ('gpt' in ckpt_dir) or ('deepseek' in ckpt_dir):
        flag_llm = 'api'
        api_model = ckpt_dir
        if api_model == 'gpt':
            api_model = 'gpt-4o-mini'

        api_provider = resolve_provider(api_model)
        api_client = build_client_for_provider(api_provider)
    else:

        generator = Llama.build(
            ckpt_dir='Model/' + ckpt_dir,
            tokenizer_path='Model/' + tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    print('generating output')
    all_prompts = load_prompts(path)

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    answer = []
    if flag_llm == 'api':
        for batch in _chunks(all_prompts, max_batch_size):
            for p in batch:
                ans = call_chat(
                    api_client,
                    api_model,
                    p,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_gen_len,
                    provider=api_provider,
                )
                answer.append(ans)
    else:
        for batch in _chunks(all_prompts, max_batch_size):
            results = generator.text_completion(
                batch,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for res in results:
                answer.append(res['generation'])

    out_name = ckpt_dir.replace("/", "_")

    with open(
        f"./Inputs&Outputs/{path}/outputs-{out_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json",
        'w', encoding='utf-8'
    ) as file:
        file.write(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
