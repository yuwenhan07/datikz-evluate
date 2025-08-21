"""
使用 qwen 运行
transformer verision is 4.52.4
"""
# * 用于生成 TikZ LaTeX 代码的脚本，测试对应的模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from datasets import load_dataset
import json
from tqdm import tqdm
import shutil  
import subprocess

model_name = "/mnt/data/model/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("nllg/datikz-v3", split="test")

for i in tqdm(range(len(ds))):
    example = ds[i]
    prompt = example["caption"]
    messages = [
        {"role": "system", "content": "You are a TikZ LaTeX diagram generation assistant capable of producing semantically accurate and structurally clear TikZ LaTeX code based on user prompts."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # * 提取代码块
    match = re.search(r"```latex\s*(.*?)\s*```", response, re.DOTALL)
    latex_code = match.group(1) if match else response
    result = {
        "prompt": prompt,
        "response": response,
        "latex_code": latex_code,
        "ground_truth": example["code"],
    }

    # * 保存结果到文件
    os.makedirs("output/output-tex", exist_ok=True)
    os.makedirs("output/original-output", exist_ok=True)

    with open(f"output/original-output/sample_{i}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"success write original json into sample_{i}.json")

    with open(f"output/output-tex/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)
        print(f"success write tex into sample_{i}.tex")