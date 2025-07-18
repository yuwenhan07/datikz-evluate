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

ds = load_dataset("nllg/datikz", split="test")
save = []

for i in tqdm(range(312, len(ds))):
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

    result = {
        "prompt": prompt,
        "response": response,
        "ground_truth": example["code"],
    }

    os.makedirs("output/output-tex", exist_ok=True)
    os.makedirs("output/groundtruth-tex", exist_ok=True)


    # 提取代码块
    match = re.search(r"```latex\s*(.*?)\s*```", response, re.DOTALL)
    latex_code = match.group(1) if match else response

    tex_path = f"output/output-tex/sample_{i}.tex"
    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)
    print(f"success write tex into sample_{i}.tex")

    # 保存 Ground Truth
    with open(f"output/groundtruth-tex/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(example["code"])