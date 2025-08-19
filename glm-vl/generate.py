"""
使用 GLM_env 运行
"""
import re
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import json
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6,7,8,9"

MODEL_PATH = "/home/yuwenhan/model/GLM-4.1V-9B-Thinking"

model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

ds = load_dataset("nllg/datikz-v3", split="test")

for i in tqdm(range(len(ds)), desc="Processing samples"):
    example = ds[i]
    prompt = example["caption"]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4048)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False
    )

    patterns = [
    r"```(?:latex|tex)?\s*(.*?)\s*```",  # markdown代码块
    r"(\\documentclass{standalone}.*?\\end{document})",  # standalone文档
    r"(\\documentclass{article}.*?\\end{document})",  # article文档
    r"(\\begin{document}.*?\\end{document})",  # 有document体但无class
    r"(\\begin{tikzpicture}.*?\\end{tikzpicture})",  # 纯TikZ结构
    r"(\\documentclass.*?\\end{document})"  # 所有documentclass
    ]

    match = None
    for pattern in patterns:
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            break

    latex_code = match.group(1).strip() if match else output_text.strip()

    result = {
        "prompt": prompt,
        "response": output_text,
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