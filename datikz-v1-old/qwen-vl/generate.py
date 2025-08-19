'''
Author: Wenhan Yu
LastEditTime: 2025-07-03 22:16:04
Date: 2025-06-30 21:38:51
Version: 1.0
Description: 
'''
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "9"  # 使用单卡避免CUDA资源耗尽

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "/mnt/data/model/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
    "/mnt/data/model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# processor = AutoProcessor.from_pretrained("/mnt/data/model/Qwen2.5-VL-32B-Instruct")
processor = AutoProcessor.from_pretrained("/mnt/data/model/Qwen2.5-VL-7B-Instruct")

ds = load_dataset("nllg/datikz", split="test")

for i in tqdm(range(len(ds))):
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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    patterns = [
        r"```(?:latex|tex)?\s*(.*?)\s*```",
        r"(\\documentclass{standalone}.*?\\end{document})",
        r"(\\documentclass{article}.*?\\end{document})",
        r"(\\begin{document}.*?\\end{document})",
        r"(\\begin{tikzpicture}.*?\\end{tikzpicture})",
        r"(\\documentclass.*?\\end{document})"
    ]

    match = None
    for pattern in patterns:
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            break

    latex_code = match.group(1).strip() if match else output_text.strip()

    os.makedirs("output/output-qwen", exist_ok=True)
    with open(f"output/output-qwen/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)