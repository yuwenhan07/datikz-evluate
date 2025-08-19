import re
import os
from datasets import load_dataset
from glob import glob
from tqdm import tqdm
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7,8,9"

MODEL_PATH = "/home/yuwenhan/model/GLM-4.1V-9B-Thinking"

# 加载模型和处理器
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

image_paths = sorted(glob("/home/yuwenhan/Tikz/evaluate/groundtruth/groundtruth-pdf&png/*.png"))
ds = [{"image": Image.open(path).convert("RGB"), "caption": os.path.basename(path)} for path in image_paths]

# 创建输出目录
os.makedirs("output/output-tex-img", exist_ok=True)

 # 推理前 N 条样本
for i, example in enumerate(tqdm(ds)):
    image = example["image"]  # 已确认是 PIL.Image 类型
    prompt = example["caption"]

    # 构建多模态输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["image"]  # 关键：必须是 "image" 而不是 "img"
                },
                {
                    "type": "text",
                    "text": "please confer this image and caption, generate the latex code: " + prompt
                }
            ]
        }
    ]

    # print(messages)

    # 模型输入处理
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # print(inputs)

    # 模型生成
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

    # 保存 LaTeX 文件
    with open(f"output/output-tex-img/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)