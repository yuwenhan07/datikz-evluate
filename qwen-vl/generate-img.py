from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import re
import json
from glob import glob
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/data/model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/mnt/data/model/Qwen2.5-VL-7B-Instruct")

# 读取test_metadata.json（包含caption、image_path、code等信息）
metadata_path = "../save_eval/datikz_test_data/test_metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)  # metadata是列表，每个元素是一个样本字典

# 构建数据集ds：从metadata中提取image_path、caption、code，关联图片
# 注意：metadata中的image_path是相对路径，需要拼接绝对路径前缀
base_dir = "../save_eval/"  # 基础路径，用于拼接image_path
ds = []
for item in metadata:
    # 拼接图片绝对路径（metadata中的image_path是"datikz_test_data/images/test_0.png"）
    img_abs_path = os.path.join(base_dir, item["image_path"])
    
    # 检查图片文件是否存在
    if not os.path.exists(img_abs_path):
        print(f"Warning: 图片文件不存在 {img_abs_path}，已跳过该样本")
        continue
    
    # 读取图片并添加到数据集
    try:
        image = Image.open(img_abs_path).convert("RGB")
        ds.append({
            "image": image,
            "caption": item["caption"],  # 从metadata取caption作为prompt
            "code": item["code"]  # 从metadata取code作为ground truth
        })
    except Exception as e:
        print(f"Error: 读取图片 {img_abs_path} 失败，错误信息：{e}，已跳过")
        continue

print(f"====== 有效样本数：{len(ds)} ===========")

for i, example in enumerate(tqdm(ds)):
    image = example["image"]
    prompt = example["caption"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "please confer this image and caption, generate the latex code: " + prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 注意图像必须传入
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    # 提取 LaTeX 代码
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

    # 构建结果（包含ground truth）
    result = {
        "prompt": prompt,
        "response": output_text,
        "latex_code": latex_code,
        "ground_truth": example["code"]  # 来自metadata的真实代码
    }

    # 确保输出目录存在
    os.makedirs("output/output-tex-inputwithimg", exist_ok=True)
    os.makedirs("output/original-output-inputwithimg", exist_ok=True)

    # 保存结果
    with open(f"output/original-output-inputwithimg/sample_img_{i}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"成功写入JSON: sample_img_{i}.json")
        
    with open(f"output/output-tex-inputwithimg/sample_img_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)
        print(f"成功写入TeX: sample_img_{i}.tex")