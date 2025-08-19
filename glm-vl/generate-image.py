import re
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7,8,9"

MODEL_PATH = "/home/yuwenhan/model/GLM-4.1V-9B-Thinking"

# 加载模型和处理器
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

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

# 推理过程
for i, example in enumerate(tqdm(ds)):
    image = example["image"]
    prompt = example["caption"]  # 使用metadata中的caption作为输入prompt

    # 构建多模态输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image  # 图片输入
                },
                {
                    "type": "text",
                    "text": f"please confer this image and caption, generate the latex code: {prompt}"
                }
            ]
        }
    ]

    # 模型输入处理
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 模型生成
    generated_ids = model.generate(** inputs, max_new_tokens=4048)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False
    )

    # 提取LaTeX代码
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