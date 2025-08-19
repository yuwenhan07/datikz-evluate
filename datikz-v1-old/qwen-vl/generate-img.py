from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import re
from glob import glob
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/data/model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/mnt/data/model/Qwen2.5-VL-7B-Instruct")

# 加载图像数据
image_paths = sorted(glob("/home/yuwenhan/Tikz/evaluate/groundtruth/groundtruth-pdf&png/*.png"))
ds = [{"image": Image.open(path).convert("RGB"), "caption": os.path.basename(path)} for path in image_paths]

os.makedirs("output/output-qwen", exist_ok=True)

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


    os.makedirs("output/output-img", exist_ok=True)
    with open(f"output/output-img/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)