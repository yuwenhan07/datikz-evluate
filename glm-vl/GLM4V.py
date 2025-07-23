'''
Author: Wenhan Yu
LastEditTime: 2025-07-03 21:50:32
Date: 2025-07-03 21:50:30
Version: 1.0
Description: 
'''
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7,8,9"

MODEL_PATH = "/home/yuwenhan/model/GLM-4.1V-9B-Thinking"


from PIL import Image
image_paths = "/home/yuwenhan/Tikz/evaluate/groundtruth/groundtruth-pdf&png/sample_0.png"
image = Image.open(image_paths).convert("RGB")


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "url": "https://model-demo.oss-cn-hangzhou.aliyuncs.com/Grayscale_8bits_palette_sample_image.png"
                "image": image
            },
            {
                "type": "text",
                "text": "描述一下这张图片"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)