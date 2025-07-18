from automatikz.evaluate.clipscore.clipscore import CLIPScore
from PIL import Image
from datasets import load_dataset
import os

# 清理 CUDA 内存
import torch
torch.cuda.empty_cache()

# 加载数据集
ds = load_dataset("nllg/datikz", split="test")

# 创建 CLIPScore 实例
clip_score_metric = CLIPScore()

# 读取数据集中的所有参考文本
text_references = [ds[i]["caption"] for i in range(312)]

# 定义图像路径
groundtruth_image_paths = ["/home/yuwenhan/Tikz/evaluate/qwen-coder/tikz_output/groundtruth-pdf&png/sample_{}.png".format(i) for i in range(312)]
generated_image_paths = ["/home/yuwenhan/Tikz/evaluate/qwen-coder/tikz_output/output-pdf&png/sample_{}.png".format(i) for i in range(312)]

# 读取图像并确保文件存在
def load_images(image_paths, references):
    images = []
    valid_references = []  # 保存有效的文本
    for img_path, caption in zip(image_paths, references):
        if os.path.exists(img_path):  # 确保文件存在
            images.append(Image.open(img_path))
            valid_references.append(caption)  # 保留有效的文本
        # else:
            # images.append(None)  # 图像缺失，保存为 None
            # valid_references.append(None)  # 对应的文本也保存为 None
            # print(f"Warning: {img_path} does not exist!")
    return images, valid_references

# 加载生成和真实图像，同时确保参考文本一致
groundtruth_images, valid_references_groundtruth = load_images(groundtruth_image_paths, text_references)
generated_images, valid_references_generated = load_images(generated_image_paths, text_references)
print("Loaded images and references:")
print(len(groundtruth_images), len(generated_images), len(valid_references_groundtruth), len(valid_references_generated))

# 创建 CLIPScore 实例
clip_score_metric = CLIPScore()

# 计算 CLIPScore - GroundTruth vs Text
clip_score_result = clip_score_metric.compute(references=valid_references_groundtruth, predictions=groundtruth_images)
print("GroundTruth CLIPScore Result:", clip_score_result)

# 计算 CLIPScore - Generated vs Text
clip_score_result = clip_score_metric.compute(references=valid_references_generated, predictions=generated_images)
print("Generate CLIPScore Result:", clip_score_result)
