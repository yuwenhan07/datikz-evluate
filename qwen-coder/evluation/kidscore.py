import json
import os
from PIL import Image
from automatikz.evaluate.kid.kid import KernelInceptionDistance
from datasets import load_dataset
from torchvision import transforms

# 加载数据集
ds = load_dataset("nllg/datikz", split="test")

# 加载 tex_files.json（确保文件路径正确）
json_file_path = "../output/tex_files.json"
output_dir = "../tikz_output/output-pdf&png"  # 预测图像所在目录
groundtruth_dir = "../tikz_output/groundtruth-pdf&png"  # 参考图像所在目录

references = []
predictions = []

# 确认数据集的大小
num_samples = len(ds)  # 获取数据集的样本数量
print(f"Data set size: {num_samples}")

# 设置 subset_size
subset_size = 10  # 如果数据集小于 10，将其动态调整为数据集大小
subset_size = min(subset_size, num_samples)

# 创建 KidScore 实例
kid = KernelInceptionDistance(subset_size=subset_size)

# # 定义图像预处理
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),  # 例如调整图像大小
# ])

# 从 JSON 文件中读取预测结果
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    for key, value in data.items():
        pred_image_path = os.path.join(output_dir, key.replace(".tex", ".png"))
        ref_image_path = os.path.join(groundtruth_dir, key.replace(".tex", ".png"))

        # 确保图像存在
        if os.path.exists(pred_image_path):
            pred_image = Image.open(pred_image_path)  # 打开图像
            pred_image = pred_image.convert('RGB')  # 确保图像是 RGB 格式
            predictions.append(pred_image)  # 保持为 PIL.Image 格式
            
            ref_image = Image.open(ref_image_path)  # 打开参考图像
            ref_image = ref_image.convert('RGB')
            references.append(ref_image)  # 保持为 PIL.Image 格式

# 输出最终结果
if len(references) > 0 and len(predictions) > 0:
    result = kid.compute(references=references, predictions=predictions)
    print(result)
else:
    print("Error: No valid images loaded for predictions and references.")
