from datasets import load_dataset
import json
import os
from PIL import Image
from tqdm import tqdm  # 导入tqdm库

# 1. 加载test数据集
test_ds = load_dataset("nllg/datikz-v3", split="test")

# 2. 创建保存目录（分离图像、元数据和代码）
save_root = "datikz_test_data"  # 主目录
image_dir = os.path.join(save_root, "images")  # 存放图像
code_dir = os.path.join(save_root, "codes")  # 存放代码(tex文件)
os.makedirs(image_dir, exist_ok=True)  # 确保图像目录存在
os.makedirs(code_dir, exist_ok=True)   # 确保代码目录存在

# 3. 遍历test数据集，保存图像、代码和元数据（添加tqdm进度条）
metadata_list = []  # 存储每条数据的文本信息（与图像对应）
# 使用tqdm包装迭代对象，并设置进度条描述
for idx, item in tqdm(enumerate(test_ds), total=len(test_ds), desc="保存数据集"):
    # 3.1 保存图像（用索引命名，避免重名）
    image = item["image"]  # PIL图像对象
    image_path = os.path.join(image_dir, f"test_{idx}.png")
    image.save(image_path)  # 保存为PNG格式
    
    # 3.2 保存code为tex文件
    code = item["code"]
    code_path = os.path.join(code_dir, f"test_{idx}.tex")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    # 3.3 收集元数据（文本信息），记录图像和代码路径以便关联
    metadata = {
        "index": idx,
        "caption": item["caption"],
        "code": item["code"],
        "image_path": image_path,  # 关联本地图像路径
        "code_path": code_path,    # 关联本地代码路径
        # "pdf": item["pdf"],
        "uri": item["uri"],
        "origin": item["origin"],
        # "date": item["date"]
    }
    metadata_list.append(metadata)

# 4. 保存元数据为JSON文件（方便查看文本与图像、代码的对应关系）
metadata_path = os.path.join(save_root, "test_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, ensure_ascii=False, indent=2)

print(f"test数据集已保存至：{save_root}")
print(f"图像保存在：{image_dir}")
print(f"代码(tex文件)保存在：{code_dir}")
print(f"元数据（含文本、图像路径和代码路径）保存在：{metadata_path}")