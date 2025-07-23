import json
from automatikz.evaluate.crystalbleu.crystalbleu import CrystalBLEU
from datasets import load_dataset

# 加载数据集
ds = load_dataset("nllg/datikz", split="test")

# 加载 tex_files.json（确保文件路径正确）
json_file_path = "../output/tex_files.json"

references = []
predictions = []

# 从 JSON 文件中读取预测结果
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    for key, value in data.items():
        predictions.append(value.strip())  # 将预测结果作为字符串添加

    for i in range(len(data)):
        references.append([ds[i]["code"]]) 

    # 打印加载的条目数量
    print("====" * 20)
    print(f"Loaded {len(predictions)} predictions and {len(references)} references.")

# 创建 CrystalBLEU 实例，使用正确的 k 和 n 值
metric = CrystalBLEU(corpus=references, k=500, n=4, use_cache=True)

# 计算 CrystalBLEU 分数
result = metric.compute(references=references, predictions=predictions)

# 输出结果
print(result)
