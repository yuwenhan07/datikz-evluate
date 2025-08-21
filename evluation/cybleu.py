import json
import re
from automatikz.evaluate.crystalbleu.crystalbleu import CrystalBLEU

# 加载tex_files.json（预测结果）
json_file_path = "../generate_test/output/tex_files.json"
# 加载test_metadata.json（标准答案）
metadata_file_path = "../save_eval/datikz_test_data/test_metadata.json"

# 从JSON文件中读取预测结果，并从键名提取ID
predictions = {}  # 使用字典存储 {ID: 预测结果}
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    
    # 定义正则表达式匹配键名中的ID（如sample_img_0.tex中的0）
    pattern = r"sample_img_(\d+)\.tex"
    
    for key, content in data.items():
        # 从键名中提取ID
        match = re.match(pattern, key)
        if match:
            file_id = match.group(1)  # 获取数字部分作为ID
            predictions[file_id] = content.strip()

# 从metadata文件中读取标准答案，使用index作为ID
references_dict = {}  # 使用字典存储 {index: 标准答案}
with open(metadata_file_path, "r", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
    
    # 假设metadata是一个列表，每个元素是一个包含index和code的字典
    for item in metadata:
        # 获取index作为ID（转换为字符串以便统一比较）
        item_id = str(item["index"])
        # 存储对应的code作为参考数据
        references_dict[item_id] = item["code"]

# 收集匹配的参考数据和预测结果
references = []
filtered_predictions = []

# 找出共同的ID
common_ids = set(predictions.keys()) & set(references_dict.keys())
# print(f"========== common ids:{common_ids}, len is {len(common_ids)} ==============")
common_ids = sorted(common_ids, key=int)  # 按数字排序

# 收集对应的数据
for item_id in common_ids:
    references.append([references_dict[item_id]])
    filtered_predictions.append(predictions[item_id])

# 打印加载的条目数量和统计信息
print("====" * 20)
print(f"有效匹配的条目数: {len(filtered_predictions)}")
print(f"JSON文件中的预测结果总数: {len(predictions)}")
print(f"Metadata中的标准答案总数: {len(references_dict)}")
print(f"未匹配的预测结果数: {len(predictions) - len(filtered_predictions)}")
print(f"未匹配的标准答案数: {len(references_dict) - len(references)}")

if len(references) == 0 or len(filtered_predictions) == 0:
    print("没有找到可匹配的预测结果和参考数据，无法计算分数")
else:
    # 创建CrystalBLEU实例
    metric = CrystalBLEU(corpus=references, k=500, n=4, use_cache=True)

    # 计算CrystalBLEU分数
    result = metric.compute(references=references, predictions=filtered_predictions)

    # 输出结果
    print("\nCrystalBLEU评估结果:")
    print(result)
