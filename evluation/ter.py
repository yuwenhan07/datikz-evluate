import os
from automatikz.evaluate.ter import TER
import re
from tqdm import tqdm

# 文件夹路径
groundtruth_dir = "../save_eval/datikz_test_data/codes"
output_dir = "../generate_test/output/output-tex-inputwithimg"

# 创建 TER 实例
ter_metric = TER()

# 获取两个文件夹中所有的 .tex 文件名
groundtruth_files = [f for f in os.listdir(groundtruth_dir) if f.endswith(".tex")]
output_files = [f for f in os.listdir(output_dir) if f.endswith(".tex")]

ground = []
output = []

# 遍历每一对同名的 .tex 文件
for filename in tqdm(groundtruth_files, desc="Processing files"):
    if filename in output_files:
        # 构建两个文件的路径
        groundtruth_path = os.path.join(groundtruth_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 读取 .tex 文件内容
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            groundtruth_content = f.read()
            ground.append([groundtruth_content])

        with open(output_path, "r", encoding="utf-8") as f:
            output_content = f.read()
            output.append(output_content)

# 计算 TER 分数
ter_score = ter_metric.compute(references=ground, predictions=output)
print(f"========== ter score is {ter_score} ================")