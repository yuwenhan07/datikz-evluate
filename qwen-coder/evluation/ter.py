import os
from automatikz.evaluate.ter import TER
import re

# 文件夹路径
groundtruth_dir = "/home/yuwenhan/Tikz/evaluate/qwen-coder/output/groundtruth-tex"
output_dir = "/home/yuwenhan/Tikz/evaluate/qwen-coder/output/output-tex"

# 创建 TER 实例
ter_metric = TER()

# 获取两个文件夹中所有的 .tex 文件名
groundtruth_files = [f for f in os.listdir(groundtruth_dir) if f.endswith(".tex")]
output_files = [f for f in os.listdir(output_dir) if f.endswith(".tex")]

# 去掉 LaTeX 命令的函数
def clean_latex(content):
    # 使用正则表达式移除 LaTeX 命令
    cleaned_content = re.sub(r'\\[a-zA-Z]+\*?(\[.*?\])?(\{.*?\})?', '', content)
    return cleaned_content.strip()

# 将每一行拆分成单词
def split_into_words(content):
    return content.split()

# 填充空行使行数一致
def pad_with_empty_lines(reference_lines, prediction_lines):
    # 比较两个列表的长度，确保两者一致
    while len(reference_lines) < len(prediction_lines):
        reference_lines.append("**")  # 填充空行
    while len(prediction_lines) < len(reference_lines):
        prediction_lines.append("**")  # 填充空行
    return reference_lines, prediction_lines

# 遍历每一对同名的 .tex 文件
for filename in groundtruth_files:
    if filename in output_files:
        # 构建两个文件的路径
        groundtruth_path = os.path.join(groundtruth_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 读取 .tex 文件内容
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            groundtruth_content = f.read()

        with open(output_path, "r", encoding="utf-8") as f:
            output_content = f.read()

        # 按行分割内容，生成字符串列表，并去掉无效的空行和 LaTeX 格式字符串
        groundtruth_lines = [clean_latex(line) for line in groundtruth_content.splitlines() if line.strip()]
        output_lines = [clean_latex(line) for line in output_content.splitlines() if line.strip()]

        # 如果行数不一致，则填充空行
        groundtruth_lines, output_lines = pad_with_empty_lines(groundtruth_lines, output_lines)

        # 将每行拆分成单词，确保输入格式符合预期
        groundtruth_words = [split_into_words(line) for line in groundtruth_lines if line.strip()]
        output_words = [split_into_words(line) for line in output_lines if line.strip()]

        # 计算 TER
        ter_result = ter_metric.compute(references=groundtruth_words, predictions=output_words)

        # 输出评估结果
        print(f"TER for {filename}: {ter_result['TER']}")
