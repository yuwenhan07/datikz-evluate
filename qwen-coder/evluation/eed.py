import os
from automatikz.evaluate.eed import TER  # 使用TER类

# 设置文件夹路径
groundtruth_dir = "/home/yuwenhan/Tikz/evaluate/qwen-coder/output/groundtruth-tex"
output_dir = "/home/yuwenhan/Tikz/evaluate/qwen-coder/output/output-tex"

# 初始化TER类
ter_metric = TER()

# 定义一个函数，读取文本文件并返回内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 定义评测函数
def evaluate_tex_files(groundtruth_dir, output_dir):
    # 获取groundtruth和output文件夹中的所有文件
    groundtruth_files = sorted(os.listdir(groundtruth_dir))
    output_files = sorted(os.listdir(output_dir))

    # 确保文件数目相同
    if len(groundtruth_files) != len(output_files):
        raise ValueError("groundtruth和output文件夹中的文件数目不一致！")

    # 遍历所有文件并进行评测
    for gt_file, out_file in zip(groundtruth_files, output_files):
        gt_path = os.path.join(groundtruth_dir, gt_file)
        out_path = os.path.join(output_dir, out_file)

        # 读取文件内容
        groundtruth = read_file(gt_path)
        output = read_file(out_path)

        # 将groundtruth和output包装成包含单个元素的列表
        ter_score = ter_metric.compute(references=[groundtruth], predictions=[output])

        print(ter_score)
        # 检查返回值是否为None
        if ter_score is not None:
            print(f"评测文件: {gt_file} vs {out_file}")
            print(f"TER Score: {ter_score['EED']}")  # 输出最终的TER值
        else:
            print(f"评测文件: {gt_file} vs {out_file} - 未计算有效的TER分数")

# 调用评测函数
if __name__ == "__main__":
    try:
        evaluate_tex_files(groundtruth_dir, output_dir)
    except Exception as e:
        print(f"发生错误: {e}")
