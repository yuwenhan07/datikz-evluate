import os
from tqdm import tqdm
from automatikz.evaluate.eed import TER  # 使用TER类

# 设置文件夹路径
groundtruth_dir = "../../groundtruth/groundtruth-tex"
output_dir = "../output/output-tex"

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

    sum = 0
    count = 0
    # 遍历所有文件并进行评测
    for gt_file, out_file in tqdm(zip(groundtruth_files, output_files), total=len(groundtruth_files), desc="Evaluating"):
        gt_path = os.path.join(groundtruth_dir, gt_file)
        out_path = os.path.join(output_dir, out_file)

        # 读取文件内容
        groundtruth = read_file(gt_path)
        output = read_file(out_path)

        # 将groundtruth和output包装成包含单个元素的列表
        ter_score = ter_metric.compute(references=[[groundtruth]], predictions=[output])

        # 累加分数
        if ter_score is not None:
            sum += ter_score['EED']
            count += 1
        
    # 计算平均分数
    if count > 0:
        average_score = sum / count
        print(f"平均EED分数: {average_score}")
    else:
        print("没有有效的评测结果。请检查文件内容。")

# 调用评测函数
if __name__ == "__main__":
    try:
        evaluate_tex_files(groundtruth_dir, output_dir)
    except Exception as e:
        print(f"发生错误: {e}")