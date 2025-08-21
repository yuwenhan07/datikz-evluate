import os
import re
from tqdm import tqdm
from automatikz.evaluate.eed import TER  # 使用TER类

# 设置文件夹路径
groundtruth_dir = "../save_eval/datikz_test_data/codes"
output_dir = "../generate_test/output/output-tex-inputwithimg"

# 初始化TER类
ter_metric = TER()

# 定义一个函数，读取文本文件并返回内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 从文件名中提取数字部分
def extract_number(filename, pattern):
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None

# 定义评测函数
def evaluate_tex_files(groundtruth_dir, output_dir):
    # 定义文件名模式
    gt_pattern = r"test_(\d+)\.tex"  # 匹配test_xx.tex格式
    output_pattern = r"sample_img_(\d+)\.tex"  # 匹配sample_img_xxx.tex格式
    
    # 获取groundtruth和output文件夹中的所有文件
    groundtruth_files = os.listdir(groundtruth_dir)
    output_files = os.listdir(output_dir)
    
    # 建立ground truth文件的数字到文件名的映射
    gt_number_map = {}
    for file in groundtruth_files:
        num = extract_number(file, gt_pattern)
        if num is not None:
            gt_number_map[num] = file
    
    # 建立output文件的数字到文件名的映射
    output_number_map = {}
    for file in output_files:
        num = extract_number(file, output_pattern)
        if num is not None:
            output_number_map[num] = file
    
    # 找到数字部分相同的文件对
    common_numbers = sorted(set(gt_number_map.keys()) & set(output_number_map.keys()))
    
    # 显示统计信息
    total_gt_files = len(gt_number_map)
    total_out_files = len(output_number_map)
    common_count = len(common_numbers)
    
    print(f"符合命名规则的Ground truth文件总数: {total_gt_files}")
    print(f"符合命名规则的Output文件总数: {total_out_files}")
    print(f"可匹配的文件对数: {common_count}")
    print(f"将评估 {common_count} 个文件对...\n")
    
    if common_count == 0:
        print("没有找到可匹配的文件对，无法进行评估。")
        return

    sum_score = 0
    valid_count = 0
    
    # 遍历所有匹配的文件对并进行评测
    for num in tqdm(common_numbers, total=common_count, desc="Evaluating"):
        gt_file = gt_number_map[num]
        out_file = output_number_map[num]
        
        gt_path = os.path.join(groundtruth_dir, gt_file)
        out_path = os.path.join(output_dir, out_file)

        # 读取文件内容
        groundtruth = read_file(gt_path)
        output = read_file(out_path)

        # 将groundtruth和output包装成包含单个元素的列表
        ter_score = ter_metric.compute(references=[[groundtruth]], predictions=[output])

        # 累加分数
        if ter_score is not None and 'EED' in ter_score:
            sum_score += ter_score['EED']
            valid_count += 1
        
    # 计算平均分数
    if valid_count > 0:
        average_score = sum_score / valid_count
        print(f"\n有效评估的文件数: {valid_count}")
        print(f"平均EED分数: {average_score}")
    else:
        print("\n没有有效的评测结果。请检查文件内容。")

# 调用评测函数
if __name__ == "__main__":
    try:
        evaluate_tex_files(groundtruth_dir, output_dir)
    except Exception as e:
        print(f"发生错误: {e}")
    