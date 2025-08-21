import json
import os
import re
from PIL import Image
from automatikz.evaluate.kid.kid import KernelInceptionDistance
from torchvision import transforms

def main():
    # 文件路径配置
    json_file_path = "../generate_test/output/tex_files.json"
    metadata_file_path = "../save_eval/datikz_test_data/test_metadata.json"
    output_dir = "../generate_test/save/png"  # 预测图像目录
    groundtruth_dir = "../save_eval/datikz_test_data/images"  # 参考图像目录

    # 1. 从预测文件中提取ID和文件名映射
    prediction_files = {}  # {ID: 文件名}
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        pattern = r"sample_img_(\d+)\.tex"  # 匹配预测文件名模式
        
        for filename in data.keys():
            match = re.match(pattern, filename)
            if match:
                file_id = match.group(1)
                prediction_files[file_id] = filename

    print(f"从JSON文件中加载了 {len(prediction_files)} 个预测结果")

    # 2. 从metadata中获取参考数据的index
    reference_ids = set()
    with open(metadata_file_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
        
        for item in metadata:
            item_id = str(item["index"])
            reference_ids.add(item_id)

    print(f"从metadata中加载了 {len(reference_ids)} 个参考数据索引")

    # 3. 找到共同存在的ID
    common_ids = sorted(
        set(prediction_files.keys()) & reference_ids, 
        key=int  # 按数字排序
    )
    print(f"找到 {len(common_ids)} 个双方都存在的ID")

    if not common_ids:
        print("没有找到匹配的ID，无法进行评估")
        return

    # 5. 加载并预处理图像
    references = []
    predictions = []
    missing_files = []

    for item_id in common_ids:
        # 构建预测图像路径
        pred_tex_filename = prediction_files[item_id]
        pred_img_filename = pred_tex_filename.replace(".tex", ".png")
        pred_image_path = os.path.join(output_dir, pred_img_filename)
        
        # 构建参考图像路径（假设命名为test_xxx.png）
        ref_img_filename = f"test_{item_id}.png"
        ref_image_path = os.path.join(groundtruth_dir, ref_img_filename)
        
        # 检查文件是否存在
        if not os.path.exists(pred_image_path):
            missing_files.append(f"预测图像: {pred_image_path}")
            continue
            
        if not os.path.exists(ref_image_path):
            missing_files.append(f"参考图像: {ref_image_path}")
            continue
        
        # 加载和预处理图像
        try:
            # 处理预测图像
            with Image.open(pred_image_path) as img:
                img = img.convert('RGB')
                predictions.append(img)
            
            # 处理参考图像
            with Image.open(ref_image_path) as img:
                img = img.convert('RGB')
                references.append(img)
        except Exception as e:
            print(f"处理ID为 {item_id} 的图像时出错: {str(e)}")

    # 6. 输出文件缺失情况
    if missing_files:
        print(f"\n发现 {len(missing_files)} 个缺失文件:")
        for file in missing_files[:5]:  # 只显示前5个
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  - 以及 {len(missing_files)-5} 个更多文件")

    # 7. 检查有效图像数量
    print(f"\n成功加载 {len(references)} 对有效图像")
    
    if len(references) < 2 or len(predictions) < 2:
        print("有效图像数量不足，无法计算KID分数（至少需要2对图像）")
        return

    # 8. 计算KID分数
    subset_size = min(10, len(references))  # 动态调整子集大小
    print(f"使用 {subset_size} 个子集计算KID分数")
    
    # 创建 KidScore 实例
    kid = KernelInceptionDistance(subset_size=subset_size)
    result = kid.compute(references=references, predictions=predictions)
    
    # 9. 输出结果
    print("\n===== KID评估结果 =====")
    print(f"KID均值: {result}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    