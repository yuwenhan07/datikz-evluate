import os
import traceback
from automatikz.infer import TikzDocument 

# ? groundtruth-tex 目录包含 TikZ 源码文件
# input_dir = "./output/groundtruth-tex"  # 输入目录，包含 .tex 文件
# output_dir = "./tikz_output/groundtruth-pdf&png"  # 输出目录，保存 PDF 和 PNG 文件
# os.makedirs(output_dir, exist_ok=True)

# ? 如果需要处理 output/output-tex 中的文件，可以取消注释以下两行
input_dir = "./output/output-img"  # 输入目录，包含 .tex 文件
output_dir = "./output/output-img-pdf&png"  # 输出目录，保存 PDF 和 PNG 文件
os.makedirs(output_dir, exist_ok=True)

start_processing = False  # Variable to track when to start processing

for filename in os.listdir(input_dir):
    if filename.endswith(".tex"):
        # Start processing from the specified file
        if filename == "sample_11.tex":
            start_processing = True

        # Skip files before the specified file
        if not start_processing:
            continue

        tex_path = os.path.join(input_dir, filename)

        # 读取 tex 内容
        try:
            with open(tex_path, "r", encoding="utf-8") as f:
                tex_code = f.read()
        except Exception as e:
            print(f"❌ 无法读取文件 {filename}: {e}")
            continue

        # 创建 TikzDocument 实例
        tikzdoc = TikzDocument(code=tex_code)

        # 输出文件前缀（无扩展名）
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\n🛠️ 正在处理：{filename}...")

        try:
            # 保存 PDF
            if tikzdoc.pdf:
                pdf_path = os.path.join(output_dir, name_without_ext + ".pdf")
                tikzdoc.save(pdf_path)
                print(f"✅ 已保存 PDF 到 {pdf_path}")

            # 保存 PNG
            if tikzdoc.has_content:
                png_path = os.path.join(output_dir, name_without_ext + ".png")
                img = tikzdoc.rasterize()
                img.save(png_path)
                print(f"✅ 已保存 PNG 到 {png_path}")

            # 编译失败时输出日志
            if tikzdoc.compiled_with_errors:
                log_path = os.path.join(output_dir, name_without_ext + ".log")
                with open(log_path, "w", encoding="utf-8") as log_file:
                    log_file.write(tikzdoc.log)
                print(f"⚠️ 编译 {filename} 时出错！日志已保存至 {log_path}")

        except Exception as e:
            print(f"❌ 编译或保存 {filename} 失败: {e}")
            traceback.print_exc()
