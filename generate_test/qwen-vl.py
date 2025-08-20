from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import re
import json
import traceback
from tqdm import tqdm
import torch
from typing import List, Optional, Tuple
from automatikz.infer import TikzDocument  # 导入TikzDocument类

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# 1. 加载模型和处理器（显式指定设备）
try:
    print("正在加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/mnt/data/model/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto"
    ).to("cuda")  # 强制使用GPU
    processor = AutoProcessor.from_pretrained("/mnt/data/model/Qwen2.5-VL-7B-Instruct")
    print(f"模型加载成功，设备：{model.device}")
except Exception as e:
    print(f"模型加载失败：{e}")
    traceback.print_exc()
    exit(1)

# 2. 读取数据集
metadata_path = "../save_eval/datikz_test_data/test_metadata.json"
try:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except Exception as e:
    print(f"读取metadata失败：{e}")
    exit(1)

base_dir = "../save_eval/"
ds = []
for item in metadata:
    img_abs_path = os.path.join(base_dir, item["image_path"])
    if not os.path.exists(img_abs_path):
        print(f"Warning: 图片不存在 {img_abs_path}，已跳过")
        continue
    try:
        image = Image.open(img_abs_path).convert("RGB")
        ds.append({
            "image": image,
            "caption": item["caption"],
            "code": item["code"]
        })
    except Exception as e:
        print(f"Error: 读取图片 {img_abs_path} 失败: {e}，已跳过")
        continue

if len(ds) == 0:
    print("无有效样本，程序退出")
    exit(1)


# 3. 核心：使用TikzDocument进行生成与修复
def parse_latex_errors(log: str, rootfile: str = "temp.tex") -> dict:
    """解析编译日志，返回{行号: 错误信息}"""
    errors = {}
    error_pattern = re.compile(
        rf"^{re.escape(rootfile)}:(\d+):\s*(.*?)(?=\n[^:]+:|$)",
        re.MULTILINE | re.DOTALL
    )
    for match in error_pattern.finditer(log):
        line = int(match.group(1))
        msg = match.group(2).strip()
        errors[line] = msg
    
    if not errors and re.search(r"Emergency stop|Fatal error", log, re.IGNORECASE):
        errors[0] = "Fatal error during compilation"
    return errors


def generate_and_repair(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_attempts: int = 5,
    return_all: bool = False
) -> Tuple[TikzDocument, List[TikzDocument]]:
    """使用TikzDocument进行生成与修复（带详细日志）"""
    all_attempts = []

    def _generate(snippet: str = "") -> str:
        """生成LaTeX代码片段（带异常处理）"""
        try:
            print(f"\n----- 开始生成代码（已有片段长度：{len(snippet)}）-----")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": (
                            f"Please generate LaTeX code based on the image and description (continue to improve the following code):\n"
                            f"Existing code:\n{snippet}\n"
                            f"Description to be supplemented: {prompt}"
                        )}
                    ]
                }
            ]

            # 处理对话模板
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 处理输入（文本+图像）
            inputs = processor(
                text=[text], images=[image], return_tensors="pt", padding=True
            ).to(model.device)

            # 模型生成
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,  # 临时减少生成长度，避免显存溢出
                do_sample=True,
                temperature=0.7
            )

            # 解码输出
            output_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            # 提取LaTeX代码
            patterns = [
                r"```(?:latex|tex)?\s*(.*?)\s*```",
                r"(\\documentclass{standalone}.*?\\end{document})",
                r"(\\documentclass{article}.*?\\end{document})",
                r"(\\begin{document}.*?\\end{document})",
                r"(\\begin{tikzpicture}.*?\\end{tikzpicture})",
                r"(\\documentclass.*?\\end{document})"
            ]
            match = None
            for pattern in patterns:
                match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
                if match:
                    print(f"匹配到LaTeX模式：{pattern[:30]}...")
                    break

            if not match:
                print("警告：未匹配到任何LaTeX代码模式，返回原始文本")
                return output_text.strip()
            return match.group(1).strip()
        except Exception as e:
            print(f"_generate函数出错：{e}")
            traceback.print_exc()
            return ""  # 出错时返回空字符串

    def _recursive_repair(
        prompt: str,
        attempts_left: int,
        snippet: str = "",
        offset: int = 1,
        prev_first_error: Optional[int] = None
    ) -> TikzDocument:
        """递归修复函数（带日志）"""
        print(f"\n----- 修复尝试：剩余次数 {attempts_left} -----")
        new_code = _generate(snippet)
        full_code = snippet + new_code if snippet else new_code
        print(f"生成的完整代码长度：{len(full_code)}")

        # 使用TikzDocument进行编译
        try:
            tikz_doc = TikzDocument(code=full_code)
            print(f"编译状态：{'成功' if tikz_doc.has_content else '失败'}")
            all_attempts.append(tikz_doc)
        except Exception as e:
            print(f"TikzDocument初始化/编译出错：{e}")
            # 出错时手动创建一个失败的文档
            class DummyDoc:
                has_content = False
                compiled_with_errors = True
                log = f"TikzDocument error: {e}"
                code = full_code
            all_attempts.append(DummyDoc())
            return DummyDoc()

        # 检查是否编译成功或达到最大尝试次数
        if tikz_doc.has_content:
            print("编译成功，返回当前文档")
            return tikz_doc
        if attempts_left <= 0:
            print("达到最大尝试次数，返回为空")
            return None

        # 解析错误
        errors = parse_latex_errors(tikz_doc.log)
        if not errors:
            print("未解析到错误，返回当前文档")
            return tikz_doc
        print(f"解析到错误：{errors}")

        first_error = min(errors.keys())
        # 调整偏移量
        if first_error != prev_first_error:
            offset = 1
            print(f"错误位置变化，重置偏移量为 {offset}")
        else:
            offset = min(4 * offset, 4096)
            print(f"错误位置不变，偏移量调整为 {offset}")
        
        # 计算需要保留的代码
        lines = full_code.splitlines(keepends=True)
        keep_lines = max(first_error - offset, 0)
        new_snippet = "".join(lines[:keep_lines])
        print(f"保留前 {keep_lines} 行代码（总长度：{len(new_snippet)}）")

        # 递归修复
        return _recursive_repair(
            prompt=prompt,
            attempts_left=attempts_left - 1,
            snippet=new_snippet,
            offset=offset,
            prev_first_error=first_error
        )

    final_doc = _recursive_repair(prompt, attempts_left=max_attempts)
    return final_doc, all_attempts

# 编译并保存PDF、PNG和日志的函数
def compile_and_save(tex_code, sample_id, output_pdf_dir, output_png_dir, output_log_dir):
    """编译LaTeX代码并保存PDF、PNG和日志"""
    
    # 文件名（使用样本ID）
    filename = f"sample_img_{sample_id}"
    
    try:
        # 创建TikzDocument实例
        tikzdoc = TikzDocument(code=tex_code)
        
        # 保存PDF
        if tikzdoc.pdf:
            pdf_path = os.path.join(output_pdf_dir, f"{filename}.pdf")
            tikzdoc.save(pdf_path)
            print(f"✅ 已保存 PDF 到 {pdf_path}")
        
        # 保存PNG
        if tikzdoc.has_content:
            png_path = os.path.join(output_png_dir, f"{filename}.png")
            img = tikzdoc.rasterize()
            img.save(png_path)
            print(f"✅ 已保存 PNG 到 {png_path}")
        
        # 保存错误日志
        if tikzdoc.compiled_with_errors:
            log_path = os.path.join(output_log_dir, f"{filename}.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(tikzdoc.log)
            print(f"⚠️ 编译 {filename} 时可能出错！日志已保存至 {log_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 编译或保存 {filename} 失败: {e}")
        traceback.print_exc()
        return False
    

skip_list = []
# 4. 主循环：生成、修复并保存结果
try:
    # 只处理前5个样本便于测试
    for i, example in enumerate(tqdm(ds, desc="Processing samples")):
        print(f"\n====== 处理样本 {i} ======")
        image = example["image"]
        prompt = example["caption"]

        # 生成并修复
        final_doc, all_attempts = generate_and_repair(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            max_attempts=3,  # 减少尝试次数便于快速测试
            return_all=True
        )

        # 构建结果（添加final_doc非空判断）
        result = {
            "prompt": prompt,
            "final_latex_code": final_doc.code if final_doc is not None else "",
            "compiled_successfully": final_doc.has_content if final_doc is not None else False,
            "ground_truth": example["code"],
            "attempts": len(all_attempts)
        }

        # 保存结果
        output_tex_dir = "output/output-tex-inputwithimg"
        output_json_dir = "output/original-output-inputwithimg"
        save_png_dir = "save/png"
        save_pdf_dir = "save/pdf"
        save_log_dir = "save/log"
        os.makedirs(output_tex_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        os.makedirs(save_pdf_dir, exist_ok=True)
        os.makedirs(save_png_dir, exist_ok=True)
        os.makedirs(save_log_dir, exist_ok=True)

        # 保存JSON结果（无论final_doc是否为None都保存）
        with open(f"{output_json_dir}/sample_img_{i}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存TeX文件（仅当final_doc不为None时）
        if final_doc is not None:
            with open(f"{output_tex_dir}/sample_img_{i}.tex", "w", encoding="utf-8") as tex_file:
                tex_file.write(final_doc.code)
            print(f"样本 {i} 处理完成，尝试次数：{len(all_attempts)}，编译成功：{final_doc.has_content if final_doc is not None else False}")
            compile_and_save(final_doc.code, i, save_pdf_dir, save_png_dir, save_log_dir)

        else:
            skip_list.append(i)
            print(f"样本 {i} 处理失败，尝试次数：{len(all_attempts)}，返回结果为None")
except Exception as e:
    print(f"主循环出错：{e}")
    traceback.print_exc()

print(f"============ 所有跳过的条目 {skip_list} ==================")