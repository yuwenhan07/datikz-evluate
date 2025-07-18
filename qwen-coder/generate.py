from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from datasets import load_dataset
import json
from tqdm import tqdm
import shutil  
import subprocess

model_name = "/mnt/data/model/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("nllg/datikz", split="test")
save = []

for i in tqdm(range(2)):
    example = ds[i]
    prompt = example["caption"]
    messages = [
        {"role": "system", "content": "You are a TikZ LaTeX diagram generation assistant capable of producing semantically accurate and structurally clear TikZ LaTeX code based on user prompts."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    result = {
        "prompt": prompt,
        "response": response,
        "ground_truth": example["code"],
    }
    save.append(result)

    with open("qwen_coder_results.json", "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False)
        f.write("\n")

    os.makedirs("output-tex", exist_ok=True)
    os.makedirs("output-png", exist_ok=True)
    os.makedirs("output-pdf", exist_ok=True)
    os.makedirs("groundtruth-tex", exist_ok=True)
    os.makedirs("groundtruth-png", exist_ok=True)
    os.makedirs("groundtruth-pdf", exist_ok=True)

    # 提取代码块
    match = re.search(r"```latex\s*(.*?)\s*```", response, re.DOTALL)
    latex_code = match.group(1) if match else response

    tex_path = f"output-tex/sample_{i}.tex"
    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_code)
    print(f"success write tex into sample_{i}.tex")

    # 保存 Ground Truth
    with open(f"groundtruth-tex/sample_{i}.tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(example["code"])

    # 编译 tex 为 pdf 并进行处理
    def compile_latex(tex_path, output_dir, pdf_output_path):
        try:
            result = subprocess.run(
                ["pdftex", tex_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # 输出日志供调试
            print(result.stderr)

            # 检查 LaTeX 编译中是否包含 "Undefined control sequence" 错误
            if "Undefined control sequence" in result.stderr:
                print("Error: Undefined control sequence detected in LaTeX file.")
                return False
            return True

        except subprocess.CalledProcessError as e:
            print(f"Latex compilation failed for {tex_path} with error: {e.stderr}")
            return False

    def convert_pdf_to_png(pdf_path, png_path):
        try:
            subprocess.run(
                ["convert", "-density", "300", pdf_path, "-quality", "90", png_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert PDF to PNG: {e.stderr}")
            return False

    # === 编译 Ground Truth 的 tex 为 PDF 和 PNG ===
    ground_tex_path = f"groundtruth-tex/sample_{i}.tex"
    ground_pdf_path = ground_tex_path.replace(".tex", ".pdf")
    ground_png_path = f"groundtruth-png/groundtruth_{i}.png"
    final_ground_pdf_path = f"groundtruth-pdf/groundtruth_{i}.pdf"

    if not compile_latex(ground_tex_path, "groundtruth-tex", ground_pdf_path):
        print(f"Compilation failed for ground truth sample_{i}, skipping PNG conversion.")
        continue  # 跳过本次迭代

    shutil.copy(ground_pdf_path, final_ground_pdf_path)
    
    # 只在 Ground Truth PDF 编译成功时才进行 PNG 转换
    convert_pdf_to_png(ground_pdf_path, ground_png_path)

    print(f"Generated Ground Truth PDF: {final_ground_pdf_path}")
    print(f"Generated Ground Truth PNG: {ground_png_path}")

    # 编译并转换输出
    pdf_path = tex_path.replace(".tex", ".pdf")
    png_path = f"output-png/sample_{i}.png"
    final_pdf_path = f"output-pdf/sample_{i}.pdf"

    if not compile_latex(tex_path, "output-tex", pdf_path):
        print(f"Compilation failed for sample_{i}, skipping PNG conversion.")
        continue  # 跳过本次迭代

    shutil.copy(pdf_path, final_pdf_path)
    
    # 只在 PDF 编译成功时才进行 PNG 转换
    convert_pdf_to_png(pdf_path, png_path)
    
    print(f"Generated PDF: {final_pdf_path}")
    print(f"Generated PNG: {png_path}")
   