import json
from automatikz.evaluate.crystalbleu.crystalbleu2 import CrystalBLEU

# 假设有一些参考代码和机器生成的代码
references = [["\\documentclass{article}\\begin{document}Hello World\\end{document}"],["\\documentclass{article}\\begin{document}Hello World\\end{document}"]]
predictions = ["\\documentclass{article}\\begin{document}Hello World\\end{document}"]

# 创建 CrystalBLEU 实例，使用正确的k和n值
metric = CrystalBLEU(corpus=references, k=1, n=1, use_cache=False)

# 计算 CrystalBLEU 分数
result = metric.compute(references=references, predictions=predictions)

# 输出结果
print(result)
