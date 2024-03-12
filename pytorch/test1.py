# python
 # 将给定的公式重新组织，使用Python的Matplotlib库以图形的方式展示出来

import matplotlib.pyplot as plt
import numpy as np

# 使用 LaTeX 风格的文本渲染器
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

# 创建一个大的画布来显示所有的公式
fig, axs = plt.subplots(5, 1, figsize=(10, 8))

# 关闭所有坐标轴，因为我们只展示公式
for ax in axs:
    ax.axis('off')

# 设置公式
formulas = [
    r"$\frac{\partial \hat{y}}{\partial z} = \sigma(z) \cdot (1 - \sigma(z)) = \hat{y} \cdot (1 - \hat{y})$",
    r"$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$",
    r"$\frac{\partial L}{\partial z} = \hat{y} - y$",
    r"$\hat{y} = \sigma(z)$",
    r"$ L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] $"
]

# 在每个子图中显示一个公式
for ax, formula in zip(axs, formulas):
    ax.text(0.5, 0.5, formula, fontsize=16, va='center', ha='center')

# 调整布局并显示图形
plt.tight_layout()
plt.show()