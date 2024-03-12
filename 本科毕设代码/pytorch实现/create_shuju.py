import numpy as np
import pandas as pd

# 设置模拟数据的参数
num_examples = 1000  # 数据样本数量
num_features = 10  # 特征数量

# 生成随机数据
data = np.random.rand(num_examples, num_features + 1)  # +1 是因为我们包含了目标列

# 为简化示例，我们让目标值（第一列）部分依赖于特征值的线性组合加上一些随机噪声
# 这里仅作为示例，实际模型训练应使用真实数据集
data[:, 0] = np.sum(data[:, 1:5], axis=1) * 0.5 + np.random.rand(num_examples) * 0.1

# 转换为 DataFrame 并命名列
column_names = ['Target'] + [f'Feature_{i}' for i in range(1, num_features + 1)]
df = pd.DataFrame(data, columns=column_names)

# 查看生成的数据
print(df.head())

# 保存生成的数据到 CSV 文件
df.to_csv('simulated_data.csv', index=False)