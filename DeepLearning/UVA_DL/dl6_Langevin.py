# 首先，指定一个pdf(x)。
# 用Langevin采样算法采样得到一系列数据点x。
# 绘制直方图和pdf(x)图像检验一致性
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义目标 PDF (未归一化)
def target_p(x):
    return 0.3 * np.exp(-(x - 2)**2 / (2 * 0.5**2)) + \
           0.7 * np.exp(-(x + 2)**2 / (2 * 1.0**2))

# 2. 定义梯度 grad(log p(x))
def grad_log_p(x):
    # 数值求导或解析求导。这里为了演示清晰使用解析形式的简化版
    term1 = 0.3 * np.exp(-(x - 2)**2 / 0.5) * (-(x - 2) / 0.25)
    term2 = 0.7 * np.exp(-(x + 2)**2 / 2) * (-(x + 2) / 1.0)
    return (term1 + term2) / (target_p(x) + 1e-10)

# 3. 朗之万采样参数
n_samples = 10000  # 采样总数
delta = 0.1        # 步长 (Step size)
x = 0.0            # 初始点
samples = []

# 4. 采样循环
for i in range(n_samples):
    noise = np.random.normal(0, np.sqrt(2 * delta))
    x = x + delta * grad_log_p(x) + noise
    samples.append(x)

# 5. 可视化检验
x_range = np.linspace(-6, 6, 1000)
plt.figure(figsize=(10, 6))

# 绘制采样点的直方图
# 理论上，前1000个点也是符合分布的
plt.hist(samples[1000:], bins=100, density=True, alpha=0.6, color='skyblue', label='Langevin Samples')

# 绘制理论 PDF 曲线 (需归一化)
y_theory = target_p(x_range)
# 简单的数值归一化以便对比
y_theory /= np.trapezoid(y_theory, x_range) 
plt.plot(x_range, y_theory, color='red', lw=2, label='True PDF (Normalized)')

plt.title("Langevin Dynamics Sampling Verification")
plt.legend()
plt.show()

