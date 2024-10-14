import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

# 假设我们有一个简单的时间序列数据
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = 5 * x + np.random.normal(0, 0.5, n)

# 将数据存储为 pandas DataFrame
df = pd.DataFrame({"x": x, "y": y})

# 使用 PyMC3 进行贝叶斯回归
with pm.Model() as model:
    # 定义先验
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # 定义线性回归模型
    mu = alpha + beta * df['x']
    
    # 定义似然
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=df['y'])
    
    # 进行采样
    trace = pm.sample(2000, return_inferencedata=False)

# 绘制采样结果
pm.plot_posterior(trace, var_names=["alpha", "beta", "sigma"])
plt.show()

# 预测未来时间点的分布
x_pred = np.linspace(1, 1.5, 50)  # 假设我们要预测未来的 x
with model:
    # 预测新的点
    post_pred = pm.sample_posterior_predictive(trace, var_names=["alpha", "beta", "sigma"])

# 计算预测分布
y_pred = post_pred["alpha"].mean() + post_pred["beta"].mean() * x_pred

# 绘制预测分布
plt.plot(x, y, label='Observed Data')
plt.plot(x_pred, y_pred, label='Predicted Data', color='r')
plt.fill_between(x_pred, np.percentile(y_pred, 2.5, axis=0), np.percentile(y_pred, 97.5, axis=0), color='r', alpha=0.3)
plt.legend()
plt.show()
