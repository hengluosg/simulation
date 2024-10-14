import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel  # RBF kernel
import seaborn as sns
import pandas as pd
# 示例目标函数 f(theta, x, xi)
def F(theta, x ,xi):
    
    return np.sum((theta - x)**2) + xi

# 随机梯度下降算法（SGD）的示例实现
def sgd(theta_init, x, lr, num_iters):
    theta = theta_init
    for _ in range(num_iters):
        grad = 2 * (theta - x)  # 目标函数 f 的梯度
        theta = theta - lr * grad
    return theta

# 离线阶段实现
def offline_stage(n, T, lr, covariate_dim):
    np.random.seed(1)
    covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
    theta_estimates = []
    for i in range(n):
        x_i = covariates_points[i]
        theta_values = []
        for _ in range(T):
            np.random.seed(1)
            theta_init = np.random.randn(covariate_dim)  # 初始化θ
            theta_t = sgd(theta_init, x_i, lr, T)
            theta_values.append(theta_t)
        theta_bar = np.mean(theta_values, axis=0)
        theta_estimates.append(theta_bar)
    return covariates_points, np.array(theta_estimates)


# 在线阶段实现
def knn_online_stage(covariates_points, theta_estimates, x, k):
    # 计算新协变量点 x 与离线阶段得到的协变量点集之间的欧氏距离
    distances = euclidean_distances([x], covariates_points).flatten()
    
    # 找到 k 个最近邻的协变量点的索引
    nearest_neighbors_indices = np.argsort(distances)[:k]
    
    # 获取 k 个最近邻点的 θ 的估计值
    nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    
    # 计算 θ 的平均值
    theta_hat = np.mean(nearest_theta_estimates, axis=0)
    
    return theta_hat






#KRR算法的在线阶段实现
def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4):
    K_phi = rbf_kernel(covariates_points, covariates_points)  # 核矩阵
    k_phi_x = rbf_kernel([x], covariates_points).flatten()    # 新点与训练点之间的核
    theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))) @ theta_estimates
    return theta_x

# 自定义实现的KRR算法
# def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4, gamma=1.0):
#     # 计算核矩阵K_phi
#     n = covariates_points.shape[0]
#     K_phi = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             K_phi[i, j] = np.exp(-gamma * np.linalg.norm(covariates_points[i] - covariates_points[j])**2)
    
#     # 计算核向量k_phi_x
#     k_phi_x = np.array([np.exp(-gamma * np.linalg.norm(x - covariates_points[i])**2) for i in range(n)])
    
#     # 计算KRR的预测值
#     theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(n)) @ theta_estimates)
    
#     return theta_x



def f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    #xi_samples = np.random.normal(0, 0, num_samples)
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)

def true_theta_function(x, num_samples):
    def objective(theta):
        return f(theta, x, num_samples)
    initial_guess = np.zeros(covariate_dim)
    result = minimize(objective, initial_guess, method='BFGS').x
    return result

# 将数据转换为适合Seaborn使用的格式
def prepare_data_for_plot(n_values, mse_by_k, mse_krr, k_values):
    data = []
    for k in k_values:
        for i, n in enumerate(n_values):
            data.append({"n": n, "MSE": mse_by_k[k][i], "Method": f"k-NN (k={k})"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Method": "KRR"})
    return pd.DataFrame(data)




if __name__ == '__main__':
    lowbound = 0
    upbound = 4
    total_budget = 5000 
    lr = 0.001
    covariate_dim = 1
    num_samples = 1000  # Number of samples for true theta calculation
    # 生成测试数据集 (在线阶段需要的)
    num_test_points = 1000
    np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    
    
    n_values = np.arange(4,101,4)
    #n_values = np.arange(100,401,100)
 
    # 不同的 k 值
    k_values = [1,  5, 10]  # 可以根据需要调整

    # # 存储每个k值下的MSE
    # mse_by_k = {k: [] for k in k_values}

    # # 遍历不同的k值
    # for k in k_values:
    #     # 遍历不同的n值
    #     for n in n_values:
    #         T = total_budget // n  # 计算对应的 T 值
            
    #         # 离线阶段
    #         covariates_points, theta_estimates = offline_stage(n, T, lr, covariate_dim)
            
    #         # 在线阶段，计算预测值和真实值的 MSE
    #         predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
    #         mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
            

    #         mse_by_k[k].append(mse)
    #         print(f"k = {k}, n = {n}, T = {T}, MSE = {mse}")


    # 存储每个k值下的MSE
    mse_by_k = {k: [] for k in k_values}


    
    # 遍历不同的n值
    for n in n_values:
        T = total_budget // n  # 计算对应的 T 值
        
        # 离线阶段
        covariates_points, theta_estimates = offline_stage(n, T, lr, covariate_dim)
        
        # 遍历不同的k值
        for k in k_values:
            # 在线阶段，计算预测值和真实值的 MSE
            predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
            mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
            
            mse_by_k[k].append(mse)
            print(f"k = {k}, n = {n}, T = {T}, MSE = {mse}")








    mse_krr = []

    for n in n_values:
        T = total_budget // n
        covariates_points, theta_estimates = offline_stage(n, T, lr, covariate_dim)
        predicted_theta_values = np.array([krr_online_stage(covariates_points, theta_estimates, x) for x in test_points])
        mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
        mse_krr.append(mse)
        print(f"KRR, n = {n}, T = {T}, MSE = {mse}")


    # 准备数据
    df = prepare_data_for_plot(n_values, mse_by_k, mse_krr, k_values)

    # 绘制MSE随n变化的曲线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='MSE', hue='Method', marker='o')

    plt.xlabel('Number of covariate points (n)')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of covariate points (n) for different methods')
    plt.grid(True)
    plt.show()




    