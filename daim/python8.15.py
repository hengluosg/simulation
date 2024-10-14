



from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

# 示例目标函数 f(theta, x, xi)
def F(theta, x ,xi):
    
    return np.sum((theta - x)**2) + xi

# 随机梯度下降算法（SGD）的示例实现
def sgd(theta_init, x, lr, num_iters):
    theta = theta_init
    for _ in range(num_iters):
        # 示例中的随机变量 xi
        grad = 2 * (theta - x)  # 目标函数 f 的梯度
        theta = theta - lr * grad
    return theta

# 离线阶段实现
def offline_stage(n, T, lr, covariate_dim):
    # 生成协变量点集 x_1, x_2, ..., x_n
    np.random.seed(1)
    covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
    
    # 存储每个协变量点的 θ 的估计值
    theta_estimates = []
    
    # 第一阶段
    for i in range(n):
        x_i = covariates_points[i]
        theta_values = []
        for _ in range(T):
            # 使用SGD算法找到每个协变量点下的最优θ
            np.random.seed(1)
            theta_init = np.random.randn(covariate_dim)  # 初始化θ
            theta_t = sgd(theta_init, x_i, lr, T)
            theta_values.append(theta_t)
        # 计算 θ 的平均值
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



def f(theta, x, num_samples):
    # 使用蒙特卡洛仿真来估计期望
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)




def true_theta_function(x, num_samples):
    # Define the true theta function, including noise considerations

    def objective(theta):
        return f(theta, x, num_samples)
    

    initial_guess = np.zeros(covariate_dim)
    result = minimize(objective, initial_guess, method='BFGS').x
    
    
    return result

if __name__ == '__main__':






    # 设置参数

    lowbound = -4
    upbound = 4
    total_budget = 1000  # 总预算 B
    lr = 0.01  # 学习率
    covariate_dim = 6 # 协变量的维度
    num_samples = 1000  # Number of samples for true theta calculation
    # 生成测试数据集 (在线阶段需要的)
    num_test_points = 1000
    np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    #print(total_budget**(covariate_dim/(covariate_dim+2)))
    n = int(total_budget**(covariate_dim/(covariate_dim+2)))

    print(n)
    # 不同的 n 值
    #n_values = [5, 10, n, 20, 30, 50]  # 可以根据需要调整
    n_values = np.arange(4,101,2)


    # 不同的 k 值
    k_values = [1,  5, 10]  # 可以根据需要调整

    # 存储每个k值下的MSE
    mse_by_k = {k: [] for k in k_values}

    # 遍历不同的k值
    for k in k_values:
        # 遍历不同的n值
        for n in n_values:
            T = total_budget // n  # 计算对应的 T 值
            
            # 离线阶段
            covariates_points, theta_estimates = offline_stage(n, T, lr, covariate_dim)
            
            # 在线阶段，计算预测值和真实值的 MSE
            predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
            mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
            print(true_theta_values.shape, predicted_theta_values.shape)

            mse_by_k[k].append(mse)
            print(f"k = {k}, n = {n}, T = {T}, MSE = {mse}")

    # 绘制MSE随n变化的曲线图
    plt.figure(figsize=(10, 6))
    for k in k_values:
        #plt.plot(n_values, np.log2(mse_by_k[k]), marker='o', label=f'k = {k}')
        plt.plot(n_values, mse_by_k[k], marker='o', label=f'k = {k}')

    plt.xlabel('Number of covariate points (n)')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of covariate points (n) for different k values')
    plt.legend()
    plt.grid(True)
    plt.show()













# import numpy as np
# from scipy.optimize import minimize
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# def noisy_objective_function(theta, x):
#     noise = np.random.normal(0, 0.1)  # Noise with mean 0 and standard deviation 0.1
#     return  (x - theta)**2 + noise

# def sgd(theta_init, x, lr, iterations):
#     theta = theta_init
#     for _ in range(iterations):
#         gradient = 2 * (x - theta ) 
#         theta = theta - lr * gradient
#     return theta

# def offline_stage(n, T, lr):
#     covariates_points = np.random.uniform(low=-10, high=10, size=n)
#     theta_estimates = []
#     for x_i in covariates_points:
#         theta_values = [sgd(np.random.randn(), x_i, lr, T) for _ in range(T)]
#         theta_bar = np.mean(theta_values)
#         theta_estimates.append(theta_bar)
#     return covariates_points, np.array(theta_estimates)

# def knn_online_stage(covariates_points, theta_estimates, x, k):

#     distances = euclidean_distances([x], covariates_points).flatten()
#     #distances = euclidean_distances([x], covariates_points.reshape(-1, 1)).flatten()
#     nearest_indices = np.argsort(distances)[:k]
#     nearest_theta_estimates = theta_estimates[nearest_indices]
#     theta_hat = np.mean(nearest_theta_estimates)
#     return theta_hat






# def true_theta_function(x):
#     # Define true theta function, here it's an example
#     return minimize(lambda theta: noisy_objective_function(theta, x), x0=0).x

# # Parameters
# total_budget = 100  # Total budget B
# lr = 0.01  # Learning rate
# n_values = [5, 10, 20, 30, 50]  # Different n values
# k_values = [1, 3, 5]  # Different k values
# num_test_points = 20  # Number of test points

# # Generate test points and true theta values
# test_points = np.random.uniform(low=-10, high=10, size=num_test_points)
# true_theta_values = np.array([true_theta_function(x) for x in test_points])

# mse_by_k = {k: [] for k in k_values}

# for k in k_values:
#     for n in n_values:
#         T = total_budget // n
#         covariates_points, theta_estimates = offline_stage(n, T, lr)
#         predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
#         mse = mean_squared_error(true_theta_values, predicted_theta_values)
#         mse_by_k[k].append(mse)
#         print(f"k = {k}, n = {n}, T = {T}, MSE = {mse}")

# # Plot MSE vs. Number of covariate points (n) for different k values
# plt.figure(figsize=(10, 6))
# for k in k_values:
#     plt.plot(n_values, mse_by_k[k], marker='o', label=f'k = {k}')

# plt.xlabel('Number of covariate points (n)')
# plt.ylabel('MSE')
# plt.title('MSE vs Number of covariate points (n) for different k values')
# plt.legend()
# plt.grid(True)
# plt.show()













