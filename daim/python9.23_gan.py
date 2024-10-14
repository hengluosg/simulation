# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics import mean_squared_error
# from scipy.optimize import minimize
# from sklearn.metrics.pairwise import rbf_kernel  # RBF kernel
# import seaborn as sns
# import pandas as pd
# from itertools import product

# # 示例目标函数 f(theta, x, xi)
# def F(theta, x ,xi):
    
#     return np.sum((theta - x)**2) + xi



# def f(theta, x, num_samples):
#     np.random.seed(1)
#     xi_samples = np.random.normal(0, 0, num_samples)  # 随机样本
#     #xi_samples = np.random.normal(0, 0, num_samples)
#     costs = [F(theta, x, xi) for xi in xi_samples]
#     return np.mean(costs)


# def grad_f(theta, x, num_samples):
#     np.random.seed(1)
#     xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
#     grads = [4 * (theta - x)**3 for xi in xi_samples]  # F 对 theta 的导数
#     return np.mean(grads, axis=0)  # 对每个样本的梯度取平均
# # SGD with Learning Rate Decay

# def sgd_with_lr_decay( theta_0, x, eta_0, gamma, T):
#     theta = theta_0
    
#     theta_values = []  # 记录每次迭代的theta值
#     for t in range(1, T + 1):
        
#         #grad = grad_f(theta, x, num_samples)

#         grad = 2 * (theta - x)
#         # 学习率衰减
#         #eta_t = eta_0 / t        #(1 + gamma * t)
#         eta_t = eta_0 / (1 + gamma * t)
#         # 更新参数
#         #eta_t = eta_0 
#         theta = theta - eta_t * grad
        
        
#         theta_values.append(theta)
        
    
#     return theta_values
# def grid_sample(d, lowbound, upbound, point):
#     # 根据给定的总点数和维度，计算每个维度的点数 n
#     n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    
#     # 在每个维度生成 n 个均匀间隔的点
#     grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
#     #print(len(grid_1d))
#     # 在 d 维空间生成网格点
#     grid_points = np.array(list(product(grid_1d, repeat=d)))
    
#     # 如果生成的点数大于目标点数，则随机选择 point 个点
#     if len(grid_points) > point:
#         indices = np.random.choice(len(grid_points), size=point, replace=False)
#         grid_points = grid_points[indices]
#     #print(grid_points)
#     return grid_points


# # 离线阶段实现
# def offline_stage(n, T, eta_0, gamma, covariate_dim):
#     #np.random.seed(1)
#     covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
#     #covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
#     #print(n, covariates_points.shape)
#     theta_estimates = []
#     for i in range(n):
#         x_i = covariates_points[i]
        
        
#         np.random.seed(1)
#         theta_init = np.random.randn(covariate_dim)  # 初始化θ

        
#         theta_values = sgd_with_lr_decay( theta_init, x_i, eta_0, gamma, T)
        
#         theta_bar = np.mean(theta_values, axis=0)
#         theta_estimates.append(theta_bar)
#     return covariates_points, np.array(theta_estimates)


# # 在线阶段实现
# def knn_online_stage(covariates_points, theta_estimates, x, k):
#     # 计算新协变量点 x 与离线阶段得到的协变量点集之间的欧氏距离
#     distances = euclidean_distances([x], covariates_points).flatten()
    
#     # 找到 k 个最近邻的协变量点的索引
#     nearest_neighbors_indices = np.argsort(distances)[:k]
    
#     # 获取 k 个最近邻点的 θ 的估计值
#     nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    
#     # 计算 θ 的平均值
#     theta_hat = np.mean(nearest_theta_estimates, axis=0)
    
#     return theta_hat




# #KRR算法的在线阶段实现
# def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4):
#     K_phi = rbf_kernel(covariates_points, covariates_points)  # 核矩阵
#     k_phi_x = rbf_kernel([x], covariates_points).flatten()    # 新点与训练点之间的核
#     theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))) @ theta_estimates
#     return theta_x



# def true_theta_function(x, num_samples):
#     def objective(theta):
#         return f(theta, x, num_samples)
#     initial_guess = np.zeros(covariate_dim)
#     result = minimize(objective, initial_guess, method='BFGS').x
#     return result



# if __name__ == '__main__':
#     lowbound = 0
#     upbound = 10
#     total_budget = 5000
#     eta_0 = 0.1  # 初始学习率
#     gamma = 0.01  # 学习率衰减率
#     covariate_dim = 3
#     num_samples = 1000  # Number of samples for true theta calculation
#     # 生成测试数据集 (在线阶段需要的)
#     num_test_points = 1000
#     np.random.seed(1)
#     test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
#     true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    
#     #假设 n = 100  T =20
#     n_values = [100,200,300]
#     T =20
#     k = 3
#     mse_krr = []
#     mse_knn = []
#     for n in n_values:
        
        
#         # 离线阶段
#         covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma, covariate_dim)
        
        
        
#         # 在线阶段，计算预测值和真实值的 MSE
#         predicted_theta_values = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
#         mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
        
#         mse_knn.append(mse)
        

#         predicted_theta_values = np.array([krr_online_stage(covariates_points, theta_estimates, x) for x in test_points])
#         mse = mean_squared_error(np.array(true_theta_values), np.array(predicted_theta_values))
#         mse_krr.append(mse)
#         print(f"KRR, n = {n}, T = {T}, MSE = {mse}")


#     df = pd.DataFrame()
#     df['knn'] = mse_knn 
#     df['krr'] = mse_krr 
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, marker='o')

#     plt.xlabel('Number of covariate points (n)')
#     plt.ylabel('MSE')
#     plt.title('MSE vs Number of covariate points (n) for different methods')
#     plt.grid(True)
#     plt.show()




# from scipy.optimize import minimize
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics.pairwise import rbf_kernel  # RBF kernel
# import seaborn as sns
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from itertools import product

# def true_theta_function(x, num_samples):
#     def objective(theta):
#         return f(theta, x, num_samples)
#     initial_guess = np.zeros(covariate_dim)
#     result = minimize(objective, initial_guess, method='BFGS').x
#     return result


# # 定义与代码2中的 objective_function 一致的目标函数
# def F(theta, x, xi):
#     quadratic_term = np.sum((theta - x) ** 2)
#     return quadratic_term + xi

# # Function to compute average cost
# def f(theta, x, num_samples):
#     np.random.seed(1)
#     xi_samples = np.random.normal(0, 0.1, num_samples)  # Random samples
#     costs = [F(theta, x, xi) for xi in xi_samples]
#     return np.mean(costs)

# # SGD with learning rate decay
# def sgd_with_lr_decay(theta_0, x, eta_0, gamma, T):
#     theta = theta_0
#     theta_values = []  # Store theta values for each iteration
#     for t in range(1, T + 1):
#         grad = 2 * (theta - x)
#         eta_t = eta_0 / (1 + gamma * t)  # Learning rate decay
#         theta = theta - eta_t * grad
#         theta_values.append(theta)
#     return theta_values

# # Grid sampling function
# def grid_sample(d, lowbound, upbound, point):
#     n = int(round(point ** (1.0 / d))) + 1
#     grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
#     grid_points = np.array(list(product(grid_1d, repeat=d)))
#     if len(grid_points) > point:
#         indices = np.random.choice(len(grid_points), size=point, replace=False)
#         grid_points = grid_points[indices]
#     return grid_points

# # Offline stage implementation with matrix inversion
# def offline_stage(n, T, eta_0, gamma, covariate_dim, lambda_param=1e-4):
#     covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
#     theta_estimates = []
    
#     # Compute the RBF kernel matrix (K_phi) and its inverse once
#     K_phi = rbf_kernel(covariates_points, covariates_points)
#     K_phi_inv = np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))  # Precompute inverse

#     for i in range(n):
#         x_i = covariates_points[i]
#         theta_init = np.random.randn(covariate_dim)  # Initialize theta
#         theta_values = sgd_with_lr_decay(theta_init, x_i, eta_0, gamma, T)
#         theta_bar = np.mean(theta_values, axis=0)
#         theta_estimates.append(theta_bar)
    
#     #return covariates_points, np.array(theta_values), K_phi_inv  # Return the inverse matrix
#     return covariates_points, theta_estimates, K_phi_inv
# # Method 1 solver using precomputed inverse matrix
# def method1_solver(covariates_points, theta_estimates, x, K_phi_inv):
#     k_phi_x = rbf_kernel(x.reshape(1, -1), covariates_points).flatten()  # Ensure x is reshaped to 2D
#     theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
#     return theta_x

# # GAN Generator and Discriminator classes
# class Generator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Generator, self).__init__()
#         # self.fc = nn.Sequential(
#         #     nn.Linear(input_dim, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, output_dim)
#         # )

#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),  # 增加神经元数量
#             nn.LeakyReLU(0.2),  # 使用 LeakyReLU 激活函数
#             nn.Linear(128, 256),  # 增加隐藏层
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, output_dim)
#         )





#     def forward(self, x):
#         return self.fc(x)

# # class Discriminator(nn.Module):
# #     def __init__(self, input_dim):
# #         super(Discriminator, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(input_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         return self.fc(x)


# def wgan_discriminator_loss(D_real, D_fake):
#     return -(torch.mean(D_real) - torch.mean(D_fake))

# def wgan_generator_loss(D_fake):
#     return -torch.mean(D_fake)


# class Discriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.fc(x)

# def adversarial_learning(generator, discriminator, train_data, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D, lambda_theta=1000, alpha=1):
#     optimizer_G = optim.Adam(generator.parameters(), lr=eta_G)
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=eta_D)

#     discriminator_losses = []
#     generator_losses = []

#     for t in range(num_iterations):
#         for x in train_data:
#             x = torch.tensor(x).float().unsqueeze(0)

#             theta_1 = torch.tensor(method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv))
#             f_theta_1 = objective_function(theta_1, x).float()

#             theta_2 = generator(x).float()
#             f_theta_2 = objective_function(theta_2, x)

#             optimizer_D.zero_grad()

#             D_real = discriminator(f_theta_1)
#             D_fake = discriminator(f_theta_2)

#             loss_D = wgan_discriminator_loss(D_real, D_fake)
#             loss_D.backward(retain_graph=True)
#             optimizer_D.step()

#             optimizer_G.zero_grad()
#             D_fake = discriminator(f_theta_2)
#             loss_G = wgan_generator_loss(D_fake)

#             loss_theta = torch.mean((theta_1 - theta_2) ** 2)
#             total_loss_G = lambda_theta * loss_theta + alpha * loss_G
#             total_loss_G.backward()
#             optimizer_G.step()

#         discriminator_losses.append(loss_D.item())
#         generator_losses.append(total_loss_G.item())
#         if t % 5 == 0:
#             print(f"Iteration {t}/{num_iterations}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")

#     # 绘制损失曲线
#     plt.plot(discriminator_losses, label='Discriminator Loss')
#     plt.plot(generator_losses, label='Generator Loss')
#     plt.legend()
#     plt.show()

#     return generator


# # # Adversarial learning (GAN) training
# # def adversarial_learning(generator, discriminator, train_data, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D, lambda_theta=1000, alpha=1):
# #     criterion = nn.BCELoss()
# #     #criterion = nn.MSELoss()  # 替换原来的 nn.BCELoss()

# #     optimizer_G = optim.Adam(generator.parameters(), lr=eta_G)
# #     optimizer_D = optim.Adam(discriminator.parameters(), lr=eta_D)
    
# #     for t in range(num_iterations):
# #         #print(t)
# #         for x in train_data:
# #             x = torch.tensor(x).float().unsqueeze(0)
            
# #             # 计算 theta_1: 使用 method1_solver
# #             theta_1 = torch.tensor(method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv))
# #             f_theta_1 = objective_function(theta_1, x).float()

# #             # 从生成器生成 theta_2
# #             theta_2 = generator(x).float()
# #             f_theta_2 = objective_function(theta_2, x)

# #             real_labels = torch.ones(x.size(0), 1)
# #             fake_labels = torch.zeros(x.size(0), 1)

# #             # D_real = discriminator(f_theta_1)
# #             # D_fake = discriminator(f_theta_2)
# #             # loss_D_real = criterion(D_real, real_labels.view_as(D_real))
# #             # loss_D_fake = criterion(D_fake, fake_labels.view_as(D_fake))

# #             # loss_D = loss_D_real + loss_D_fake

# #             # # 更新判别器
# #             # optimizer_D.step()

# #             optimizer_D.zero_grad()  # Ensure gradients are zeroed before updating

# #             D_real = discriminator(f_theta_1)
# #             D_fake = discriminator(f_theta_2)

# #             loss_D_real = criterion(D_real, real_labels.view_as(D_real))
# #             loss_D_fake = criterion(D_fake, fake_labels.view_as(D_fake))
# #             loss_D = loss_D_real + loss_D_fake

# #             loss_D.backward(retain_graph=True)  # Backpropagate the loss for D
# #             optimizer_D.step()  # Update the discriminator

# #             # for _ in range(3):  # 这里可以调整生成器的训练次数
# #             #     optimizer_G.zero_grad()
# #             #     D_fake = discriminator(f_theta_2)
# #             #     loss_G = criterion(D_fake, real_labels.view_as(D_fake))
# #             #     loss_theta = torch.mean((theta_1 - theta_2) ** 2)
# #             #     total_loss_G = lambda_theta * loss_theta + alpha * loss_G
# #             #     total_loss_G.backward()
# #             #     optimizer_G.step()
            

# #             # Train Generator
# #             optimizer_G.zero_grad()
# #             D_fake = discriminator(f_theta_2)
# #             loss_G = criterion(D_fake, real_labels.view_as(D_fake))

# #             loss_theta = torch.mean((theta_1 - theta_2) ** 2)
# #             total_loss_G = lambda_theta * loss_theta + alpha * loss_G
# #             total_loss_G.backward()
# #             optimizer_G.step()
# #         print(f"Iteration {t}/{num_iterations}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")
# #         if t % 100 == 0:
# #             print(f"Iteration {t}/{num_iterations}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")

# #     return generator

# # Objective function with non-convexity
# def objective_function(theta, x):
#     quadratic_term = torch.sum((theta - x) ** 2, dim=-1)
#     return quadratic_term 

# # Main script
# if __name__ == '__main__':
#     np.random.seed(1)
#     torch.manual_seed(1)

#     lowbound = 0
#     upbound = 4
#     total_budget = 5000
#     eta_0 = 0.1  # Initial learning rate
#     gamma = 0.01  # Learning rate decay rate
#     covariate_dim = 6
#     num_samples = 1000
#     num_test_points = 1000
#     np.random.seed(1)
#     test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    
#     # 划分 test_points，一半用于训练，一半用于测试
#     split_index = len(test_points) // 2
#     train_points = test_points[:split_index]
#     test_points = test_points[split_index:]

#     true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
#     test_points = torch.tensor(test_points).float()
#     n_values = [50, 100, 150]
#     T = 300
#     num_iterations = 100
#     eta_G = 0.001
#     eta_D = 0.0001
#     mse_method1 = []
#     mse_gan = []
#     for n in n_values:
#         covariates_points, theta_estimates, K_phi_inv = offline_stage(n, T, eta_0, gamma, covariate_dim)
#         print(covariates_points[1], theta_estimates[1])
#         # 初始化生成器和判别器
#         generator = Generator(covariate_dim, covariate_dim)
#         discriminator = Discriminator(1)
        
#         # 训练GAN，使用train_points
#         trained_generator = adversarial_learning(generator, discriminator, train_points, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D)

#         # 使用 method1_solver 和 GAN 对 test_points 进行预测
#         predicted_theta_values_method1 = np.array([method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in test_points])
#         #predicted_theta_values_gan = np.array([trained_generator(torch.tensor(x).float()) for x in test_points])
#         predicted_theta_values_gan = np.array([trained_generator(torch.tensor(x).float()).detach().numpy() for x in test_points])


#         #predicted_theta_values_gan = np.array([trained_generator(torch.tensor(x).float().unsqueeze(0)).detach().numpy() for x in test_points])

#         # 计算 MSE
#         mse = mean_squared_error(true_theta_values, predicted_theta_values_method1)
#         mse_method1.append(mse)
#         mse = mean_squared_error(true_theta_values, predicted_theta_values_gan)
#         mse_gan.append(mse)
#         print(f"Method 1 (KRR), n = {n}, T = {T}, MSE = {mse_method1}")
#         print(f"GAN, n = {n}, T = {T}, MSE = {mse_gan}")

#     # Plotting results
#     df = pd.DataFrame()
#     df['KRR'] = mse_method1 # Example MSE for plotting
#     df['KRR_GAN'] = mse_gan
#     df.index = n_values
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, marker='o')
#     plt.xlabel('Number of covariate points (n)')
#     plt.ylabel('MSE')
#     plt.title('MSE comparison between KRR and KRR-GAN')
#     plt.grid(True)
#     plt.show()



# from scipy.optimize import minimize
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics.pairwise import rbf_kernel
# import seaborn as sns
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from itertools import product

# # 定义与 GAN 训练一致的目标函数
# def F(theta, x, xi):
#     quadratic_term = np.sum((theta - x) ** 2)
#     return quadratic_term + xi

# # 计算损失函数的均值
# def f(theta, x, num_samples):
#     np.random.seed(1)
#     xi_samples = np.random.normal(0, 0.1, num_samples)  # Random samples
#     costs = [F(theta, x, xi) for xi in xi_samples]
#     return np.mean(costs)

# # SGD with learning rate decay
# def sgd_with_lr_decay(theta_0, x, eta_0, gamma, T):
#     theta = theta_0
#     theta_values = []
#     for t in range(1, T + 1):
#         grad = 2 * (theta - x)
#         eta_t = eta_0 / (1 + gamma * t)
#         theta = theta - eta_t * grad
#         theta_values.append(theta)
#     return theta_values

# # Grid sampling function
# def grid_sample(d, lowbound, upbound, point):
#     n = int(round(point ** (1.0 / d))) + 1
#     grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
#     grid_points = np.array(list(product(grid_1d, repeat=d)))
#     if len(grid_points) > point:
#         indices = np.random.choice(len(grid_points), size=point, replace=False)
#         grid_points = grid_points[indices]
#     return grid_points

# # Offline stage implementation with matrix inversion
# def offline_stage(n, T, eta_0, gamma, covariate_dim, lambda_param=1e-4):
#     covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
#     theta_estimates = []
    
#     # Compute RBF kernel and its inverse once
#     K_phi = rbf_kernel(covariates_points, covariates_points)
#     K_phi_inv = np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

#     for i in range(n):
#         x_i = covariates_points[i]
#         theta_init = np.random.randn(covariate_dim)
#         theta_values = sgd_with_lr_decay(theta_init, x_i, eta_0, gamma, T)
#         theta_bar = np.mean(theta_values, axis=0)
#         theta_estimates.append(theta_bar)
    
#     return covariates_points, theta_estimates, K_phi_inv

# # Method 1 solver using precomputed inverse matrix
# def method1_solver(covariates_points, theta_estimates, x, K_phi_inv):
#     k_phi_x = rbf_kernel(x.reshape(1, -1), covariates_points).flatten()
#     theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
#     return theta_x

# # GAN Generator and Discriminator classes
# class Generator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Generator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, output_dim)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class Discriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.2),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.fc(x)

# # WGAN 损失函数
# def wgan_generator_loss(D_fake):
#     return -torch.mean(D_fake)

# def wgan_discriminator_loss(D_real, D_fake):
#     return torch.mean(D_fake) - torch.mean(D_real)

# # Adversarial learning with WGAN loss
# def adversarial_learning(generator, discriminator, train_data, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D, lambda_theta=1000, alpha=1):
#     optimizer_G = optim.Adam(generator.parameters(), lr=eta_G)
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=eta_D)
    
#     for t in range(num_iterations):
#         for x in train_data:
#             x = torch.tensor(x).float().unsqueeze(0)

#             theta_1 = torch.tensor(method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv))
#             f_theta_1 = objective_function(theta_1, x).float()

#             theta_2 = generator(x).float()
#             f_theta_2 = objective_function(theta_2, x)

#             optimizer_D.zero_grad()
#             D_real = discriminator(f_theta_1)
#             D_fake = discriminator(f_theta_2)

#             loss_D = wgan_discriminator_loss(D_real, D_fake)
#             loss_D.backward(retain_graph=True)
#             optimizer_D.step()

#             optimizer_G.zero_grad()
#             D_fake = discriminator(f_theta_2)
#             loss_G = wgan_generator_loss(D_fake)

#             loss_theta = torch.mean((theta_1 - theta_2) ** 2)
#             total_loss_G = lambda_theta * loss_theta + alpha * loss_G
#             total_loss_G.backward()
#             optimizer_G.step()

#         if t % 5 == 0:
#             print(f"Iteration {t}/{num_iterations}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")

#     return generator

# # Objective function
# def objective_function(theta, x):
#     quadratic_term = torch.sum((theta - x) ** 2, dim=-1)
#     return quadratic_term 
# def true_theta_function(x, num_samples):
#     def objective(theta):
#         return f(theta, x, num_samples)
#     initial_guess = np.zeros(covariate_dim)
#     result = minimize(objective, initial_guess, method='BFGS').x
#     return result
# # Main script
# if __name__ == '__main__':
#     lowbound = 0
#     upbound = 4
#     total_budget = 5000
#     eta_0 = 0.1
#     gamma = 0.01
#     covariate_dim = 4
#     num_samples = 1000
#     num_test_points = 1000
#     np.random.seed(1)
#     test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    
#     # Split test_points into train and test sets
#     split_index = len(test_points) // 2
#     train_points = test_points[:split_index]
#     test_points = test_points[split_index:]

#     true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
#     test_points = torch.tensor(test_points).float()
#     n_values = [ 100, 150,250]
#     T = 100
#     num_iterations = 50
#     eta_G = 0.001
#     eta_D = 0.001
#     mse_method1 = []
#     mse_gan = []

#     for n in n_values:
#         covariates_points, theta_estimates, K_phi_inv = offline_stage(n, T, eta_0, gamma, covariate_dim)
#         print(covariates_points[1], theta_estimates[1])

#         # Initialize Generator and Discriminator
#         generator = Generator(covariate_dim, covariate_dim)
#         discriminator = Discriminator(1)

#         # Train GAN using train_points
#         trained_generator = adversarial_learning(generator, discriminator, train_points, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D)

#         # Predict using Method 1 (KRR) and GAN
#         predicted_theta_values_method1 = np.array([method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in test_points])
#         predicted_theta_values_gan = np.array([trained_generator(torch.tensor(x).float()).detach().numpy() for x in test_points])

#         # Calculate MSE
#         mse = mean_squared_error(true_theta_values, predicted_theta_values_method1)
#         mse_method1.append(mse)
#         mse = mean_squared_error(true_theta_values, predicted_theta_values_gan)
#         mse_gan.append(mse)

#         print(f"Method 1 (KRR), n = {n}, T = {T}, MSE = {mse_method1}")
#         print(f"GAN, n = {n}, T = {T}, MSE = {mse_gan}")

#     # Plotting results
#     df = pd.DataFrame()
#     df['KRR'] = mse_method1  # Example MSE for plotting
#     df['KRR_GAN'] = mse_gan
#     df.index = n_values
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, marker='o')
#     plt.xlabel('Number of covariate points (n)')
#     plt.ylabel('MSE')
#     plt.title('MSE comparison between KRR and KRR-GAN')
#     plt.grid(True)
#     plt.show()


from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

# 定义与 GAN 训练一致的目标函数
def F(theta, x, xi):
    quadratic_term = np.sum((theta - x) ** 2)
    return quadratic_term + xi

# 计算损失函数的均值
def f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 0.1, num_samples)  # Random samples
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)

# SGD with learning rate decay
def sgd_with_lr_decay(theta_0, x, eta_0, gamma, T):
    theta = theta_0
    theta_values = []
    for t in range(1, T + 1):
        grad = 2 * (theta - x)
        eta_t = eta_0 / (1 + gamma * t)
        theta = theta - eta_t * grad
        theta_values.append(theta)
    return theta_values

# Grid sampling function
def grid_sample(d, lowbound, upbound, point):
    n = int(round(point ** (1.0 / d))) + 1
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points

# Offline stage implementation with matrix inversion
def offline_stage(n, T, eta_0, gamma, covariate_dim, lambda_param=1e-4):
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    theta_estimates = []
    
    # Compute RBF kernel and its inverse once
    K_phi = rbf_kernel(covariates_points, covariates_points)
    K_phi_inv = np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

    for i in range(n):
        x_i = covariates_points[i]
        theta_init = np.random.randn(covariate_dim)
        theta_values = sgd_with_lr_decay(theta_init, x_i, eta_0, gamma, T)
        theta_bar = np.mean(theta_values, axis=0)
        theta_estimates.append(theta_bar)
    
    return covariates_points, theta_estimates, K_phi_inv

# Method 1 solver using precomputed inverse matrix
def method1_solver(covariates_points, theta_estimates, x, K_phi_inv):
    k_phi_x = rbf_kernel(x.reshape(1, -1), covariates_points).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x

# GAN Generator and Discriminator classes
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# WGAN 损失函数
def wgan_generator_loss(D_fake):
    return -torch.mean(D_fake)

def wgan_discriminator_loss(D_real, D_fake):
    return torch.mean(D_fake) - torch.mean(D_real)

# Adversarial learning with WGAN loss and L2 regularization
def adversarial_learning(generator, discriminator, train_data, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D, lambda_theta=1, alpha=1, weight_decay=1e-4):
    optimizer_G = optim.Adam(generator.parameters(), lr=eta_G, weight_decay=weight_decay)  # L2 正则化
    optimizer_D = optim.Adam(discriminator.parameters(), lr=eta_D, weight_decay=weight_decay)  # L2 正则化
    
    for t in range(num_iterations):
        for x in train_data:
            x = torch.tensor(x).float().unsqueeze(0)

            theta_1 = torch.tensor(method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv))
            f_theta_1 = objective_function(theta_1, x).float()-100

            theta_2 = generator(x).float()
            f_theta_2 = objective_function(theta_2, x)

            optimizer_D.zero_grad()
            D_real = discriminator(f_theta_1)
            D_fake = discriminator(f_theta_2)

            loss_D = wgan_discriminator_loss(D_real, D_fake)
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            optimizer_G.zero_grad()
            D_fake = discriminator(f_theta_2)
            loss_G = wgan_generator_loss(D_fake)

            loss_theta = torch.mean((theta_1 - theta_2) ** 2)
            total_loss_G = lambda_theta * loss_theta + alpha * loss_G
            total_loss_G.backward()
            optimizer_G.step()

        if t % 5 == 0:
            print(f"Iteration {t}/{num_iterations}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")

    return generator

# Objective function
def objective_function(theta, x):
    quadratic_term = torch.sum((theta - x) ** 2, dim=-1)
    return quadratic_term 

# true_theta_function for generating ground truth
def true_theta_function(x, num_samples):
    def objective(theta):
        return f(theta, x, num_samples)
    initial_guess = np.zeros(covariate_dim)
    result = minimize(objective, initial_guess, method='BFGS').x
    return result

# Main script
if __name__ == '__main__':
    lowbound = 0
    upbound = 4
    total_budget = 5000
    eta_0 = 0.1
    gamma = 0.01
    covariate_dim = 4
    num_samples = 1000
    num_test_points = 1000
    np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    
    # Split test_points into train and test sets
    split_index = len(test_points) // 2
    train_points = test_points[:split_index]
    test_points = test_points[split_index:]

    true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    test_points = torch.tensor(test_points).float()
    n_values = [100, 150, 250]
    T = 100
    num_iterations = 50
    eta_G = 0.001
    eta_D = 0.001
    weight_decay = 1e-4  # L2 正则化系数

    mse_method1 = []
    mse_gan = []

    for n in n_values:
        covariates_points, theta_estimates, K_phi_inv = offline_stage(n, T, eta_0, gamma, covariate_dim)

        # Initialize Generator and Discriminator
        generator = Generator(covariate_dim, covariate_dim)
        discriminator = Discriminator(1)

        # Train GAN using train_points
        trained_generator = adversarial_learning(generator, discriminator, train_points, covariates_points, theta_estimates, K_phi_inv, num_iterations, eta_G, eta_D, weight_decay=weight_decay)

        # Predict using Method 1 (KRR) and GAN
        predicted_theta_values_method1 = np.array([method1_solver(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in test_points])
        predicted_theta_values_gan = np.array([trained_generator(torch.tensor(x).float()).detach().numpy() for x in test_points])

        # Calculate MSE
        mse = mean_squared_error(true_theta_values, predicted_theta_values_method1)
        mse_method1.append(mse)
        mse = mean_squared_error(true_theta_values, predicted_theta_values_gan)
        mse_gan.append(mse)

        print(f"Method 1 (KRR), n = {n}, T = {T}, MSE = {mse_method1}")
        print(f"GAN, n = {n}, T = {T}, MSE = {mse_gan}")

    # Plotting results
    df = pd.DataFrame()
    df['KRR'] = mse_method1
    df['KRR_GAN'] = mse_gan
    df.index = n_values
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, marker='o')
    plt.xlabel('Number of covariate points (n)')
    plt.ylabel('MSE')
    plt.title('MSE comparison between KRR and KRR-GAN with L2 regularization')
    plt.grid(True)
    plt.show()
