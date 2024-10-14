# import numpy as np

# # 定义 RBF 核函数，支持多维 x
# def rbf_kernel(X1, X2, sigma=1.0):
#     K = np.zeros((X1.shape[0], X2.shape[0]))
#     for i in range(X1.shape[0]):
#         for j in range(X2.shape[0]):
            
#             K[i, j] = np.exp(-np.linalg.norm(X1[i] - X2[j])**2 / (2 * sigma**2))
#     return K


# def kernel_ridge_regression(X_train, y_train, lambda_reg=1.0, sigma=1.0):
    
#     K = rbf_kernel(X_train, X_train, sigma)
    
 
#     n = K.shape[0]
#     K_reg = K + lambda_reg * np.eye(n)
    

#     alpha = np.linalg.solve(K_reg, y_train)
    
#     return alpha, X_train


# def predict(X_train, X_new, alpha, sigma=1.0):

#     K_new = rbf_kernel(X_new, X_train, sigma)
    

#     y_pred = K_new.dot(alpha)
    
#     return y_pred


# def generate_random_data(num_samples, num_features):

#     X = np.random.rand(num_samples, num_features)
#     y = np.random.rand(num_samples)
#     return X, y


# num_samples = 5
# num_features = 3


# X_train, y_train = generate_random_data(num_samples, num_features)


# alpha, X_train = kernel_ridge_regression(X_train, y_train, lambda_reg=0.1, sigma=1.0)


# X_new, _ = generate_random_data(3, num_features)
# y_pred = predict(X_train, X_new, alpha, sigma=1.0)

# print("Predictions:", y_pred)









# #
# import numpy as np
# from scipy.special import kv, gamma

# # 定义 Matérn 
# def matern_kernel(X1, X2, nu=1.5, length_scale=1.0):
#     """
#     Matérn kernel function.
#     Parameters:
#     - X1, X2: Arrays of input data points.
#     - nu: Smoothness parameter of the Matérn kernel.
#     - length_scale: Length scale parameter.

#     Returns:
#     - Kernel matrix K.
#     """
#     K = np.zeros((X1.shape[0], X2.shape[0]))
#     for i in range(X1.shape[0]):
#         for j in range(X2.shape[0]):
#             r = np.linalg.norm(X1[i] - X2[j]) / length_scale
#             if r == 0:
#                 K[i, j] = 1
#             else:
#                 factor = (2**(1 - nu)) / gamma(nu)
#                 matern_part = (r**nu) * kv(nu, r)
#                 K[i, j] = factor * matern_part * np.exp(-r)
#     return K


# def kernel_ridge_regression(X_train, y_train, lambda_reg=1.0, nu=1.5, length_scale=1.0):

#     K = matern_kernel(X_train, X_train, nu=nu, length_scale=length_scale)
    

#     n = K.shape[0]
#     K_reg = K + lambda_reg * np.eye(n)
    

#     alpha = np.linalg.solve(K_reg, y_train)
    
#     return alpha, X_train


# def predict(X_train, X_new, alpha, nu=1.5, length_scale=1.0):

#     K_new = matern_kernel(X_new, X_train, nu=nu, length_scale=length_scale)
    

#     y_pred = K_new.dot(alpha)
    
#     return y_pred


# def generate_random_data(num_samples, num_features):
    
#     X = np.random.rand(num_samples, num_features)
#     y = np.random.rand(num_samples)
#     return X, y


# num_samples = 5
# num_features = 3

# # 生成随机训练数据
# X_train, y_train = generate_random_data(num_samples, num_features)

# alpha, X_train = kernel_ridge_regression(X_train, y_train, lambda_reg=0.1, nu=1.5, length_scale=1.0)


# X_new, _ = generate_random_data(3, num_features)
# y_pred = predict(X_train, X_new, alpha, nu=1.5, length_scale=1.0)

# print("Predictions:", y_pred)





import numpy as np
from sklearn.metrics.pairwise import rbf_kernel  # RBF kernel
def gaussian_kernel_matrix(theta, theta_prime):
    sigma = 1.0
    # Calculate the squared Euclidean distance matrix
    sq_dist = np.sum((theta[:, np.newaxis] - theta_prime[np.newaxis, :]) ** 2, axis=2)
    
    # Calculate the Gaussian kernel matrix
    kernel_matrix = np.exp(-sq_dist / (2 * sigma**2))
    
    return kernel_matrix

# 定义 RBF 核函数，支持多维 x
def rbf_kernel1(X1, X2, sigma=1.0):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            
            K[i, j] = np.exp(-np.linalg.norm(X1[i] - X2[j])**2 / (2 * sigma**2))
    return K



# Example usage
theta = np.array([[1, 2],[2, 3]])
theta_prime = np.array([[1, 1], [4, 4]])
sigma = 1.0

kernel_matrix = gaussian_kernel_matrix(theta, theta_prime)
print("Gaussian Kernel Matrix:\n", kernel_matrix)


kernel_matrix1 = rbf_kernel1(theta, theta_prime)
print("Gaussian Kernel Matrix:\n", kernel_matrix1)



kernel_matrix2 = rbf_kernel(theta, theta_prime)
print("Gaussian Kernel Matrix:\n", kernel_matrix2)