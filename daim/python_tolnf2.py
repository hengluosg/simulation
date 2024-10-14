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
            f_theta_1 = objective_function(theta_1, x).float()

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
