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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def blackbox_function(w, x):

    noise = np.random.normal(0, 0)  # 添加噪声
    return np.sum((w - x) ** 2) + noise

def blackbox_function1(theta, x):
    theta = torch.tensor(theta, dtype=torch.float32)  # 将 theta 转换为 Tensor
    x = torch.tensor(x, dtype=torch.float32) 
    noise = torch.normal(mean=0, std=0, size=(1,))
    quadratic_term = torch.sum((theta - x) ** 2, dim=-1)
    return quadratic_term  + noise 


def sgd_with_lr_decay(f, w_init, x, T, a_0=0.1, epsilon=1e-5, gamma=0.1):
  
    w = w_init
    dim = len(w)  
    w_history = []  
    
    for t in range(1, T + 1):

        a_t = a_0 / (t ** gamma)  
        

        grad_est = np.zeros(dim)
        for i in range(dim):
            w_plus = np.copy(w)
            w_plus[i] += epsilon
            f_plus = f(w_plus, x)

            w_minus = np.copy(w)
            w_minus[i] -= epsilon
            f_minus = f(w_minus, x)

            grad_est[i] = (f_plus - f_minus) / (2 * epsilon)


        w = w - a_t * grad_est

  
        w_history.append(np.copy(w))

    return w_history


# def sgd_with_lr_decay(f, w_init, x, T, a_0=1, c_0=1, gamma=0.10, delta=0.60):
    
    w = w_init
    dim = len(w) 
    w_history = [] 
    n_samples = 20
    for t in range(1, T + 1):
        a_t = a_0 / (t ** gamma)  
        c_t = c_0 / (t ** delta)  

        grad_mean = np.zeros(dim)  
        
        for i in range(n_samples):
            
            delta_vec = np.random.choice([-1, 1], size=dim)
            
            
            f_plus = f(w + c_t * delta_vec, x)  
            f_minus = f(w - c_t * delta_vec, x)  
            
            grad_est = (f_plus - f_minus) / (2 * c_t) * delta_vec  
            
            
            grad_mean = grad_mean + (grad_est - grad_mean) / (i + 1)
        

        w = w - a_t * grad_mean
        
  
        w_history.append(np.copy(w))

    return w_history

# SGD with learning rate decay
# def sgd_with_lr_decay(theta_0, x, eta_0, gamma, T):
#     theta = theta_0
#     theta_values = []
#     for t in range(1, T + 1):
#         grad = 2 * (theta - x)
#         eta_t = eta_0 / (1 + gamma * t)
#         theta = theta - eta_t * grad
#         theta_values.append(theta)
#     return theta_values







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
def offline_stage(n, T,  covariate_dim, lambda_param=1e-4):
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    theta_estimates = np.zeros((n,covariate_dim))
    
    # Compute RBF kernel and its inverse once
    K_phi = rbf_kernel(covariates_points, covariates_points)
    K_phi_inv = np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

    for i in range(n):
        x_i = covariates_points[i]
        theta_init = np.random.randn(covariate_dim)

        theta_values = sgd_with_lr_decay(blackbox_function, theta_init, x_i, T)

        theta_bar = np.mean(theta_values, axis=0)
        theta_estimates[i] = theta_bar
        

    K_phi_theta = rbf_kernel(theta_estimates, theta_estimates)
    K_phi_theta_inv = np.linalg.inv(K_phi_theta + lambda_param * np.eye(K_phi_theta.shape[0]))
    
    return covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv

# Method 1 solver using precomputed inverse matrix
def optimal_solution_prediction(covariates_points, theta_estimates, x, K_phi_inv):
    k_phi_x = rbf_kernel(x.reshape(1, -1), covariates_points).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x

def objective_function_prediction(theta_estimates, theta, K_phi_theta_inv,f):
    theta_estimates = torch.tensor(theta_estimates, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    
    K_phi_theta_inv = torch.tensor(K_phi_theta_inv, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)

    theta = theta.unsqueeze(0)
    
    k_phi_theta = torch.tensor(rbf_kernel(theta.reshape(1, -1), theta_estimates.numpy()), dtype=torch.float32).flatten()
    

    
    
    f_theta_x = torch.mm(k_phi_theta.unsqueeze(0), torch.mm(K_phi_theta_inv, f.unsqueeze(1)))
    
    return f_theta_x

def grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f):
    sigma=1.0
    
    # Convert numpy arrays to tensors if they are not already
    theta_estimates = torch.tensor(theta_estimates, dtype=torch.float32)
    if theta_prime.dim() == 1:
        theta_prime = torch.tensor(theta_prime, dtype=torch.float32).unsqueeze(0)
    else:
        theta_prime = torch.tensor(theta_prime, dtype=torch.float32)
    
    K_phi_theta_inv = torch.tensor(K_phi_theta_inv, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)

    # Calculate the Gaussian kernel between theta_prime and theta_estimates
    k_phi_theta = torch.tensor(rbf_kernel(theta_prime.detach().numpy().reshape(1, -1), theta_estimates.detach().numpy()), dtype=torch.float32).flatten()
    
    # Calculate the function prediction
    #f_theta_x = torch.mm(k_phi_theta.unsqueeze(0), torch.mm(K_phi_theta_inv, f.unsqueeze(1)))

    
    # Calculate the derivative of the Gaussian kernel matrix
    diff = theta_estimates - theta_prime
    
    k_phi_prime = -(diff / sigma**2) * k_phi_theta.unsqueeze(1)  # Apply the derivative formula
    # # Calculate the gradient of the function prediction
    grad_f_theta_x = torch.mm(k_phi_prime.T, torch.mm(K_phi_theta_inv, f.unsqueeze(1)))

    return  grad_f_theta_x



# def iterative_optimization(theta_estimates, initial_theta_prime, K_phi_theta_inv, f, iterations = 1000):
   
#     theta_prime = torch.tensor(initial_theta_prime, dtype=torch.float32)

#     # Ensure all inputs are tensors
#     theta_estimates = torch.tensor(theta_estimates, dtype=torch.float32)
#     K_phi_theta_inv = torch.tensor(K_phi_theta_inv, dtype=torch.float32)
#     f = torch.tensor(f, dtype=torch.float32)
#     theta_values = []

#     grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
    
#     for i in range(iterations):
#         # Calculate the gradient of the function at theta_prime
#         grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
#         eta_t = eta_0 / (1 + gamma * (1 + i))
#         # Update theta_prime using the gradient
#         grad_norm = torch.norm(grad_f_theta_x).item()
#         #print(grad_norm )
#         theta_prime = theta_prime - eta_t * grad_f_theta_x.squeeze()

#         # Optional: print intermediate results
#         #print(f"Iteration {i+1}: Theta' = {theta_prime.numpy()}")


#         theta_values.append(theta_prime.detach())

#     theta_values_tensor = torch.stack(theta_values)
#     mean_theta_values = torch.mean(theta_values_tensor, dim=0)
#     #print(mean_theta_values.shape)
#     return mean_theta_values

# def iterative_optimization(theta_estimates, initial_theta_prime, K_phi_theta_inv, f, iterations=100):
#     tol=1e-4
#     theta_prime = torch.tensor(initial_theta_prime, dtype=torch.float32)

#     # Ensure all inputs are tensors
#     theta_estimates = torch.tensor(theta_estimates, dtype=torch.float32)
#     K_phi_theta_inv = torch.tensor(K_phi_theta_inv, dtype=torch.float32)
#     f = torch.tensor(f, dtype=torch.float32)
#     theta_values = []

#     # Step 1: Compute the initial gradient at initial_theta_prime
#     grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
#     grad_norm = torch.norm(grad_f_theta_x).item()

#     if grad_norm < tol:
#         print(f"Converged with initial theta_prime, gradient norm {grad_norm}")
#         return theta_prime

#     # Initialize variables for monitoring improvement
#     f_previous = objective_function_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f).item()
#     no_improvement_count = 0

#     # Step 2: If not converged, continue with iterations
#     for i in range(iterations):
#         # Learning rate update rule
#         eta_t = eta_0 / (1 + gamma * (i + 1))

#         # Gradient clipping
#         #grad_f_theta_x = torch.clamp(grad_f_theta_x, -1.0, 1.0)

#         # Update theta_prime using the gradient
#         theta_prime += eta_t * grad_f_theta_x.squeeze()

#         # Recompute the gradient after the update
#         grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
#         grad_norm = torch.norm(grad_f_theta_x).item()

#         # Recalculate objective function to check improvement
#         f_current = objective_function_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f).item()
#         if f_current < f_previous:
#             f_previous = f_current
#             no_improvement_count = 0
#         else:
#             no_improvement_count += 1

#         # Save the current value of theta_prime if better
#         theta_values.append(theta_prime.detach())

#         # Check for convergence or early stopping
#         if grad_norm < tol or no_improvement_count > 10:
#             print(f"Converged after {i + 1} iterations with gradient norm {grad_norm}")
#             break

#     # Convert the list of theta values to a tensor and compute the mean
#     if theta_values:
#         theta_values_tensor = torch.stack(theta_values)
#         mean_theta_values = torch.mean(theta_values_tensor, dim=0)
#         return mean_theta_values
#     else:
#         # Return the last updated value of theta_prime if no values were saved
#         return theta_prime.detach()








def iterative_optimization(theta_estimates, initial_theta_prime, K_phi_theta_inv, f, iterations=100,eta_0 = 0.001,gamma = 0.01):
    
    theta_prime = torch.tensor(initial_theta_prime, dtype=torch.float32)

    # Ensure all inputs are tensors
    theta_estimates = torch.tensor(theta_estimates, dtype=torch.float32)
    K_phi_theta_inv = torch.tensor(K_phi_theta_inv, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)
    theta_values = []
    tol = 1e-2
    # Step 1: Compute the initial gradient at initial_theta_prime
    grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
    grad_norm = torch.norm(grad_f_theta_x).item()
    #print(f"Initial gradient norm: {grad_norm}")
    
    # Check if the initial gradient norm is already smaller than the tolerance
    if grad_norm < tol:
        print(f"Converged with initial theta_prime, gradient norm {grad_norm}")
        return theta_prime

    # Step 2: If not converged, continue with iterations
    for i in range(iterations):
        # Learning rate update rule
        #eta_t = eta_0 / (1 + gamma ** (1 + i))
        eta_t = eta_0 / ( (1 + i) ** gamma)
        
        # Update theta_prime using the gradient
        theta_prime = theta_prime + eta_t * grad_f_theta_x.squeeze()
        
        # Recompute the gradient after the update
        grad_f_theta_x = grad_f_theta_x_prediction(theta_estimates, theta_prime, K_phi_theta_inv, f)
        grad_norm = torch.norm(grad_f_theta_x).item()
        #print(f"Iteration {i+1} gradient norm: {grad_norm}")
        
        # Check for convergence, stop if gradient norm is smaller than the tolerance
        if grad_norm < tol:
            #print(f"Converged after {i+1} iterations with gradient norm {grad_norm}")
            break

        # Save the current value of theta_prime
        theta_values.append(theta_prime.detach())

    # Convert the list of theta values to a tensor and compute the mean
    if theta_values:  # Check if list is not empty
        theta_values_tensor = torch.stack(theta_values)
        mean_theta_values = torch.mean(theta_values_tensor, dim=0)
        return mean_theta_values
    else:
        # Return the last updated value of theta_prime if no values were saved
        return theta_prime.detach()


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
def adversarial_learning(generator, discriminator, train_data, covariates_points, theta_estimates, K_phi_inv,K_phi_theta_inv, num_iterations, eta_G, eta_D, lambda_theta=0.5, alpha=0.5, weight_decay=1e-4):
    optimizer_G = optim.Adam(generator.parameters(), lr=eta_G, weight_decay=weight_decay)  # L2 正则化
    optimizer_D = optim.Adam(discriminator.parameters(), lr=eta_D, weight_decay=weight_decay)  # L2 正则化
    
    for t in range(num_iterations):
        for x in train_data:
            f = [blackbox_function1(theta,x) for theta in  theta_estimates]
            x = torch.tensor(x).float().unsqueeze(0)
            
            #theta_1 = torch.tensor(optimal_solution_prediction(covariates_points, theta_estimates, x.numpy(), K_phi_inv))
            theta_1 = torch.tensor(optimize_theta_for_point(x.numpy(), covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv))
            
            f_theta_0 = objective_function_prediction(theta_estimates, theta_1,  K_phi_theta_inv,f) 
            #print("0",f_theta_0)
            # if f_theta_0 < 1e-3:
            #     updated_theta = theta_1
            #     f_theta_1 = f_theta_0 
            # else:
                
            #     updated_theta = iterative_optimization(theta_estimates, theta_1, K_phi_theta_inv, f)
            #     f_theta_1= objective_function_prediction(theta_estimates, updated_theta,  K_phi_theta_inv,f) 
            #     if f_theta_1 > f_theta_0:
            #         updated_theta = theta_1
            #         f_theta_1 = f_theta_0 


            # updated_theta = iterative_optimization(theta_estimates, theta_1, K_phi_theta_inv, f)
            # f_theta_1= objective_function_prediction(theta_estimates, updated_theta,  K_phi_theta_inv,f) 
            #print("1",f_theta_1)
            theta_2 = generator(x).float()
            #print(type(theta_estimates),type(theta_2),type(x),type(K_phi_theta_inv))
            #f_theta_2 = objective_function(theta_2, x)
            f_theta_2 = objective_function_prediction(theta_estimates, theta_2, K_phi_theta_inv,f)
            #print(f_theta_2 , f_theta_1,f_theta_0)
            #print("2",f_theta_2)

            optimizer_D.zero_grad()
            D_real = discriminator(f_theta_0)
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







# def objective_function_prediction(theta_estimates, theta, x, K_phi_thaeta_inv):
#     print("Theta shape:", theta.shape)
#     print("X shape:", x.shape)
#     theta_reshaped = theta.reshape(x.shape)
#     k_phi_theta = rbf_kernel(theta.reshape(1, -1), theta_estimates).flatten()
#     f_theta_x =  np.dot(k_phi_theta, K_phi_thaeta_inv) @ objective_function(theta, x)
    
#     return f_theta_x

# GAN Generator and Discriminator classes


# # Objective function
# def objective_function(theta, x):
#     theta = torch.tensor(theta, dtype=torch.float32)  # 将 theta 转换为 Tensor
#     x = torch.tensor(x, dtype=torch.float32) 
#     quadratic_term = torch.sum((theta - x) ** 2, dim=-1)
#     return quadratic_term 

# # true_theta_function for generating ground truth
# def true_theta_function(x, num_samples):
#     def objective(theta):
#         return f(theta, x, num_samples)
#     initial_guess = np.zeros(covariate_dim)
#     result = minimize(objective, initial_guess, method='BFGS').x
#     return result
def optimize_theta_for_point(x, covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv):
    f = [blackbox_function(theta, x) for theta in theta_estimates]
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    theta_1 = torch.tensor(optimal_solution_prediction(covariates_points, theta_estimates, x_tensor.numpy(), K_phi_inv))
    updated_theta = iterative_optimization(theta_estimates, theta_1, K_phi_theta_inv, f)
    return updated_theta.detach().numpy()


# Main script
if __name__ == '__main__':
    lowbound = 0
    upbound = 8
    total_budget = 50000
    
    covariate_dim = 6
    num_samples = 1
    num_test_points = 400
    np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))

    # Split test_points into train and test sets
    split_index = len(test_points) // 2
    train_points = test_points[:split_index]
    test_points = test_points[split_index:]

    #true_theta_values_train = np.array([true_theta_function(x, num_samples) for x in train_points])
    #true_theta_values_test = np.array([true_theta_function(x, num_samples) for x in test_points])
    true_theta_values_train = np.array([x for x in train_points])
    true_theta_values_test = np.array([x for x in test_points])

    train_points = torch.tensor(train_points).float()
    test_points = torch.tensor(test_points).float()

    n_values = [100,300,400,500]
    T = 100
    num_iterations = 30
    eta_G = 0.001
    eta_D = 0.001
    weight_decay = 1e-4  # L2 正则化系数

    mse_method1_train = []
    mse_method1_test = []
    mse_gan_train = []
    mse_gan_test = []
    mse_gan_train1 = []
    mse_gan_test1 = []


    

    for n in n_values:
        
        covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv = offline_stage(n, T,  covariate_dim)
        print(mean_squared_error(covariates_points, theta_estimates))

        # Initialize Generator and Discriminator
        
        # Initialize Generator and Discriminator
        generator = Generator(covariate_dim, covariate_dim)
        discriminator = Discriminator(1)

        # Train GAN using train_points
        trained_generator = adversarial_learning(generator, discriminator, train_points, covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv, num_iterations, eta_G, eta_D, weight_decay=weight_decay)
        

        # Predict using Method 1 (KRR) and GAN for training set
        predicted_theta_values_train_method1 = np.array([optimal_solution_prediction(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in train_points])
        predicted_theta_values_train_gan = np.array([optimize_theta_for_point(x.numpy(), covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv) for x in train_points])
        predicted_theta_values_train_gan1 = np.array([trained_generator(torch.tensor(x).float()).detach().numpy() for x in train_points])
        # Predict using Method 1 (KRR) and GAN for test set
        predicted_theta_values_test_method1 = np.array([optimal_solution_prediction(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in test_points])
        predicted_theta_values_test_gan = np.array([optimize_theta_for_point(x.numpy(), covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv) for x in test_points])
        predicted_theta_values_test_gan1 = np.array([trained_generator(torch.tensor(x).float()).detach().numpy() for x in test_points])
        # Calculate MSE for training set
        mse_train = mean_squared_error(true_theta_values_train, predicted_theta_values_train_method1)
        mse_method1_train.append(mse_train)
        mse_train = mean_squared_error(true_theta_values_train, predicted_theta_values_train_gan)
        mse_gan_train.append(mse_train)
        mse_train = mean_squared_error(true_theta_values_train, predicted_theta_values_train_gan1)
        mse_gan_train1.append(mse_train)


        # Calculate MSE for test set
        mse_test = mean_squared_error(true_theta_values_test, predicted_theta_values_test_method1)
        mse_method1_test.append(mse_test)
        mse_test = mean_squared_error(true_theta_values_test, predicted_theta_values_test_gan)
        mse_gan_test.append(mse_test)
        mse_test = mean_squared_error(true_theta_values_test, predicted_theta_values_test_gan1)
        mse_gan_test1.append(mse_test)



        print(f"Method 1 (KRR), n = {n}, T = {T}, Train MSE = {mse_method1_train[-1]}, Test MSE = {mse_method1_test[-1]}")
        print(f"optimal_and_objection_Test, n = {n}, T = {T}, Train MSE = {mse_gan_train[-1]}, Test MSE = {mse_gan_test[-1]}")
        print(f"GAN, n = {n}, T = {T}, Train MSE = {mse_gan_train1[-1]}, Test MSE = {mse_gan_test1[-1]}")

    #fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting results for training set
    # df_train = pd.DataFrame()
    # df_train['KRR_Train'] = mse_method1_train
    # df_train['GAN_Train'] = mse_gan_train
    # df_train.index = n_values

    # sns.lineplot(data=df_train, marker='o', ax=axes[0])
    # title_text = f'MSE comparison on Training Set (Covariate Dim: {covariate_dim})'
    # axes[0].set_xlabel('Number of covariate points (n)')
    # axes[0].set_ylabel('MSE')
    # axes[0].set_title(title_text, fontsize=14)
    # axes[0].grid(True)
    # axes[0].legend(title="Method")

    # # Plotting results for test set
    # df_test = pd.DataFrame()
    # df_test['KRR_Test'] = mse_method1_test
    # df_test['GAN_Test'] = mse_gan_test
    # df_test.index = n_values
    # title_text = f'MSE comparison on Test Set (Covariate Dim: {covariate_dim})'
    # sns.lineplot(data=df_test, marker='o', ax=axes[1])
    # axes[1].set_xlabel('Number of covariate points (n)')
    # axes[1].set_ylabel('MSE')
    # axes[1].set_title(title_text, fontsize=14)
    # axes[1].grid(True)
    # axes[1].legend(title="Method")

    # # Display the plot
    # plt.tight_layout()
    # plt.show()


    # Plotting results for training set
    # df_train = pd.DataFrame()
    # df_train['KRR_Train'] = mse_method1_train
    # df_train['GAN_Train'] = mse_gan_train
    # df_train.index = n_values

    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=df_train, marker='o')
    # plt.xlabel('Number of covariate points (n)')
    # plt.ylabel('MSE')

    # plt.title('MSE comparison between KRR and GAN on Training Set')
    # plt.grid(True)
    # plt.legend(title="Method")
    # plt.show()

    # Plotting results for test set
    df_test = pd.DataFrame()
    df_test['KRR_Test'] = mse_method1_test
    df_test['optimal_and_objection_Test'] = mse_gan_test
    df_test['GAN'] = mse_gan_test1
    df_test.index = n_values
    title_text = f'MSE comparison on Test Set (Covariate Dim: {covariate_dim})'
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_test, marker='o')
    plt.xlabel('Number of covariate points (n)')
    plt.ylabel('MSE')
    plt.title(title_text, fontsize=14)
    plt.grid(True)
    plt.legend(title="Method")
    plt.show()




# # Main script
# if __name__ == '__main__':
#     lowbound = 0
#     upbound = 8
#     total_budget = 5000
#     eta_0 = 0.1
#     gamma = 0.01
#     covariate_dim = 8
#     num_samples = 1000     #随噪声项
#     num_test_points = 1000
#     np.random.seed(1)
#     test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    
#     # Split test_points into train and test sets
#     split_index = len(test_points) // 2
#     train_points = test_points[:split_index]
#     test_points = test_points[split_index:]

#     true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
#     test_points = torch.tensor(test_points).float()
#     n_values = [10]
#     T = 500
#     num_iterations = 30
#     eta_G = 0.001
#     eta_D = 0.001
#     weight_decay = 1e-4  # L2 正则化系数

#     mse_method1 = []
#     mse_gan = []

#     for n in n_values:
#         covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv = offline_stage(n, T, eta_0, gamma, covariate_dim)
#         print(mean_squared_error(covariates_points, theta_estimates))
#         # Initialize Generator and Discriminator
#         generator = Generator(covariate_dim, covariate_dim)
#         discriminator = Discriminator(1)

#         # Train GAN using train_points
#         trained_generator = adversarial_learning(generator, discriminator, train_points, covariates_points, theta_estimates, K_phi_inv, K_phi_theta_inv, num_iterations, eta_G, eta_D, weight_decay=weight_decay)

#         # Predict using Method 1 (KRR) and GAN
#         predicted_theta_values_method1 = np.array([optimal_solution_prediction(covariates_points, theta_estimates, x.numpy(), K_phi_inv) for x in test_points])
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
#     df['KRR'] = mse_method1
#     df['KRR_GAN'] = mse_gan
#     df.index = n_values
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, marker='o')
#     plt.xlabel('Number of covariate points (n)')
#     plt.ylabel('MSE')
#     plt.title('MSE comparison between KRR and KRR-GAN with L2 regularization')
#     plt.grid(True)
#     plt.show()