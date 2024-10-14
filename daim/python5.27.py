import numpy as np

import matplotlib.pyplot as plt 
import sklearn.gaussian_process as gp
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import seaborn as sns
def testfun(theta,x ):
    
    
    loss = np.sum((theta-x)**2)
    noise = np.random.normal(0, noise_std, 1)[0]
    return loss + noise

def gradient_descent(initial_theta, x, num_iters, tol=1e-6):
    
    theta = initial_theta.copy()
    n = len(theta)
    
    for i in range(num_iters):
        # 计算梯度
        eps = 1e-6
        grad = np.zeros_like(theta)
        for j in range(n):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            grad[j] = (testfun(theta_plus,x) - testfun(theta_minus, x)) / (2 * eps)
        
        # 更新参数
        alpha = 1/ (i+1)
        theta = theta - alpha * grad
        
        # 检查停止标准
        if np.linalg.norm(grad) < tol:
            break
    return theta



def descent_kw(x):
    
    initial_theta = np.ones(d)
    optimized_theta = np.zeros((replication, d))
    for i in range(replication):
        optimized_theta[i,:] = gradient_descent(initial_theta, x , num_iters,tol=1e-6)
    optimized_theta_avg = np.mean(optimized_theta,axis= 0)
    return optimized_theta_avg

def generate(m1):

    """input:numbers ; output: x and theta""" 
    consvallw = np.full((d,),0)  
    consvalup = np.full((d,),up_bound)  
    dnum = np.full((d,),10)
    vallist = dict()  
    for di in range(d):
        vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
    X1 , theta1 = np.zeros((m1,d)) , np.zeros((m1,d)) 
    for di in range(d):
        X1[:,di] = np.random.choice(vallist[di],size=(m1,))
    
    # X1[0] = np.zeros(d)
    # print(X1)
    for i in range(m1):
        theta1[i] = descent_kw(X1[i])
    return X1 ,theta1
def generate_m2(m2):
    X2 = np.random.uniform([up_bound]*m2 *d).reshape(m2 ,d)
    theta2 = np.zeros((m2,d)) 
    for i in range(m2):
        theta2[i] = X2[i]  ##descent_kw(X2[i])
    return X2 ,theta2

def k_nearest_neighbor(X1, theta1 , X2, theta2,K ):
    neigh = NearestNeighbors(n_neighbors = K)
    neigh.fit(X1)
    distances, indices =neigh.kneighbors(X2)
    optimal_delta = np.zeros(m2)
    optimal_delta_k = np.zeros(m2)
    for j in range(m2):
        x_neighbor_index = indices[j]

        ######### k_mean
        results= np.mean(np.array([theta1[t] for t in x_neighbor_index]),axis= 0)
        
        optimal_delta_k[j] = testfun(results,X2[j]) - testfun(theta2[j],X2[j]) 


    # optimal_delta.sort()
    # flag = optimal_delta[int((1 - alpha)*m2)]
    #optimal_delta_k.sort()
    #flag_k = optimal_delta_k[int((1 - alpha)*m2)]
    
    return optimal_delta_k


if __name__ == '__main__':
    d =  2
    n = 50
    m2= 100
    up_bound  =  4
    replication = 100
    num_iters = 1000
    noise_std = 0.001
    X1, theta1 =  generate(n)
    X2 ,theta2 =  generate_m2(m2)
    optimal_delta_knn= k_nearest_neighbor(X1, theta1 , X2, theta2, 5)
    optimal_delta_knn1= k_nearest_neighbor(X1, theta1 , X2, theta2, 10)
    optimal_delta_knn2= k_nearest_neighbor(X1, theta1 , X2, theta2, 15)
    data = pd.DataFrame()
    data[r'K =5'] = optimal_delta_knn #[np.log2(i) for i in optimal_delta_knn]
    data[r'K =10'] = optimal_delta_knn1
    data[r'K =15'] = optimal_delta_knn2
    sns.set( font_scale = 2)
    sns.displot(data=data, bins=100, kde=True)
    plt.tick_params(labelsize=30)
    plt.legend()
    plt.show()
    


    
  
    
    # for i in range(replication):

    #     optimized_theta[i,:] = gradient_descent(initial_theta, num_iters,tol=1e-6)


    # optimized_theta_avg = np.mean(optimized_theta,axis= 0)

    #print(f"Optimized parameters: {optimized_theta_avg}")
