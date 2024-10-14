
import numpy  as np
import matplotlib.pyplot as plt 
import sklearn.gaussian_process as gp
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import seaborn as sns

# def testfun(x,tm):

#     y = np.sum((x-(tm-2)**2)**2)
#     return y 

def testfun(x,tm):

    y = np.sum((x-tm)**2)
    return y 

# def testfun(x,tm):
#     d = x.shape[0]
#     sum1 = [(i+1)* (x[i]-np.sqrt(tm[i]))**2 for i in range(d)]

#     y = np.sum(sum1)
#     return y 

# def testfun(x,tm):

#     y1 = 10 * d  
#     y2 = np.sum((x-tm)**2) 
#     y3 = np.sum(10 *np.cos(2 * np.pi*(x - tm )))

#     return y1 +y2 -y3



# def testfun(x,tm):
#     sum1 = [(x[i]-np.sqrt(tm[i]))**2 for i in range(d)]

#     y = np.sum(sum1)
#     return y 

# def testfun(x,tm):
#     sum1 = [(x[i]-np.sqrt(tm[i]))**2 for i in range(d)]

#     y = np.sum(sum1)
#     return y 
# def testfun(x,tm):
#     sum1 = tm * np.sin(1/np.linalg.norm(tm ,2 ))

    

#     y = np.sum((x-tm)**2)
#     return y 



def descent_kw(tm , theta):
    d = tm.shape[0]

    consvallw = np.full((d,),0)  # lower bound of solution's value

    consvalup = np.full((d,),up_bound)  # upper bound of solution's value

    dnum = np.full((d,),201)

    vallist = dict()  # dictionary to store possible values in each dimension
    for di in range(d):
        vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
        

    actions0= np.zeros((d))
    for di in range(d):
        actions0[di] =  np.random.choice(vallist[di],size=1).item()
    
    if type(theta) == str:
        x0 = actions0
        
    else:
        
        x0 = theta
   


    def objective_function(x):
        return testfun(x, tm)

    result = minimize(objective_function, x0, method='BFGS')
    return result.x



# def generate(m1):

#     """input:numbers ; output: x and theta""" 
#     consvallw = np.full((d,),0)  
#     consvalup = np.full((d,),up_bound)  
#     dnum = np.full((d,),20)
#     vallist = dict()  
#     for di in range(d):
#         vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
#     X1 , theta1 = np.zeros((m1,d)) , np.zeros((m1,d)) 
#     for di in range(d):
#         X1[:,di] = np.random.choice(vallist[di],size=(m1,))
    
#     # X1[0] = np.zeros(d)
#     # print(X1)
#     for i in range(m1):
#         theta1[i] = descent_kw(X1[i], "initialize")
#     return X1 ,theta1


def get_grid_points(num_divisions, ranges):
    
    if not ranges:
        return []
    
    current_range = ranges[0]
    min_val, max_val = current_range
    min_val, max_val =  min_val+0.2, max_val-0.2
    step = (max_val - min_val) / (num_divisions - 1)
    
    current_coords = [min_val + i * step for i in range(num_divisions)]
    
    if len(ranges) == 1:
        return [[coord,] for coord in current_coords]
    else:
        sub_points = get_grid_points(num_divisions, ranges[1:])
        return [[coord,] + sub_point for coord in current_coords for sub_point in sub_points]
def generate(m1):
    X1 , theta1 = np.zeros((m1,d)) , np.zeros((m1,d))
    ten_dim_points = get_grid_points(4, [(0, up_bound)] * d)
    length = len(ten_dim_points)
    index = np.random.randint(0, length, size=m1)
    for i in range(m1):

        X1[i] =  np.array(ten_dim_points[index[i]])
        theta1[i] = descent_kw(X1[i], "initialize")
    
    return X1 ,theta1









def generate_m2(m2):
    X2 = np.random.uniform([up_bound]*m2 *d).reshape(m2 ,d)
    theta2 = np.zeros((m2,d)) 
    for i in range(m2):
        theta2[i] = descent_kw(X2[i], "initialize")
    return X2 ,theta2


def k_nearest_neighbor(X1, theta1 , X2, theta2,K ):
    neigh = NearestNeighbors(n_neighbors = K)
    neigh.fit(X1)
    distances, indices =neigh.kneighbors(X2)
    optimal_delta = np.zeros(m2)
    optimal_delta_k = np.zeros(m2)
    for j in range(m2):
        x_neighbor_index = indices[j]

        # ######### pick one best
        # results = [testfun(theta1[t],X2[j]) for t in x_neighbor_index]
        # optimal_delta[j] = np.min(results) - testfun(theta2[j],X2[j]) 
        # #########
        

        
        
        ######### k_mean
        results= np.mean(np.array([theta1[t] for t in x_neighbor_index]),axis= 0)
        #print(np.array([theta1[t] for t in x_neighbor_index]),results, theta2[j])
        optimal_delta_k[j] = testfun(results,X2[j]) - testfun(theta2[j],X2[j]) 


    # optimal_delta.sort()
    # flag = optimal_delta[int((1 - alpha)*m2)]
    optimal_delta_k.sort()
    flag_k = optimal_delta_k[int((1 - alpha)*m2)]
    
    return flag_k





def nearest_neighbor(X1, theta1 , X2, theta2,K=1):
 

    neigh = NearestNeighbors(n_neighbors = K)
    neigh.fit(X1)
    distances, indices =neigh.kneighbors(X2)

    optimal_delta = np.zeros(m2)

    for j in range(m2):
        x_neighbor_index = indices[j]
        
        temp  = theta1[x_neighbor_index[0]]
        optimal_delta[j] =testfun(temp,X2[j])  - testfun(theta2[j],X2[j]) 
        #print("nn",temp,theta2[j])
    optimal_delta.sort()
    flag = optimal_delta[int((1 - alpha)*m2)]
    return flag


# def nearest_neighbor(X1, theta1 , X2, theta2,K=1):
 

#     neigh = NearestNeighbors(n_neighbors = K)
#     neigh.fit(X1)
#     distances, indices =neigh.kneighbors(X2)

#     optimal_delta = np.zeros(m2)

#     for j in range(m2):
#         x_neighbor_index = indices[j]
        
#         temp  = theta1[x_neighbor_index[0]]
#         optimal_delta[j] =testfun(X2[j],temp)  - testfun(X2[j],theta2[j]) 

    
#     return optimal_delta



def gaussian_kernel_function(x1,x2):
    sigma = 1
    y = np.linalg.norm(x1-x2)
    kernel = np.exp(-y**2 / (2*sigma**2))
    return kernel

def K_hat_inv(n,X):
    K_hat = np.zeros((n,n))    
    for i in range(n):
        for j in range(n):
            K_hat[i,j] = gaussian_kernel_function(X[i],X[j])
    temp = K_hat + lambda1 *n * np.eye(N=n,k=0)
    temp_inv = np.linalg.inv(temp) 
    return temp_inv



def krr(X1, theta1 , X2, theta2):
    n = len(X1)
    optimal_delta = np.zeros(m2)
    K_hat_inv_ = K_hat_inv(n,X1)
    for i, x0 in enumerate(X2):
        z0 = np.zeros((1,d))
        for j in range(d):
            a = K_hat_inv_ @ theta1[:,j] 
            z0[:,j] = np.array(sum([gaussian_kernel_function(X1[i],x0)*a[i] for i in range(n)]) )
        
        optimal_delta[i] = testfun(z0.T ,X2[i])  - testfun(theta2[i] ,X2[i]) 

    optimal_delta.sort()
    flag = optimal_delta[int((1 - alpha)*m2)]
    return flag


# def calculate_delta(X1,theta1):
#     K = 1
#     m1 = len(X1)
#     delta = np.zeros(m1)
#     neigh = NearestNeighbors(n_neighbors = K+1)
#     neigh.fit(X1)
#     distances, indices =neigh.kneighbors(X1)
#     for j in range(m1):
#         x_neighbor_index = indices[j][1:]
#         temp  = theta1[x_neighbor_index[0]]
#         delta[j] =testfun(X1[j],temp)  - testfun(X1[j],theta1[j]) 
    
#     return delta


def picture_plot(data):
    sns.set( font_scale = 2)
    
    


    #sns.histplot(data=data_picture1,kde=True)
    
    #sns.lineplot(data=data, markers='o', markersize=10)
    sns.lineplot(data=data,marker='o', linestyle='-',markersize=10)
    number = 100 * (1- alpha)
    plt.ylabel(r'$log_{2}(\Delta_{%d\%%})$' % number, fontsize=40)
    #plt.xticks(m, labels)
    #plt.tight_layout()
    
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.xlabel(r'$log_{2}(m1)$',fontsize=30) 
    plt.title(r'$dimension = {} $'.format(d), fontsize=30)
    plt.tick_params(labelsize=30)
    plt.show()


class exp_kernel:
    def __init__(self ,length = 1,sigma_f = 1):
        self.length = length
        self.sigma_f = sigma_f
    def __call__(self , x1 ,x2):
        y = np.linalg.norm(x1-x2)
        return float(self.sigma_f*np.exp(-y**2 / (2*self.length*2)))
def K_matrix(x1,x2,kernel_function):
    return np.array([[kernel_function(a,b) for a in x1] for b in x2])

class GPR:
    def __init__(self ,data_x ,data_y ,online_x ,online_y ,kernel_function = exp_kernel(),noise = 1e-7 ):
        self.data_x = data_x
        self.data_y = data_y
        self.online_x = online_x
        self.online_y = online_y
        self.kernel_function = kernel_function
        self.noise = noise
        self.cov_matrix_inv = np.linalg.inv(K_matrix(data_x ,data_x , kernel_function) + (noise )* np.identity(len(data_x)))
    def predict(self):
        K_data_x = K_matrix(self.data_x , self.online_x ,self.kernel_function)  # m ,n
        K_x_x = K_matrix(self.online_x , self.online_x ,self.kernel_function)           #m ,m 
        mean = (K_data_x @ self.cov_matrix_inv @ self.data_y) 
        matrix = K_x_x - K_data_x @ self.cov_matrix_inv @  K_data_x.T

        optimal_delta = np.zeros(m2)

        for j in range(m2):
            optimal_delta[j] =testfun(mean[j] ,self.online_x[j])  - testfun(self.online_y[j],self.online_x[j]) 
        optimal_delta.sort()
        flag = optimal_delta[int((1 - alpha)*m2)]
        return flag

class KRR:
    def __init__(self ,data_x ,data_y ,online_x ,online_y ,kernel_function = exp_kernel(),lambda1 = 1e-7 ):
        self.data_x = data_x
        self.data_y = data_y
        self.online_x = online_x
        self.online_y = online_y
        self.kernel_function = kernel_function     
        self.cov_matrix_inv = np.linalg.inv(K_matrix(data_x ,data_x , kernel_function) + (lambda1 )* len(data_x) *np.identity(len(data_x)))
    def predict(self):
        K_data_x = K_matrix(self.data_x , self.online_x ,self.kernel_function)  # m ,n
        descion_z = (K_data_x @ self.cov_matrix_inv @ self.data_y) 
        
        optimal_delta = np.zeros(m2)

        for j in range(m2):
            optimal_delta[j] =testfun(descion_z[j] ,self.online_x[j])  - testfun(self.online_y[j],self.online_x[j]) 
        optimal_delta.sort()
        flag = optimal_delta[int((1 - alpha)*m2)]
        return flag
    
    def predict_distribution(self):
        K_data_x = K_matrix(self.data_x , self.online_x ,self.kernel_function)  # m ,n
        descion_z = (K_data_x @ self.cov_matrix_inv @ self.data_y) 
        
        optimal_delta = np.zeros(m3)

        for j in range(m3):
            optimal_delta[j] =testfun(descion_z[j] ,self.online_x[j])  - testfun(self.online_y[j],self.online_x[j]) 
        
        
        optimal_delta1 =  optimal_delta.copy()
        optimal_delta.sort()
        flag = optimal_delta[int(np.ceil((1 - alpha)*m3))]
        return flag , optimal_delta1



def picture_distribution(d, m1 , off_flag, optimal_delta_krr_test,flag_test):
    sns.set()

    number = 100 * (1- alpha)

    #plt.ylabel(r'$log_{2}(\Delta_{%d\%%})$' % number, fontsize=40)
    
    data_picture = pd.DataFrame()

    data_picture[r'$\Delta$'] = optimal_delta_krr_test
    
    sns.histplot(data=data_picture)

    plt.xlabel(r'$\Delta$',fontsize=30) 
    #plt.title(r'Online:Distribution of '+'$ \Delta$'+' with n_test =500,n = %d,d = %d'% (m1,d))
    plt.title(r'Online: Distribution of $ \Delta$ with n_test =500, n = %d, d = %d' % (m1, d))

    
    plt.axvline(off_flag, 0, 0.5,color='r',label = r'$Offline:\Delta_{%d\%%}$'% number)
    plt.axvline(flag_test, 0, 0.5,color='g',label = r'$Online:\Delta_{%d\%%}$'% number)

    plt.tick_params(labelsize=30)
    plt.legend()
    plt.show()

    pro= np.sum([int(i < off_flag)  for i in optimal_delta_krr_test]) / len(optimal_delta_krr_test)
    print(pro,off_flag)


if __name__ == '__main__':
    

    #lambda1 = 1e-7
    d =  6
    alpha =0.05
    up_bound = 4
    m1 , m2 ,m3= 300 ,200,500
    # X1 ,theta1 =  generate(m1)
    # X2 ,theta2 =  generate_m2(m2)
    # optimal_delta_nn=  nearest_neighbor(X1, theta1 , X2, theta2,K = 1)
    # optimal_delta_krr= krr(X1, theta1 , X2, theta2)


    # picture_plot(optimal_delta_krr)
    # picture_plot(optimal_delta_nn)
    #m = [50,100,150,200,250,300,350,400,450,500,550,600,650,700]
    t = 11
    flag = np.arange(6, t, 1)
    labels = ['$2^{%s}$' % i for i in flag]
    m = [2**i for i in flag]
    
    optimal_delta_nn = np.zeros(len(m))
    optimal_delta_knn= np.zeros(len(m))
    optimal_delta_krr= np.zeros(len(m))
    optimal_delta_gpr= np.zeros(len(m))
    for i ,m1 in enumerate(m):
        X1 ,theta1 =  generate(m1)
        X2 ,theta2 =  generate_m2(m2)
        optimal_delta_nn[i]=  nearest_neighbor(X1, theta1 , X2, theta2)
        optimal_delta_knn[i] = k_nearest_neighbor(X1, theta1 , X2, theta2,K= 10)
       
        model1  = KRR(X1, theta1 , X2, theta2,lambda1 = 1e-7/m1)
        optimal_delta_krr[i] = model1.predict()
        model  = GPR(X1, theta1 , X2, theta2,noise = 1e-7/m1)
        optimal_delta_gpr[i] = model.predict()

   


        if int(np.log2(m1))== 7:
            X3 ,theta3 =  generate_m2(m3)
            
            model1  = KRR(X1, theta1 , X3, theta3,lambda1 = 1e-7/m1)
            flag_test, optimal_delta_krr_test =model1.predict_distribution()
            off_flag = optimal_delta_krr[i]
            
            picture_distribution(d,m1 , off_flag, optimal_delta_krr_test,flag_test)

        
    data = pd.DataFrame()
    data.index = [np.log2(i) for i in m]
    data['nearest neighbor'] =  [np.log2(i) for i in optimal_delta_nn]
    data['K nearest neighbor'] = [np.log2(i) for i in optimal_delta_knn]
    data['kernel ridge regression'] = [np.log2(i) for i in optimal_delta_krr]
    #data['gaussian process regression'] = [np.log2(i) for i in optimal_delta_gpr]
    picture_plot(data)
    
    



    

    

   











    

    
  


   



    