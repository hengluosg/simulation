
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
        actions0[di] =  np.random.choice(vallist[di],size=1)
    
    if type(theta) == str:
        x0 = actions0
        
    else:
        
        x0 = theta
   


    def objective_function(x):
        return testfun(x, tm)

    result = minimize(objective_function, x0, method='BFGS')
    return result.x



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
        theta1[i] = descent_kw(X1[i], "initialize")
    return X1 ,theta1


def generate_m2(m2):
    #np.random.seed(2)
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
    plt.xlabel(r'$log_{2}(n)$',fontsize=30) 
    plt.title(r'$dimension = {} $'.format(d), fontsize=30)
    plt.tick_params(labelsize=30)
    plt.show()


def picture_plot1(df):
    sns.set( font_scale = 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, column in enumerate(df):
        sns.lineplot(data=column,ax = axes[i],marker='o', linestyle='-',markersize=10)
        number = 100 * (1- alpha)
        axes[i].set_ylabel(r'$log_{2}(\Delta_{%d\%%})$' % number, fontsize=10)
        #plt.xticks(m, labels)
        
        
        #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        axes[i].set_xlabel(r'$log_{2}(n)$',fontsize=10) 
        axes[i].set_title(r'$d = {} $'.format(d1[i]), fontsize=10)
        axes[i].tick_params(labelsize=10)
        #axes[i].tight_layout()
    plt.savefig("1-4.png")
    plt.show()








class exp_kernel:
    def __init__(self ,length = 1,sigma_f = 1):
        self.length = length
        self.sigma_f = sigma_f
    def __call__(self , x1 ,x2):
        y = np.linalg.norm(x1-x2)
        return float(self.sigma_f*np.exp(-y**2 / (2*self.length*2)))
  
class matern_kernel:
    def __init__(self ,l=1.0, nu=1.5):
        self.l = l
        self.nu= nu
    def __call__(self , x1 ,x2):
        dist = np.linalg.norm(x1 - x2)
        #term1 = (2**(1 - self.nu)) / np.math.gamma(self.nu)
        term2 = (np.sqrt(2 * self.nu) * dist) / self.l
        term1 = (np.sqrt(2 * self.nu) * dist**2) / (3*self.l**2)

        term3 = np.exp(-term2) 

        #term4 = (1+term2) * term3
        term4 = (1 + term2 + term1) * term3
        
        
        kernel = term4

        return float(kernel)
        







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

def picture_distribution(df):
    sns.set()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, column in enumerate(df):
        d, m1 , optimal_delta_krr_test,flag_test, off_flag = df[i]
        
        


        number = 100 * (1- alpha)


        ax = sns.histplot(data=optimal_delta_krr_test,ax = axes[i],bins=100,kde=True)
        #sns.displot(optimal_delta_krr_test,ax = axes[i],bins=100)
        #sns.kdeplot(optimal_delta_krr_test, fill=True)
        kde_data = ax.get_lines()[0].get_data()
        ax.fill_between(kde_data[0], kde_data[1], alpha=0.3,color='skyblue')
        axes[i].set_xlabel(r'$\Delta$',fontsize=10) 
        axes[i].set_title(r'Online:Distribution of '+'$ \Delta$'+' with n_test =%d,n = %d,d = %d'% (len(optimal_delta_krr_test),m1,d))
        axes[i].axvline(off_flag, 0, 0.5,color='r',label = r'$Offline:\Delta_{%d\%%}$'% number)
        axes[i].axvline(flag_test, 0, 0.5,color='g',label = r'$Online:\Delta_{%d\%%}$'% number)
        axes[i].tick_params(labelsize=10)
        axes[i].legend()
        pro= np.sum([int(i < off_flag)  for i in optimal_delta_krr_test]) / len(optimal_delta_krr_test)
        print(pro,off_flag)
    #plt.savefig("1-5.png")
    plt.show()

    # 


if __name__ == '__main__':
    

    #lambda1 = 1e-7
    d =  10
    alpha =0.05
    up_bound = 3
    m1 , m2 ,m3= 300 ,200 ,200
    index_num = 1
    # X1 ,theta1 =  generate(m1)
    # X2 ,theta2 =  generate_m2(m2)
    # optimal_delta_nn=  nearest_neighbor(X1, theta1 , X2, theta2,K = 1)
    # optimal_delta_krr= krr(X1, theta1 , X2, theta2)


    # picture_plot(optimal_delta_krr)
    # picture_plot(optimal_delta_nn)
    #m = [50,100,150,200,250,300,350,400,450,500,550,600,650,700]
    data1 = []
    t = 11
    flag = np.arange(3, t, 1)
    
    labels = ['$2^{%s}$' % i for i in flag]
    m = [2**i for i in flag]
    
    replication = 1
    optimal_delta_nn = np.ones(len(m))
    optimal_delta_knn= np.ones(len(m))
    optimal_delta_krr= np.ones(len(m))
    optimal_delta_gpr= np.ones(len(m))
    optimal_delta_nn1 = np.ones((replication ,len(m)))
    optimal_delta_knn1= np.ones((replication ,len(m)))
    optimal_delta_krr1= np.ones((replication ,len(m)))
    optimal_delta_gpr1= np.ones((replication ,len(m)))



    vallist_para= dict()
    d1 = [4,6,8]
    for t ,d in enumerate(d1):
        
        for j in range(replication):
            for i ,m1 in enumerate(m):
                X1 ,theta1 =  generate(m1)
                X2 ,theta2 =  generate_m2(m2)


                ############################################################
                # optimal_delta_nn[i]=  nearest_neighbor(X1, theta1 , X2, theta2)
                # optimal_delta_knn[i] = k_nearest_neighbor(X1, theta1 , X2, theta2,K= 6)
            
                # model1  = KRR(X1, theta1 , X2, theta2,lambda1 = 1e-3/m1)
                # optimal_delta_krr[i] = model1.predict()
                # # model  = GPR(X1, theta1 , X2, theta2,noise = 1e-3/m1)
                # # optimal_delta_gpr[i] = model.predict()


                # #parameter = []
                # if m1 == m[index_num] and j == (replication - 1):
                #     X3 ,theta3 =  generate_m2(m3)
                    
                #     model1  = KRR(X1, theta1 , X3, theta3,lambda1 = 1e-3/m1)
                #     flag_test, optimal_delta_krr_test =model1.predict_distribution()
                #     off_flag = optimal_delta_krr[i]
                #     parameter  = [d,m1 , optimal_delta_krr_test,flag_test,off_flag]
                    
                    #picture_distribution(d,m1 , off_flag, optimal_delta_krr_test,flag_test)

                #######################################################################




                optimal_delta_nn[i]=  nearest_neighbor(X1, X1 , X2, theta2)
                optimal_delta_knn[i] = k_nearest_neighbor(X1, X1 , X2, theta2,K= 6)
            
                model1  = KRR(X1, X1 , X2, theta2,lambda1 = 1e-3/m1)
                optimal_delta_krr[i] = model1.predict()
                # model  = GPR(X1, theta1 , X2, theta2,noise = 1e-3/m1)
                # optimal_delta_gpr[i] = model.predict()


                #parameter = []
                if m1 == m[index_num] and j == (replication - 1):
                    X3 ,theta3 =  generate_m2(m3)
                    
                    model1  = KRR(X1, X1 , X3, theta3,lambda1 = 1e-3/m1)
                    flag_test, optimal_delta_krr_test =model1.predict_distribution()
                    off_flag = optimal_delta_krr[i]
                    parameter  = [d,m1 , optimal_delta_krr_test,flag_test,off_flag]

                
            # optimal_delta_nn1[j,:] =  [np.log2(i) for i in optimal_delta_nn]
            # optimal_delta_knn1[j,:] =  [np.log2(i) for i in optimal_delta_knn]
            # optimal_delta_krr1[j,:] =  [np.log2(i) for i in optimal_delta_krr]
            # optimal_delta_gpr1[j,:] =  [np.log2(i) for i in optimal_delta_gpr]
            optimal_delta_nn1[j,:] =  optimal_delta_nn
            optimal_delta_knn1[j,:] =  optimal_delta_knn
            optimal_delta_krr1[j,:] =  optimal_delta_krr
            #optimal_delta_gpr1[j,:] =  optimal_delta_gpr
        
        
            

        # print(optimal_delta_krr1)
        # print(np.mean(optimal_delta_krr1,axis =0))
        data = pd.DataFrame()
        data.index = [np.log2(i) for i in m]
        # data['nearest neighbor'] =  [np.log2(i) for i in optimal_delta_nn]
        # data['K nearest neighbor'] = [np.log2(i) for i in optimal_delta_knn]
        # data['kernel ridge regression'] = [np.log2(i) for i in optimal_delta_krr]
        # data['gaussian process regression'] = [np.log2(i) for i in optimal_delta_gpr]

        

        # print(optimal_delta_nn1,optimal_delta_knn1,optimal_delta_krr1,optimal_delta_gpr1)
        data['NN'] =  [np.log2(i) for i in np.mean(optimal_delta_nn1,axis =0)]
        data['KNN'] = [np.log2(i) for i in np.mean(optimal_delta_knn1,axis =0)]  
        data['KRR'] = [np.log2(i) for i in np.mean(optimal_delta_krr1,axis =0)] 
        #data['GPR'] = [np.log2(i) for i in np.mean(optimal_delta_gpr1,axis =0)]

        # data['NN'] =  np.mean(optimal_delta_nn1,axis =0)
        # data['KNN'] = np.mean(optimal_delta_knn1,axis =0)
        # data['KRR'] = np.mean(optimal_delta_krr1,axis =0)
        # data['GPR'] = np.mean(optimal_delta_gpr1,axis =0)
        #picture_plot(data
        data1.append(data)


        # parameter.append(data['KRR'][7-1])
        #off_flag =  np.mean(optimal_delta_krr1,axis =0)[4]
        #parameter.append(off_flag)
        vallist_para[t] =  parameter
    picture_distribution(vallist_para)
    picture_plot1(data1)
    

    

    

   











    

    
  


   



    