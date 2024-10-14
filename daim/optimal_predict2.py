import numpy  as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import r2_score
# import cvxpy as cp

#from keras.models import Sequential
#from keras.layers import Dense, Activation
from sklearn.metrics import mean_squared_error,r2_score



# def testfun(x,tm):

#     z = A @ (x-tm)
#     y = np.sum(z**2)
#     return y
    
# def testfun(x,tm):
#     z = A @ (x-tm)
#     y1 ,y2= 0, 0
#     for i in range(d):
#         y1 += np.cos((z[i])/np.sqrt(i+1))
#     y2 = np.sum(z**2) /4000
#     y = y2 - y1 + 1
#     return y


# def testfun(x,tm):
#     z = A @ (x-tm)

#     y1 = 10 * d
#     y2 = np.sum(z**2)
#     y3 = np.sum(10 *np.cos(2 * np.pi*(z )))

#     return y1 +y2 -y3



# def testfun(x,tm):

#     y = np.sum((x-tm)**2)
#     return y


# def testfun(x,tm):
    
#     y1 ,y2= 0, 0
#     for i in range(d):
#         y1 += np.cos((x[i]-tm[i])/np.sqrt(i+1))
#     y2 = np.sum((x-tm)**2) /4000
#     y = y2 - y1 + 1
#     return y


def testfun(x,tm):

    y1 = 10 * d
    y2 = np.sum((x-tm)**2)
    y3 = np.sum(10 *np.cos(2 * np.pi*(x - tm )))

    return y1 +y2 -y3



def gaussian_kernel_function(x1,x2):
    sigma = 1
    y = np.linalg.norm(x1-x2)
    kernel = np.exp(-y**2 / (2*sigma**2))
    return kernel


def K_hat_(X):
    n = len(X)
    K_hat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K_hat[i,j] = gaussian_kernel_function(X[i],X[j])
    return K_hat

def K_hat_inv(n,K_hat,lambda1):
    temp = K_hat + lambda1 *n * np.eye(N=n,k=0)
    temp_inv = np.linalg.inv(temp)
    return temp_inv

def K_X_x0_(X,x0):
    K_X_x0 = np.array([gaussian_kernel_function(X[i],x0) for i in range(len(X))])
    return K_X_x0



def Objective_Prediction(theta,X):
    Y_theta_X = np.zeros((len(theta),n))
    for j in range(len(theta)):
        Y_theta_X[j] = np.array([testfun(theta[j],X[i]) for i in range(n)])
    
    h_theta_x0 = K_hat.T @ K_hat_inv_ @ Y_theta_X.T
    
    return h_theta_x0

def descent_kw(tm , theta):    # tm  is x ;
    
    
    
    d = tm.shape[0]

    consvallw = np.full((d,),0)  # lower bound of solution's value

    consvalup = np.full((d,),up_bound/d)  # upper bound of solution's value

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
    
    
    N = 500

    dim  =  x0.shape[0]
    
    e = np.ones(dim)
    
    e = np.diag(e)
    

    for n in range(1 ,N+1):
        
        a_n = n**(-1)
        
        c_n = n**(-1/3)
        
        
        delta  = [testfun(x0 + c_n * e[i], tm) - testfun(x0 , tm) for i in range(dim)]
        
        
        x1 = x0 - [a_n * t / c_n for  t in delta]
       
        x0 = x1
        
    optimal = testfun(x0, tm)
        
    return x0 ,optimal


def Optimal_Prediction(theta,x0):
    z0 = np.zeros((1,d))
    for j in range(d):
        a = K_hat_inv_ @ theta[:,j]
        z0[:,j] = np.array(sum([gaussian_kernel_function(X[i],x0)*a[i] for i in range(n)]) )
    
    return z0





def algorithm_1(m, k, d):
    ##first stage
    # x = []  # tm
    # for i in range(m):
    #     temp = np.random.uniform([10] * d)
    #     x.append(temp)

    theta = []
    theta_new = [0] * m
    for i in range(m):
        theta.append(descent_kw(x[i], "initialize")[0])

        # theta.append(descent_spsa(x[i] ,"initialize")[0])

    # second stage
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors = k+1)
    neigh.fit(x)

    for j in range(m):
        x_neighbor_index = neigh.kneighbors([x[j]], return_distance=False)

        # x_neighbor = [x[t] for  t  in x_neighbor_index[0]]

        results = [descent_kw(x[j], theta[t]) for t in x_neighbor_index[0]]

        # results = [descent_spsa(x[j],theta[t]) for t  in x_neighbor_index[0]]

        index = np.argmin([x[1] for x in results])

        theta_new[j] = results[index][0]

        # index = np.argmin([solve(x_neighbor[l],theta[j])[1] for l in range(k+1)])
        # theta_new[j] = solve(x_neighbor[index],theta[j])[0]
    print("algorithm_1 is over")
    return np.array(theta_new), np.array(x)






def algorithm_2(theta , x ,num):
    d = x[0].shape[0]
    #num = 1000
    #print(len(x))
    theta2 = [0] * num


    # X =  np.random.uniform([10] *num)
    # X = []
    # for i in range (num):
    #     temp = np.random.uniform([10] *d)
    #     X.append(temp)
    X = X0
    
    for i in range(num):
        index1  =np.argmin([np.linalg.norm(x[j] - X[i], ord = d)   for j in range(len(x))])
        
        theta2[i] = theta[index1]
    
    f = [testfun(theta2[i],X[i]) for i in range(num)]
    print("algorithm_2 is over")
    return theta2 ,f







if __name__ == '__main__':
    #lambda1 = 0.001
    
    k = 2# neighbor nums
    d =  4
    up_bound = 3*d
    #n = 200
    error =[]
    error1 =[]
    error2 =[]
    error3 =[]
    error4 =[]
    op_error =[]
    op_error1 =[]
    op_error2 =[]
    op_error3 =[]
    op_error4 =[]
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i][j] = 1
    
    # n = 100
    # N = 20000
    
    # #np.random.seed(1)
    # true = np.zeros((n,N))
    # random_theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
    # X = np.random.uniform([up_bound/d]*n *d).reshape(n ,d)
    # K_hat_inv_ =  K_hat_inv(n,X)
    # y_pre1 = Objective_Prediction2(random_theta, X)
    

    # for i in range(n):
    #     x0 = X[i]
    #     for j in range(N):
    #         #X_train[j,i] = np.hstack([theta[j],x0])
    #         true[i,j] = testfun(random_theta[j],x0)
    # print("curve fitting r2 score:", r2_score(y_pre1, true))

    num1 = [400,600,800]
    for n in num1:
        sample_length = 500
        #X =  np.random.uniform(0,up_bound/d,size=(n,d))
        consvallw = np.full((d,),0)  # lower bound of solution's value
        consvalup = np.full((d,),up_bound/d)  # upper bound of solution's value
        dnum = np.full((d,),n)
        dnum_sample = np.full((d,),sample_length)
        vallist = dict()  # dictionary to store possible values in each dimension
        vallist1 = dict()  # dictionary to store possible values in each dimension
        for di in range(d):
            vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
            vallist1[di] = np.linspace(consvallw[di],consvalup[di],dnum_sample[di])
        theta = np.array(pd.DataFrame(vallist))
        np.random.seed(3)
        theta = np.random.uniform([up_bound/d]*n *d).reshape(n ,d)
        X0 = np.array(pd.DataFrame(vallist1))
        np.random.seed(10)
        X0 = np.random.uniform([up_bound/d]*sample_length *d).reshape(sample_length ,d)
        

        X = theta.copy()
        x = X    #offline data
        #print(X.shape)
        
        lambda1 = 0.000001

        K_hat = K_hat_(X)

        K_hat_inv_ = K_hat_inv(n,K_hat,lambda1)

        theta_kw, x_kw= algorithm_1(n, k*d, d)
        theta_kw1 = theta_kw.copy()
        
        print(np.linalg.norm(theta_kw - X))

        
        for _ in range(1):
            N = 10000
            #np.random.seed(1)
            true = np.zeros((n,N))
            random_theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
            #y_pre1 = Objective_Prediction(random_theta, X)

            
        
            for i in range(n):
                x0 = X[i]
                for j in range(N):
                    #X_train[j,i] = np.hstack([theta[j],x0])
                    true[i,j] = testfun(random_theta[j],x0)

            y_pre1 = true
            from sklearn.metrics import mean_squared_error
            print("curve fitting r2 score:", r2_score(y_pre1, true))
            print("mse:", mean_squared_error(y_pre1, true))
            

            #print(y_pre1.shape)
            theta_kw2 = theta_kw.copy()
            for i in range(n):
                #print(theta_kw1.shape)
                
                y_pre = y_pre1[i,:]
                the = random_theta[ np.argmin(y_pre) ]

                
                if testfun(theta_kw1[i], X[i]) - testfun(the, X[i])  >0 :
                    print(i)
                    theta_kw1[i] = the
                theta_kw2[i] = the
            print("over")
            print(np.linalg.norm(theta_kw1 - X))
            print(np.linalg.norm(theta_kw2 - X))
        lambda1 = 0.0001
        K_hat_inv_ = K_hat_inv(n,K_hat,lambda1)
        theta2 ,f = algorithm_2(theta_kw, x_kw ,sample_length)
        #error.append(np.mean(f)+ d  -1)   # Griewank function
        error.append(np.mean(f))
        op_error.append(np.linalg.norm(theta2-X0)**2/sample_length)
        
        
        #Predict:use krr algorithm to predict online data
        predict = np.zeros((sample_length,d))  #h_theta_x0
        predict1 = np.zeros((sample_length,d))  #h_theta_x0
        predict2 = np.zeros((sample_length,d))  #h_theta_x0
        predict3 = np.zeros((sample_length,d))  #h_theta_x0
        predict4 = np.zeros((sample_length,d))  #h_theta_x0

        for i in range(sample_length):
            x0 = X0[i]
            predict[i] = Optimal_Prediction(theta_kw,x0)
            predict1[i] = Optimal_Prediction(X,x0)
            #predict2[i] = Optimal_Prediction(z_theta_kw ,x0)
            predict3[i] = Optimal_Prediction(theta_kw1 ,x0)
            predict2[i] = Optimal_Prediction(theta_kw2 ,x0)

        
        print(np.linalg.norm(theta_kw1 - X))
  
        f1 = [testfun(predict[i],X0[i]) for i in range(sample_length)]
        #error1.append(np.mean(f1)+ d  -1)  # Griewank function
        error1.append(np.mean(f1))
        op_error1.append(np.linalg.norm(predict-X0)**2/sample_length)

        f2 = [testfun(predict1[i],X0[i]) for i in range(sample_length)]
        #error1.append(np.mean(f1)+ d  -1)  # Griewank function
        error2.append(np.mean(f2))
        op_error2.append(np.linalg.norm(predict1-X0)**2/sample_length)

        # #print(z_theta_kw.shape,X0.shape)
        f3 = [testfun(predict2[i],X0[i]) for i in range(sample_length)]
        #error1.append(np.mean(f1)+ d  -1)  # Griewank function
        error3.append(np.mean(f3))
        op_error3.append(np.linalg.norm(predict2-X0)**2/sample_length)


        f4 = [testfun(predict3[i],X0[i]) for i in range(sample_length)]
        #error1.append(np.mean(f1)+ d  -1)  # Griewank function
        error4.append(np.mean(f4))
        op_error4.append(np.linalg.norm(predict3-X0)**2/sample_length)
        print("krr_r2 score:", r2_score(predict3, X0))


        

    import seaborn as sns


    sns.set()
    data_picture = pd.DataFrame()
    data_picture.index = [i for i in num1]
    data_picture["the nearest neighbor_predict"] = error

    data_picture[r"$\hat{\theta}^{*}(x)$_krr_predict"] = error1
    data_picture["optimal_krr_predict"] = error2
    data_picture["random_krr_predict"] = error3
    data_picture["object_and_optimal"] = error4

    #colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37"]
    f, axs = plt.subplots(1, 2,figsize=(25, 10))
    sns.lineplot(data=data_picture, ax = axs[0],linewidth=3,marker='o',ms=10,mew=1,mec='k')
    axs[0].set_xlabel("numbers",fontsize=40)
    axs[0].set_ylabel(r'$E_{x}[f(\theta,x) -f(\theta^{*},x)] $', fontsize=30)
    #axs[0].set_title(r'$sphere-dimension = {} $'.format(d),fontsize=30)
    #axs.set_title(r'$griewank-dimension = {} $'.format(d),fontsize=30)
    axs[0].set_title(r'$rastrigin-dimension = {} $'.format(d),fontsize=30)
    #axs.set_title(r'$Schwefel-dimension = {} $'.format(d),fontsize=30)
    axs[0].legend()
    axs[0].tick_params(labelsize=30)
    #plt.savefig('Objective'+'.png',dpi=700)
    #plt.show()






    data_picture1 = pd.DataFrame()
    data_picture1.index = [i for i in num1]
    
    data_picture1["the nearest neighbor_predict"] = op_error

    data_picture1[r"$\hat{\theta}^{*}(x)$_krr_predict"] = op_error1
    data_picture1["optimal_krr_predict"] = op_error2
    data_picture1["random_krr_predict"] =op_error3
    data_picture1["object_and_optimal"] = op_error4
    #sns.set()

    #colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37"]
    #f, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=data_picture1,ax = axs[1], linewidth=3,marker='o',ms=10,mew=1,mec='k')
    axs[1].set_xlabel("numbers",fontsize=40)
    axs[1].set_ylabel(r'$E_{x}(\theta - \theta^{*})^{2} $', fontsize=30)
    #axs[1].set_title(r'$sphere-dimension = {} $'.format(d),fontsize=30)
    #axs[1].set_title(r'$griewank-dimension = {} $'.format(d),fontsize=30)
    axs[1].set_title(r'$rastrigin-dimension = {} $'.format(d),fontsize=30)
    #ax.set_title(r'$Schwefel-dimension = {} $'.format(d),fontsize=30)
    axs[1].legend()
    axs[1].tick_params(labelsize=30)
    plt.savefig('Objective_and'+'_Optimal_{}'.format(d)+'.png',dpi=700)
    #plt.show()









# import numpy  as np
# import matplotlib.pyplot as plt
# import sklearn.gaussian_process as gp
# import pandas as pd
# from timeit import default_timer as timer
# from sklearn.metrics import r2_score
# #import cvxpy as cp
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from sklearn.metrics import mean_squared_error,r2_score
#
#
#
# # def testfun(x,tm):
#
# #     z = A @ (x-tm)
# #     y = np.sum(z**2)
# #     return y
#
# # def testfun(x,tm):
# #     z = A @ (x-tm)
# #     y1 ,y2= 0, 0
# #     for i in range(d):
# #         y1 += np.cos((z[i])/np.sqrt(i+1))
# #     y2 = np.sum(z**2) /4000
# #     y = y2 - y1 + 1
# #     return y
#
#
# # def testfun(x,tm):
# #     z = A @ (x-tm)
#
# #     y1 = 10 * d
# #     y2 = np.sum(z**2)
# #     y3 = np.sum(10 *np.cos(2 * np.pi*(z )))
#
# #     return y1 +y2 -y3
#
#
#
# def testfun(x,tm):
#
#     y = np.sum((x-tm)**2)
#     return y
#
#
# # def testfun(x,tm):
#
# #     y1 ,y2= 0, 0
# #     for i in range(d):
# #         y1 += np.cos((x[i]-tm[i])/np.sqrt(i+1))
# #     y2 = np.sum((x-tm)**2) /4000
# #     y = y2 - y1 + 1
# #     return y
#
#
# # def testfun(x,tm):
#
# #     y1 = 10 * d
# #     y2 = np.sum((x-tm)**2)
# #     y3 = np.sum(10 *np.cos(2 * np.pi*(x - tm )))
#
# #     return y1 +y2 -y3
#
#
#
# def gaussian_kernel_function(x1,x2):
#     sigma = 1
#     y = np.linalg.norm(x1-x2)
#     kernel = np.exp(-y**2 / (2*sigma**2))
#     return kernel
#
#
# def K_hat_(X):
#     n = len(X)
#     K_hat = np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             K_hat[i,j] = gaussian_kernel_function(X[i],X[j])
#     return K_hat
#
# def K_hat_inv(n,K_hat,lambda1):
#     temp = K_hat + lambda1 *n * np.eye(N=n,k=0)
#     temp_inv = np.linalg.inv(temp)
#     return temp_inv
#
# def K_X_x0_(X,x0):
#     K_X_x0 = np.array([gaussian_kernel_function(X[i],x0) for i in range(len(X))])
#     return K_X_x0
#
#
#
# def Objective_Prediction2(theta,X):
#     length = len(K_hat_inv_)
#
#     Y_theta_X = np.zeros((length,n))
#     K_X_x0 = np.zeros((length,len(theta)))
#     for i in range(length):
#         for j in range(len(theta)):
#             K_X_x0[i,j] = gaussian_kernel_function(random_theta[i],theta[j])
#     for j in range(length):
#         Y_theta_X[j] = np.array([testfun(random_theta[j],X[i]) for i in range(n)])
#
#
#     h_theta_x0 = K_X_x0.T @ K_hat_inv_ @ Y_theta_X
#
#     return h_theta_x0.T
#
# def descent_kw(tm , theta):    # tm  is x ;
#
#
#
#     d = tm.shape[0]
#
#     consvallw = np.full((d,),0)  # lower bound of solution's value
#
#     consvalup = np.full((d,),up_bound/d)  # upper bound of solution's value
#
#     dnum = np.full((d,),201)
#
#     vallist = dict()  # dictionary to store possible values in each dimension
#     for di in range(d):
#         vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
#
#
#     actions0= np.zeros((d))
#     for di in range(d):
#         actions0[di] =  np.random.choice(vallist[di],size=1)
#
#     if type(theta) == str:
#         x0 = actions0
#
#     else:
#
#         x0 = theta
#
#
#     N = 500
#
#     dim  =  x0.shape[0]
#
#     e = np.ones(dim)
#
#     e = np.diag(e)
#
#
#     for n in range(1 ,N+1):
#
#         a_n = n**(-1)
#
#         c_n = n**(-1/3)
#
#
#         delta  = [testfun(x0 + c_n * e[i], tm) - testfun(x0 , tm) for i in range(dim)]
#
#
#         x1 = x0 - [a_n * t / c_n for  t in delta]
#
#         x0 = x1
#
#     optimal = testfun(x0, tm)
#
#     return x0 ,optimal
#
#
# def Optimal_Prediction(theta,x0):
#     z0 = np.zeros((1,d))
#     for j in range(d):
#         a = K_hat_inv_1 @ theta[:,j]
#         z0[:,j] = np.array(sum([gaussian_kernel_function(X[i],x0)*a[i] for i in range(n)]) )
#
#     return z0
#
#
#
#
#
# def algorithm_1(m, k, d):
#     ##first stage
#     # x = []  # tm
#     # for i in range(m):
#     #     temp = np.random.uniform([10] * d)
#     #     x.append(temp)
#
#     theta = []
#     theta_new = [0] * m
#     for i in range(m):
#         theta.append(descent_kw(x[i], "initialize")[0])
#
#         # theta.append(descent_spsa(x[i] ,"initialize")[0])
#
#     # second stage
#     from sklearn.neighbors import NearestNeighbors
#     neigh = NearestNeighbors(n_neighbors = k+1)
#     neigh.fit(x)
#
#     for j in range(m):
#         x_neighbor_index = neigh.kneighbors([x[j]], return_distance=False)
#
#         # x_neighbor = [x[t] for  t  in x_neighbor_index[0]]
#
#         results = [descent_kw(x[j], theta[t]) for t in x_neighbor_index[0]]
#
#         # results = [descent_spsa(x[j],theta[t]) for t  in x_neighbor_index[0]]
#
#         index = np.argmin([x[1] for x in results])
#
#         theta_new[j] = results[index][0]
#
#         # index = np.argmin([solve(x_neighbor[l],theta[j])[1] for l in range(k+1)])
#         # theta_new[j] = solve(x_neighbor[index],theta[j])[0]
#     print("algorithm_1 is over")
#     return np.array(theta_new), np.array(x)
#
#
#
#
#
#
# def algorithm_2(theta , x ,num):
#     d = x[0].shape[0]
#     #num = 1000
#     #print(len(x))
#     theta2 = [0] * num
#
#
#     # X =  np.random.uniform([10] *num)
#     # X = []
#     # for i in range (num):
#     #     temp = np.random.uniform([10] *d)
#     #     X.append(temp)
#     X = X0
#
#     for i in range(num):
#         index1  =np.argmin([np.linalg.norm(x[j] - X[i], ord = d)   for j in range(len(x))])
#
#         theta2[i] = theta[index1]
#
#     f = [testfun(theta2[i],X[i]) for i in range(num)]
#     print("algorithm_2 is over")
#     return theta2 ,f
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     #lambda1 = 0.001
#
#     k = 2# neighbor nums
#     d =  2
#     up_bound = 3*d
#     #n = 200
#     error =[]
#     error1 =[]
#     error2 =[]
#     error3 =[]
#     error4 =[]
#     op_error =[]
#     op_error1 =[]
#     op_error2 =[]
#     op_error3 =[]
#     op_error4 =[]
#     A = np.zeros((d, d))
#     for i in range(d):
#         for j in range(i, d):
#             A[i][j] = 1
#
#     # n = 100
#     # N = 20000
#
#     # #np.random.seed(1)
#     # true = np.zeros((n,N))
#     # random_theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
#     # X = np.random.uniform([up_bound/d]*n *d).reshape(n ,d)
#     # K_hat_inv_ =  K_hat_inv(n,X)
#     # y_pre1 = Objective_Prediction2(random_theta, X)
#
#
#     # for i in range(n):
#     #     x0 = X[i]
#     #     for j in range(N):
#     #         #X_train[j,i] = np.hstack([theta[j],x0])
#     #         true[i,j] = testfun(random_theta[j],x0)
#     # print("curve fitting r2 score:", r2_score(y_pre1, true))
#
#     num1 = [200,300,400]
#
#     N =2000
#
#     lambda1 = 0.000001
#     #np.random.seed(1)
#
#     random_theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
#     #random_theta = np.array(pd.DataFrame(vallist))
#     K_hat = K_hat_(random_theta)
#     K_hat_inv_ = K_hat_inv(N,K_hat,lambda1)
#
#
#     for n in num1:
#         sample_length = 500
#         #X =  np.random.uniform(0,up_bound/d,size=(n,d))
#         consvallw = np.full((d,),0)  # lower bound of solution's value
#         consvalup = np.full((d,),up_bound/d)  # upper bound of solution's value
#         dnum = np.full((d,),2000)
#         #dnum = np.full((d,),n)
#         dnum_sample = np.full((d,),sample_length)
#         vallist = dict()  # dictionary to store possible values in each dimension
#         vallist1 = dict()  # dictionary to store possible values in each dimension
#         for di in range(d):
#             vallist[di] = np.linspace(consvallw[di],consvalup[di],dnum[di])
#             vallist1[di] = np.linspace(consvallw[di],consvalup[di],dnum_sample[di])
#         theta = np.array(pd.DataFrame(vallist))
#         np.random.seed(3)
#         theta = np.random.uniform([up_bound/d]*n *d).reshape(n ,d)
#         X0 = np.array(pd.DataFrame(vallist1))
#         np.random.seed(10)
#         X0 = np.random.uniform([up_bound/d]*sample_length *d).reshape(sample_length ,d)
#
#
#         X = theta.copy()
#         x = X    #offline data
#         #print(X.shape)
#         K_hat1 = K_hat_(X)
#         lambda1 = 0.000001
#
#         # K_hat = K_hat_(X)
#
#         # K_hat_inv_ = K_hat_inv(n,K_hat,lambda1)
#
#         theta_kw, x_kw= algorithm_1(n, k*d, d)
#         theta_kw1 = theta_kw.copy()
#
#         print(np.linalg.norm(theta_kw - X))
#
#
#         # N =4000
#         # lambda1 = 0.000001
#         # #np.random.seed(1)
#         # true = np.zeros((n,N))
#         # random_theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
#         # #random_theta = np.array(pd.DataFrame(vallist))
#         # K_hat = K_hat_(random_theta)
#         # K_hat_inv_ = K_hat_inv(N,K_hat,lambda1)
#         N =1000
#         true = np.zeros((n,N))
#         for i in range(n):
#                 x0 = X[i]
#                 for j in range(N):
#                     #X_train[j,i] = np.hstack([theta[j],x0])
#                     true[i,j] = testfun(random_theta[j],x0)
#                 the = random_theta[ np.argmin(true[i,:]) ]
#                 if testfun(theta_kw1[i], X[i]) - testfun(the, X[i])  >0 :
#                     #print(np.argmin(y_pre))
#                     theta_kw1[i] = the
#         print("over")
#         print(np.linalg.norm(theta_kw1 - X))
#
#
#         for _ in range(2):
#             N = 10000
#             #np.random.seed(1)
#
#             theta = np.random.uniform([up_bound/d]*N *d).reshape(N ,d)
#
#
#             y_pre1 = Objective_Prediction2(theta, X)
#
#
#
#
#             # from sklearn.metrics import mean_squared_error
#             # print("curve fitting r2 score:", r2_score(y_pre1, true))
#             # print("mse:", mean_squared_error(y_pre1, true))
#
#
#             #print(y_pre1.shape)
#             #theta_kw2 = theta_kw.copy()
#             for i in range(n):
#                 #print(theta_kw1.shape)
#
#                 y_pre = y_pre1[i,:]
#                 the = theta[ np.argmin(y_pre) ]
#
#
#                 if testfun(theta_kw1[i], X[i]) - testfun(the, X[i])  >0 :
#
#                     theta_kw1[i] = the
#
#             print("over")
#             print(np.linalg.norm(theta_kw1 - X))
#             #print(np.linalg.norm(theta_kw2 - X))
#         lambda1 = 0.0001
#         K_hat_inv_1 = K_hat_inv(n,K_hat1,lambda1)
#         theta2 ,f = algorithm_2(theta_kw, x_kw ,sample_length)
#         #error.append(np.mean(f)+ d  -1)   # Griewank function
#         error.append(np.mean(f))
#         op_error.append(np.linalg.norm(theta2-X0)**2/sample_length)
#
#
#         #Predict:use krr algorithm to predict online data
#         predict = np.zeros((sample_length,d))  #h_theta_x0
#         predict1 = np.zeros((sample_length,d))  #h_theta_x0
#         predict2 = np.zeros((sample_length,d))  #h_theta_x0
#         predict3 = np.zeros((sample_length,d))  #h_theta_x0
#         predict4 = np.zeros((sample_length,d))  #h_theta_x0
#
#         for i in range(sample_length):
#             x0 = X0[i]
#             predict[i] = Optimal_Prediction(theta_kw,x0)
#             predict1[i] = Optimal_Prediction(X,x0)
#             #predict2[i] = Optimal_Prediction(z_theta_kw ,x0)
#             predict3[i] = Optimal_Prediction(theta_kw1 ,x0)
#             #predict2[i] = Optimal_Prediction(theta_kw2 ,x0)
#
#
#         print(np.linalg.norm(theta_kw1 - X))
#
#         f1 = [testfun(predict[i],X0[i]) for i in range(sample_length)]
#         #error1.append(np.mean(f1)+ d  -1)  # Griewank function
#         error1.append(np.mean(f1))
#         op_error1.append(np.linalg.norm(predict-X0)**2/sample_length)
#
#         f2 = [testfun(predict1[i],X0[i]) for i in range(sample_length)]
#         #error1.append(np.mean(f1)+ d  -1)  # Griewank function
#         error2.append(np.mean(f2))
#         op_error2.append(np.linalg.norm(predict1-X0)**2/sample_length)
#
#         # #print(z_theta_kw.shape,X0.shape)
#         # f3 = [testfun(predict2[i],X0[i]) for i in range(sample_length)]
#         # #error1.append(np.mean(f1)+ d  -1)  # Griewank function
#         # error3.append(np.mean(f3))
#         # op_error3.append(np.linalg.norm(predict2-X0)**2/sample_length)
#
#
#         f4 = [testfun(predict3[i],X0[i]) for i in range(sample_length)]
#         #error1.append(np.mean(f1)+ d  -1)  # Griewank function
#         error4.append(np.mean(f4))
#         op_error4.append(np.linalg.norm(predict3-X0)**2/sample_length)
#         print("krr_r2 score:", r2_score(predict3, X0))
#
#
#
#
#     import seaborn as sns
#
#
#     sns.set()
#     data_picture = pd.DataFrame()
#     data_picture.index = [i for i in num1]
#     data_picture["the nearest neighbor_predict"] = error
#
#     data_picture[r"$\hat{\theta}^{*}(x)$_krr_predict"] = error1
#     data_picture["optimal_krr_predict"] = error2
#     #data_picture["random_krr_predict"] = error3
#     data_picture["object_and_optimal"] = error4
#
#     #colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37"]
#     f, axs = plt.subplots(1, 2,figsize=(25, 10))
#     sns.lineplot(data=data_picture, ax = axs[0],linewidth=3,marker='o',ms=10,mew=1,mec='k')
#     axs[0].set_xlabel("numbers",fontsize=40)
#     axs[0].set_ylabel(r'$E_{x}[f(\theta,x) -f(\theta^{*},x)] $', fontsize=30)
#     axs[0].set_title(r'$sphere-dimension = {} $'.format(d),fontsize=30)
#     #axs.set_title(r'$griewank-dimension = {} $'.format(d),fontsize=30)
#     #axs[0].set_title(r'$rastrigin-dimension = {} $'.format(d),fontsize=30)
#     #axs.set_title(r'$Schwefel-dimension = {} $'.format(d),fontsize=30)
#     axs[0].legend()
#     axs[0].tick_params(labelsize=30)
#     #plt.savefig('Objective'+'.png',dpi=700)
#     #plt.show()
#
#
#
#
#
#
#     data_picture1 = pd.DataFrame()
#     data_picture1.index = [i for i in num1]
#
#     data_picture1["the nearest neighbor_predict"] = op_error
#
#     data_picture1[r"$\hat{\theta}^{*}(x)$_krr_predict"] = op_error1
#     data_picture1["optimal_krr_predict"] = op_error2
#     #data_picture1["random_krr_predict"] =op_error3
#     data_picture1["object_and_optimal"] = op_error4
#     #sns.set()
#
#     #colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37"]
#     #f, ax = plt.subplots(figsize=(20, 10))
#     sns.lineplot(data=data_picture1,ax = axs[1], linewidth=3,marker='o',ms=10,mew=1,mec='k')
#     axs[1].set_xlabel("numbers",fontsize=40)
#     axs[1].set_ylabel(r'$E_{x}(\theta - \theta^{*})^{2} $', fontsize=30)
#     axs[1].set_title(r'$sphere-dimension = {} $'.format(d),fontsize=30)
#     #axs[1].set_title(r'$griewank-dimension = {} $'.format(d),fontsize=30)
#     #axs[1].set_title(r'$rastrigin-dimension = {} $'.format(d),fontsize=30)
#     #ax.set_title(r'$Schwefel-dimension = {} $'.format(d),fontsize=30)
#     axs[1].legend()
#     axs[1].tick_params(labelsize=30)
#     plt.savefig('Objective_and'+'_Optimal_{}'.format(d)+'.png',dpi=700)
#     #plt.show()
#
#
#
#









