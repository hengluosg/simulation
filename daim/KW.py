import numpy  as np
import matplotlib.pyplot as plt   
import pandas as pd

def testfun(x,tm):
    y = np.sum((x-tm)**2)
    return y

def descent_kw(tm , theta):    # tm  is x ;
    
    
    
    d = tm.shape[0]

    consvallw = np.full((d,),0)  # lower bound of solution's value

    consvalup = np.full((d,),10)  # upper bound of solution's value

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

    for n in range(1 ,N+1):
        
        a_n = n**(-1)
        
        c_n = n**(-1/3)
        
        
        delta  = [testfun(x0[i] + c_n * e[i], tm) - testfun(x0[i] , tm) for i in range(dim)]
        
        
        x1 = x0 - [a_n * t for  t in delta]
       
        x0 = x1
        
    optimal = testfun(x0, tm)
        
    return x0 ,optimal





def algorithm_1(m, k ,d ):
    ##first stage
    x = []  #tm
    for i in range ( m ):
        temp = np.random.uniform([10] *d)
        x.append(temp)
        
        
    theta =[]
    theta_new =[0] * m
    for i in range(m):

        theta.append(descent_kw(x[i] ,"initialize")[0])
    #second stage
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(x)
   
    for j in range(m):

        x_neighbor_index  =   neigh.kneighbors([x[j]], return_distance=False)

        x_neighbor = [x[t] for  t  in x_neighbor_index[0]]

        results = [descent_kw(x_neighbor[l],theta[j]) for l in range(k+1)]
        index =  np.argmin([x[1] for x in results])
        theta_new[j] = results[index][0]

        # index = np.argmin([solve(x_neighbor[l],theta[j])[1] for l in range(k+1)])
        # theta_new[j] = solve(x_neighbor[index],theta[j])[0]
    print("algorithm_1 is over")
    return theta_new , x 
        


























def algorithm_2(theta , x ,num):
    d = x[0].shape[0]
    #num = 1000
    #print(len(x))
    theta2 = [0] * num


    # X =  np.random.uniform([10] *num)
    X = []
    for i in range (num):
        temp = np.random.uniform([10] *d)
        X.append(temp)

    m = len(x)
    for i in range(num):
        index1  =np.argmin([np.linalg.norm(x[j] - X[i], ord = d)   for j in range(len(x))])
        
        theta2[i] = theta[index1] 
    
    f = [testfun(theta2[i],X[i]) for i in range(num)]
    print("algorithm_2 is over")
    return theta2 ,f

if __name__ == '__main__':


    for d in [1,2]:
        k= 2
        if d ==1 :
            m1 = [5**d ,15**d ,20**d,40**d, 60**d ,100**d ]
        elif d == 2:
            m1 = [3**d ,5**d ,10**d,15**d, 20**d ,25**d ]
        elif d == 3:
            m1 = [3**d ,5**d ,8**d,10**d, 12**d ,15**d ]
        else:
            m1 = [3**d ,4**d ,5**d,6**d, 7**d ,8**d ]
        i = 0
        error = []
        for m in m1:
            num1 =[1000]
            
            theta , x = algorithm_1(m, k ,d )
            
            for num in num1:
                theta2 ,f = algorithm_2(theta , x ,num)
                # sphere function
                error.append(np.mean(f))

                # Griewank function
                #error.append(np.mean(f + d  -1))

                # Rastrigin function:
                # error.append(np.mean(f))




        plt.figure(figsize=(20, 10))
        plt.plot([np.log(i) for i in m1], [np.log(x) for x in error], marker='o', color='red',label = "Sphere_Function")
        plt.xlabel(r'$log(m)$')
        #plt.ylabel("error_mean")
        plt.ylabel(r'$log(f(\theta ,x)-f(\theta* ,x))$')
        plt.title(r'$dimension = {} $'.format(d))
        plt.legend()
        #path = " ./C:/Users/Admin/Desktop/pythoncode/博二上/code/numerical_sphere/"
      
        #plt.savefig("C:/Users/Admin/Desktop/pythoncode/博二上/code/numerical_sphere/dimension{}_1.png".format(d))

        plt.show()

