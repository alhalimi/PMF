import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.io import loadmat
import matplotlib.gridspec as gridspec
import random
import time
from scipy.spatial import Delaunay
import numpy as np
import math
from scipy.spatial import ConvexHull  
import matplotlib.pyplot as plt
import pmfgeneral8
import scipy
from pmfgeneral8 import pmf





iter = 20
data=[]
NumIterations=5000

for d in range(iter):  
    r=3+d # Number of Components
    m=r # Number of Mixtures
    N=800 # Number of Samples
    a=np.zeros((NumIterations,iter));
    a1=np.zeros((NumIterations,iter));
    a2=np.zeros((NumIterations,iter));
    restartcount=np.zeros((NumIterations))

    k = d
    Inp=2.0*np.random.rand(r,N)-1 # generate random vecotrs U[-1,1]^r
    A=(np.random.randn(m,r)) # Mixing matrix
    Y=np.dot(A,Inp)
    pmf2=pmf(Y,r)
    ns=np.sign(np.sum(pmf2.R[:,2]))
    pmf2.R[:,2]=pmf2.R[:,2]*ns
    pmf2.H[:,2]=pmf2.H[:,2]*ns
    pmf2.H=pmf2.H@np.random.rand(r,r)
    
    pmf2.BoundMax=np.array([[1,1,1]])
    pmf2.BoundMax=np.ones((1,r))
    pmf2.BoundMin=-np.ones((1,r))
    pmf2.SparseList=np.array([],dtype=object)
    pmf2.SetGroundTruth(Inp.T)
    pmf2.SetGroundTruthW(A)
    pmf2.MaxIterations=1
    pmf2.algorithm_iterations=1000
    pmf2.step_size_rule='const'
    pmf2.checklocalstationary=1
    pmf2.detstabilityvalue=1e-5
    pmf2.detwindow=100
    pmf2.distancethreshold=1.0
    pmf2.algorithm_iterations=1000
    pmf2.general(verbose=1,step_size_rule='relaxation',HProjectionPeriod=1,gradnormalization2=1,step_size_scale=5,Exponential=1,exbase=0.995,Nesterov=0,NesterovEta=0.01,Adam=0,beta0=0.99,beta2=0.99,eta=3e-1,epsilon=1e-5)
    pmf2.algorithm_iterations=NumIterations
    pmf2.general(verbose=1,step_size_rule='relaxation',HProjectionPeriod=1,gradnormalization2=1,step_size_scale=0.5,Exponential=1,exbase=0.995,Nesterov=0,NesterovEta=0.01,Adam=0,beta0=0.99,beta2=0.99,eta=3e-2,epsilon=1e-5)
    a2[:,k]=10*np.log10(pmf2.SIR)
    data.append((a2[4500][k],r))
    
    #if (10*np.log10(pmf2.SIR[-1])<20):
    #  break
    a[:,k]=a2[:,k]
    plt.plot((a2[:,k]))
    plt.xlabel('Iterations')
    plt.ylabel('Signal-to-Interference Ratio (dB)')
    plt.grid()
    restartcount[k]=pmf2.restarts
    print(k)
    #  plt.show()
print(data)


def coefficient(x, y):
    """ Return R^2"""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


# data for N=500
pts1 = [(37.3294181656948, 3 ),  (31.904759642916, 8), (8.378918718201835, 13), (-1.6612707892028957, 18), (-5.336133435807739, 23), (-5.496083357900332, 28), (-6.120785465599998, 33), (-6.799752726081142, 38), (-6.196887803676603, 43), (-5.055225421598074, 48), (-6.175338871788499, 53), (-5.228520819634576, 58), (-6.2913373771865295, 63), (-6.120507231882453, 68), (-6.384797979753881, 73), (-6.46853987617545, 78), (-6.0816224985437195, 83), (-7.458995570455418, 88), (-8.513668617323985, 93), (-8.891017993699368, 98)]
# R^2 = 0.772355000791419, y ≈ -12.33586832343497log(x)+43.64435277528207



# data for N=1000
pts2 = [(48.804043450469564, 3), (34.875074795230674, 8), (5.987916345509251, 13), (1.7513451214066469, 18), (-4.601788586783594, 23), (-4.930293432243577, 28), (-5.532810342388749, 33), (-7.217613873707048, 38), (-5.75624965366066, 43), (-6.169980202815065, 48), (-5.573487954449928, 53), (-6.120916580583842, 58), (-6.984482294631751, 63), (-6.161829370192187, 68), (-6.778926476028352, 73), (-6.56970406591271, 78), (-6.680345753427128, 83), (-7.587800993518288, 88), (-7.190572859745586, 93), (-8.216038748842605, 98)]
# R^2 = 0.772355000791419, y ≈ -14.501831443457313log(x)+52.342301978695076

# data for N=2000
pts3 = [(58.0174167754859, 3), (46.937385774741244, 5), (41.473581467024864, 7), (42.583110764330236, 9), (40.4314467741774, 11), (7.249958711710411, 13), (2.3770091547221326, 15), (-0.40252674776649666, 17), (-1.093545071513222, 19), (-3.181606821098883, 21), (-5.703166185840053, 23), (-5.406910812594335, 25), (-9.068550839814932, 27), (-6.7929562489385376, 29), (-4.198014804671681, 31), (-5.1934857070175555, 33), (-5.436487616149439, 35), (-6.393692485507105, 37), (-7.013462653207723, 39), (-7.330409787118853, 41)]
# R^2 = 0.8639820656543005, y ≈ -28.869774947691898log(x)+92.07786712237902

# data for N=800
pts4 = [(42.616959553665936, 3), (35.21470870803952, 4), (44.54268819264623, 5), (36.92667938711544, 6), (36.6804711931248, 7), (22.298923781667852, 8), (33.67918337139968, 9), (13.54154145662515, 10), (28.405277443362834, 11), (9.113919379252827, 12), (4.734774514320023, 13), (5.10831571754379, 14), (-0.13158473782199945, 15), (1.1205901413777284, 16), (-3.135082955453905, 17), (-1.5523049235676964, 18), (-3.646122638739424, 19), (-1.505988228248142, 20), (-2.525699020122849, 21), (-4.270008560238809, 22)]
# y ≈ -28.877036407490092log(x)+83.84526370090853, R^2 = 0.8653194246502698




plt.ylabel('Signal-to-Interference Ratio (dB)')
plt.xlabel('Dimension')
x_val = [x[1] for x in pts4]
y_val = [x[0] for x in pts4]
plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
plt.show()

# exponential regression
data_lst = [pts1, pts2, pts3, pts4]
for i in range(4):
    print(i)
    x_data = [np.log(x[1]) for x in data_lst[i]]
    y_data = [x[0] for x in data_lst[i]]
    result = np.polyfit(x_data, y_data, 1)
    print(result)
    print(f"y ≈ {result[0]}log(x)+{result[1]}\n")
    print(f"R^2 = {coefficient(x_data, y_data)}")