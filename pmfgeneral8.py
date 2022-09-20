import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import scipy as sp
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from sympy import Plane, Point3D
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.iterables import multiset_permutations
from IPython.display import display, Latex, Math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import ortho_group # Requires version 0.18 of scipy
import numpy as np
import math

class pmf:
    def __init__(self,Y,r):
        # r is the intermediate dimension (dimension of row vectors of H)
        self.r=r
        # Data Matrix is allocated in the object
        self.Y=Y
        shapeY=Y.shape
        # Data Matrix Dimensionns
        # Data vector size
        self.rY=shapeY[0]
        # Number of Data vectors
        self.cY=shapeY[1]
        # Default step size rule for step size is mu0/sqrt(k+1)
        self.step_size_rule='invsqrt'
        # Step size scaling default is mu0=10
        self.step_size_scale=10;
        # Default number of algorithm iterations
        self.algorithm_iterations=10000
        # It is presumed that there is no ground truth
        self.ground_truth=False
        self.iter=0
        self.verbose=0
        
        # obtain SVD of the data matrix Y
        U,S,V=np.linalg.svd(self.Y,full_matrices=False)
        self.U=U
        self.S=S
        self.V=V
        self.B=np.eye(self.r)
        self.W=np.zeros((self.rY,self.r))
        self.F=np.eye(self.r)

        # Initialize H
        self.H=np.transpose(V[0:self.r,:])
        self.R=self.H.copy()
        self.batchsize=5000
        self.W=self.Y@np.linalg.pinv(np.transpose(self.H))
        
        # y-hs fro norm
        self.err_fro=np.zeros(0)
        # h'-h fro norm
        self.fro=np.zeros(0)
        # determinant objective
        self.detobj=np.zeros(0)
        # sparsity sparsitycost
        self.sparsitycost=np.zeros(0)
        # sparsity linftycost
        self.linftycost=np.zeros(0)
        # Distance to Boundary list
        self.distlist=np.zeros(0)
        
        # General Polytope Variables 
        # Maximum Values
        self.BoundMax=np.ones((1,r))
        # Minimum Values
        self.BoundMin=-1*np.ones((1,r))
        # Number of Alternating Iterations for Projection
        self.MaxIterations=10
        # Feasible point by default
        self.feasible_point = 0.00*np.ones(r)
        # Gradient Update Parameters
        # Divide gradient by its frobenius norm
        self.gradnormalization=0  
        # Divide gradient by square of its frobenius norm (unless gradnormalization=1)
        self.gradnormalization2=0
        # Add noise to gradient (with decaying power)
        self.gradnoiseon = 0
        # H Projection Period
        self.HProjectionPeriod=1

        # Adam Parameters
        self.Adam=0
        self.beta1=0.9
        self.beta2=0.999
        self.epsilon=1e-8
        self.eta=0.01

        # Exponetial parameters
        self.Exponential=0
        self.exbase=0.99

        # Nesterov Parameters
        self.gamm=0.9
        self.NesterovEta=1.0

	# Noisy PMF
        self.tau=0.1
        self.nlamda=1
        self.p=1
        self.Lt=5

        # Joint projection to polytope and subspace
        self.enablejointprojection=0

        # Check local stationary
        self.checklocalstationary=0
        self.detstabilityvalue=1e-4
        self.distancethreshold=0.4
        self.detwindow=2000
        self.restarts=0

    ###################################################################
    #                GENERAL                                          #
    ###################################################################    



    def noisy_general(self, **kwargs):
        # Y=WH^T and the rows of H are assumed to be in l_1-norm ball

        shapeY = self.Y.shape
        self.m_dB=np.zeros((self.r,self.r))
        self.v_dB = np.zeros((self.r, self.r))
        self.prevv_dB = np.zeros((self.r, self.r))
        SS = np.diag(self.S)
        R = self.R.copy()
        #self.W=self.Y@np.linalg.pinv(np.transpose(self.H))

        #self.H = np.random.rand(self.cY,self.r)
        #self.W = ortho_group.rvs(dim=self.r)
        #self.W = np.random.rand(self.rY,self.r)
        #self.H=H.copy()

        #self.ProjectHtoPolytope()

        for k, v in kwargs.items():
            if k == 'H':
                self.H = v
            elif k == 'tau':
                self.tau = v
            elif k == 'p':
                self.p = v
            elif k == 'nlamda':
                self.nlamda = v
            elif k == 'step_size_rule':
                self.step_size_rule = v
            elif k == 'step_size_scale':
                self.step_size_scale = v
            elif k == 'algorithm_iterations':
                self.algorithm_iterations = v
            elif k == 'beta1':
                self.beta1 = v
            elif k == 'beta2':
                self.beta2 = v
            elif k == 'eta':
                self.eta = v
            elif k == 'epsilon':
                self.epsilon = v
            elif k == 'Adam':
                self.Adam = v
            elif k == 'Exponential':
                self.Exponential = v
            elif k == 'exbase':
                self.exbase = v
            elif k == 'gradnormalization':
                self.gradnormalization = v
            elif k == 'gradnormalization2':
                self.gradnormalization2 = v
            elif k == 'enablejointprojection':
                self.enablejointprojection=v
            elif k == 'gradnoiseon':
                self.gradnoiseon=v
            elif k == 'Nesterov':
                self.Nesterov = v
            elif k == 'NesterovEta':
                self.NesterovEta = v
            elif k == 'gamma':
                self.gamm = v
            elif k == 'HProjectionPeriod':
                self.HProjectionPeriod = v
            elif k=='checklocalstationary':
                self.checklocalstationary=v
            # Verbose mode
            elif k == 'verbose':
                self.verbose = v

        # initialize column scaling for H
        #B = R.T@self.H
        #B = np.random.rand(self.r,self.r)
        #self.H = R@B
        

	#self.F=np.eye(self.r)
        #self.H = R@B
        nk=0
        self.W=self.U[:,:self.r]
        self.We=np.eye(self.cY)
        kk=0
        q=1

        if (self.ground_truth):
            Hg=self.Hg.T
            Hnorm=np.sqrt(np.sum(Hg**2,axis=1))
            HD=np.diag(1/Hnorm)
            self.Hn=np.dot(HD,Hg)
            self.rmsangle=np.zeros(self.algorithm_iterations)
            #self.SIR=np.zeros(self.algorithm_iterations)

        dif=self.H
        for k in range(self.algorithm_iterations):
            kk=kk+1
            self.distlist=np.append(self.distlist,self.DistancetoBoundaryValues())
            #if (self.checklocalstationary>0):
            #    if (self.CheckDetObjStabilize(self.detwindow) < self.detstabilityvalue) and (self.DistancetoBoundaryValues() > self.distancethreshold):
            #        W=ortho_group.rvs(dim=self.r)
            #        kk=1
            #        self.restarts=self.restarts+1
            self.iter = k
            #self.detobj = np.append(self.detobj, np.abs(np.linalg.det(np.transpose(self.W)@self.W)))
            nk=nk+1
            #print(self.W)
            #print(self.H)
            #print(np.transpose((np.transpose(self.W)@self.W@np.transpose(self.H) - np.transpose(self.W)@self.Y)/(100*np.linalg.norm(np.transpose(self.W)@self.W,2))))

            #--------------SLOW------------
            #self.H = self.H - np.transpose((np.transpose(self.W)@self.W@np.transpose(self.H) - np.transpose(self.W)@self.Y)/(20*np.linalg.norm(np.transpose(self.W)@self.W,2)))
            #self.ProjectHtoPolytope()
	    #if k>self.algorithm_iterations/2
            #    self.nlamda=self.nlamda-0.1
            #-------------FAST-----------------
            fastH=self.H.copy()
            self.H=self.H+((q-1)/((1+math.sqrt(1+4*q*q))/2))*(dif)
            self.H = self.H - np.transpose((np.dot(np.transpose(self.W),(np.dot(self.W,np.transpose(self.H)) - self.Y))/(self.Lt*np.linalg.norm(np.transpose(self.W)@self.W, 2))))
            q = (1+math.sqrt(1+4*q*q))/2
            self.ProjectHtoPolytope()
            dif = self.H - fastH
            #self.ProjectHtoPolytope()
            
            #if (np.mod(k, self.HProjectionPeriod)==0):
            ## Update H
            #    self.H = R@B
            #    self.ProjectHtoPolytope()
            #    B = R.T@self.H
            #self.B = B
            #self.H = R@B
            if (np.mod(k, 5000) == 0) and (self.verbose > 0):
                print(k)
            #self.We=np.eye(self.cY)
      
            #nk=nk+1
	
            self.W = np.dot(np.dot(self.Y,self.H),np.linalg.inv(np.dot(np.transpose(self.H),self.H)+self.nlamda*self.F))
	    
            self.F = np.linalg.inv(np.dot(np.transpose(self.W),self.W)+self.tau*np.eye(self.r))
            self.detobj = np.append(self.detobj, np.abs(np.linalg.det((self.H.T)@self.H)))
            self.err_fro = np.append(self.err_fro, np.linalg.norm(self.Y-self.W@(self.H.T),'fro'))
            if (self.ground_truth):
                self.CalculateRMSAngle(k)
                self.CalculateSIR()
                noisy_dp=np.linalg.pinv(self.Wg)@self.W
                max_pos=np.argmax(abs(noisy_dp), axis=1)
                dp=np.zeros([self.r,self.r])
                dp[[range(self.r)],max_pos]=1
                dp=dp*np.sign(noisy_dp)
                W_corr = self.W@dp.T
                self.fro = np.append(self.fro, np.linalg.norm(W_corr-self.Wg,'fro')/np.linalg.norm(self.Wg,'fro'))
                
	    #E_y = self.Y-self.W@(self.H.T)
            E_y =  (self.Y-self.W@(self.H.T))
            for j in range(self.cY):
                self.We[j,j] = 0.5*self.p/(np.linalg.norm(E_y[:,j],2)**(1-0.5*self.p))

    def general(self, **kwargs):
        # Y=WH^T and the rows of H are assumed to be in l_1-norm ball

        shapeY = self.Y.shape
        self.m_dB=np.zeros((self.r,self.r))
        self.v_dB = np.zeros((self.r, self.r))
        self.prevv_dB = np.zeros((self.r, self.r))
        SS = np.diag(self.S)
        R = self.R.copy()
        #self.H=H.copy()
        self.ProjectHtoPolytope()

        for k, v in kwargs.items():
            if k == 'H':
                self.H = v
            elif k == 'step_size_rule':
                self.step_size_rule = v
            elif k == 'step_size_scale':
                self.step_size_scale = v
            elif k == 'algorithm_iterations':
                self.algorithm_iterations = v
            elif k == 'beta1':
                self.beta1 = v
            elif k == 'beta2':
                self.beta2 = v
            elif k == 'eta':
                self.eta = v
            elif k == 'epsilon':
                self.epsilon = v
            elif k == 'Adam':
                self.Adam = v
            elif k == 'Exponential':
                self.Exponential = v
            elif k == 'exbase':
                self.exbase = v
            elif k == 'gradnormalization':
                self.gradnormalization = v
            elif k == 'gradnormalization2':
                self.gradnormalization2 = v
            elif k == 'enablejointprojection':
                self.enablejointprojection=v
            elif k == 'gradnoiseon':
                self.gradnoiseon=v
            elif k == 'Nesterov':
                self.Nesterov = v
            elif k == 'NesterovEta':
                self.NesterovEta = v
            elif k == 'gamma':
                self.gamm = v
            elif k == 'HProjectionPeriod':
                self.HProjectionPeriod = v
            elif k=='checklocalstationary':
                self.checklocalstationary=v
            # Verbose mode
            elif k == 'verbose':
                self.verbose = v

        # initialize column scaling for H
        B = R.T@self.H
        #B = np.random.rand(self.r,self.r)
        #self.H = R@B
        kk=0
        for k in range(self.algorithm_iterations):
            kk=kk+1
            self.distlist=np.append(self.distlist,self.DistancetoBoundaryValues())
            if (self.checklocalstationary>0):
                if (self.CheckDetObjStabilize(self.detwindow) < self.detstabilityvalue) and (self.DistancetoBoundaryValues() > self.distancethreshold):
                    B=np.random.rand(self.r,self.r)
                    kk=1
                    self.restarts=self.restarts+1
            self.iter = k
            self.detobj = np.append(self.detobj, np.abs(np.linalg.det(B)))
            self.sparsitycost = np.append(
                self.sparsitycost, np.max(np.sum(np.abs(self.H), axis=1)))
            
            # Determine Step Size
            if self.step_size_rule == 'invsqrt':
                muk = self.step_size_scale/np.sqrt(kk+1.0)
            elif self.step_size_rule == 'inv':
                muk = self.step_size_scale/(kk+1.0)
            elif self.step_size_rule=='relaxation':
                err=0
                for ind1 in range(self.r):
                    err=err+np.abs(self.BoundMax[0,ind1]-np.max(self.H[:,ind1]))+np.abs(np.min(self.H[:,ind1])-self.BoundMin[0,ind1])
                muk = err*self.step_size_scale
            else:
                muk=self.step_size_scale

            # Update B
            dB = np.transpose(np.linalg.inv(B))+0.001*self.gradnoiseon*np.random.randn(self.r,self.r)
            #dB = (np.linalg.pinv(B))+self.gradnoiseon * np.random.randn(self.r, self.r)/(k+1)
            if (self.gradnormalization==1):
                dB = dB/(np.linalg.norm(dB, 'fro')+1e-5)
            elif (self.gradnormalization2 == 1):
                dB = dB/(np.linalg.norm(dB, 'fro')**2+1e-5)
            self.m_dB = self.beta1*self.m_dB + (1-self.beta1)*dB
            self.v_dB = self.beta2*self.v_dB + (1-self.beta2)*(dB**2)
            m_dB_corr = self.m_dB/(1-self.beta1**(k+1))
            v_dB_corr = self.v_dB/(1-self.beta2**(k+1))

            if (self.Adam==1):
                Update = self.eta * (m_dB_corr/(np.sqrt(v_dB_corr)+self.epsilon))/np.sqrt(kk+1.0)
            elif (self.Exponential==1):
                Update=self.step_size_scale*(self.exbase**kk)*dB
            elif (self.Nesterov==1):
                self.v_dB=self.gamm*self.prevv_dB
                GradW=np.transpose(np.linalg.pinv(B+self.v_dB))
                self.v_dB = self.gamm*self.prevv_dB+self.NesterovEta*GradW
                #B=B-self.v_dB
                self.prevv_dB=self.v_dB
                Update=self.v_dB
            else:
                Update=muk*dB#/np.linalg.norm(dB,'fro')
            B = B+Update
            if (np.mod(k, self.HProjectionPeriod)==0):
            # Update H
                self.H = R@B
                self.ProjectHtoPolytope()
                B = R.T@self.H
            self.B = B
            self.H = R@B
            if (np.mod(k, 1000) == 0) and (self.verbose > 0):
                print(k)
            
            if (self.ground_truth):
                self.CalculateSIR()
                self.W = self.Y@np.linalg.pinv(np.transpose(self.H))
 
                #print(self.W.shape)
                W_for_sort = abs(self.W[0,:])
                #print(Wcolsum.shape)
                Wf_sort = np.argsort (W_for_sort)
                #print(Wsumsort.shape)
                Wsort=self.W[:,Wf_sort]
                #print(Wsort.shape)
                self.fro = np.append(self.fro, np.linalg.norm(abs(Wsort)-abs(self.Wg),'fro'))

        SS = np.diag(self.S)
        self.W = self.U[:, 0:self.r]@SS[0:self.r,
                                        0:self.r]@np.transpose(np.linalg.inv(B))
        self.B = B

    ###################################################################
    #                          ANTISPARSE                             #
    ###################################################################
    def antisparse(self,**kwargs):
        # Y=WH^T and the rows of H are assumed to be in l_\infty ball

        
        shapeY=self.Y.shape
        # obtain SVD of the data matrix Y
        #U,S,V=np.linalg.svd(self.Y,full_matrices=False)

        SS=np.diag(self.S)   
        R=self.H.copy()
        # Normalize to [-1,1] interval
        self.H=self.H/np.max(np.abs(self.H))
        # initialize B matrix used in linearly combining columns of V
        
        
        
        
        # Check potential arguments entered by the user
        for k,v in kwargs.items():
            # Initial value of H
            if k=='H':
                self.H=v
            # Step size rule selection
            elif k=='step_size_rule':
                self.step_size_rule=v
            # Step size scale
            elif k=='step_size_scale':
                self.step_size_scale=v
            # Number of algorithm iterations
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v
        
        B=R.T@self.H
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.linftycost=np.append(self.linftycost,np.max(np.abs(self.H)))
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            # Determine Step Size based on the step size rule
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            RR=R@B
            # By projecting elements of H to [-1,1] interval
            self.H=RR*(RR>=-1.0)*(RR<=1.0)+(RR>1.0)*1.0-1.0*(RR<-1.0)
            B=R.T@self.H
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            # If ground truth exists calculate the corresponding SIR level
            if (self.ground_truth):
                self.CalculateSIR()
            
            
        # Obtaining the corresponding estimated W
        # SS is the diagonal singular values matrix
        SS=np.diag(self.S)
        # W is U*S*B^{-T}
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B
    
    
    
    
    ###################################################################
    #                              SPARSE                             #
    ###################################################################
    def sparse(self,**kwargs):
        # Y=WH^T and the rows of H are assumed to be in l_1-norm ball

        shapeY=self.Y.shape

            
        SS=np.diag(self.S)    
        R=self.H.copy()
        #self.H=H.copy()
        self.ProjectHtoL1NormBall();
        


        for k,v in kwargs.items():
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v
        
        # initialize column scaling for H
        B=R.T@self.H
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            self.H=R@B
            self.ProjectHtoL1NormBall();
            B=R.T@self.H
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   
    
    
    
    
    ###################################################################
    #              NONNEGATIVE ANTISPARSE                             #
    ###################################################################
    def nonnegative_antisparse(self,**kwargs):

        
        shapeY=self.Y.shape

        # Initialize H
        H=self.H
        SS=np.diag(self.S)
        for k in range(self.r):
            mv=np.max(np.abs(H[:,k]))
            mvi = np.asarray(np.where(np.abs(H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(H[mvi[0,0],k]);
            H[:,k]=H[:,k]*np.sign(H[mvi[0,0],k]);

        H=H/np.max(H)/2
        H=H*(H>0)*(H<=1.0)+1.0*(H>1)  
        self.H=H.copy()   
        R=self.R


        for k,v in kwargs.items():
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

                
        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.linftycost=np.append(self.linftycost,np.max(np.abs(self.H-0.5)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.pinv(B))
            # Update H
            RR=R@B
            self.H=RR*(RR>0)*(RR<=1)+1.0*(RR>1)
            B=R.T@self.H
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            if (self.ground_truth):
                self.CalculateSIR()
            
            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B
        
    
    ###################################################################
    #                 NONNEGATIVE  SPARSE                             #
    ###################################################################
    def nonnegative_sparse(self,**kwargs):

        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
            
        SS=np.diag(self.S)    
        R=self.H.copy()
        #self.H=H.copy()
        self.ProjectHtoL1NormBall();


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            self.H=R@B
            for mm in range(2):
                self.ProjectHtoL1NormBall();
                RR=self.H.copy();
                self.H=RR*(RR>0)
                B=R.T@self.H
            
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   


    ###################################################################
    #                          ANTISPARSE    BATCH                    #
    ###################################################################
    def antisparse_batch(self,**kwargs):

        
        shapeY=self.Y.shape

        # Initialize H
        H=self.H
        H=self.H/np.max(np.abs(H))/2
        SS=np.diag(self.S)
   
        R=self.R

        #B=np.random.randn(self.r,self.r)
        for k,v in kwargs.items():
            if k=='H':
                self.H=v
                B=R.T@v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
             # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            
            indexk=np.random.randint(0,self.cY,(1,self.batchsize))

            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            RB=np.squeeze(R[indexk,:])
            RR=RB@B
            
            HBO=RR
            
            self.HB=RR*(RR>=-1.0)*(RR<=1.0)+(RR>1.0)*1.0-1.0*(RR<-1.0)
            B=B+RB.T@(self.HB-HBO)
            self.B=B
            self.H[indexk,:]=self.HB
            if (np.mod(k,100)==0):
                if (self.verbose>0):
                    print(k)

                self.detobj=np.append(self.detobj, np.linalg.det(B))
                self.linftycost=np.append(self.linftycost,np.max(np.abs(self.H)))   
                
            if (self.ground_truth):
                self.CalculateSIR()
            
            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B

        
    ###################################################################
    #                 SPARSE  BATCH                                   #
    ###################################################################
    def sparse_batch(self,**kwargs):

        
        shapeY=self.Y.shape


            
        SS=np.diag(self.S)    
        R=self.R.copy()
        #self.H=H.copy()
        self.ProjectHtoL1NormBall();


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            indexk=np.random.randint(0,self.cY,(1,self.batchsize))

            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            
            RB=np.squeeze(R[indexk,:])
            self.HB=RB@B
            
            HBO=self.HB
            for mm in range(2):
                self.ProjectHtoL1NormBall_batch();
                RR=self.HB.copy();
                self.HB=RR
                B=B+RB.T@(self.HB-HBO)
            
            self.B=B
            self.H[indexk,:]=self.HB
            if (np.mod(k,500)==0):
                if (self.verbose>0):
                    print(k)

                self.detobj=np.append(self.detobj, np.linalg.det(B))
                self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B 
    
    ###################################################################
    #              NONNEGATIVE ANTISPARSE    BATCH                    #
    ###################################################################
    def nonnegative_antisparse_batch(self,**kwargs):

        
        shapeY=self.Y.shape

        # Initialize H
        H=self.H
        SS=np.diag(self.S)
        for k in range(self.r):
            mv=np.max(np.abs(H[:,k]))
            mvi = np.asarray(np.where(np.abs(H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(H[mvi[0,0],k]);
            H[:,k]=H[:,k]*np.sign(H[mvi[0,0],k]);
            
        H=H/np.max(H)/2
        self.H=H.copy()   
        R=self.R


        for k,v in kwargs.items():
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            
            indexk=np.random.randint(0,self.cY,(1,self.batchsize))

            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            RB=np.squeeze(R[indexk,:])
            RR=RB@B
            
            HBO=RR
            
            self.HB=RR*(RR>0)*(RR<=1)+1.0*(RR>1)
            B=B+RB.T@(self.HB-HBO)
            self.B=B
            self.H[indexk,:]=self.HB
            if (np.mod(k,100)==0):
                if (self.verbose>0):
                    print(k)

                self.detobj=np.append(self.detobj, np.linalg.det(B))
                self.linftycost=np.append(self.linftycost,np.max(np.abs(self.H-0.5)))   
                
            if (self.ground_truth):
                self.CalculateSIR()
            
            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B
        
    ###################################################################
    #                 NONNEGATIVE  SPARSE  BATC                       #
    ###################################################################
    def nonnegative_sparse_batch(self,**kwargs):

        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
            
        SS=np.diag(self.S)    
        R=self.H.copy()
        #self.H=H.copy()
        self.ProjectHtoL1NormBall();


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            indexk=np.random.randint(0,self.cY,(1,self.batchsize))

            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            
            RB=np.squeeze(R[indexk,:])
            self.HB=RB@B
            
            HBO=self.HB
            for mm in range(2):
                self.ProjectHtoL1NormBall_batch();
                RR=self.HB.copy();
                self.HB=RR*(RR>0)
                B=B+RB.T@(self.HB-HBO)
            
            self.B=B
            self.H[indexk,:]=self.HB
            if (np.mod(k,100)==0):
                if (self.verbose>0):
                    print(k)
                    
                self.detobj=np.append(self.detobj, np.linalg.det(B))
                self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   

    ###################################################################
    #                 NONNEGATIVE  SPARSE  BATCH MODIFIED             #
    ###################################################################
    def nonnegative_sparse_batch_modified(self,**kwargs):

        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
            
        SS=np.diag(self.S)    
        R=self.H.copy()
        #self.H=H.copy()
        self.ProjectHtoL1NormBall();
        sl1=self.H.T;
        sl1shape=sl1.shape
        sl1m=np.array([(sl1[0,:]).reshape((sl1shape[1],1)), (sl1[1,:]*(sl1[1,:]<0.6)+0.6*(sl1[1,:]>=0.6)).reshape((sl1shape[1],1))])
        sl1m=np.squeeze(sl1m)
        self.H=sl1m.T


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@self.H                
        for k in range(self.algorithm_iterations):
            self.iter=k
            indexk=np.random.randint(0,self.cY,(1,self.batchsize))

            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            
            RB=np.squeeze(R[indexk,:])
            self.HB=RB@B
            
            HBO=self.HB.copy()
            for mm in range(6):
                self.ProjectHtoL1NormBall_batch();
                RR=self.HB.copy();
                self.HB=RR*(RR>0)
                sl1=self.HB.T;
                sl1shape=sl1.shape
                sl1m=np.array([(sl1[0,:]).reshape((sl1shape[1],1)), (sl1[1,:]*(sl1[1,:]<0.6)+0.6*(sl1[1,:]>=0.6)).reshape((sl1shape[1],1))])
                sl1m=np.squeeze(sl1m)
                self.HB=sl1m.T
                
            B=B+RB.T@(self.HB-HBO)
            
            self.B=B
            self.H[indexk,:]=self.HB
            if (np.mod(k,100)==0):
                if (self.verbose>0):
                    print(k)
                    
                self.detobj=np.append(self.detobj, np.linalg.det(B))
                self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   



    ###################################################################
    #                 NONNEGATIVE  SPARSE   MAPPING                   #
    ###################################################################
    def nonnegative_sparse_mapping(self,**kwargs):

        R=self.H.copy()        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
        
        HH=self.H.copy()
        HH=HH*(HH>0)
        self.H=np.sqrt(HH)

         
        SS=np.diag(self.S)    

        #self.H=H.copy()
        self.ProjectHtoL1NormBall();
        HH=self.H*self.H  


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@HH 
        #print(B)               
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)
                
            # Update B
            B=B+muk*np.transpose(np.linalg.inv(B))
            # Update H
            HH=R@B
            HH=HH*(HH>0)
            self.H=np.sqrt(HH)

            for mm in range(2):
                self.ProjectHtoL1NormBall();
                RR=self.H.copy();
                self.H=RR*(RR>0)
                
            HH=self.H*self.H
            B=R.T@HH
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   



    ###################################################################
    #                 NONNEGATIVE  SPARSE   MAPPING 2                  #
    ###################################################################
    def nonnegative_sparse_mapping2(self,**kwargs):

        R=self.H.copy()        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
        
        HH=self.H.copy()
        HH=HH*(HH>0)
        self.H=np.sqrt(HH)

         
        SS=np.diag(self.S)    

        #self.H=H.copy()
        self.ProjectHtoL1NormBall();
        HH=self.H*self.H  


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@HH 
        #print(B)               
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)

            RH=self.H.T@self.H
            Qder=R*(1/(self.H+0.01))   
            derB=np.linalg.pinv(RH)@self.H.T@Qder 
            # Update B
            B=B+muk*derB
            # Update H
            HH=R@B
            HH=HH*(HH>0)
            self.H=np.sqrt(HH)

            for mm in range(2):
                self.ProjectHtoL1NormBall();
                RR=self.H.copy();
                self.H=RR*(RR>0)
                
            HH=self.H*self.H
            B=R.T@HH
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B   

    ###################################################################
    #                 NONNEGATIVE  SPARSE   MAPPING 3                 #
    ###################################################################
    def nonnegative_sparse_mapping3(self,**kwargs):

        R=self.H.copy()        
        shapeY=self.Y.shape

        for k in range(self.r):
            mv=np.max(np.abs(self.H[:,k]))
            mvi = np.asarray(np.where(np.abs(self.H[:,k]) == mv))
            self.U[:,k]=self.U[:,k]*np.sign(self.H[mvi[0,0],k]);
            self.H[:,k]=self.H[:,k]*np.sign(self.H[mvi[0,0],k]);
        
        HH=self.H.copy()
        HH=HH
        self.H=HH*HH

         
        SS=np.diag(self.S)    

        #self.H=H.copy()
        self.ProjectHtoL1NormBall();
        self.H=(self.H>0)*self.H
        self.ProjectHtoL1NormBall();
        self.H=(self.H>0)*self.H
        HH=np.sqrt(self.H)  


        for k,v in kwargs.items():
            self.iter=k
            if k=='H':
                self.H=v
            elif k=='step_size_rule':
                self.step_size_rule=v
            elif k=='step_size_scale':
                self.step_size_scale=v
            elif k=='algorithm_iterations':
                self.algorithm_iterations=v
            # Verbose mode
            elif k=='verbose':
                self.verbose=v

        # initialize column scaling for H
        B=R.T@HH 
        #print(B)               
        for k in range(self.algorithm_iterations):
            self.iter=k
            self.detobj=np.append(self.detobj, np.linalg.det(B))
            self.sparsitycost=np.append(self.sparsitycost,np.max(np.sum(np.abs(self.H),axis=1)))
            # Determine Step Size
            if self.step_size_rule=='invsqrt':
                muk=self.step_size_scale/np.sqrt(k+1.0)
            elif self.step_size_rule=='inv':
                muk=self.step_size_scale/(k+1.0)

            RH=self.H.T@self.H
            Qder=R*(HH)   
            derB=np.linalg.pinv(RH)@self.H.T@Qder 
            # Update B
            B=B+muk*derB
            # Update H
            HH=R@B
            HH=HH
            self.H=HH*HH

            for mm in range(8):
                self.ProjectHtoL1NormBall();
                RR=self.H.copy();
                self.H=RR*(RR>0)
                
            HH=np.sqrt(self.H)
            B=R.T@HH
            self.B=B
            if (np.mod(k,100)==0) and (self.verbose>0):
                print(k)
                
            if (self.ground_truth):
                self.CalculateSIR()

            
        SS=np.diag(self.S)
        self.W=self.U[:,0:self.r]@SS[0:self.r,0:self.r]@np.transpose(np.linalg.inv(B))
        self.B=B  

   #####################################################################################
   #                     UTILITY FUNCTIONS                                             #
   #####################################################################################
    
    def ProjectHtoL1NormBall(self):
        Hshape=self.H.shape
        #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
        lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
        #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))
        
        u=-np.sort(-np.abs(self.H),axis=1)
        sv=np.cumsum(u,axis=1)
        q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
        rho=np.max(q,axis=1)
        rho=rho.astype(int)
        lindex=np.linspace(1,Hshape[0],Hshape[0])-1
        lindex=lindex.astype(int)
        theta=np.maximum(0,np.reshape((sv[[lindex,rho]]-1)/(rho+1),(Hshape[0],1)))
        ww=np.abs(self.H)-theta
        self.H=np.sign(self.H)*(ww>0)*ww

    def ProjectSubHtoL1NormBall(self,indxs):
        HH=self.H[:,indxs]
        Hshape=HH.shape
        #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
        lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
        #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))
        
        u=-np.sort(-np.abs(HH),axis=1)
        sv=np.cumsum(u,axis=1)
        q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
        rho=np.max(q,axis=1)
        rho=rho.astype(int)
        lindex=np.linspace(1,Hshape[0],Hshape[0])-1
        lindex=lindex.astype(int)
        theta=np.maximum(0,np.reshape((sv[[lindex,rho]]-1)/(rho+1),(Hshape[0],1)))
        ww=np.abs(HH)-theta
        Out=np.sign(HH)*(ww>0)*ww
        return Out

    def ProjectHtoBoundaryRectangle(self):
        Hshape = self.H.shape
        self.resignH2()
        BoundMaxlist = np.dot(np.ones((Hshape[0],1)),self.BoundMax)
        BoundMinlist = np.dot(np.ones(( Hshape[0],1)),self.BoundMin)
        CheckMin = 1.0*(self.H > BoundMinlist)
        a = 1-2.0*(np.sum(CheckMin, axis=0) == 0)*(self.BoundMin == 0)
        AA = np.diag(np.reshape(a, (self.r,)))
        self.H=np.dot(self.H,AA)
        CheckMax = 1.0*(self.H < BoundMaxlist)
        CheckMin = 1.0*(self.H > BoundMinlist)
        self.H = self.H*CheckMax*CheckMin+(1-CheckMin)*BoundMinlist+(1-CheckMax)*BoundMaxlist
    
    def ProjectSparseComponents(self):
        NumGroups=self.SparseList.shape[0]
        if NumGroups>0:
            for k in range(NumGroups):
                self.H[:, np.array(self.SparseList[k],dtype=int)]=self.ProjectSubHtoL1NormBall(np.array(self.SparseList[k], dtype=int));

    def ProjectHtoPolytope(self):
        for k in range(self.MaxIterations):
            self.ProjectHtoBoundaryRectangle();
            self.ProjectSparseComponents();
            if (self.enablejointprojection==1):
                B = self.R.T@self.H
                self.H=self.R@B

    def ProjectHtoL1NormBall_batch(self):
        Hshape=self.HB.shape
        #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
        lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
        #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))
        
        u=-np.sort(-np.abs(self.HB),axis=1)
        sv=np.cumsum(u,axis=1)
        q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
        rho=np.max(q,axis=1)
        rho=rho.astype(int)
        lindex=np.linspace(1,Hshape[0],Hshape[0])-1
        lindex=lindex.astype(int)
        theta=np.maximum(0,np.reshape((sv[[lindex,rho]]-1)/(rho+1),(Hshape[0],1)))
        ww=np.abs(self.HB)-theta
        self.HB=np.sign(self.HB)*(ww>0)*ww

    def GeneratePolyhedralConstraintMatrix(self,**kwargs):
        for k, v in kwargs.items():
            if k == 'feasible_point':
                self.feasible_point = v
        A=np.concatenate((np.eye(self.r),-1*self.BoundMax.T),axis=1)
        A2=np.concatenate((-1*np.eye(self.r),self.BoundMin.T),axis=1)
        A=np.concatenate((A,A2),axis=0)
        for k in range(self.SparseList.shape[0]):
            NumElements=np.size(self.SparseList[k])
            d=np.arange(2**NumElements)
            SignsList=2*(((d[:,None] & (1 << np.arange(NumElements)))) > 0).astype(int)-1
            for l in range(SignsList.shape[0]):
                v=np.zeros((1,self.r))
                v[0,np.array(self.SparseList[k],dtype=int)]=SignsList[l,:]
                v=np.concatenate((v,np.reshape(np.array(-1),(1,1))),axis=1)
                A=np.concatenate((A,v),axis=0)
        self.PolyhedralConstraintMatrix=A
        self.HalfSpaces = HalfspaceIntersection(A, self.feasible_point)
        self.verts = self.HalfSpaces.intersections
        self.hull = ConvexHull(self.verts)
        self.faces = self.hull.simplices

    def CheckPolytope(self):
        indxs = np.arange(self.verts.shape[0])
        V = self.verts.T
        print('{:d} Vertices:'.format(V.shape[1]))
        #print(V)
        self.display_matrix(V)
        Vp = np.linalg.pinv(V)
        checkval = 0
        for indxp in multiset_permutations(indxs):
            Vindxp = V[:, indxp]
            G = Vindxp@Vp
            if (np.linalg.norm(G@V-Vindxp) < 1e-6):
                GG = G*(abs(G) > 1e-3)
                #if (np.linalg.norm(np.abs(GG)-np.sign(np.abs(GG))) > 1e-6):
                if ((np.linalg.norm(np.sign(np.abs(GG)).sum(axis=0) - np.ones((1, self.r))) > 1e-6)):
                    #print(np.array_str(G, precision=3, suppress_small=True))
                    checkval = checkval+1
        if (checkval > 0):
            print('Not Identifiable!')
        else:
            print('Identifiable!')
        if (self.r==3):
            self.Plot3DPolytope()
                 
                
        
   #####################################################################################
   #                PERFORMANCE MEASUREMENT FUNCTIONS                                  #
   #####################################################################################
        
        
        
    def SetGroundTruth(self,H):
        self.ground_truth=True
        self.pHt=np.linalg.pinv(H)
        self.SIR=np.array([])
        self.Hg = H
        
    def SetGroundTruthW(self,W):
        self.ground_truth=True
        self.Wg = W
        
            
    def CalculateSIR(self):
        G=self.pHt@self.H
        Gmax=np.diag(np.max(abs(G),axis=0))
        P=1.0*((np.abs(G)@np.linalg.inv(Gmax))>0.95)
        T=G@P.T
        SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        self.SIR=np.append(self.SIR,SIRV)
        self.T=T
        self.G=G
        
    def CalculateRMSAngle(self,k):
        Hest=self.H.T
        Hestnorm=np.sqrt(np.sum(Hest**2,axis=1))
        HestD=np.diag(1/Hestnorm)
        Hestn=np.dot(HestD,Hest)
        HH=np.dot(Hestn,self.Hn.T)
        angls=np.arccos(np.max(np.abs(HH),axis=1))
        self.rmsangle[k]=np.linalg.norm(angls)/np.sqrt(self.r)

    def DistancetoBoundaryValues(self):
        err=0
        for ind1 in range(self.r):
            err=err+np.abs(self.BoundMax[0,ind1]-np.max(self.H[:,ind1]))+np.abs(np.min(self.H[:,ind1])-self.BoundMin[0,ind1])
        return(err)
    def CheckDetObjStabilize(self,numsamples):
        if numsamples<len(self.detobj):
            ddetobj=np.diff(self.detobj)
            out=np.std(np.diff(ddetobj[-numsamples:]))
        else:
            out=100
        return(out)
   #####################################################################################
   #                          MORE GENERAL  FUNCTIONS                                  #
   #####################################################################################
    def display_matrix(self,array):
        data = ''
        for line in array:
            if len(line) == 1:
                data += ' %.3f &' % line + r' \\\n'
                continue
            for element in line:
                data += ' %.3f &' % element
            data += r' \\' + '\n'
        display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))

    def simplify(self,triangles):
        """
        Simplify an iterable of triangles such that adjacent and coplanar triangles form a single face.
        Each triangle is a set of 3 points in 3D space.
        """

        # create a graph in which nodes represent triangles;
        # nodes are connected if the corresponding triangles are adjacent and coplanar
        G = nx.Graph()
        G.add_nodes_from(range(len(triangles)))
        for ii, a in enumerate(triangles):
            for jj, b in enumerate(triangles):
                if (ii < jj):  # test relationships only in one way as adjacency and co-planarity are bijective
                    if self.is_adjacent(a, b):
                        if self.is_coplanar(a, b, np.pi / 180.):
                            G.add_edge(ii, jj)

        # triangles that belong to a connected component can be combined
        components = list(nx.connected_components(G))
        simplified = [set(self.flatten(triangles[index] for index in component)) for component in components]

        # need to reorder nodes so that patches are plotted correctly
        reordered = [self.reorder(face) for face in simplified]

        return reordered
    def resignH(self):
        a = 1-2*(np.sum(self.H, axis=0) < 0)*(self.BoundMin == 0)
        AA = np.diag(np.reshape(a, (self.r,)))
        self.H=np.dot(self.H,AA)
        self.R=self.H.copy()
        self.U[:, 0:self.r] = self.U[:, 0:self.r]@AA

    def resignH2(self):
        a = 1-2*(np.sum(self.H, axis=0) < 0)*(self.BoundMin == 0)
        AA = np.diag(np.reshape(a, (self.r,)))
        self.H = np.dot(self.H, AA)
        

    def is_adjacent(self,a, b):
        # i.e. triangles share 2 points and hence a side
        return len(set(a) & set(b)) == 2
    
    def is_coplanar(self,a, b, tolerance_in_radians=0):
        a1, a2, a3 = a
        b1, b2, b3 = b
        plane_a = Plane(Point3D(a1), Point3D(a2), Point3D(a3))
        plane_b = Plane(Point3D(b1), Point3D(b2), Point3D(b3))
        if not tolerance_in_radians:  # only accept exact results
            return plane_a.self.is_coplanar(plane_b)
        else:
            angle = plane_a.angle_between(plane_b).evalf()
            angle %= np.pi  # make sure that angle is between 0 and np.pi
            return (angle - tolerance_in_radians <= 0.) or \
                ((np.pi - angle) - tolerance_in_radians <= 0.)

    flatten = lambda self,l: [item for sublist in l for item in sublist]

    def reorder(self,vertices):
        """
        Reorder nodes such that the resulting path corresponds to the "hull" of the set of points.

        Note:
        -----
        Not tested on edge cases, and likely to break.
        Probably only works for convex shapes.

        """
        if len(vertices) <= 3:  # just a triangle
            return vertices
        else:
            # take random vertex (here simply the first)
            reordered = [vertices.pop()]
            # get next closest vertex that is not yet reordered
            # repeat until only one vertex remains in original list
            vertices = list(vertices)
            while len(vertices) > 1:
                idx = np.argmin(self.get_distance(reordered[-1], vertices))
                v = vertices.pop(idx)
                reordered.append(v)
            # add remaining vertex to output
            reordered += vertices
            return reordered

    def get_distance(self,v1, v2):
        v2 = np.array(list(v2))
        difference = v2 - v1
        ssd = np.sum(difference**2, axis=1)
        return np.sqrt(ssd)

    def Plot3DPolytope(self,**kwargs):
        faces = self.faces
        verts = self.verts

        ax = a3.Axes3D(plt.figure())
        ax.dist = 10
        ax.azim = 30
        ax.elev = 10
        for k, v in kwargs.items():
            if k=='dist':
                ax.dist=v
            elif k=='azim':
                ax.azim=v
            elif k=='elev':
                ax.elev=v

        MinLim = 1.5*self.BoundMin*(self.BoundMin < -0.5)-0.5*np.ones((1, self.BoundMin.shape[1]))*(self.BoundMin > -0.5)
        MaxLim = 1.5*self.BoundMax*(self.BoundMax >0.5)+0.5*np.ones((1, self.BoundMax.shape[1]))*(self.BoundMax < 0.5)
        ax.set_xlim([MinLim[0,0], MaxLim[0,0]])
        ax.set_ylim([MinLim[0,1], MaxLim[0,1]])
        ax.set_zlim([MinLim[0,2], MaxLim[0,2]])

        triangles = []
        for s in faces:
            sq = [
                (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
                (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
                (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2])
            ]
            triangles.append(sq)

        new_faces = self.simplify(triangles)
        for sq in new_faces:
            f = a3.art3d.Poly3DCollection([sq],alpha=0.4)
            f.set_color('b')#colors.rgb2hex(sp.rand(3)))
            f.set_edgecolor('k')
            f.set_alpha(0.4)
            ax.add_collection3d(f)
        

            
            
            
