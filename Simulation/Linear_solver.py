import taichi as ti
import numpy as np
import random,collections

@ti.data_oriented
class LinearSolver:
    def GraphColoring(self):
        n = self.n
        g = [[]for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j:continue
                if self.A[i,j]!=0.0:g[i].append(j)
        maxStuckNum,stuckNum = 20,0
        maxColorNum = max(2,max([len(g[i])for i in range(n)])//7.5)
        palette = [set([i for i in range(maxColorNum)])for _ in range(n)]
        u = set([i for i in range(n)])
        c = [0 for _ in range(n)]
        nc = [maxColorNum for _ in range(n)]
        while len(u)>0:
            for i in u:
                c[i] = random.choice(list(palette[i]))
            t = set()
            for i in u:
                distinct = True
                for j in g[i]:
                    if c[j]==c[i]:
                        distinct = False
                        break
                if distinct:
                    for j in g[i] :
                        if c[i] in palette[j]:
                            palette[j].remove(c[i])
                else: t.add(i)
                if len(palette[i])==0:
                    palette.add(nc[i])
                    nc[i]+=1
            if len(u)==len(t):
                stuckNum+=1
                if stuckNum>=maxStuckNum:
                    stuckNum = 0
                    r = random.choice(list(u))
                    palette[r].add(nc[r])
                    nc[r]+=1
            u = t
        m = collections.defaultdict(list)
        for i in range(n):m[c[i]].append(i)
        reorder = []
        for v in m.values():
            reorder.append(np.array(v))
        return reorder

    def SetAb(self,newA,newb):
        if newb.shape!=(self.n,) or newA.shape!=(self.n,self.n):
            print("shape mismatch.\ncurrent solver'n is %d.\nplease instantiate a new LinearSolver"%self.n)
        self.A.from_numpy(newA)
        self.b.from_numpy(newb)

    @ti.kernel
    def Residual(self):
        for i in self.r:   self.r[i] = self.b[i]
        for i,j in self.A: self.r[i] -= self.A[i,j]*self.x[j]

    @ti.kernel
    def GetMaxResidual(self)->ti.f64:
        ret = ti.f64(0.0)
        for i in self.r: ti.atomic_max(ret,ti.abs(self.r[i]))
        return ret

    def __init__(self,n=None,A=None,b=None,iterationNum=100,epsilon=1e-7,rho = 1.0-1e-4):
        if n is None : self.n = b.shape[0]
        else:          self.n = n
        self.eps = epsilon
        self.itrNum = iterationNum
        
        self.A  = ti.field(dtype = ti.f64,shape = (self.n,self.n))
        self.b  = ti.field(dtype = ti.f64,shape = self.n)
        if A is not None and b is not None: self.SetAb(A,b)
        
        self.x  = ti.field(dtype = ti.f64,shape = self.n)
        self.r  = ti.field(dtype = ti.f64,shape = self.n)
        self.t  = ti.field(dtype = ti.f64,shape = self.n)
        self.p  = ti.field(dtype = ti.f64,shape = self.n) #temporary vector for intermedate variables

        # LU direct solver
        self.L = ti.field(dtype = ti.f64,shape = (self.n,self.n))
        self.U = ti.field(dtype = ti.f64,shape = (self.n,self.n))
        self.P = ti.field(dtype = ti.f64,shape = (self.n,self.n))

        # chebyshev acceleration
        self.y   = ti.field(dtype = ti.f64,shape = self.n)  # new y          
        self.y1   = ti.field(dtype = ti.f64,shape = self.n) # previous y          
        self.y2  = ti.field(dtype = ti.f64,shape = self.n)  # previous previous y 

    @ti.kernel
    def StepJacobi(self):
        for i in range(self.n):
            bi = self.b[i]
            for j in range(self.n):
                if i==j:continue
                bi-=self.A[i,j]*self.x[j]
            self.t[i] = bi/self.A[i,i]
        for i in range(self.n):
            self.x[i] = self.t[i]

    # parallel gauss seidel
    @ti.kernel
    def StepGaussSeidel(self,partition:ti.types.ndarray(),partitionSize:ti.u32):
        for s in range(partitionSize):
            i = partition[s]
            bi = self.b[i]
            for j in range(self.n):
                if i==j:continue
                bi-=self.A[i,j]*self.x[j]
            self.x[i] = bi/self.A[i,i]
            
    @ti.kernel
    def StepConjugateGradient(self):
        # calculate t = A*p
        for i in range(self.n):
            self.t[i] = 0.0
            for j in range(self.n):
                self.t[i] += self.A[i,j]*self.p[j]
        # calculate alpha
        alphaNumerator   = ti.f64(0.0)
        alphaDenominator = ti.f64(0.0)
        for i in range(self.n):
            alphaNumerator   += self.r[i]*self.r[i]
            alphaDenominator += self.t[i]*self.p[i] 
        alpha = alphaNumerator/alphaDenominator
        # update x and compute new residual(rn = r - alpha*Ap or rn = b - Ax) and beta
        betaNumerator = ti.f64(0.0) 
        betaDenominator = alphaNumerator
        for i in range(self.n):
            self.x[i] = self.x[i] + alpha*self.p[i]
            self.r[i] = self.r[i] - alpha*self.t[i]
            betaNumerator += self.r[i]*self.r[i]
        beta = betaNumerator/betaDenominator
        #update p
        for i in range(self.n):
            self.p[i] = self.r[i] + beta *self.p[i]

    @ti.kernel
    def StepChebyshev(self,w:ti.f64):
        for i in self.y: self.y[i] = w * (self.x[i] - self.y2[i]) + self.y2[i]

    def Chebyshev(self,Method = 'Jacobi', rho=0.8+1e-4):
        stepMethod = getattr(self,'Step'+Method)
        self.x.fill(0)
        stepMethod() 
        self.y2.copy_from(self.x)
        stepMethod() 
        self.y1.copy_from(self.x)
        w  = 2/(2-rho*rho)
        _  = 2
        self.Residual()
        while _<self.itrNum and np.max(np.abs(self.r.to_numpy()))>self.eps:
            self.x.copy_from(self.y1)
            stepMethod() 
            self.StepChebyshev(w)
            self.y2.copy_from(self.y1)
            self.y1.copy_from(self.y)
            w = 4/(4-rho*rho*w)
            self.Residual()
            _+=1
        return _

    def Jacobi(self):
        _ = 0
        self.x.fill(0)
        self.Residual()
        while _<self.itrNum and self.GetMaxResidual()>self.eps:
            self.StepJacobi()
            self.Residual()
            _+=1
        return _

    def GaussSeidel(self):
        _ = 0
        reorder = self.GraphColoring()
        self.x.fill(0) 
        self.Residual()
        while _<self.itrNum and self.GetMaxResidual()>self.eps:
            for partition in reorder:
                self.StepGaussSeidel(partition,partition.size)
            self.Residual()
            _+=1
        return _

    def ConjugateGradient(self):
        _ = 0
        self.x.fill(0) 
        self.Residual()
        self.p.copy_from(self.r)
        while _<self.itrNum and self.GetMaxResidual()>self.eps:
            self.StepConjugateGradient()
            _+=1
        return _

    # P 时初等行变换矩阵:  A = PLU 
    def DirectLU(self):
        self.L.fill(0) 
        self.P.fill(0)
        self.U.copy_from(self.A)
        for i in range(self.n):self.P[i,i] = 1
        for i in range(self.n):
            if ti.abs(self.U[i,i])>1e-4: continue
            for j in range(self.n):
                if ti.abs(self.U[j,i]) <= 1e-4 or ti.abs(self.U[i,j])<=1e-4: continue
                for k in range(self.n): 
                    self.U[i,k],self.U[j,k] = self.U[j,k],self.U[i,k]
                    self.P[i,k],self.P[j,k] = self.P[j,k],self.P[i,k]
                break
        for i in range(self.n):
            self.L[i,i] = 1.0
            for j in range(i+1,self.n):
                self.L[j,i] = self.U[j,i]/self.U[i,i]
                for k in range(i+1):        self.U[j,k] = 0.0
                for k in range(i+1,self.n): self.U[j,k] -= self.L[j,i]*self.U[i,k]
        
        self.x.fill(0)
        for j in range(self.n):
            for i in range(self.n):
                self.x[j] += self.P[j,i]*self.b[i] 
        for j in range(self.n):
            for i in range(j):
                self.x[j] -= self.L[j,i]*self.x[i]
        for j in range(self.n-1,-1,-1):
            for i in range(j+1,self.n):
                self.x[j] -= self.U[j,i]*self.x[i]
            self.x[j] /= self.U[j,j]


def GenerateTestAb_Iterative(n):
    A = np.zeros((n,n))
    b = np.zeros((n,))
    for i in range(n):
        A[i,i] = 2.5
        if i+1<=n-1: A[i,i+1] = -1
        if i-1>=0:   A[i,i-1] = -1
    A[0,n-1]=1
    A[n-1,0]=1
    for i in range(n):
        b[i] = 0.0
    b[0] = 1
    from numpy import linalg as LA
    D = np.diag(np.diag(A))
    L = np.tril(A)
    w, v = LA.eig(-LA.inv(D)@(A-D))
    w, v = LA.eig(-LA.inv(L)@(A-L))
    # print(w)
    return A,b

def GenerateTestAb_Direct(n):
    from numpy import random
    npA = np.zeros((n,n))
    npb = np.zeros((n,))
    for j in range(n):
        for i in range(n):
            npA[j,i] = random.rand()*9
    for i in range(n): npb[i] = random.rand()*9
    return npA,npb


def LinearSolverTestMain():
    ti.init(ti.cpu)
    n  = 7
    npA,npb = GenerateTestAb_Iterative(n)
    npx = np.linalg.solve(npA,npb)
    ls = LinearSolver(A = npA,b = npb)
    # ls.Jacobi() # warm up

    import time
    print('{0:20}   {1:10}  {2:10}  {3:12} {4:10}'.format('Method','Time(s)','Itr-Cnt','Max-Residual','lsx-npx'))

    for method in [ls.Jacobi,ls.GaussSeidel,ls.ConjugateGradient,ls.Chebyshev]:
        start = time.process_time()
        cnt = method()
        end = time.process_time()
        print('{0:20} {1:10}{2:10} {3:15.3e} {4:10.3e} '.format(method.__name__,end-start,cnt,np.max(np.abs(ls.r.to_numpy())),np.max(np.abs(npx-ls.x.to_numpy())) ))

    # maxerr = 0
    # n = 50
    # ls = LinearSolver(n)
    # for _ in range(1):
    #     print(_)
    #     npA,npb = GenerateTestAb_Direct(n)
    #     ls.SetAb(npA,npb)
    #     ls.DirectLU()
    #     err = np.max(np.abs(np.linalg.solve(npA,npb) - ls.x.to_numpy()))
    #     maxerr = max(err,maxerr)
    # print(maxerr)

LinearSolverTestMain()