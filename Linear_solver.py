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
            print("ERROR: shape not match")
        self.A,self.b = newA,newb

    @ti.kernel
    def Residual(self):
        for i in range(self.n):
            self.r[i] = self.b[i]
            for j in range(self.n):
                self.r[i]-=self.A[i,j]*self.x[j]

    def __init__(self,n,A=None,b=None,iterationNum=100,epsilon=1e-6):
        self.n = n
        self.eps = epsilon
        self.itrNum = iterationNum
        
        self.A  = A 
        self.b  = b 
        
        self.x    = ti.field(dtype = ti.f64,shape = n)
        self.r  = ti.field(dtype = ti.f64,shape = n)
        self.t  = ti.field(dtype = ti.f64,shape = n)
        self.p  = ti.field(dtype = ti.f64,shape = n) #temporary vector for intermedate variables

        self.L = ti.field(dtype = ti.f64,shape = (n,n))
        self.U = ti.field(dtype = ti.f64,shape = (n,n))
        self.P = ti.field(dtype = ti.f64,shape = (n,n))

        self.x.fill(0) #initial guess

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

    def Jacobi(self):
        _ = 0
        self.Residual()
        while _<self.itrNum and np.max(np.abs(self.r.to_numpy()))>self.eps:
            self.StepJacobi()
            self.Residual()
            _+=1
        return _

    def GaussSeidel(self):
        _ = 0
        reorder = self.GraphColoring()
        self.Residual()
        while _<self.itrNum and np.max(np.abs(self.r.to_numpy()))>self.eps:
            for partition in reorder:
                self.StepGaussSeidel(partition,partition.size)
            self.Residual()
            _+=1
        return _

    def ConjugateGradient(self):
        _ = 0
        self.Residual()
        self.p.copy_from(self.r)
        while _<self.itrNum and np.max(np.abs(self.r.to_numpy()))>self.eps:
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

def LinearSolverTestMain():
    ti.init(ti.cpu)
    n  = 7
    b  = ti.field(dtype = ti.f64,shape=n)
    A  = ti.field(dtype = ti.f64,shape=(n,n))

    for i in range(n):
        A[i,i] = 2.5
        if i+1<=n-1: A[i,i+1] = -1
        if i-1>=0:   A[i,i-1] = -1
    A[0,n-1]=-1
    A[n-1,0]=-1
    for i in range(n):
        b[i] = 0.0
    b[0] = 1

    ls = LinearSolver(n,A,b)

    import time

    print('{0:10}   {1:10}  {2:10}  {3:10}'.format('Method','Time(s)','Itr Cnt','Max Residual'))

    ls.x.fill(0)
    start = time.process_time()
    cnt = ls.Jacobi()
    end = time.process_time()
    print('{0:10}{1:10}{2:10} {3:10}'.format('Jacobi',end-start,cnt,np.max(np.abs(ls.r.to_numpy()))))
    
    ls.x.fill(0)
    start = time.process_time()
    cnt = ls.GaussSeidel()
    end = time.process_time()
    print('{0:10}{1:10}{2:10} {3:10}'.format('GS',end-start,cnt,np.max(np.abs(ls.r.to_numpy()))))

    ls.x.fill(0)
    start = time.process_time()
    cnt = ls.ConjugateGradient()
    end = time.process_time()
    print('{0:10}{1:10}{2:10} {3:10}'.format('CG',end-start,cnt,np.max(np.abs(ls.r.to_numpy()))))

    from numpy import random
    n  = 50
    b  = ti.field(dtype = ti.f64,shape=n)
    A  = ti.field(dtype = ti.f64,shape=(n,n))

    maxerr = 0
    for _ in range(10):
        print(_)
        npA = np.zeros((n,n))
        npb = np.zeros((n,))
        # npA = np.array([ [0.0,2,1],[5,8,1],[4,4,1] ] )
        # npb = np.array([13,-9,6])
        # print(npA.shape)
        # print(npb.shape)
        for j in range(n):
            for i in range(n):
                npA[j,i] = random.rand()*10
        for i in range(n): npb[i] = random.rand()*10

        A.from_numpy(npA)
        b.from_numpy(npb)
        ls = LinearSolver(n,A,b)
        ls.DirectLU()
        err = np.max(np.abs(np.linalg.solve(npA,npb) - ls.x.to_numpy()))
        maxerr = max(err,maxerr)
    print(maxerr)

LinearSolverTestMain()