import taichi as ti
import numpy as np
import time,random
from collections import defaultdict

ti.init(arch=ti.gpu)

n  = 7

x  = ti.field(dtype = ti.f32,shape=n) 
xn = ti.field(dtype = ti.f32,shape=n) # x new
b  = ti.field(dtype = ti.f32,shape=n)
r  = ti.field(dtype = ti.f32,shape=n) # residual
A  = ti.field(dtype = ti.f32,shape=(n,n))

partitions      = ti.field(dtype = ti.i32,shape = n)
partitionOffset = ti.field(dtype = ti.i32,shape = n+1)

rn = ti.field(dtype = ti.f32,shape=n)
p  = ti.field(dtype = ti.f32,shape=n)
#temporary vector for intermedate variables
t  = ti.field(dtype = ti.f32,shape=n)

# r = b - Ax or r = xn-x
@ti.kernel
def Residual():
    for i in range(n):
        r[i] = b[i]
        for j in range(n):
            r[i]-=A[i,j]*x[j]

@ti.kernel
def Jacobi():
    for i in range(n):
        bi = b[i]
        for j in range(n):
            if i==j:continue
            bi-=A[i,j]*x[j]
        xn[i] = bi/A[i,i]
    for i in range(n):
        x[i] = xn[i]

# NOTE : taichi Gauss-Seidel can be buggy because of kernel's auto parallel.
# Use red-black Gauss-Seidel instead,or use a single loop to block parallel.
@ti.kernel
def GaussSeidel():
    for _ in range(1):
        for i in range(n):
            bi = b[i]
            for j in range(n):
                if i==j:continue
                bi-=A[i,j]*x[j]
            x[i] = bi/A[i,i]

@ti.kernel
def ParallelGaussSeidel():
    for k in range(n):
        for s in range(partitionOffset[k],partitionOffset[k+1]):
            i = partitions[s]
            bi = b[i]
            for j in range(n):
                if i==j:continue
                bi-=A[i,j]*x[j]
            x[i] = bi/A[i,i]

@ti.kernel
def ConjugateGradient():
    # calculate t = A*p
    for i in range(n):
        t[i] = 0.0
        for j in range(n):
            t[i] += A[i,j]*p[j]
    # calculate alpha
    alphaNumerator   = 0.0
    alphaDenominator = 0.0
    for i in range(n):
        alphaNumerator   += r[i]*r[i]
        alphaDenominator += t[i]*p[i] 
    alpha = alphaNumerator/alphaDenominator
    # update x and compute new residual(rn = r - alpha*Ap or rn = b - Ax) and beta
    betaNumerator = 0.0 
    betaDenominator = alphaNumerator
    for i in range(n):
        x[i] = x[i] + alpha*p[i]
        r[i] = r[i] - alpha*t[i]
        betaNumerator += r[i]*r[i]
    beta = betaNumerator/betaDenominator
    #update p
    for i in range(n):
        p[i] = r[i] + beta *p[i]

@ti.kernel
def Initialize():
    for i in range(n):
        x[i] = 0.0
        A[i,i] = 2.5
        if i+1<=n-1: A[i,i+1] = -1
        if i-1>=0:   A[i,i-1] = -1
    A[0,n-1]=-1
    A[n-1,0]=-1
    for i in range(n):
        b[i] = 0.0
    b[0] = 1
    for i in range(n):
        r[i] = b[i]
        for j in range(n):
            r[i]-=A[i,j]*x[j]
        p[i] = r[i]

# https://gist.github.com/Erkaman/b34b3531e209a1db38e259ea53ff0be9
def GraphColoring(A):
    g = [[]for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j:continue
            if A[i,j]!=0.0:g[i].append(j)
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
    m = defaultdict(list)
    for i in range(n):m[c[i]].append(i)
    p,pn = [],[]
    for v in m.values():
        p.extend(v)
        pn.append(len(v))
    assert len(p)==n
    while len(pn)<n:pn.append(0)
    po = [0]
    for i in range(n-1):
        po.append(po[-1]+pn[i])
    po.append(n)
    partitions.from_numpy(np.array(p))
    partitionOffset.from_numpy(np.array(po))

#有些矩阵迭代会发散
def LinearSolver(iterateMethod = Jacobi,iterationNum=100,epsilon=1e-6):
    Initialize()
    GraphColoring(A.to_numpy())
    _ = 0
    start = time.process_time()
    while _<iterationNum and np.max(np.abs(r.to_numpy()))>epsilon:
        iterateMethod()
        Residual()
        _+=1
    timePassed = time.process_time()-start
    if _<iterationNum: print("iteration count =",_+1,"\n x =",x)
    else:              print("No Convergence ! x = ",x)
    return timePassed


f = 0.0
for _ in range(100):
    f += LinearSolver(iterateMethod = ConjugateGradient)
print(f/100)

# Method                   averge time
# ParallelGaussSeidel       0.0190625
# Jacobi                    0.0246875
# GaussSeidel               0.01546875
# ConjugateGradient         0.005625