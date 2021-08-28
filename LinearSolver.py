import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n  = 7

x  = ti.field(dtype = ti.f32,shape=n)
xn = ti.field(dtype = ti.f32,shape=n)
b  = ti.field(dtype = ti.f32,shape=n)
r  = ti.field(dtype = ti.f32,shape=n)
A  = ti.field(dtype = ti.f32,shape=(n,n))

# preconditioner
M  = ti.field(dtype = ti.f32,shape=(n,n))

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

#NOTE : taichi Gauss-Seidel can be buggy because of kernel's 
# auto parallel. Use red-black Gauss-Seidel instead. or use a 
# single loop to block parallel.
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
        M[i,i] = A[i,i]
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

#有些矩阵迭代会发散
def LinearSolver(iterateMethod = Jacobi,iterationNum=100,epsilon=1e-6):
    Initialize()
    _ = 0
    while _<iterationNum and np.max(np.abs(r.to_numpy()))>epsilon:
        iterateMethod()
        Residual()
        _+=1
    if _<iterationNum: print("iteration count =",_+1,"\n x =",x)
    else:              print("No Convergence ! x = ",x)

LinearSolver(iterateMethod = ConjugateGradient)