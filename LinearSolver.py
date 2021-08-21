import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

iterationNum = 100
epsilon      = 1e-6
n            = 100

x  = ti.field(dtype = ti.f32,shape=n)
xn = ti.field(dtype = ti.f32,shape=n)
b  = ti.field(dtype = ti.f32,shape=n)
r  = ti.field(dtype = ti.f32,shape=n)
A  = ti.field(dtype = ti.f32,shape=(n,n))

rn = ti.field(dtype = ti.f32,shape=n)
p  = ti.field(dtype = ti.f32,shape=n)

# r = b - Ax or r = xn-x
@ti.kernel
def Residual():
    for i in range(n):
        r[i] = b[i]
        for j in range(n):
            r[i]-=A[i,j]*x[j]
        p[i] = r[i]

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

@ti.kernel
def GaussSeidel():
    for i in range(n):
        bi = b[i]
        for j in range(n):
            if i==j:continue
            bi-=A[i,j]*xn[j]
        xn[i] = bi/A[i,i]
    for i in range(n):
        x[i] = xn[i]

@ti.kernel
def SuccessiveOverRelax(w:ti.f32):
    for i in range(n):
        bi = b[i]
        for j in range(n):
            if i==j:continue
            bi-=A[i,j]*xn[j]
        xn[i] = bi/A[i,i]
    for i in range(n):
        x[i] = xn[i]*w+(1-w)*x[i]

@ti.kernel
def GradientDescent():
    alpha = 0.0
    denominator = 0.0
    for i in range(n):
        alpha += r[i]*r[i]
        Ari = 0.0
        for j in range(n):
            Ari += A[i,j]*r[j]
        denominator += r[i]*Ari 
    alpha/=denominator
    for i in range(n):
        x[i] += alpha*r[i]

@ti.kernel
def ConjugateGradient():
    # calculate alpha
    alphaNumerator   = 0.0
    alphaDenominator = 0.0
    for i in range(n):
        alphaNumerator += r[i]*r[i]
        Ari = 0.0
        for j in range(n):
            Ari += A[i,j]*p[j]
        alphaDenominator += Ari * p[i] 
    alpha = alphaNumerator/alphaDenominator
    # update x and compute new residual(rn = r - alpha*Ap or rn == b - Ax)
    for i in range(n):
        x[i] += alpha*r[i]
        Api = 0.0
        for j in range(n):
            Api += A[i,j]*p[j]
        rn[i] = r[i] - alpha*Api 
    #compute beta
    betaNumerator = 0.0 
    betaDenominator = 0.0
    for i in range(n):
        betaNumerator += rn[i]*rn[i]
        betaDenominator += r[i] *r[i]
    beta = betaNumerator/betaDenominator
    #update p
    for i in range(n):
        p[i] = rn[i] + beta *p[i]

# A = [  2.5,-1,0,0,...., -1
#         -1,2.5,-1,0,0,...
#         0,-1,2.5,-1,0,0,...
#     ]
@ti.kernel
def Initialize():
    for i in range(n):
        x[i] = 0
        b[i] = 0
    b[0] = 1
    for i in range(n):
        for j in range(n):
            A[i,j] = 0
    for i in range(n):
        A[i,i] = 2.5
        if i+1<n: A[i,i+1]=-1 
        if i-1>=0:A[i,i-1]=-1
    A[0,n-1]=-1
    A[n-1,0]=-1


def LinearSolver():
    Initialize()
    Residual()
    for _ in range(iterationNum):
        # Jacobi()
        # GaussSeidel()
        # SuccessiveOverRelax(1.2)
        # GradientDescent()
        ConjugateGradient()
        Residual()
        if np.max(np.abs(r.to_numpy()))<epsilon:
            print("iteration count = ",_)
            # print("x = ",x)
            print("r = ",r)
            return
    print("No Convergence !")

LinearSolver()