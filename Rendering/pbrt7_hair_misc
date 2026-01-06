import numpy as np
from matplotlib import pyplot as plt
import taichi as ti
from math import pi

ti.init(arch=ti.cpu)

@ti.func
def I0(x):
    ret,x2i,ifact,i4 = 0.0,1.0,1.0,1.0
    for i in ti.static(range(10)):
        if i>1: ifact *= i
        ret += x2i/(i4*ifact**2)
        x2i *= x*x
        i4 *= 4
    return ret

@ti.func
def LogI0(x):
    ret = ti.log(I0(x))
    if x >12 :ret = x + 0.5*(ti.log(1/x) + 1/(8*x) - ti.log(2*pi))
    return ret

@ti.func
def Sinh(x):return 0.5*(ti.exp(x) - ti.exp(-x))
@ti.func
def Mp(thetaI,thetaO,v):
    cosi,sini,coso,sino = ti.cos(thetaI),ti.sin(thetaI),ti.cos(thetaO),ti.sin(thetaO)
    a,b = cosi*coso/v,sini*sino/v
    return ti.exp(LogI0(a)-b-1/v+0.6931+ti.log(1/(2*v))) if v<0.1 else ti.exp(-b)*I0(a)/(Sinh(1/v)*2*v) 
@ti.kernel
def Mps(input:ti.types.ndarray(dtype=ti.f64, ndim=1), output:ti.types.ndarray(dtype=ti.f64, ndim=1),thetaO:ti.f64):
    for i in range(input.shape[0]): output[i] = Mp(input[i],thetaO,0.02)

@ti.func
def Logisitic(x,s): return ti.exp(-x/s)/(s*(1+ti.exp(-x/s))**2)
@ti.func
def LogisticCDF(x,s): return 1/(1+ti.exp(-x/s))   
@ti.func
def TrimmedLogistic(x,s,a,b): return Logisitic(x,s)/(LogisticCDF(b,s)-LogisticCDF(a,s))
@ti.kernel
def Tls(input:ti.types.ndarray(dtype=ti.f64, ndim=1), output:ti.types.ndarray(dtype=ti.f64, ndim=1),s:ti.f64):
    for i in range(input.shape[0]): output[i] = TrimmedLogistic(input[i],s,-pi,pi)

# # input = np.linspace(-pi/2,pi/2,512).astype(np.float64)
# input = np.linspace(-pi,pi,512).astype(np.float64)
# output0 = np.zeros_like(input)   
# output1 = np.zeros_like(input)   
# # output2 = np.zeros_like(input)   
# # Mps(input,output0,-1.0)
# # Mps(input,output1,-1.3)
# # Mps(input,output2,-1.4)

# # Tls(input,output0,0.1)
# # Tls(input,output1,0.5)

# plt.plot(input,output0)
# plt.plot(input,output1)
# # plt.plot(input,output2)
# plt.show()

#这个类来验证使用eta'计算的折射角和实际折射角投影到法平面是一致的
class ModifiedIORChecker:
    def __init__(self, eta=1.55):
        self.eta = eta
    # θ是光线和法平面(yoz)的夹角。   φ是法平面上(投影的)光线和bitangent(z)的夹角
    def  Verify(self,phi,theta):
    #A. compute the transmitted direction wt from w and then project wt into the normal plane, 
        # 先求光线的坐标
        w = np.array([np.sin(theta),np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi)])
        # α, β分别是 w与normal(y)的夹角  和 w投影在表面(xoz)与tangent(x)的夹角
        # cos(α)=y  tan(β)=z/x
        alpha,beta = np.arccos(w[1]),np.arctan2(w[2],w[0])
        sin_alpha,cos_alpha = np.sin(alpha),np.cos(alpha)   
        sin_alpha_t = sin_alpha/self.eta
        cos_alpha_t = np.sqrt(1 - sin_alpha_t*sin_alpha_t)
        # 求折射后的光线坐标
        wt = np.array([-sin_alpha_t*np.cos(beta),-cos_alpha_t,-sin_alpha_t*np.sin(beta)])
        # print(w,wt)
        gamma_t_A = np.arctan(wt[2]/wt[1])
    #B. use modified index of refraction 
        eta_p = np.sqrt(self.eta*self.eta - np.sin(theta)*np.sin(theta))/np.cos(theta)
        sin_gamma_t = np.sin(phi-np.pi/2)/eta_p
        gamma_t_B = np.arcsin(sin_gamma_t)
        if np.abs(gamma_t_A - gamma_t_B)>1e-6:
            print(f"{phi:2f} {theta:2f} {gamma_t_A:2f} {gamma_t_B:2f} {gamma_t_A-gamma_t_B:2f}")

    def RunRandomTests(self,num_tests=10):
        np.random.seed(42)
        for _ in range(num_tests):
            # 对于wo，phi应该总是在 0-pi之间的
            phi,theta = np.random.rand()*np.pi,(np.random.rand()-0.5)*np.pi
            self.Verify(phi,theta)

ModifiedIORChecker().RunRandomTests(10)
