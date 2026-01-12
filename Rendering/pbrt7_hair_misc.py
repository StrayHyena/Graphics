import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import taichi as ti
import taichi.math as tm
from math import pi

ti.init(arch=ti.cpu)
eta=1.55

@ti.data_oriented
class M:
    def __init__(self):
        n = 512
        self.i = ti.field(ti.f64, n)
        self.o = ti.field(ti.f64, n)
        self.i.from_numpy(np.linspace(-pi/2,pi/2,n).astype(np.float64))
    @ti.func
    def I0(x):
        ret,x2i,ifact,i4 = ti.cast(0.0,ti.f64),ti.cast(1.0,ti.f64),ti.cast(1.0,ti.f64),ti.cast(1.0,ti.f64)
        for i in ti.static(range(10)):
            if i>1: ifact *= i
            ret += x2i/(i4*ifact**2)
            x2i *= x*x
            i4 *= 4
        return ret
    @ti.func
    def LogI0(x):
        ret = ti.cast(ti.log(M.I0(x)),ti.f64)
        if x >12 :ret = x + 0.5*(ti.log(1/x) + 1/(8*x) - ti.log(2*pi))
        return ret
    @ti.func
    def Sinh(x):return 0.5*(ti.exp(x) - ti.exp(-x))
    @ti.func
    def M(thetaI,thetaO,v):
        cosi,sini,coso,sino = ti.cos(thetaI),ti.sin(thetaI),ti.cos(thetaO),ti.sin(thetaO)
        a,b = cosi*coso/v,sini*sino/v
        return ti.exp(M.LogI0(a)-b-1/v+0.6931+ti.log(1/(2*v))) if v<0.1 else ti.exp(-b)*M.I0(a)/(M.Sinh(1/v)*2*v)
    @ti.kernel
    def Mps(self,thetaO:ti.f64):
        for i in self.o: self.o[i] = M.M(self.i[i],thetaO,0.02)
    # Figure 9.46
    def Plot(self):
        for theta in [-1.0,-1.3,-1.4]:
            self.Mps(theta)
            plt.plot(self.i.to_numpy(), self.o.to_numpy())
        plt.show()

@ti.data_oriented
class N:
    def __init__(self,beta_n = 0.1):
        self.n,self.beta_n = 512,beta_n
        self.s = 0.626657069 * (0.265 * beta_n + 1.194 * beta_n**2 +5.372 * beta_n**22)
        self.i = ti.field(ti.f64, self.n)
        self.h = ti.field(ti.f64, self.n)
        self.o = ti.field(ti.f64, self.n)
        self.i.from_numpy(np.linspace(-pi,pi,self.n).astype(np.float64))
        self.h.from_numpy(np.linspace(-1, 1, self.n).astype(np.float64))
        self.plt,self.ax =  plt.subplots(1)
        self.ax.spines['bottom'].set_position('zero')  # 将底部坐标轴移到y=0的位置
        self.ax.spines['left'].set_position('zero')
    @ti.func
    def Logisitic(x,s): return ti.exp(-x/s)/(s*(1+ti.exp(-x/s))**2)
    @ti.func
    def LogisticCDF(x,s): return 1/(1+ti.exp(-x/s))
    # 类似于微表面的NDF(微观法向与half vector之间的夹角越小值越大);这个函数应该在零附近很大
    # 对于头发的bxdf,我们要衡量两个(法平面内的)方向之间的差值。①是采样的入射光线wi ②根据wo计算(即Phi()函数)出的完全镜面反射/折射的入射光线
    @ti.func
    def TrimmedLogistic(x,s,a,b): return N.Logisitic(x,s)/(N.LogisticCDF(b,s)-N.LogisticCDF(a,s))
    @ti.kernel
    def TrimmedLogistics(self,s:ti.f64):
        for i in self.o: self.o[i] = N.TrimmedLogistic(self.i[i],s,-pi,pi)
    # Figure 9.53
    def PlotTrimmedLogistic(self):
        for s in [0.1,0.5]:
            self.TrimmedLogistics(s)
            self.ax.plot(self.i.to_numpy(), self.o.to_numpy())
        plt.show()
    @ti.func
    def Phi(p,h):
        eta = 5
        gamma_o = ti.asin(h)
        gamma_t = ti.asin(ti.sin(gamma_o)/eta)
        return 2*p*gamma_t-2*gamma_o+p*pi
    @ti.kernel
    def Phis(self):
        for i in self.h:self.o[i] = N.Phi(1,self.h[i])
    # Figure 9.52
    def PlotPhis(self):
        self.Phis()
        self.ax.plot(self.h.to_numpy(), self.o.to_numpy())
        self.ax.set_yticks(np.arange(0, 3, 0.5) * np.pi)
        self.ax.set_yticklabels([f'{frac}π' for frac in np.arange(0, 3, 0.5)])
        plt.show()
    # 这么想,Phi()虽然以γo为视角计算了其偏转,但是γo和φo其实指代的是同一个法平面的方向wo (γo和φo是wo在不同的坐标系下的表达)
    # 所以 完全镜面反射/折射之后的出射角φi是 φo+Phi()  , 采样的出射角是φ'。所以，TrimmedLogistic衡量的应该是 φ'-(φo+Phi())
    @ti.func
    def N(p,h,s,phi):
        dphi = phi-N.Phi(p,h)
        while dphi>pi:dphi -= 2*pi
        while dphi<-pi:dphi += 2*pi
        return N.TrimmedLogistic(dphi,s,-pi,pi)
    @ti.kernel
    def Ns(self,p:ti.f64,h:ti.f64):
        for i in self.i: self.i[i] = i/(self.n-1)*2*pi
        for i in self.o: self.o[i] = N.N(p, h, self.s,self.i[i])
    # Figure 9.54
    def PlotPolarNs(self):
        for h in [-0.5,0.3]:
            self.Ns(1,h)
            pts = []
            for i in range(self.n): pts.append(self.o[i]*np.array([np.cos(self.i[i]),np.sin(self.i[i])]))
            pts = np.array(pts)
            self.ax.plot(pts[:,0],pts[:,1],label=f'{h}')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.legend()
        plt.show()

# N().PlotPhis()
# N().PlotTrimmedLogistic()
# N().PlotPolarNs()

#这个类来验证使用eta'计算的折射角和实际折射角投影到法平面是一致的
class ModifiedIORChecker:
    @staticmethod
    # θ是光线和法平面(yoz)的夹角。   φ是法平面上(投影的)光线和bitangent(z)的夹角
    def  Verify(phi,theta):
    # 首先要明确gamma_t是和y正轴的夹角。而不是常见的和y负轴的夹角
    #A. compute the transmitted direction wt from w and then project wt into the normal plane, 
        # 先求光线的坐标
        w = np.array([np.sin(theta),np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi)])
        # α, β分别是 w与normal(y)的夹角  和 w投影在表面(xoz)与tangent(x)的夹角
        # cos(α)=y  tan(β)=z/x
        alpha,beta = np.arccos(w[1]),np.arctan2(w[2],w[0])
        sin_alpha,cos_alpha = np.sin(alpha),np.cos(alpha)   
        sin_alpha_t = sin_alpha/eta
        cos_alpha_t = np.sqrt(1 - sin_alpha_t*sin_alpha_t)
        # 求折射后的光线坐标
        wt = np.array([-sin_alpha_t*np.cos(beta),-cos_alpha_t,-sin_alpha_t*np.sin(beta)])
        gamma_t_A = np.arctan(abs(wt[2]/wt[1]))
    #B. use modified index of refraction
        eta_p = np.sqrt(eta*eta - np.sin(theta)*np.sin(theta))/np.cos(theta)
        gamma_o = phi-np.pi/2  # 注意gamma_o可能是负的  gammo是y向-z旋转的角度
        sin_gamma_t = np.sin(abs(gamma_o))/eta_p # 先算和-y夹角的gamma_t
        gamma_t_B = np.arcsin(sin_gamma_t)
        # if np.abs(gamma_t_A - gamma_t_B)>1e-6:
        print(f"φ {np.rad2deg(phi):<7.2f} θ {np.rad2deg(theta):<7.2f} γ {np.rad2deg(gamma_o):<7.2f} {np.rad2deg(gamma_t_A):<7.2f} {np.rad2deg(gamma_t_B):<7.2f} {gamma_t_A-gamma_t_B:<7.2f}")
    # 我们还可以看一下θt
        theta_t_A = np.arcsin(wt[0])
        theta_t_B = w[0]/eta
        print(f"θtA {np.rad2deg(theta_t_A):<7.2f} θtB {np.rad2deg(theta_t_B):<7.2f}")

    @staticmethod
    def RunRandomTests(num_tests=10):
        np.random.seed(42)
        for _ in range(num_tests):
            # 对于wo，phi应该总是在 0-pi之间的
            phi,theta = np.random.rand()*np.pi,(np.random.rand()-0.5)*np.pi
            ModifiedIORChecker.Verify(phi,theta)

ModifiedIORChecker.RunRandomTests(10)
