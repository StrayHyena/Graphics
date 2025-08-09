import torch,math
import numpy as np
from plot import Ploter

class Utils:
    @staticmethod
    def ArmijoWolfe(f,x,d,alpha = 0.5):
        c1,c2 = 1e-3,0.9
        grad = Utils.Gradient(f,x)
        Armijo = lambda alpha: f(x+d*alpha) <= f(x) + c1*alpha*np.dot(grad,d)
        Wolfe  = lambda alpha: np.dot(Utils.Gradient(f,x+d*alpha),d) >= c2*np.dot(grad,d)
        for _ in range(int(1e3)):
            if Armijo(alpha) and Wolfe(alpha): return alpha
            if not Armijo(alpha):
                alpha = -np.dot(grad,d)*alpha**2/(2*f(x+alpha*d)-f(x)-np.dot(grad,d)*alpha)
            if not Wolfe(alpha):
                alpha += np.dot(Utils.Gradient(f,x+d*alpha),d)*alpha/(np.dot(grad,d) - np.dot(Utils.Gradient(f,x+d*alpha),d) )
        if not Armijo(alpha):print('warning: alpha ',alpha ,' is not conform Armijo')
        if not Wolfe(alpha): print('warning: alpha ',alpha ,' is not conform Wolfe')

    @staticmethod
    def Visualize(f,pts,method_name):
        if len(pts)>=2: print(method_name,' iter num ',len(pts)-1,' Min point ',pts[-1])
        plter = Ploter()
        plter.DrawFunction(f,xrange=(-10,10),yrange=(-4,4), draw_type='s')
        if len(pts)>=2:plter.DrawLines(pts)
        plter.Show()

    @staticmethod
    def Gradient(f,x):
        x = torch.from_numpy(x).requires_grad_(True)
        if x.grad: x.grad.zero_()
        z = f(x)
        z.backward()
        return x.grad.numpy()

    @staticmethod
    def Hessian(f,x):
        x = torch.from_numpy(x).requires_grad_(True)
        h = torch.autograd.functional.hessian(f,x)
        return h.numpy()

class GradientMethod:
    @staticmethod
    def GD(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        xs = np.array([x0])
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1].copy()
            d = -Utils.Gradient(f,x)
            if np.linalg.norm(d)<threshold:break
            alpha = Utils.ArmijoWolfe(f,x,d,1e-3)
            x += alpha*d
            xs = np.concatenate((xs,[x]))
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

    @staticmethod
    def BB(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        alpha_min,alpha_max = 1e-5,1e5
        xs = np.array([x0])
        eta,c1 = 0.5,0.5
        Cs,Qs = [f(xs[-1])],[1]
        alpha = Utils.ArmijoWolfe(f,xs[-1],-Utils.Gradient(f,xs[-1]))
        
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1]
            C = Cs[-1]
            Q = Qs[-1]
            d = -Utils.Gradient(f,x)
            
            if np.linalg.norm(d)<threshold:break
            
            x_next = x + d*alpha
            for __ in range(10):
                if f(x_next) <= C + c1*alpha*np.dot(Utils.Gradient(f,x),d): break
                alpha *= eta
                x_next = x+d*alpha

            s = x_next - x
            y = Utils.Gradient(f,x_next) - Utils.Gradient(f,x)
            alpha_BB1 = np.dot(s,y)/np.dot(y,y)
            alpha_BB2 = np.dot(s,s)/np.dot(s,y)
            if _%2==1:  alpha = alpha_BB1
            else:       alpha = alpha_BB2
            alpha = min(max(alpha,alpha_min),alpha_max)

            x += alpha*d
            xs = np.concatenate((xs,[x]))
            Qs.append(Q*eta+1)
            Cs.append((eta*C*Q+f(x_next))/Qs[-1])

        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

class NewtonMethod:
    @staticmethod
    def BFGS(f,x0 = np.array([0,0.]),maxIterNum=int(1e2),threshold = 1e-4):
        xs = np.array([x0])
        H = np.eye(2)
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1]
            d = -H@Utils.Gradient(f,x)
            if np.linalg.norm(d)<threshold:break
            alpha = Utils.ArmijoWolfe(f,x,d)
            x_1 = x + alpha*d
            
            s = x_1 - x
            y = Utils.Gradient(f,x_1) - Utils.Gradient(f,x)
            rho = 1/np.dot(s,y)
            M = np.eye(2) - rho*np.outer(y,s)
            H = M.T@H@M + rho*np.outer(s,s)

            xs = np.concatenate((xs,[x_1]))
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

    # return H @ nabla(f)
    def DoubleLoop(f,x,ss,ys,m):
        assert len(ss)>=m
        assert len(ys)>=m
        # compute d
        q = Utils.Gradient(f,x)
        alphas = []  # [alpha_{k-1} alpha_{k-2}  alpha_{k-m}  ] 
        # 注意，公式里的 s^{k-1} 现在就是  ss[-1],
        # i 从 -1 到 -m,对应公式里的  k-1 到 k-m
        for i in range(-1,-m-1,-1):
            alphas.append(1.0/np.dot(ss[i],ys[i])*np.dot(ss[i],q))
            q -= alphas[-1]*ys[i]
        assert len(alphas)==m
        r = np.dot(ss[-1],ys[-1])/np.dot(ys[-1],ys[-1]) * q
        # i 从 -m 到 -1,对应公式里的  k-m 到 k-1, 现在alphas是从 k-1,...k-m,  所以alpha_{k-m} 对应 alphas[-1]
        # 对alphas inverse. alphas是从 k-m,...k-1。 alphas[-1]对应公式里的 alpha{k-1} ,alphas[-m]对应公式里的 alpha{k-m} ,
        alphas = alphas[::-1]
        for i in range(-m,0):
            beta = 1.0/np.dot(ss[i],ys[i]) * np.dot(ys[i],r)
            r += (alphas[i]-beta)*ss[i]
        return r

    @staticmethod
    def LBFGS(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        ss,ys = [],[]
        xs = np.array([x0])

        # 多走一步，让DoubleLoop可以启动
        x = xs[-1]
        d = -Utils.Gradient(f,x)
        alpha = Utils.ArmijoWolfe(f,x,d)
        x_next = x + alpha*d
        xs = np.concatenate((xs,[x_next]))
        ss.append(x_next-x)
        ys.append(Utils.Gradient(f,x_next)-Utils.Gradient(f,x))

        # 注意，只有当x^(k+1)被计算出来的时候，s^k才被计算出来
        m = 5
        for k in range(1,maxIterNum):
            if k==maxIterNum-1:return None
            x = xs[-1]

            # 当 ss,ys不足m时，就用小m
            d = -NewtonMethod.DoubleLoop(f,x,ss,ys,min(k,m))
            if np.linalg.norm(d)<threshold:break
            alpha = Utils.ArmijoWolfe(f,x,d)
            x_next = x + alpha*d

            ss.append( x_next - x )
            ys.append( Utils.Gradient(f,x_next) - Utils.Gradient(f,x))
            xs = np.concatenate((xs,[x_next]))
            
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

class TrustRegionMethod:
    # 用截断共轭梯度来求解子问题
    def SubProblem(f,x,radius,grad,B):
        s = np.zeros((2,))
        r = grad.copy()
        p = -r.copy()
        r0 = r.copy()
        if np.linalg.norm(p)<1e-4:return s
        for _ in range(int(1e4)):
            pBp = p.T@B@p
            alpha = np.dot(r,r)/pBp
            s_next = s + alpha*p
            if pBp <= 0 or np.linalg.norm(s_next)>=radius:  # alpha <=0 means p.T@B@p<=0
                rs = np.roots([np.dot(p,p),2*np.dot(p,s),np.dot(s,s)-radius**2])
                for root in sorted(rs):
                    if root > 0:return s+root*p
            assert alpha > 0
            r_next  = r + alpha*B@p
            if np.linalg.norm(r_next)<1e-4*np.linalg.norm(r0):
                return s_next
            beta = np.dot(r_next,r_next)/np.dot(r,r)
            p = beta*p-r_next
            s = s_next
            r = r_next  
        print('should not be here',p,s)

    @staticmethod
    def TrustRegion(f,x0 = np.array([0,0.]),maxIterNum=int(1e4),threshold = 1e-4):
        radius_max,radius =  10,0.1
        rho1_bar,rho2_bar,gamma1,gamma2 = 0.25,0.75,0.25,2
        eta = 0.8*rho1_bar
        xs = np.array([x0])
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1].copy()
            B = Utils.Hessian(f,x)
            g = Utils.Gradient(f,x)
            if np.linalg.norm(g)<threshold:break
            m = lambda d: f(x)+np.dot(g,d)+0.5*d.T@B@d
            # d是下降步长加方向。由于子问题从零开始猜测，所以d也是子问题的极小值点
            d = TrustRegionMethod.SubProblem(m,x,radius,g,B)
            rho = (f(x) - f(x+d))/(m(np.zeros((2,)))-m(d))

            if rho<rho1_bar:radius *= gamma1
            elif rho>rho2_bar or abs(np.linalg.norm(d)-radius)<1e-5: radius = min(gamma2*radius,radius_max)

            if rho>eta: xs = np.concatenate((xs,[x+d]))

        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

# P461  用于机器学习的随机优化方法,出于演示的目的,这里没用随机
class StochasticMethod:
    @staticmethod
    def Momentum(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        xs = np.array([x0])
        lr,mu = 0.1,0.75
        # 为了得到v的初始值，要先进行一步
        v  = -lr*Utils.Gradient(f,xs[-1])
        xs = np.concatenate((xs,[xs[-1]+v]))
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1].copy()
            d = -Utils.Gradient(f,x)
            if np.linalg.norm(d)<threshold:break

            v = mu*v + lr*d
            x = x + v

            xs = np.concatenate((xs,[x]))
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts
    
    # 缺点是 学习率过快的衰减到零
    @staticmethod
    def AdaGrad(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        xs = np.array([x0])
        lr = 0.0001
        G,epsilon  = np.zeros(xs[-1].shape),1e-8*np.ones(xs[-1].shape)
        for _ in range(maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1].copy()
            g = Utils.Gradient(f,x)
            if np.linalg.norm(g)<threshold:break
            
            G += g*g
            x -= lr*g/np.sqrt(G+epsilon)

            xs = np.concatenate((xs,[x]))
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

    @staticmethod
    def Adam(f,x0 = np.array([0,0.]),maxIterNum=int(1e5),threshold = 1e-4):
        xs = np.array([x0])
        # lr 一般取0.001 
        lr,rho1,rho2 = 0.1,0.9,0.999
        S,M,epsilon = np.zeros(xs[-1].shape),np.zeros(xs[-1].shape),np.ones(xs[-1].shape)
        for _ in range(1,maxIterNum):
            if _==maxIterNum-1:return None
            x = xs[-1].copy()
            g = Utils.Gradient(f,x)
            if np.linalg.norm(g)<threshold:break

            S = rho1*S + (1 - rho1) * g
            M = rho2*M + (1 - rho2) * g * g
            S_hat = S/(1-rho1**_)
            M_hat = M/(1-rho2**_)
            x = x - lr*S_hat/np.sqrt(M_hat+epsilon)

            xs = np.concatenate((xs,[x]))
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

def UnconstrainedTest(method_name):
    f = lambda x: 1*x[0]**2 + 10*x[1]**2
    if method_name=='TR':       pts = TrustRegionMethod.TrustRegion(f,[10.,1.])
    elif method_name=='BFGS':   pts = NewtonMethod.BFGS(f,[10.,1.])
    elif method_name=='LBFGS':  pts = NewtonMethod.LBFGS(f,[10.,1.])
    elif method_name=='GD':     pts = GradientMethod.GD(f,[10.,1.])
    elif method_name=='BB':     pts = GradientMethod.BB(f,[10.,1.])
    elif method_name=='Mom':    pts = StochasticMethod.Momentum(f,[10.,1.])
    # elif method_name=='AD':     pts = StochasticMethod.AdaGrad(f,[10.,1.])
    elif method_name=='ADAM':   pts = StochasticMethod.Adam(f,[10.,1.])
    Utils.Visualize(f,pts,method_name)

# 约束优化： 参数c表示约束。是一个字典。key是类型，有两种，'eq','ineq'。ineq 表示 c(x)<=0  ----------------------
class PenaltyMethod:
    @staticmethod
    def Penalty(f,cons,x0,maxIterNum=int(1e3),threshold = 1e-4):
        sigma = 1
        rho = 2
        penalty = lambda x:  (    sum([max(c(x),0)**2 for c in cons.get('ineq',[])])  \
                                + sum([c(x)**2        for c in cons.get('eq',[])])  )
        xs = np.array([x0])
        for _ in range(maxIterNum):
            x = xs[-1].copy()
            pts = GradientMethod.GD(lambda x: f(x) + sigma*penalty(x) ,x)
            if pts is None:  #  pts is None 表示发散
                sigma*=rho
                continue
            x = pts[-1][:2]
            xs = np.concatenate((xs,[x]))
            if sigma*penalty(x)<threshold:break
            else: sigma*=rho
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

class AugmentedLagrangianMethod:
    @staticmethod
    # 只考虑等式约束的增广拉格朗日法
    def AL_EQ(f,cons,x0):
        eq_cons = cons.get('eq',[])
        sigma = 1
        rho = 2
        xs = np.array([x0])
        l  = np.zeros((len(eq_cons)))
        penalty = lambda x: sum([c(x)**2 for c in eq_cons])
        for _ in range(int(1e3)):
            x = xs[-1].copy()
            pts = GradientMethod.GD(lambda x: f(x) + sum([l[i]*c(x) for i,c in enumerate(eq_cons)]) + sigma*penalty(x) ,x ) 
            if pts is None:  #  pts is None 表示发散
                sigma*=rho
                continue
            x = pts[-1][:2]
            xs = np.concatenate((xs,[x]))
            if penalty(x)<1e-8:break
            else:
                l+=sigma*np.array([ c(x) for c in eq_cons ]) 
                sigma*=rho
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

    @staticmethod
    # 只考虑不等式约束的增广拉格朗日法
    def AL_INEQ(f,cons,x0):
        ineq_cons = cons.get('ineq',[])
        sigma = 10 # sigma为1的话,可能会陷入循环(看eta,epsilon的计算)
        alpha,beta = 0.5,0.8
        rho = 2
        eta,epsilon = 1/sigma, 1/(sigma**alpha)
        eta_stop, epsilon_stop = 1e-5, 1e-5
        xs = np.array([x0])
        mu  = np.zeros(len(ineq_cons))

        # P319 (7.2.12) 第二行的项  注意现在只公式里关注i∈I的部分
        penalty = lambda x: sum([ max(mu[i]/sigma + c(x),0)**2 - (mu[i]/sigma)**2  for i,c in enumerate(ineq_cons)])
        # P319 最后一个公式 衡量违反程度
        violation = lambda x: math.sqrt(sum( [max(c(x),-mu[i]/sigma)**2 for i,c in enumerate(ineq_cons)] ) )
        for _ in range(int(1e5)):
            x = xs[-1].copy()
            # 本轮优化的目标函数
            object_func = lambda x: f(x) + sigma/2*penalty(x)
            # P321 算法7.6  步骤3的判别条件有误，应该是对x求梯度
            pts = GradientMethod.GD( object_func ,x,threshold=eta )
            x = pts[-1][:2]
            xs = np.concatenate((xs,[x]))
            if pts is not None and  violation(x)<epsilon:
                if violation(x)<=epsilon_stop and np.linalg.norm(Utils.Gradient(object_func,x))<=eta_stop: break
                # 如果解x满足粗约束，说明此罚因子sigma可能有效，就不改它。进而把粗约束变细
                else:
                    for i in range(len(mu)):
                        mu[i] = max(mu[i]+sigma*ineq_cons[i](x),0)
                    eta,epsilon = eta/sigma,epsilon/(sigma**beta)
            else:
                sigma*=rho
                eta,epsilon = 1/sigma,1/(sigma**alpha)
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

    @staticmethod
    def AL(f,cons,x0):
        ineqs,eqs = cons.get('ineq',[]),cons.get('eq',[])
        sigma,rho,alpha,beta = 10,2,0.5,0.8
        eta_stop, epsilon_stop = 1e-5, 1e-5
        eta,epsilon = 1/sigma, 1/(sigma**alpha)
        xs = np.array([x0])
        mu,lam  = np.zeros(len(ineqs)),np.zeros(len(eqs))
        
        # 这一项是 P319 (7.2.12)公式中系数有σ的那两项
        penalty   = ( lambda x:  
            sum([ c(x)**2  for c in eqs]) + 
            sum([ max(mu[i]/sigma + c(x),0)**2 - (mu[i]/sigma)**2  for i,c in enumerate(ineqs)]) )
        violation = ( lambda x: math.sqrt( 
            sum([ c(x)**2 for c in eqs     ])  + 
            sum([ max(c(x),-mu[i]/sigma)**2 for i,c in enumerate(ineqs) ]) ) ) 
        
        for _ in range(int(1e5)):
            x = xs[-1].copy()
            object_func = lambda x: f(x) + sum([ lam[i]*c(x) for i,c in enumerate(eqs)  ]) + sigma/2*penalty(x)
        
            pts = GradientMethod.GD( object_func ,x,threshold=eta )
        
            x = pts[-1][:2] # 这里的2是因为，测试的x是二维的（2个自由度）
            xs = np.concatenate((xs,[x]))
        
            if pts is not None and  violation(x)<epsilon:
                if violation(x)<=epsilon_stop and np.linalg.norm(Utils.Gradient(object_func,x))<=eta_stop: break
                else:
                    for i in range(len(lam)):lam[i] = lam[i]+sigma*eqs[i](x)
                    for i in range(len(mu)): mu[i]  = max(mu[i]+sigma*ineqs[i](x),0)
                    eta,epsilon = eta/sigma,epsilon/(sigma**beta)
            else:
                sigma*=rho
                eta,epsilon = 1/sigma,1/(sigma**alpha)
        zs  = [f(p) for p in xs]
        pts = np.c_[xs,zs]
        return pts

def ConstrainedTest(method_name):
    f = lambda x: 1*x[0]**2 + 10*x[1]**2
    cons = {'ineq':[ lambda x:-0.5*(x[0]-1)+x[1],lambda x:-0.5*(x[0]-1)-x[1] ],
            'eq': [lambda x: x[0]-x[1]**2-1 ]}
    if method_name=='P':    pts = PenaltyMethod.Penalty(f,cons,[10.0,1.0])
    elif method_name=='AL': pts = AugmentedLagrangianMethod.AL(f,cons,[10.0,1.0])
    
    # 就不可视化了，毕竟迭代的点的目标函数已经有罚项了
    print(method_name,' iter num ',len(pts)-1,' Min point ',pts[-1])

UnconstrainedTest('BB')
# ConstrainedTest('AL')

# 演示ADMM。 min f(x) = x[0]**2 + 10*x[1]**2 等价于
# min  f1(x) + f2(z) = 0.7*x[0]**2+3*x[1]**2-2*x[0]*x[1]-x[0]-x[1]  +  0.3*z[0]**2+7*z[1]**2+2*z[0]*z[1]+z[0]+z[1] 
# s.t  x = z
# def ADMM():
#     f1 = lambda x: 7*x[0]**2+3*x[1]**2-2*x[0]*x[1]-x[0]-x[1]
#     f2 = lambda z: 3*z[0]**2+7*z[1]**2+2*z[0]*z[1]+z[0]+z[1]
#     # print(np.linalg.eigvals(  Utils.Hessian(f,np.array([1,1.]))) )
#     x,z = np.array([1,10.]),np.array([1,10.])
#     y = np.zeros((2,))
#     sigma,rho = 10,2
#     for _ in range(int(1e3)):
#         L1 = lambda x: f1(x) + (x[0]-z[0])*y[0]+(x[1]-z[1])*y[1] + sigma/2*((x[0]-z[0])**2 + (x[1]-z[1])**2)
#         ptsx = GradientMethod.GD(L1,x)
#         x1 = ptsx[-1][:2]
#         print(_,'x ',x1,' start x ',x,' y ',y,' sigma ',sigma, ' z ',z)
#         L2 = lambda z: f2(z) + (x1[0]-z[0])*y[0]+(x1[1]-z[1])*y[1] + sigma/2*((x1[0]-z[0])**2 + (x1[1]-z[1])**2)
#         ptsz = GradientMethod.GD(L2,z)
#         z1 = ptsz[-1][:2]
#         y += sigma*(x1-z1)
#         if np.linalg.norm(x-z)<1e-8 and np.linalg.norm(z1-z)<1e-8:break
#         x = x1
#         z = z1
#         sigma*=rho
#     print(x,z)

# ADMM()