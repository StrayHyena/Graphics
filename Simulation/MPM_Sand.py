import taichi as ti

# ti.init(arch=ti.cpu,debug=True,cpu_max_num_threads=1)
ti.init(arch=ti.gpu)

frameCnt = ti.field(dtype = ti.i32,shape=())
Np   = 10000  # particle number
x    = ti.Vector.field(2,dtype = ti.f32,shape = Np)
v    = ti.Vector.field(2,dtype = ti.f32,shape = Np)
Fe   = ti.Matrix.field(2,2,dtype = ti.f32,shape = Np)
Fp   = ti.Matrix.field(2,2,dtype = ti.f32,shape = Np)
C    = ti.Matrix.field(2,2,dtype = ti.f32,shape = Np)
q    = ti.field(dtype = ti.f32,shape = Np)
Vol  = ti.field(dtype = ti.f32,shape = Np)

Ng = 256  # grid size
dx = 1/Ng
vg = ti.Vector.field(2,dtype = ti.f32,shape = (Ng,Ng))
fg = ti.Vector.field(2,dtype = ti.f32,shape = (Ng,Ng))
mg = ti.field(dtype = ti.f32 , shape = (Ng,Ng))

pi = 3.141592653589793238462643383279502884197169399375105821
E,nu = 3.537e5,0.3
rho = 2200
mu0,lambda0 =E/(2*(1+nu)) ,nu*E/((1+nu)*(1-2*nu))
h = (35.0*(pi/180),9.0*(pi/180),0.2*(pi/180),10.0*(pi/180))
dt = 1e-4
m = 0.2**2*rho/Np
bound = 3

# w_ip = N_i(x_p) = N(x_p-i*h)
@ti.func
def N(x):
    ret,x = 0.0,ti.abs(x)
    if x<0.5:           ret = 0.75-x*x
    elif 0.5<=x<1.5:    ret = 0.5*(1.5-x)**2
    return ret

@ti.func
def Nd1(x): # 1st order derivative
    ret,absx = 0.0,ti.abs(x)
    if absx<0.5:            ret = -2*x
    elif 0.5<=absx<1.5:     ret = (absx-1.5)*(absx/x)
    return ret

# x should in grid space(i.e. world position / dx) , not world space
@ti.func
def Weight(x):
    return N(x[0])*N(x[1])

@ti.func
def WeightGradient(x):
    wx,wy,wdx,wdy = N(x[0]),N(x[1]),Nd1(x[0]),Nd1(x[1])
    return ti.Vector([wdx*wy/dx,wdy*wx/dx])

@ti.func
def Ln(T):  # apply ln(x) to diagonal elements
    ret = ti.Matrix.zero(float,2,2)
    ret[0,0] = ti.log(T[0,0])
    ret[1,1] = ti.log(T[1,1])
    return ret

@ti.kernel
def Initialize():
    frameCnt[None] = 0
    for i,j in mg:
        mg[i,j] = 0
        fg[i,j] = vg[i,j] = [0,0]
    for pi in x:
        x[pi]   = ti.Vector([0.4,0.5]) + ti.Vector([ti.random()*0.2,ti.random()*0.2])
        v[pi]   = ti.Vector([0.0,0.0])
        Fe[pi]  = ti.Matrix.identity(float,2)
        Fp[pi]  = ti.Matrix.identity(float,2)
        C[pi]   = ti.Matrix.zero(float,2,2)
        Vol[pi] = q[pi] = 0.0
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                nx = x[pi]/dx - (base+offset)
                w = [N(nx[0]),N(nx[1])]
                mg[base+offset]+=w[0]*w[1]*m
    for pi in x:
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                nx = x[pi]/dx - (base+offset)
                w = [N(nx[0]),N(nx[1])]
                Vol[pi] += w[0]*w[1]*mg[base+offset]
        Vol[pi] = m*dx*dx/Vol[pi]

@ti.kernel
def SubStep():
    frameCnt[None]+=1
    for i,j in mg:
        mg[i,j] = 0
        fg[i,j] = vg[i,j] = [0,0]
    for pi in x:
        F         = Fe[pi]
        U,sigma,V = ti.svd(F)
        lnSigma   = Ln(sigma)
        invSigma  = sigma.inverse()
        dPsi_dF   = U @ (2*mu0*invSigma@lnSigma + lambda0*lnSigma.trace()*invSigma) @ V.transpose()
        dPsi_dF   = Vol[pi]*dPsi_dF @ F.transpose()
        gridCoord = x[pi]/dx
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                ij = base + ti.Vector([i,j])
                weight = Weight(gridCoord-ij)
                mg[ij] += weight*m
                vg[ij] += (v[pi] + C[pi]@(ij*dx-x[pi]))*weight*m
                fg[ij] -= dPsi_dF @ WeightGradient(gridCoord-ij)
                # if frameCnt[None]==debugFrameID and ij[0]==64: print('I',pi,ij,mg[ij],vg[ij],fg[ij],dPsi_dF,WeightGradient(gridCoord-ij))

    for i,j in mg:
        if mg[i,j] <= 0:continue
        vg[i,j]    /= mg[i,j]
        vg[i,j][1] += dt*-50
        vg[i,j]    += dt*fg[i,j]/mg[i,j]
        if i<=bound or i>=Ng-bound :vg[i,j][0] = 0.0
        if j<=bound or j>=Ng-bound :vg[i,j][1] = 0.0
        # if frameCnt[None]==debugFrameID and i<70: print('II',i,j,vg[i,j],mg[i,j],fg[i,j])

    for pi in x:
        newV = ti.Vector.zero(float,2)
        B    = ti.Matrix.zero(float,2,2)
        T    = ti.Matrix.zero(float,2,2)

        gridCoord = x[pi]/dx
        base = int(gridCoord-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                ij = base + ti.Vector([i,j])
                weight = Weight(gridCoord - ij)
                newV +=  weight*vg[ij]
                B    +=  weight*vg[ij].outer_product(ij*dx-x[pi])
                T    +=  vg[ij].outer_product( WeightGradient(gridCoord-ij))

        # if debugFrameID==frameCnt[None]:print(pi,newV,x[pi],Fe[pi])
        
        C[pi] = B*4/(dx*dx)
        v[pi] = newV
        x[pi] += dt*v[pi]
        Fe[pi] = (ti.Matrix.identity(float,2)+dt*T)@Fe[pi]
       
        Psi_F     = h[0]+(h[1]*q[pi]-h[3])*ti.exp(-h[2]*q[pi])
        alpha     = ti.sqrt(2.0/3.0)*2.0*ti.sin(Psi_F)/(3.0-ti.sin(Psi_F))
        U,sigma,V = ti.svd(Fe[pi])
       
        T  = sigma
        dq = 0.0
        e  = Ln(sigma)
        tre     = e.trace()
        eHat    = e - tre/2 * ti.Matrix.identity(float,2)
        eHatF   = ti.sqrt( (eHat.transpose()@eHat).trace() )
        gamma   = eHatF + (lambda0/mu0 + 1)*tre*alpha
        if eHatF==0 or tre > 0.0:
            T  = ti.Matrix.identity(float,2)
            dq = ti.sqrt( (e.transpose()@e).trace() )
        elif gamma > 0.0:
            H       = e-gamma*eHat/eHatF
            T[0,0]  = ti.exp(H[0,0])
            T[1,1]  = ti.exp(H[1,1])
            dq      = gamma

        Fe[pi]    = U@T@V.transpose()
        Fp[pi]    = V@T.inverse()@sigma@V.transpose()@Fp[pi]
        q[pi]     += dq

def Main():
    Initialize()
    window = ti.ui.Window("MPM Sand",(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        SubStep()
        canvas.set_background_color((245/255,245/255,245/255))
        canvas.circles(x,radius = 0.001,color=(183/255,134/255,11/255))
        window.show()

Main()