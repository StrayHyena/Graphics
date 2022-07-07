import taichi as ti

ti.init(arch = ti.cpu)

n,l = 5,0.015
# n,l = 3,0.2
m = n-1
dt = 0.001
x     = ti.Vector.field(2,ti.f32,n)
dx    = ti.Vector.field(2,ti.f32,n)
xn    = ti.Vector.field(2,ti.f32,n)
v     = ti.Vector.field(2,ti.f32,n)
mi    = ti.field(ti.f32,n)   # mass inverse
lam   = ti.field(ti.f32,m)   # lambda       : Lagrange mutliplier
k     = ti.field(ti.f32,m)   # stiffness       : Lagrange mutliplier
d     = ti.field(ti.f32,m)   # damping ratio       : Lagrange mutliplier
at    = ti.field(ti.f32,m)   # alpha tilde  : 1/(dt*dt* stiffness)
bt    = ti.field(ti.f32,m)   # beta  tilde       : constraint damping

@ti.kernel
def Initialize():
    for i in x:
        x[i] = [0.5 - i*l ,0.5]
        v[i] = [0.0,0.0]
        mi[i] = 1.0
    mi[0] = 0.0
    for i in lam:
        lam[i] = 0.0
        k[i] = 1e3
        d[i] = 0.001
        at[i]  = 1/(dt*dt*k[i])
        bt[i]  = k[i]*d[i]*dt*dt

@ti.kernel
def Predict():
    for i in x:    xn[i] = x[i] + dt*v[i]+dt*dt*ti.Vector([0,-2])*mi[i]
    for i in lam:  lam[i] = 0.0

@ti.kernel
def SolveConstraint():
    for i in dx:  dx[i] = [0.0,0.0]
    for j in lam:
        C =  (xn[j]-xn[j+1]).norm() - l
        dC_dxj1 = (xn[j+1]-xn[j]).normalized()
        dC_dxj  = -dC_dxj1
    
        # dLam = -(C + at[j]*lam[j])/( at[j] + (dC_dxj.norm_sqr() * mi[j] + dC_dxj1.norm_sqr() * mi[j+1]) )
        dLam = -(C + at[j]*lam[j] +(dC_dxj.dot(xn[j]-x[j])+dC_dxj1.dot(xn[j+1]-x[j+1])) * at[j]*bt[j]/dt )/( at[j] + (dC_dxj.norm_sqr() * mi[j] + dC_dxj1.norm_sqr() * mi[j+1])* (1+at[j]*bt[j]/dt) )
        dx[j]   += dC_dxj *  dLam * mi[j]
        dx[j+1] += dC_dxj1 * dLam * mi[j+1]
        lam[j]  += dLam
    for i in xn:
        xn[i]   += dx[i] 


@ti.kernel
def Update():
    for i in x:
        v[i] = (xn[i]-x[i])/dt
        x[i] = xn[i]

def Main():
    Initialize()
    window = ti.ui.Window('XPBD_Chain',(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        Predict()
        for _ in range(50):
            SolveConstraint()
        Update()

        canvas.set_background_color((0.7,0.7,0.7))
        canvas.circles(x,radius = 0.005,color=(0.0,0.0,0.0))
        window.show()

Main()