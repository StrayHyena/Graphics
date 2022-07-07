import taichi as ti
ti.init(ti.cpu)

n = 10
a,r = 0.3,0.005 # scale, radius
dt = 0.001

x   = ti.Vector.field(2,ti.f32,n)
xr  = ti.Vector.field(2,ti.f32,n)  # x rest 
xn  = ti.Vector.field(2,ti.f32,n)  # x next
v   = ti.Vector.field(2,ti.f32,n)

@ti.kernel
def Initialize():
    for i in x:
        v[i] = [0.0,0.0]
        xr[i] = [ti.random()*a+0.35,ti.random()*a+0.35]
        x[i]  = xr[i]
        xn[i] = xr[i]

@ti.kernel
def Substep():
    xrc  = ti.Vector([0.0,0.0])
    xnc  = ti.Vector([0.0,0.0])
    for i in xn:
        v[i]  += ti.Vector([0.0,-2.0])*dt
        xn[i] = x[i] + v[i]*dt
        xrc += xr[i]
        xnc += xn[i]
    xrc /=n
    xnc /=n
    A_pq = ti.Matrix.zero(ti.f32,2,2)
    for i in x:
        A_pq += (xn[i]-xnc)@(xr[i]-xrc).transpose()
    R,_ = ti.polar_decompose(A_pq)
    for i in x:
        xn[i] = ti.math.clamp(R@(xr[i]-xrc)+xnc,r,1.0-r)
        v[i] = (xn[i]-x[i])/dt
        x[i] = xn[i]

def Main():
    Initialize()
    window = ti.ui.Window('Shape Matching',(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        Substep()

        canvas.set_background_color((0.54,0.89,0.999))
        canvas.circles(x,radius = r,color=(0.78,0.31,0.03))
        window.show()

Main()