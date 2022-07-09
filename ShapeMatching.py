import taichi as ti
ti.init(ti.cpu)

n = 10
a,r = 0.3,0.005 # scale, radius
dt = 0.001

x   = ti.Vector.field(2,ti.f32,n)
xr  = ti.Vector.field(2,ti.f32,n)  # x rest 
xn  = ti.Vector.field(2,ti.f32,n)  # x next
v   = ti.Vector.field(2,ti.f32,n)
R   = ti.Matrix.field(2,2,ti.f32,())
# a rotation with radian w, Rotation(w)
# the deformation matrix is A,the raotion part of it is R_
# use Newton method to solve non-linear equation 
# w = w + dw, Rotation(w) will colser to R_
# where dw = -(A[0,1]*ti.cos(w)-A[1,0]*ti.cos(w)+A[0,0]*ti.sin(w)+A[1,1]*ti.sin(w))/(-A[0,1]*ti.sin(w)+A[1,0]*ti.sin(w)+A[0,0]*ti.cos(w)+A[1,1]*ti.cos(w))

@ti.func
def ExtractRotation(A,R,maxIter):
    for _ in range(maxIter):
        a0 = ti.Vector([A[0,0],A[1,0]])
        a1 = ti.Vector([A[0,1],A[1,1]])
        r0 = ti.Vector([R[0,0],R[1,0]])
        r1 = ti.Vector([R[0,1],R[1,1]])
        w = -(a0.cross(r0) + a1.cross(r1))/(ti.abs(a0.dot(r0) + a1.dot(r1))+1e-9)
        R = ti.Matrix([[ti.cos(w),-ti.sin(w)],[ti.sin(w),ti.cos(w)]]) @ R
    return R

@ti.kernel
def Initialize():
    for i in x:
        v[i] = [0.0,0.0]
        xr[i] = [ti.random()*a+0.35,ti.random()*a+0.35]
        x[i]  = xr[i]
        xn[i] = xr[i]
    R[None] = ti.Matrix.identity(ti.f32,2)

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
    R[None] = ExtractRotation(A_pq,R[None],10)
    for i in x:
        xn[i] = ti.math.clamp(R[None]@(xr[i]-xrc)+xnc,r,1.0-r)
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