import taichi as ti

ti.init(arch = ti.cpu)

dt = 0.001                                      # time step
g = 9.81                                        # gravity constant
p = ti.Vector.field(2,dtype = ti.f32,shape = 3) # position
l = ti.field(dtype = ti.f32,shape = 2)          # length
m = ti.field(dtype = ti.f32,shape = 2)          # mass
ys= ti.field(dtype = ti.f32,shape = 4)          # [ theta1, theta2, omega1, omega2 ]
indices = ti.field(dtype = ti.i32,shape = 4)    # line indices for drawing lines

def Initialize():
    for i in range(2):
        m[i] = 1
        l[i] = 0.2
        indices[2*i] = i
        indices[2*i+1] = i+1
    p[0] = ti.Vector([0.5,0.5]) # base position
    ys[0] = 3.1416*1.5/2
    ys[1] = 0
    ys[2] = ys[3] = 0

@ti.func
def f(y):
    b0 = -(m[0]+m[1])*g*ti.sin(y[0])-m[1]*l[1]*ti.sin(y[0]-y[1])*y[3]*y[3]
    b1 = l[0]*ti.sin(y[0]-y[1])*y[2]*y[2]-g*ti.sin(y[1])
    b = ti.Vector([b0,b1])

    a00 = (m[0]+m[1])*l[0]
    a01 = m[1]*l[1]*ti.cos(y[0]-y[1])
    a10 = l[0]*ti.cos(y[0]-y[1])
    a11 = l[1]
    A_inv = ti.Matrix([[a11,-a01],[-a10,a00]])/(a00*a11-a01*a10)
    acc = A_inv@b
    return ti.Vector([ y[2] , y[3] , acc[0], acc[1] ]) #[ d(theta1)/dt, d(theta2)/dt, d(omega1)/dt, d(omega2)/dt]  

# 4th order Runge-Kutta to solve ODEs
@ti.kernel
def RK4():
    k1 = f(ti.Vector([ys[0],ys[1],ys[2],ys[3]]))
    k2 = f(ti.Vector([ys[0]+dt*k1[0]/2,ys[1]+dt*k1[1]/2,ys[2]+dt*k1[2]/2,ys[3]+dt*k1[3]/2]))
    k3 = f(ti.Vector([ys[0]+dt*k2[0]/2,ys[1]+dt*k2[1]/2,ys[2]+dt*k2[2]/2,ys[3]+dt*k2[3]/2]))
    k4 = f(ti.Vector([ys[0]+dt*k3[0],ys[1]+dt*k3[1],ys[2]+dt*k3[2],ys[3]+dt*k3[3]]))
    for i in ti.static(range(4)):
        ys[i]=ys[i]+(dt/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
    # compute position using theta （从广义坐标得到笛卡尔坐标）
    for i in range(1,3):
        p[i] = p[i-1] + ti.Vector([l[i-1]*ti.sin(ys[i-1]),-l[i-1]*ti.cos(ys[i-1])])

def Main():
    Initialize()
    window = ti.ui.Window("Pendulum",(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        RK4()

        T = 0.5*(m[0]+m[1])*(l[0]**2)*(ys[2]**2)+0.5*m[1]*(l[1]**2)*(ys[3]**2)+m[1]*l[0]*l[1]*ti.cos(ys[0]-ys[1])*ys[2]*ys[3]
        V = -(m[0]+m[1])*l[0]*g*ti.cos(ys[0])-m[1]*l[1]*g*ti.cos(ys[1])
        window.GUI.begin("state info", 0, 0, 0.2, 0.1)
        window.GUI.text("kinetic      "+"{:.4f}".format(T))
        window.GUI.text("potential    "+"{:.4f}".format(V))
        window.GUI.text("total energy "+"{:.6f}".format(T+V))
        window.GUI.end()
        
        canvas.set_background_color((218/255,221/255,216/255))
        canvas.circles(p,radius = 0.01,color=(31/255,110/255,212/255))
        canvas.lines(p,indices = indices,width = 0.005, color=(57/255,186/255,232/255))
        window.show()

Main()