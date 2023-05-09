import taichi as ti

ti.init(arch=ti.gpu)

# assume mass of all points == [1,1,1,....,1]
RES        = 400                         # resolution
N          = 30
length     =  0.02
g          = -9.8                        # gravity
deltaT     = 0.01                        # time step
anchor     = (0.5,1.0)                   # anchor point for position[0]
iterNum    = 5

p       = ti.Vector.field(2,dtype = ti.f32,shape = N)  # positions
pp      = ti.Vector.field(2,dtype = ti.f32,shape = N)  # previous positions intermediate variable of positions
v       = ti.Vector.field(2,dtype = ti.f32,shape = N)  # velocities
stiffness = ti.field(dtype = ti.f32,shape = N)  

@ti.kernel
def initialize():
    for i in p:
        p[i]         = ti.Vector([anchor[0]-length*i,anchor[1]])
        pp[i]        = p[i]
        v[i]         = ti.Vector([0,0])
        stiffness[i] = 0.9

@ti.kernel
def PredictPositions():
    for i in p:
        v[i][1] += deltaT  * g
        p[i]    += deltaT*v[i]
    
@ti.kernel
def ConstrainsProject():
    for i in p:
        if i==0:
            p[i]   = ti.Vector(anchor)
        else:
            pii_1          = ti.Vector([p[i][0]-p[i - 1][0],p[i][1]-p[i - 1][1]])
            deltaP         = 0.5*pii_1*(length/pii_1.norm() - 1)*stiffness[i]
            p[i]   += deltaP
            p[i-1] -= deltaP

@ti.kernel
def Update():
    p[0] = ti.Vector(anchor)
    for i in p:
        v[i] = (p[i] - pp[i])/deltaT
        pp[i] = p[i]

def Main():
    initialize()
    gui = ti.GUI('Position Based Dynamics -- rope',res = (RES,RES),background_color = 0x666666)
    while not gui.get_event(gui.ESCAPE):
        PredictPositions()
        for _ in range(iterNum):
            ConstrainsProject()
        Update()
        for i in range(N-1):
            gui.line(p[i],p[i+1],color=0xf7ff00,radius=1.5)
        gui.show() 

Main()