import taichi as ti

ti.init(arch=ti.cpu)

dx = 1/64
elementShape = (16,10)
vertexShape  = (elementShape[0]+1,elementShape[1]+1)
elementN     = elementShape[0]*elementShape[1]*2
vertexN      = vertexShape[0]*vertexShape[1]
area         = dx*dx/2
dt = 1e-5                        # time step
g  = -9.81                       # gravity constant
E,nu = 5e6,0.45
mu0,lambda0 =E/(2*(1+nu)) ,nu*E/((1+nu)*(1-2*nu))

p    = ti.Vector.field(2,dtype = ti.f32,shape = vertexN)     # postioins
v    = ti.Vector.field(2,dtype = ti.f32,shape = vertexN)     # velocities
a    = ti.Vector.field(2,dtype = ti.f32,shape = vertexN)     # acceleration
e    = ti.Vector.field(3,dtype = ti.i32,shape = elementN)    # element
DmInv= ti.Matrix.field(2,2,dtype = ti.f32,shape = elementN)    # element
indices = ti.field(dtype = ti.i32,shape=6*elementN)

@ti.kernel
def Initialize():
    base = ti.Vector([(1.0-dx*elementShape[0])/2.0,0.95-elementShape[1]*dx])
    for i in p:
        p[i]    = ti.Vector([i%vertexShape[0],i//vertexShape[0]])*dx + base
        v[i]    = ti.Vector([i//vertexShape[0]-i%vertexShape[1],-2*(i%vertexShape[0]-elementShape[0])])*0.05
        a[i]    = ti.Vector([0.0,g])
    for i in e: 
        offsety,offsetx = i//(2*elementShape[0]),(i%(2*elementShape[0]))
        offset = (offsety*vertexShape[0] + offsetx//2 )* ti.Vector([1,1,1])
        if i%2==0: e[i] = ti.Vector([0,    1,         vertexShape[0]])   + offset
        else:      e[i] = ti.Vector([1,vertexShape[0],vertexShape[0]+1]) + offset
        for j in ti.static(range(6)): indices[6*i+j] = e[i][j%3]
        DmInv[i] = ti.Matrix.cols([ p[e[i][1]]-p[e[i][0]] ,p[e[i][2]]-p[e[i][0]] ]).inverse()

@ti.kernel
def Step():
    # compute acceleration
    for i in e:
        Ds      = ti.Matrix.cols([ p[e[i][1]] - p[e[i][0]] ,p[e[i][2]] - p[e[i][0]] ])
        F       = Ds @ DmInv[i]
        Strain  = (F.transpose()@F - ti.Matrix.identity(float,2)) / 2               # St.Venant-Kirchhoff
        P       = F @ (2*mu0*Strain + lambda0*Strain.trace()*ti.Matrix.identity(float,2))
        dPsi_dx = P @ DmInv[i].transpose()
        a1 = ti.Vector([dPsi_dx[0,0],dPsi_dx[1,0]])
        a2 = ti.Vector([dPsi_dx[0,1],dPsi_dx[1,1]])
        acc = [-a1-a2,a1,a2]
        for j in ti.static(range(3)):  a[e[i][j]] -= area*acc[j]
    #time integration
    for i in p:
        v[i]+=a[i]*dt
        p[i]+=v[i]*dt
        if p[i][1]<0.001:
            p[i][1] = 0.001
            v[i][1] *= -1
        if p[i][0]<0.001 or p[i][0]>0.999:
            p[i][0] = ti.min(ti.max(0.001,p[i][0]),0.999)
            v[i][0] *= -1
        a[i] = ti.Vector([0.0,g])

def Main():
    Initialize()
    window = ti.ui.Window("FEM ELASTIC",(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        for _ in range(100):Step()
        canvas.set_background_color((218/255,221/255,216/255))
        canvas.lines(p,indices = indices,width = 0.0025, color=(57/255,186/255,232/255))
        window.show()

Main()