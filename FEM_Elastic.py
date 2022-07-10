import taichi as ti

ti.init(arch=ti.cpu)

elementShape = (20,2)
vertexShape  = (elementShape[0]+1,elementShape[1]+1)
elementN     = elementShape[0]*elementShape[1]*2
vertexN      = vertexShape[0]*vertexShape[1]
dx           = 1/64
area         = dx*dx/2
dt           = 16.7e-3/1000                       # time step
g            = -9.8                               # gravity constant
E,nu         = 1e6,0.0
mu0,lambda0  = E/(2*(1+nu)) ,nu*E/((1+nu)*(1-2*nu))
STVK,CR,NH   = 0,1,2

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
        v[i]    = ti.Vector([i//vertexShape[0]-i%vertexShape[1]+3,(i%vertexShape[0]-elementShape[0])])*0.5
        a[i]    = ti.Vector([0.0,g])
    for i in e: 
        offsety,offsetx = i//(2*elementShape[0]),(i%(2*elementShape[0]))
        offset = (offsety*vertexShape[0] + offsetx//2 )* ti.Vector([1,1,1])
        if i%2==0: e[i] = ti.Vector([0,    1,         vertexShape[0]])   + offset
        else:      e[i] = ti.Vector([1,vertexShape[0],vertexShape[0]+1]) + offset
        for j in ti.static(range(6)): indices[6*i+j] = e[i][j%3]
        DmInv[i] = ti.Matrix.cols([ p[e[i][1]]-p[e[i][0]] ,p[e[i][2]]-p[e[i][0]] ]).inverse()

@ti.kernel
def Step(model:ti.i32):
    # compute acceleration
    for i in e:
        Ds      = ti.Matrix.cols([ p[e[i][1]] - p[e[i][0]] ,p[e[i][2]] - p[e[i][0]] ])
        F       = Ds @ DmInv[i]
        P       = ti.Matrix.zero(float,2,2)
        if   model == CR:                                # co-rotated linear elasticity
            R,S     = ti.polar_decompose(F)               
            P       = 2*mu0*(F-R) + lambda0*(R.transpose()@F-ti.Matrix.identity(ti.f32,2))@R
        elif model == STVK:                              # St.Venant-Kirchhoff
            Strain  = (F.transpose()@F - ti.Matrix.identity(float,2)) / 2             
            P       = F @ (2*mu0*Strain + lambda0*Strain.trace()*ti.Matrix.identity(float,2))
        elif model == NH:                                # Neo-Hookean
            FInvT   = F.transpose().inverse()
            P       = mu0*(F-FInvT) + lambda0*ti.log(F.determinant())*FInvT
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
    model = CR
    constitutive_model = ['St.Venant-Kirchhoff','co-rotated','Neo-Hookean']
    Initialize()
    window = ti.ui.Window("FEM ELASTIC",(1000,1000))
    canvas = window.get_canvas()
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                model = (model+1)%3
                Initialize()
            elif window.event.key == ti.ui.ESCAPE: break
        
        for _ in range(100):Step(model)

        window.GUI.begin('SPACE to change constitutive model', 0.0, 0.0, 0.28, 0.05)
        window.GUI.text('Current : '+constitutive_model[model])
        canvas.set_background_color((218/255,221/255,216/255))
        canvas.lines(p,indices = indices,width = 0.0025, color=(57/255,186/255,232/255))
        window.GUI.end()
        window.show()

Main()