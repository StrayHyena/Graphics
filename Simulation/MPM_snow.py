import taichi as ti

# ti.init(arch=ti.cpu,debug=True,cpu_max_num_threads=1)
ti.init(arch=ti.gpu)

Np = 40000  # particle number
x    = ti.Vector.field(2,dtype = ti.f32,shape = Np)
v    = ti.Vector.field(2,dtype = ti.f32,shape = Np)
Fe   = ti.Matrix.field(2,2,dtype = ti.f32,shape = Np)
C    = ti.Matrix.field(2,2,dtype = ti.f32,shape = Np)
Vol  = ti.field(dtype = ti.f32 , shape = Np)
J    = ti.field(dtype = ti.f32,shape = Np)

Ng = 256  # grid size
dx = 1/Ng
vg = ti.Vector.field(2,dtype = ti.f32,shape = (Ng,Ng))
fg = ti.Vector.field(2,dtype = ti.f32,shape = (Ng,Ng))
mg = ti.field(dtype = ti.f32 , shape = (Ng,Ng))

E,nu = 1.4e5,0.2
COMPRESSION,STRETCH = 2.0e-2,6.0e-3
rho = 400
mu0,lambda0 =E/(2*(1+nu)) ,nu*E/((1+nu)*(1-2*nu))

dt = 1e-4/5
m = 0.15**2*2*rho/Np
bound = 3

@ti.func
def N(x):
    ret = 0.0
    x = ti.abs(x)
    if x<0.5:           ret = 0.75-x*x
    elif 0.5<=x<1.5:    ret = 0.5*(1.5-x)**2
    return ret

@ti.func
def Nd1(x): # 1st order derivative
    ret,absx = 0.0,ti.abs(x)
    if absx<0.5:            ret = -2*x
    elif 0.5<=absx<1.5:     ret = (absx-1.5)*(absx/x)
    return ret

@ti.kernel
def Initialize():
    for i,j in mg:
        mg[i,j] = 0
        fg[i,j] = vg[i,j] = [0,0]
    for pi in x:
        x[pi]   = ti.Vector([0.25,0.5]) + ti.Vector([ti.random()*0.15,ti.random()*0.15])
        if pi>Np//2:x[pi]   = ti.Vector([0.35,0.7]) + ti.Vector([ti.random()*0.15,ti.random()*0.15])
        v[pi]   = ti.Vector([0.0,0.0])
        Fe[pi]  = ti.Matrix([[1.0,0.0],[0.0,1.0]])
        C[pi]   = ti.Matrix.zero(float,2,2)
        J[pi]   = 1.0
        Vol[pi] = 0.0
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                nx = x[pi]/dx - float(base+offset)
                w = [N(nx[0]),N(nx[1])]
                mg[base+offset]+=w[0]*w[1]*m
    for pi in x:
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                nx = x[pi]/dx - float(base+offset)
                w = [N(nx[0]),N(nx[1])]
                Vol[pi] += w[0]*w[1]*mg[base+offset]
        Vol[pi] = m*dx*dx/Vol[pi]

@ti.kernel
def SubStep():
    for i,j in mg:
        mg[i,j] = 0
        fg[i,j] = vg[i,j] = [0,0]
    for pi in x:
        h = ti.exp(10*(1-J[pi]))
        mu , lamda = mu0*h,lambda0*h
        U,sigma,V = ti.svd(Fe[pi])
        Je = sigma[0,0]*sigma[1,1] 
        stress = (2*mu*(Fe[pi]-U@V.transpose())@Fe[pi].transpose()+lamda*(Je-1)*Je*ti.Matrix.identity(float,2))
        stress *= Vol[pi]
        Fe[pi] = U @ sigma @ V.transpose()
        
        gridCoord = x[pi]/dx
        base = int(x[pi]/dx-ti.Vector([0.5,0.5]))
        wx  = [N(gridCoord[0]-base[0]),N(gridCoord[0]-base[0]-1),N(gridCoord[0]-base[0]-2)]
        wy  = [N(gridCoord[1]-base[1]),N(gridCoord[1]-base[1]-1),N(gridCoord[1]-base[1]-2)]
        wdx = [Nd1(gridCoord[0]-base[0])/dx,Nd1(gridCoord[0]-base[0]-1)/dx,Nd1(gridCoord[0]-base[0]-2)/dx]
        wdy = [Nd1(gridCoord[1]-base[1])/dx,Nd1(gridCoord[1]-base[1]-1)/dx,Nd1(gridCoord[1]-base[1]-2)/dx]       
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                ij = base + ti.Vector([i,j])
                weight = wx[i]*wy[j]
                mg[ij] += weight*m
                vg[ij] += (v[pi] + C[pi]@(ij*dx-x[pi]))*weight*m
                fg[ij] -=  stress @ ti.Vector([wdx[i]*wy[j],wdy[j]*wx[i]])

    for i,j in mg:
        if mg[i,j]<=0:continue
        vg[i,j]    /= mg[i,j]
        vg[i,j][1] += dt*-50
        vg[i,j]    += dt*fg[i,j]/mg[i,j]
        if i<=bound  or i>=Ng-bound :vg[i,j][0] = 0.0
        if j<=bound  or j>=Ng-bound :vg[i,j][1] = 0.0

    for pi in x:
        newV = ti.Vector.zero(float,2)
        fea  = ti.Matrix.zero(float,2,2)
        B    = ti.Matrix.zero(float,2,2)
        gridCoord = x[pi]/dx
        base = int(gridCoord-ti.Vector([0.5,0.5]))
        wx  = [N(gridCoord[0]-base[0]),N(gridCoord[0]-base[0]-1),N(gridCoord[0]-base[0]-2)]
        wy  = [N(gridCoord[1]-base[1]),N(gridCoord[1]-base[1]-1),N(gridCoord[1]-base[1]-2)]
        wdx = [Nd1(gridCoord[0]-base[0])/dx,Nd1(gridCoord[0]-base[0]-1)/dx,Nd1(gridCoord[0]-base[0]-2)/dx]
        wdy = [Nd1(gridCoord[1]-base[1])/dx,Nd1(gridCoord[1]-base[1]-1)/dx,Nd1(gridCoord[1]-base[1]-2)/dx]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                ij = base + ti.Vector([i,j])
                weight = wx[i]*wy[j]
                fea  +=  vg[ij].outer_product(ti.Vector([ wdx[i]*wy[j],wdy[j]*wx[i] ]))
                newV +=  weight*vg[ij]
                B    +=  weight*vg[ij].outer_product(ij*dx-x[pi])
        v[pi] = newV
        C[pi] = B*4/(dx*dx)
        x[pi] += dt*v[pi]
        Fe[pi] = (ti.Matrix.identity(float,2) + dt*fea) @ Fe[pi]
        U,sigma,V = ti.svd(Fe[pi])
        for d in ti.static(range(2)):
            sigmaNewI = min(max(sigma[d,d],1-COMPRESSION),1+STRETCH)
            J[pi]*=sigma[d,d]/sigmaNewI
            sigma[d,d] = sigmaNewI
        Fe[pi] = U @ sigma @ V.transpose()
            

def Main():
    Initialize()
    window = ti.ui.Window("MPM Snow",(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        for _ in range(5): SubStep()
        canvas.set_background_color((56/255,117/255,159/255))
        canvas.circles(x,radius = 0.001,color=(238/255,238/255,238/255))
        window.show()

Main()