import taichi as ti

ti.init(arch = ti.gpu)

order = 2
# particles' number(N)|velocity(vp)|position(xp)|mass(mp)
N = 8192
vp = ti.Vector.field(2,dtype = ti.f32,shape=N)
xp = ti.Vector.field(2,dtype = ti.f32,shape=N)
Cp = ti.Matrix.field(2,2,dtype = ti.f32,shape=N)

# grid resolution(n)|velocity(vi)|mass(mi)
n = 32
vi = ti.Vector.field(2,dtype = ti.f32,shape=(n,n)) 
mi = ti.field(dtype = ti.f32,shape=(n,n)) 

dx = 1.0/n
dt = 2.0e-3

@ti.func
def BSpline(dist):
    dist = ti.abs(dist)
    ret = 0.0
    if ti.static(order==1): # linear
        if dist<1.0: ret = 1-dist
    elif ti.static(order==2): # quadratic
        if 0.5<=dist<1.5:ret = 0.5*(1.5-dist)**2
        elif dist<0.5:   ret = 0.75-dist*dist
    elif ti.static(order==3): #cubic
        if 1<=dist<2: ret = (1/6)*(2-dist)**3
        elif dist<1:  ret = 0.5*dist**3-dist**2+2/3
    return ret

@ti.kernel
def PIC():
    # clear grid data
    for gi in range(n):
        for gj in range(n):
            vi[gi,gj]=ti.Vector([0.0,0.0])
            mi[gi,gj]=0.0
    # P2G: particle to grid
    for pi in range(N):
        pos = xp[pi]
        base = (pos/dx-ti.Vector([0.5,0.5])*(order-1)).cast(int)
        ws = [ti.Vector([0.0,0.0])]*(order+1)
        for k in ti.static(range(order+1)):
            ws[k][0] = BSpline(pos.x/dx-k-base.x)
            ws[k][1] = BSpline(pos.y/dx-k-base.y)
        for gi in ti.static(range(order+1)):
            for gj in ti.static(range(order+1)):
                offset = ti.Vector([gi,gj])
                weight = ws[gi][0]*ws[gj][1]
                mi[base+offset] += weight
                vi[base+offset] += vp[pi]*weight
    for gi in range(n):
        for gj in range(n):
            if mi[gi,gj]>0:
                vi[gi,gj]/=mi[gi,gj]
    # G2P: grid to particle
    for pi in range(N):
        newV = ti.Vector([0.0,0.0])
        pos = xp[pi]
        base = (pos/dx-ti.Vector([0.5,0.5])*(order-1)).cast(int)
        ws = [ti.Vector([0.0,0.0])]*(order+1)
        for k in ti.static(range(order+1)):
            ws[k][0] = BSpline(pos.x/dx-k-base.x)
            ws[k][1] = BSpline(pos.y/dx-k-base.y)
        for gi in ti.static(range(order+1)):
            for gj in ti.static(range(order+1)):
                newV += ws[gi][0]*ws[gj][1]*vi[ti.Vector([gi,gj])+base]
        xp[pi] = xp[pi]+vp[pi]*dt
        vp[pi] = newV

@ti.kernel
def APIC():
    # clear grid data
    for gi in range(n):
        for gj in range(n):
            vi[gi,gj]=ti.Vector([0.0,0.0])
            mi[gi,gj]=0.0
    # P2G: particle to grid
    for pi in range(N):
        pos = xp[pi]
        base = (pos/dx-ti.Vector([0.5,0.5])*(order-1)).cast(int)
        ws = [ti.Vector([0.0,0.0])]*(order+1)
        for k in ti.static(range(order+1)):
            ws[k][0] = BSpline(pos.x/dx-k-base.x)
            ws[k][1] = BSpline(pos.y/dx-k-base.y)
        for gi in ti.static(range(order+1)):
            for gj in ti.static(range(order+1)):
                offset = ti.Vector([gi,gj])
                weight = ws[gi][0]*ws[gj][1]
                mi[base+offset] += weight
                vi[base+offset] += weight*(vp[pi]+Cp[pi]@((base+offset).cast(float)*dx-pos))
    for gi in range(n):
        for gj in range(n):
            if mi[gi,gj]>0:
                vi[gi,gj]/=mi[gi,gj]
    # G2P: grid to particle
    for pi in range(N):
        newV = ti.Vector([0.0,0.0])
        newC = ti.Matrix([[0.0,0.0],[0.0,0.0]])
        pos = xp[pi]
        base = (pos/dx-ti.Vector([0.5,0.5])*(order-1)).cast(int)
        ws = [ti.Vector([0.0,0.0])]*(order+1)
        for k in ti.static(range(order+1)):
            ws[k][0] = BSpline(pos.x/dx-k-base.x)
            ws[k][1] = BSpline(pos.y/dx-k-base.y)
        for gi in ti.static(range(order+1)):
            for gj in ti.static(range(order+1)):
                offset = ti.Vector([gi,gj])
                weight = ws[gi][0]*ws[gj][1]
                newV += weight*vi[base+offset]
                newC += weight*vi[base+offset].outer_product((base+offset).cast(float)-pos/dx)/dx
        xp[pi] = xp[pi]+vp[pi]*dt
        vp[pi] = newV
        if ti.static(order==2): Cp[pi] = 4*newC
        elif ti.static(order==3):Cp[pi] = 3*newC

@ti.kernel
def Initialize():
    for i in range(N):
        xp[i] = ti.Vector([ti.random()*0.6+0.2,ti.random()*0.6+0.2])
        vp[i] = ti.Vector([xp[i][1]-0.5,0.5-xp[i][0]])

@ti.kernel
def Inspector(x:ti.i32):
    momentum = ti.Vector([0.0,0.0])
    for i in range(N):momentum += vp[i]
    angularMomentum = 0.0
    for i in range(N):angularMomentum += xp[i].cross(vp[i])
    print(x,momentum,angularMomentum)

def Main():
    affine = True
    l = ['Linear','Quadratic','Cubic']
    Initialize()
    gui = ti.GUI('Particle In Cell',res = (800,800))
    for _ in range(10**3):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:break
            elif gui.event.key == ti.GUI.SPACE:
                affine=not affine
                Initialize()
                _ = 0
        
        # print(_,ti.abs(am-Inspector(_))/ti.abs(am))
        Inspector(_)
        if not affine:PIC()
        else: APIC()
        
        gui.clear(0x1494CF)
        gui.text('APIC'if affine else 'PIC',pos = (0.0,0.95))
        gui.text(l[order-1],pos =(0.0,0.9))
        gui.circles(xp.to_numpy(),color = 0xFCFF01,radius = 3)
        gui.show()

Main()