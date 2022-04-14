import taichi as ti

ti.init(arch = ti.cpu)

n = 10
nv,ne = (n+1)**2, 4*n**2+2*n 
h = 1e-2                            # time step
x = ti.Vector.field(2, ti.f32, nv)  # positions
v = ti.Vector.field(2, ti.f32, nv)  # velocities
f = ti.Vector.field(2, ti.f32, nv)  # forces
d2E = ti.Matrix.field(2,2,ti.f32,ne) # force derivative 弹簧对它的第一个顶点位置的导数 即K1 d^2E/d(x1)d(x1)
m = ti.field(ti.f32, nv)            # mass
e = ti.field(ti.i32, 2*ne)          # edge(spring) pairs
l = ti.field(ti.f32, ne)            # rest length 
k = ti.field(ti.f32, ne)            # stiffness 

MB = ti.linalg.SparseMatrixBuilder(2*nv,2*nv,2*nv)
KB = ti.linalg.SparseMatrixBuilder(2*nv,2*nv,4*ne*4)
solver = ti.linalg.SparseSolver(solver_type="LLT")

@ti.kernel
def Initialize():
    squareLen = 0.4
    lowerLeft = ( (1.0-squareLen)/2 , 0.90-squareLen )
    for i,j in ti.ndrange(n+1,n+1):
        x[j*(n+1)+i] = ti.Vector(lowerLeft) + ti.Vector([i,j])*squareLen/n
        v[j*(n+1)+i] = ti.Vector([0.0,0.0])
        m[j*(n+1)+i] = 1.0
    for i,j in ti.ndrange(n,n+1):
        e[2*(j*n+i)]     = j*(n+1)+i 
        e[2*(j*n+i) + 1] = j*(n+1)+i + 1
    start = n*(n+1)
    for i,j in ti.ndrange(n+1,n):
        e[2*(start + i*n+j)]   = j*(n+1)+i 
        e[2*(start + i*n+j)+1] = (j+1)*(n+1)+i 
    start += n*(n+1)
    for i,j in ti.ndrange(n,n):
        e[2*(start + j*n+i)]    = j*(n+1)+i 
        e[2*(start + j*n+i) +1] = (j+1)*(n+1)+i+1
    start += n*n
    for i,j in ti.ndrange(n,n):
        e[2*(start + j*n+i)]    = (j+1)*(n+1)+i 
        e[2*(start + j*n+i) +1] = j*(n+1)+i+1
    for i in l:
        l[i] = (x[e[2*i]]-x[e[2*i+1]]).norm()
        k[i] = 1000

@ti.kernel
def FillMK(MB:ti.types.sparse_matrix_builder(),KB:ti.types.sparse_matrix_builder(),fixPD:ti.i32):
    for i in f: 
        f[i] = ti.Vector([0.0,-2.0])*m[i]
        MB[2*i+0,2*i+0] += m[i]
        MB[2*i+1,2*i+1] += m[i]
    I = ti.Matrix.identity(ti.f32,2)
    for i in l:
        i0,i1 = e[2*i],e[2*i+1]
        dist01 = (x[i0]-x[i1]).norm()
        f0 = k[i]*(dist01-l[i])*(x[i0]-x[i1]).normalized()
        f[i0]-=f0
        f[i1]+=f0
        d2E[i] = k[i]*(I-l[i]/dist01* ( I - (x[i0]-x[i1])@(x[i0]-x[i1]).transpose()/dist01**2 ) )
        if fixPD: #保证 nabla^2 G 是 正定的
            U,Sigma,V = ti.svd(d2E[i])
            Sigma[0,0] = ti.max(0.0001,Sigma[0,0])
            Sigma[1,1] = ti.max(0.0001,Sigma[1,1])
            d2E[i] = U@Sigma@V.transpose()
        for ki,kj in ti.static(ti.ndrange(2,2)):
            KB[2*i0+ki,2*i0+kj] += d2E[i][ki,kj]
            KB[2*i1+ki,2*i1+kj] += d2E[i][ki,kj]
            KB[2*i0+ki,2*i1+kj] -= d2E[i][ki,kj]
            KB[2*i1+ki,2*i0+kj] -= d2E[i][ki,kj]
        
@ti.kernel
def Numpy1DToField2D(arg0:ti.template(),anp:ti.ext_arr()):
    for i in arg0: arg0[i] = ti.Vector([anp[2*i],anp[2*i+1]])

# One iteration of Newton’s method(Baraff and Witkin, 1998)----------------------------------------------
@ti.kernel
def UpdateX():
    for i in x:
        if i==nv-1 or i == nv-1-n:v[i] = ti.Vector([0.0,0.0])
        x[i] += v[i]*h

def SubStep_OneStepNewton():
    FillMK(MB,KB,0)
    K = KB.build()
    M = MB.build()
    b =  M @ v.to_numpy().reshape(2*nv) + h*f.to_numpy().reshape(2*nv)
    A = M + h*h*K
    solver.analyze_pattern(A)   
    solver.factorize(A)
    vnew = solver.solve(b)
    Numpy1DToField2D(v,vnew)
    UpdateX()
#========================================================================================================
#Exact---------------------------------------------------------------------------------------------------
dg = ti.Vector.field(2,ti.f32,nv) # g(x)'s derivative to x, i.e. Nable g
dx = ti.Vector.field(2,ti.f32,nv) # 位移增量
xn = ti.Vector.field(2,ti.f32,nv) # x next
temp = ti.Vector.field(2,ti.f32,nv) # temporary variable

@ti.kernel
def G(y:ti.template())->ti.f32:
    E = 0.0
    for i in l: E += k[i]*((y[e[2*i]]-y[e[2*i+1]]).norm()-l[i])**2
    sum = 0.0
    for i in y: sum += (y[i]-x[i]-h*v[i]).norm_sqr()*m[i]
    return 0.5*(h*h*E+sum)

@ti.kernel
def NablaG(y:ti.template()):
    for i in f:f[i] = ti.Vector([0.0,0.0])
    for i in l:
        i0,i1 = e[2*i],e[2*i+1]
        dist01 = (y[i0]-y[i1]).norm()
        f0 = k[i]*(dist01-l[i])*(y[i0]-y[i1]).normalized()
        f[i0]-=f0
        f[i1]+=f0
    for i in y: dg[i] = m[i]*(y[i]-x[i]-h*v[i])-h*h*f[i]

@ti.kernel
def FieldNorm(y:ti.template())->ti.f32:
    res = 0.0
    for i in y:res += y[i].norm_sqr()
    return ti.sqrt(res)

@ti.kernel
def FieldDot(a:ti.template(),b:ti.template())->ti.f32:
    res = 0.0
    for i in a:res += a[i].dot(b[i])
    return res

@ti.kernel
def FieldAdd(s:ti.template(),a:ti.template(),b:ti.template(),ca:ti.f32,cb:ti.f32):
    for i in s:s[i] = a[i]*ca+b[i]*cb

@ti.kernel
def Initialize_Exact():
    for i in x:
        if i==nv-1 or i == nv-1-n:continue
        v[i] += h*ti.Vector([0.0,-2.0])
        x[i] += h*v[i]
    for i in xn:xn[i] = x[i] + h*v[i]

@ti.kernel
def GradientDesent():
    for i in dx:dx[i] = -dg[i]

def LineSearch(alpha,beta,initT):
    t = initT
    while True:
        FieldAdd(temp,xn,dx,1.0,t)
        assert FieldDot(dx,dg)<=0
        if G(temp) > G(xn) + alpha*t*FieldDot(dx,dg): t*=beta
        else : break
    return t

@ti.kernel
def UpdateXV():
    for i in x:
        if i==nv-1 or i == nv-1-n:continue
        v[i] = (xn[i]-x[i])/h
        x[i] = xn[i]

def SubStep_Exact(newtonMethod):
    Initialize_Exact()
    while True:
        NablaG(xn)
        if newtonMethod:
            FillMK(MB,KB,1)
            K = KB.build()
            M = MB.build()
            A = M-h*h*K
            b = -dg.to_numpy().reshape(2*nv)
            solver.analyze_pattern(A)   
            solver.factorize(A)
            Npdx = solver.solve(b)
            Numpy1DToField2D(dx,Npdx)
        else:
            GradientDesent()
        if FieldNorm(dg) < 1e-4: break
        t = LineSearch(0.03,0.5,1)
        FieldAdd(xn,xn,dx,1.0,t)
    UpdateXV()
#========================================================================================================

def Main():
    methods = ['Exact Solution with Gradient Descent','Exact Solution with Newton\'s Method','One Step Newton']
    methodIdx = 2
    Initialize()
    window = ti.ui.Window("Implicit Mass Spring",(1000,1000))
    canvas = window.get_canvas()
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE: 
                Initialize()    
                methodIdx = (methodIdx+1)%3
            elif window.event.key == ti.ui.ESCAPE:  break

        if methodIdx==2:   SubStep_OneStepNewton()
        elif methodIdx==1: SubStep_Exact(True)
        else:              SubStep_Exact(False)
        
        window.GUI.begin('SPACE to change solver method', 0.0, 0.0, 0.35, 0.05)
        window.GUI.text('Current : '+ methods[methodIdx])
        canvas.set_background_color((245/255,245/255,245/255))
        canvas.lines(x,indices = e,width = 0.025/n, color=(57/255,186/255,232/255))
        window.show()

Main()