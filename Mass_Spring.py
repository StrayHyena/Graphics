import taichi as ti

ti.init(arch = ti.cpu)

n = 10
nv,ne = (n+1)**2, 4*n**2+2*n 
h = 1e-2                            # time step
x = ti.Vector.field(2, ti.f32, nv)  # positions
v = ti.Vector.field(2, ti.f32, nv)  # velocities
f = ti.Vector.field(2, ti.f32, nv)  # forces
df = ti.Matrix.field(2,2,ti.f32,ne) # force derivative 弹簧对它的第一个顶点位置的导数 即K1
m = ti.field(ti.f32, nv)            # mass
e = ti.field(ti.i32, 2*ne)          # edge(spring) pairs
l = ti.field(ti.f32, ne)            # rest length 
k = ti.field(ti.f32, ne)            # stiffness 

MB = ti.linalg.SparseMatrixBuilder(2*nv,2*nv,2*nv)
KB = ti.linalg.SparseMatrixBuilder(2*nv,2*nv,4*ne*4)

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
def FillMK(MB:ti.types.sparse_matrix_builder(),KB:ti.types.sparse_matrix_builder()):
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
        df[i] = k[i]*(I-l[i]/dist01* ( I - (x[i0]-x[i1])@(x[i0]-x[i1]).transpose()/dist01**2 ) )
        for ki,kj in ti.static(ti.ndrange(2,2)):
            KB[2*i0+ki,2*i0+kj] -= df[i][ki,kj]
            KB[2*i1+ki,2*i1+kj] -= df[i][ki,kj]
            KB[2*i0+ki,2*i1+kj] += df[i][ki,kj]
            KB[2*i1+ki,2*i0+kj] += df[i][ki,kj]

@ti.kernel
def UpdateVX(vnew :ti.ext_arr() ):
    for i in x:
        v[i] = ti.Vector([vnew[2*i],vnew[2*i+1]])
        if i==nv-1 or i == nv-1-n:v[i] = ti.Vector([0.0,0.0])
        x[i] += v[i]*h

def SubStep():
    FillMK(MB,KB)
    K = KB.build()
    M = MB.build()
    b =  M @ v.to_numpy().reshape(2*nv) + h*f.to_numpy().reshape(2*nv)
    A = M - h*h*K
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(A)   
    solver.factorize(A)
    vnew = solver.solve(b)
    UpdateVX(vnew)

def Main():
    Initialize()
    window = ti.ui.Window("Implicit Mass Spring",(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        SubStep()
        canvas.set_background_color((245/255,245/255,245/255))
        canvas.lines(x,indices = e,width = 0.025/n, color=(57/255,186/255,232/255))
        window.show()

Main()