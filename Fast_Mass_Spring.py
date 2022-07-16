import taichi as ti

ti.init(ti.cpu)

dt = 1e-3
rect = (10,10)     # rect[0] row point number, rect[1] col point number
aa = 0.4/(max(rect)-1)
nv = rect[0]*rect[1]
ne = rect[0]*(rect[1]-1) + rect[1]*(rect[0]-1) + 2*(rect[1]-1)*(rect[0]-1)

xc  = ti.Vector.field(2,ti.f32,nv)  # current position
xp  = ti.Vector.field(2,ti.f32,nv)  # previous position
m   = ti.field(ti.f32,nv)
f   = ti.Vector.field(2,ti.f32,nv) # external force

d   = ti.Vector.field(2,ti.f32,ne)   # projection vector
l   = ti.field(ti.f32,ne)            # rest length
k   = ti.field(ti.f32,ne)            # stiffness
e   = ti.field(ti.i32,ne*2)          # edge index

A = ti.field(ti.f32,(nv,nv))   # A = M + dt*dt * L 
J = ti.field(ti.f32,(nv,ne))   # J = Sigma_i(0->ne) a_i @ a_i.transpose()  # a_i = 1 and a_j = -1 for spring(i,j), others 0
L = ti.field(ti.f32,(nv,nv))   # L = Sigma_i(0->ne) a_i @ s_i.transpose()  # s_i = 1 for spring i, others 0
rhs = ti.Vector.field(2,ti.f32,nv)  # right_hand_side = M @ (2*xc-xp) + dt*dt*f

low = ti.field(ti.f32,(2*nv,2*nv))
up  = ti.field(ti.f32,(2*nv,2*nv))
P   = ti.field(ti.f32,(2*nv,2*nv))  # partial pivoting 
x   = ti.field(ti.f32,2*nv)
b   = ti.field(ti.f32,2*nv)


@ti.kernel
def Initialize():
    lowerLeft = 0.5*ti.Vector([ 1-aa*(rect[1]-1),1-aa*(rect[0]-1) ])
    nr,nc = rect
    for i,j in ti.ndrange(nc,nr):
        xc[j*nc+i] = lowerLeft + ti.Vector([i,j])*aa
        xp[j*nc+i] = xc[j*nc+i]
        m[j*nc+i] = 1.0
        f[j*nc+i] = [0.0,-2.2]
    for i,j in ti.ndrange(nc-1,nr):
        e[2*(j*(nc-1)+i)]     = j*nc+i 
        e[2*(j*(nc-1)+i) + 1] = j*nc+i + 1
    start = (nc-1)*nr
    for i,j in ti.ndrange(nc,nr-1):
        e[2*(start + i*(nr-1)+j)]   = j*nc+i 
        e[2*(start + i*(nr-1)+j)+1] = (j+1)*nc+i 
    start += nc*(nr-1)
    for i,j in ti.ndrange(nc-1,nr-1):
        e[2*(start + j*(nc-1)+i)]    = j*nc+i 
        e[2*(start + j*(nc-1)+i) +1] = (j+1)*nc+i+1
    start += (nc-1)*(nr-1)
    for i,j in ti.ndrange(nc-1,nr-1):
        e[2*(start + j*(nc-1)+i)]    = (j+1)*nc+i 
        e[2*(start + j*(nc-1)+i) +1] = j*nc+i+1
    for idx in l:
        i,j = e[2*idx],e[2*idx+1]
        l[idx] = (xc[i]-xc[j]).norm()
        k[idx] = 1000
        L[i,i] += k[idx] 
        L[j,j] += k[idx]
        L[i,j] -= k[idx]
        L[j,i] -= k[idx]
        J[i,idx] += k[idx]
        J[j,idx] -= k[idx]
    for I in ti.grouped(A):     A[I] = L[I]*dt*dt
    for i in m:                 A[i,i] += m[i]

    # LUDecomposition
    for i in range(2*nv):           P[i,i] = 1.0
    for i,j in ti.ndrange(nv,nv):   up[2*i,2*j] = up[2*i+1,2*j+1] = A[i,j]
    ti.loop_config(serialize=True) 
    for i in range(2*nv):
        if ti.abs(up[i,i])>1e-4: continue
        for j in range(2*nv):
            if ti.abs(up[j,i]) <= 1e-4 or ti.abs(up[i,j])<=1e-4: continue
            for k in range(2*nv): 
                up[i,k],up[j,k] = up[j,k],up[i,k]
                P[i,k],P[j,k] = P[j,k],P[i,k]
            break
    ti.loop_config(serialize=True) 
    for i in range(2*nv):
        low[i,i] = 1.0
        for j in range(i+1,2*nv):
            low[j,i] = up[j,i]/up[i,i]
            for k in range(i+1):        up[j,k] = 0.0
            for k in range(i+1,2*nv):   up[j,k] -= low[j,i]*up[i,k]

@ti.kernel
def Substep():
    for i in d:     d[i] = (xc[e[2*i]] - xc[e[2*i+1]]).normalized()*l[i]
    for i in rhs:   rhs[i] = [0.0,0.0]
    for j,i in J:   rhs[j] += dt*dt*J[j,i]*d[i]
    for i in rhs:   rhs[i] += (2*xc[i] - xp[i])*m[i] + dt*dt*f[i] 
    for i in rhs:   b[2*i],b[2*i+1] = rhs[i][0],rhs[i][1]
    for i in x:     x[i] = 0.0
    # solving (M+dt*dt*L) @ x = dt*dt*J @ d + M@(2*cx-xp) + dt*dt* f_ext  AND  do not parallelized
    for j,i in P:   x[j] += P[j,i]*b[i]
    ti.loop_config(serialize=True) 
    for j in range(2*nv):
        for i in range(j):
            x[j] -= low[j,i]*x[i]
    ti.loop_config(serialize=True) 
    for j in range(2*nv):
        k = 2*nv-1-j
        for i in range(k+1,2*nv):
            x[k] -= up[k,i]*x[i]
        x[k] /= up[k,k]
    for i in xc:
        if i==nv-1 or i==nv-rect[1]:continue
        xp[i] = xc[i]
        xc[i] = [x[2*i],x[2*i+1]]

def Main():
    Initialize()
    window = ti.ui.Window('Fast Mass Spring',(1000,1000))
    canvas = window.get_canvas()
    while not window.is_pressed(ti.ui.ESCAPE):
        Substep()

        canvas.set_background_color((0.7,0.7,0.7))
        canvas.lines(xc,0.005,e,(0,0,0))
        window.show()

Main()