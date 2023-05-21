# https://www.gorillasun.de/blog/bowyer-watson-algorithm-for-delaunay-triangulation/#the-super-triangle
import taichi as ti
from taichi.math import vec2,vec3,vec4
from time import sleep

ti.init(arch = ti.cpu)

epsilon = 1e-7
FLT_MAX = 1e10
bg_color = (0.8,0.8,0.8)
n = 100
radius = 0.01

pt_n        = ti.field(ti.u32,()) 
tri_n       = ti.field(ti.i32,())
p           = ti.Vector.field(2,ti.f32,n) 
edges       = ti.field(ti.u32,2*n*n)
edge_hash   = ti.field(ti.u32,n*(n-1)) # edge (i0,i1) where i0<i1 whose hash is i0*n+i1
tri         = ti.Vector.field(3,ti.i32,3*n)
tri_temp    = ti.Vector.field(3,ti.i32,3*n)
tri_temp1   = ti.Vector.field(3,ti.i32,3*n)

@ti.kernel
def Initialize():
    pt_n[None] = 3 + 4
    for i in p: p[i] = [-1,-1]
    pmin = vec2(1,1)*FLT_MAX
    pmax = -pmin
    for i in range(pt_n[None]):  
        p[i] = [ti.random(),ti.random()]
        ti.atomic_min(pmin, p[i])
        ti.atomic_max(pmax, p[i])
    d = 10*(pmax - pmin)
    p[0] = pmin - d*vec2(1,3)
    p[1] = vec2(pmin[0],pmax[1]) + d*vec2(-1,1)
    p[2] = pmax + d*vec2(3,1)
    tri_n[None] = 1
    tri[0] = [0,1,2]  # super triangle :-D

@ti.func
def Circumcircle(a,b,c):
    center,r2 = vec2(0,0),-1.0
    e0,e1 = b-a,c-a
    d0,d1 = vec2(e0[1],-e0[0]),vec2(e1[1],-e1[0])
    o0,o1 = (a+b)/2.0,(a+c)/2.0
    A = ti.Matrix.cols([d0,-d1])
    if ti.abs(A.determinant())>epsilon: 
        t = A.inverse()@(o1-o0)
        center = o0+t[0]*d0
        r2     = (center-a).norm_sqr()
    return center,r2

@ti.kernel
def ProcessOnePoint(pi:ti.i32):
    pos = p[pi]
    tempn,tempn1 = 0,0

    # find all triangles whose circumcircle contains point pi
    for i in range(tri_n[None]):
        center,r2 = Circumcircle(p[tri[i][0]],p[tri[i][1]],p[tri[i][2]])
        if (pos-center).norm_sqr() <= r2:
            old = ti.atomic_add(tempn, 1)
            tri_temp[old] = tri[i]
        else:
            old = ti.atomic_add(tempn1, 1)
            tri_temp1[old] = tri[i]

    # find common edges in temp triangles, connect pi with Delaunay空腔 vectex, without common edges
    for i in edge_hash:edge_hash[i] = 0
    for i in range(tempn):
        for j in ti.static(range(3)):
            e0,e1 = tri_temp[i][j],tri_temp[i][(j+1)%3]
            if e0>e1: e0,e1 = e1,e0
            edge_hash[e0*n+e1] += 1
    for i in edge_hash:
        if edge_hash[i]!=1:continue
        old = ti.atomic_add(tempn1, 1)
        tri_temp1[old] = [i//n,i%n,pi]

    tri_n[None] = tempn1
    for i in range(tempn1): tri[i] = tri_temp1[i]

@ti.kernel
def BuildEdges():
    for i in edge_hash:edge_hash[i] = 0
    for i in range(tri_n[None]):
        for j in ti.static(range(3)):
            e0,e1 = tri[i][j],tri[i][(j+1)%3]
            if e0>e1: e0,e1 = e1,e0
            if e0>=3 and e1>=3:edge_hash[e0*n+e1] += 1
    for i in edges: edges[i] = n-1
    for i in edge_hash:
        if edge_hash[i]==0:continue
        edges[2*i]   = i//n
        edges[2*i+1] = i%n

    tri_n[None] = 1
    tri[0] = [0,1,2]
    
def Main():
    Initialize()
    window = ti.ui.Window('Delaunay Trianglation',res=(1000,1000))
    gui = window.get_gui()
    canvas = window.get_canvas()
    while window.running:
        if pt_n[None] >= n-1:print('Too many points , max is ',n-1)
        if window.is_pressed(ti.ui.ESCAPE):break
        pos,selected = window.get_cursor_pos(),None
        for i in range(pt_n[None]):
            if (pos[0]-p[i][0])**2+(pos[1]-p[i][1])**2 >= radius**2: continue
            selected = i
            break
        if window.is_pressed(ti.ui.LMB) :
            if selected!=None: p[selected]=window.get_cursor_pos()
        elif window.is_pressed(ti.ui.RMB):
            p[pt_n[None]] = window.get_cursor_pos()
            pt_n[None] = min(pt_n[None]+1,n-1)
            sleep(1/2)
        elif window.is_pressed(ti.ui.MMB):
            if selected!=None:
                pt_n[None] = max(0,pt_n[None]-1)
                for i in range(selected,pt_n[None]):p[i] = p[i+1]
                for i in range(pt_n[None],n): p[i] = [-1,-1]

        with gui.sub_window(" ", 0, 0, 0.5, 0.05) as w:
            w.text("[Left mouse]- Drag; [Middle mouse]- Delete; [Right mouse]- Add")
       
        for i in range(3,pt_n[None]): 
            ProcessOnePoint(i)
        BuildEdges()

        canvas.set_background_color(bg_color)
        canvas.lines(p,0.005,indices = edges)
        canvas.circles(p,radius,(0,0,0))
        window.show()

Main()