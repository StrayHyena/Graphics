import taichi as ti
from taichi.math import sign,vec3,vec4

ti.init(arch=ti.cpu,default_fp=ti.f64)

vec2d = ti.types.vector(2,ti.f64)
vec3d = ti.types.vector(3,ti.f64)
vec4d = ti.types.vector(4,ti.f64)
mat2d = ti.types.matrix(2,2,ti.f64)
FLT_MAX = 1e7+0.01
FLT_MIN = -FLT_MAX
epsilon = 1e-10

tri_n = 8
fps = 30.0*5
collision_pass = 10
dt = 1.0/fps
h = 1e-2

pts  = ti.Vector.field(3, ti.f64, 3*tri_n)  # point position 
pts1  = ti.Vector.field(3, ti.f64, 3*tri_n) # proposed position
vel  = ti.Vector.field(3, ti.f64, 3*tri_n)  # velocity
ipl  = ti.Vector.field(3, ti.f64, 3*tri_n)  # impulse 
idx  = ti.field(ti.i32,3*tri_n)             # triangle indices
edges = ti.Vector.field(2,ti.i32,3*tri_n)   # edge inidices
el    = ti.field(ti.f64,3*tri_n)            # initial edge length
clr  = ti.Vector.field(3, ti.f64, 3*tri_n)  # color
vf = ti.Vector.field(4,ti.i32,3*tri_n*(tri_n-1) )    # point-triangle pair [point-idx, triangle-v0,triangle-v1,triangle-v2]
ee = ti.Vector.field(4,ti.i32,3*tri_n*3*(tri_n-1) )  # edge-edge pair [edge0-idx0,edge0-idx1,    edge1-idx0,edge1-idx1]
vf_n = ti.field(ti.i32,1)
ee_n = ti.field(ti.i32,1)

@ti.func
def SolveLinear(a:ti.f64,b:ti.f64):
    root_num,roots = 0,vec3d(0,0,0)
    if ti.abs(a) < epsilon and ti.abs(b) < epsilon : root_num = 1
    elif ti.abs(a)<epsilon: pass 
    else: root_num,roots[0] = 1,-b/a
    return root_num,roots

@ti.func
def SolveQuadratic(a:ti.f64,b:ti.f64,c:ti.f64):
    root_num,roots = 0,vec3d(0,0,0)
    if ti.abs(a) < epsilon : root_num,roots = SolveLinear(b,c)
    else:
        delta = b*b-4*a*c
        if delta<0.0: pass
        elif 0.0 <= delta < epsilon: root_num,roots[0],roots[1] = 1,0.5*-b/a,0.5*-b/a
        else:
            q = (b + sign(b)*ti.sqrt(delta))*(-0.5)
            root_num,roots[0],roots[1] = 2,c/q,q/a
            if roots[0]>roots[1]:roots[0],roots[1] = roots[1],roots[0]
    return root_num,roots

@ti.func
def SolveCubic(a:ti.f64,b:ti.f64,c:ti.f64,d:ti.f64,l:ti.f64=FLT_MIN,u:ti.f64=FLT_MAX):
    root_num,roots = 0,vec3d(0,0,0)
    if a==0.0: 
        rn,rs = SolveQuadratic(b,c,d)
        for i in range(rn):
            if l<rs[i]<u:
                roots[root_num] = rs[i]
                root_num+=1
    else:
        segment_num,segment_points = 2,vec4d(l,u,u,u)
        sp_num,sp = SolveQuadratic(3*a,2*b,c)
        if sp_num==1:
            if l+epsilon < sp[0] < u-epsilon:         segment_num,segment_points[1] = 3,sp[0]
        if sp_num==2:
            if l+epsilon < sp[0] < sp[1] < u-epsilon: segment_num,segment_points[1],segment_points[2] = 4,sp[0],sp[1]
            elif l+epsilon < sp[0] < u-epsilon:       segment_num,segment_points[1] = 3,sp[0]
            elif l+epsilon < sp[1] < u-epsilon:       segment_num,segment_points[1] = 3,sp[1]
        
        for i in range(segment_num-1):
            lower,upper = segment_points[i],segment_points[i+1]
            fl,fu = a*lower**3+b*lower**2+c*lower+d,a*upper**3+b*upper**2+c*upper+d
            if ti.abs(fl)<epsilon: 
                roots[root_num] = lower
                root_num+=1
                continue
            if ti.abs(fu)<epsilon:
                roots[root_num] = upper
                root_num+=1
                continue
            if fl*fu > 0: continue

            for _ in range(100):
                mid = (upper+lower)/2.0
                fm  =  a*mid**3+b*mid**2+c*mid+d
                if ti.abs(fm)<epsilon :
                    roots[root_num] = mid
                    root_num+=1
                    break
                fl,fu   = a*lower**3+b*lower**2+c*lower+d, a*upper**3+b*upper**2+c*upper+d
                if   fm*fl<0.0: upper = mid
                elif fm*fu<0.0: lower = mid 

    return root_num,roots

@ti.kernel
def Initialize():
    for i in idx:idx[i] = i
    for i in ipl:ipl[i] = vec3d(0,0,0)
    vf_n[0] = ee_n[0] = 0
    for i in range(tri_n):
        color = vec3d(0.5+0.5*ti.random(),0.5+0.5*ti.random(),0.5+0.5*ti.random())
        clr[3*i] = clr[3*i+1] = clr[3*i+2] = color
    devide = [3,2,2]
    cellsize = vec3([1.0/devide[0],1.0/devide[1],1.0/devide[2]])
    for i in range(tri_n):
        cellidx    = [i//(devide[2]*devide[1]), (i%(devide[2]*devide[1]))//devide[2],(i%(devide[2]*devide[1]))%devide[2] ]
        celloffset = cellidx*cellsize
        for j in ti.static(range(3)):
            random_vector = vec3d(ti.random(),ti.random(),ti.random()) 
            pts[3*i+j] = random_vector * cellsize + celloffset
    for i in range(tri_n):
        a,b,c = pts[3*i],pts[3*i+1],pts[3*i+2]
        n = (b-a).cross(c-a).normalized()
        tri_idx = [idx[3*i],idx[3*i+1],idx[3*i+2]]
        for j in ti.static(range(3)):
            edges[3*i+j] = [tri_idx[j],tri_idx[(j+1)%3]]
            vel[3*i+j] = n
            for k in ti.static(range(3)):
                vel[3*i+j][k] += (ti.random()-0.5)*0.1
            vel[3*i+j]*=(ti.random()*0.1-0.05+1.0)*0.1*10 #vel is normal with some jitter
    for i in edges: el[i] = (pts[edges[i][0]]-pts[edges[i][1]]).norm()

@ti.func
def MomentumAndKinetic():
    m = vec3d(0,0,0)
    k = 0.0
    for i in vel:
        m+=vel[i]
        k+=vel[i].dot(vel[i])
    return m,k

@ti.kernel
def Integration():
    for i in pts1:  pts1[i] = pts[i] + vel[i]*dt
    for i in ipl: ipl[i] = [0,0,0]
    # don't want an edge to be stretched or compressed too much
    for i in edges:
        i0,i1 = edges[i][0],edges[i][1]
        x10 = pts1[i0] - pts1[i1]
        curr_len = x10.norm()
        delta_ratio = 1 - curr_len/el[i] 
        if ti.abs(delta_ratio) < 0.2: continue
        normal = x10.normalized()
        I = delta_ratio*ti.abs((vel[i0]-vel[i1]).dot(normal))*normal + h
        ipl[i0]+=I
        ipl[i1]-=I
    for i in vel:
        vel[i]+=ipl[i]
        pts1[i] = pts[i]+vel[i]*dt

@ti.kernel
def Collision():
    for i in ipl:ipl[i] = [0,0,0]
    ee_n[0] = vf_n[0] = 0

    for i in range(tri_n):
        for j in range(tri_n):
            if i==j:continue
            for k in ti.static(range(3)):
                vf[vf_n[0]+k]=[idx[3*i+k],idx[3*j+0],idx[3*j+1],idx[3*j+2]]
            vf_n[0]+=3
    for i in edges:
        for j in range(i+1,3*tri_n):
            if edges[j][0]==edges[i][0] or edges[j][0]==edges[i][1] or edges[j][1]==edges[i][0] or edges[j][1]==edges[i][1]:continue 
            ee[ee_n[0]] = [edges[i][0],edges[i][1],edges[j][0],edges[j][1]]
            ee_n[0]+=1

    x0 = [vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0)]
    x1 = [vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0)]
    v  = [vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0),vec3d(0,0,0)]
    

    m0,k0 = MomentumAndKinetic()
    for i in range(vf_n[0]):
        for j in ti.static(range(4)): 
            x0[j] = pts[vf[i][j]]
            x1[j] = pts1[vf[i][j]]
            v[j]  = x1[j]-x0[j]
        v01,v21,v31 = v[0]-v[1],v[2]-v[1],v[3]-v[1]
        x01,x21,x31 = x0[0]-x0[1],x0[2]-x0[1],x0[3]-x0[1]
        a = v01.dot(v21.cross(v31))
        b = x01.dot(v21.cross(v31)) + v01.dot(v21.cross(x31)) + v01.dot(x21.cross(v31))  
        c = x01.dot(v21.cross(x31)) + x01.dot(x21.cross(v31)) + v01.dot(x21.cross(x31))  
        d = x01.dot(x21.cross(x31))   
        root_num,roots = SolveCubic(a,b,c,d,-epsilon,1.0+epsilon) 
        if root_num==0:continue

        xt = [x0[0]+roots[0]*v[0],x0[1]+roots[0]*v[1],x0[2]+roots[0]*v[2],x0[3]+roots[0]*v[3]]
        pa,pb,pc, ab,ac = xt[1]-xt[0],xt[2]-xt[0],xt[3]-xt[0], xt[2]-xt[1],xt[3]-xt[1]
        ws = vec3d(pb.cross(pc).norm(),pa.cross(pc).norm(),pa.cross(pb).norm())
        S = ab.cross(ac).norm()
        if S < epsilon or  ws.sum() >= S + epsilon: continue

        ws/=S
        normal = ab.cross(ac).normalized()
        if normal.dot(x0[0]-x0[2])<0.0: normal*=-1
        xb0,xb1 = ws[0]*x0[1]+ws[1]*x0[2]+ws[2]*x0[3],ws[0]*x1[1]+ws[1]*x1[2]+ws[2]*x1[3]
        Ic = ti.abs((x1[0]-x0[0] - xb1+xb0).dot(normal))/dt + h
        I_tilde = normal* Ic/(1+ws[0]*ws[0]+ws[1]*ws[1]+ws[2]*ws[2])
        Is = [I_tilde, -ws[0]*I_tilde,-ws[1]*I_tilde,-ws[2]*I_tilde]
        for j in ti.static(range(4)): ipl[vf[i][j]]+=Is[j]

    for i in range(ee_n[0]):
        for j in ti.static(range(4)): 
            x0[j] = pts[ee[i][j]]
            x1[j] = pts1[ee[i][j]]
            v[j]  = x1[j]-x0[j]
        v20,v10,v32 = v[2]-v[0],v[1]-v[0],v[3]-v[2]
        x20,x10,x32 = x0[2]-x0[0],x0[1]-x0[0],x0[3]-x0[2]
        a = v20.dot(v10.cross(v32))
        b = x20.dot(v10.cross(v32)) + v20.dot(v10.cross(x32)) + v20.dot(x10.cross(v32))  
        c = x20.dot(v10.cross(x32)) + x20.dot(x10.cross(v32)) + v20.dot(x10.cross(x32))  
        d = x20.dot(x10.cross(x32))
        if ti.abs(a)<epsilon and ti.abs(d)<epsilon and ti.abs(c)<epsilon and ti.abs(d)<epsilon :continue   # parallel 
        root_num,roots = SolveCubic(a,b,c,d,-epsilon,1.0+epsilon) 
        if root_num==0:continue

        xt = [x0[0]+roots[0]*v[0],x0[1]+roots[0]*v[1],x0[2]+roots[0]*v[2],x0[3]+roots[0]*v[3]]
        o1,o2,d1,d2,o12 = xt[0],xt[2],xt[1]-xt[0],xt[3]-xt[2],xt[2]-xt[0]
        l1,l2 = d1.norm(),d2.norm()
        if l1 < epsilon or l2 < epsilon or d1.cross(d2).norm_sqr()<epsilon: continue
        valid,t  = False,vec2d(FLT_MAX,FLT_MAX)
        index_pair = [0,1,0,2,1,2]
        for k in ti.static(range(3)):
            ii,jj = index_pair[2*k],index_pair[2*k+1]
            A = mat2d([ [d1[ii],-d2[ii]] , [d1[jj],-d2[jj] ]])
            y = vec2d(o12[ii],o12[jj])
            if ti.abs(A.determinant())>=epsilon: valid,t = True,A.inverse()@y
        if t.min() <0.0 or t.max()>1.0:continue # p lies outside l1 or l2
        ws = vec4d(1-t[0],t[0],1-t[1],t[1])
        p10,p20,p11,p21 = x0[0]*ws[0]+x0[1]*ws[1],x0[2]*ws[2]+x0[3]*ws[3],x1[0]*ws[0]+x1[1]*ws[1],x1[2]*ws[2]+x1[3]*ws[3]
        
        normal = d1.cross(d2).normalized()
        if normal.dot( p10 - p20 ) < 0.0: normal*=-1
        Ic = ti.abs((p11-p10 - p21+p20).dot(normal))/dt + h
        I_tilde =  Ic*normal/(ws[0]*ws[0]+ws[1]*ws[1]+ws[2]*ws[2]+ws[3]*ws[3])
        Is = [ ws[0]*I_tilde, ws[1]*I_tilde,-ws[2]*I_tilde, -ws[3]*I_tilde]
        for j in ti.static(range(4)): ipl[ee[i][j]]+=Is[j]

    for i in pts1:
        vel[i] += ipl[i]
        pts1[i] = pts[i]+vel[i]*dt

    m1,k1 = MomentumAndKinetic()
    if  k1 - k0 > epsilon  :
        print(k1 - k0)

    for i in pts1:
        for j in ti.static(range(3)):
            if (0.0>pts1[i][j] and vel[i][j]<0.0) or (pts1[i][j]>1.0 and vel[i][j]>0.0):
                vel[i][j] *= -1
                pts1[i][j] = ti.min(1.0,ti.max(pts1[i][j],0.0))

@ti.kernel
def UpdatePosition():
    for i in pts: pts[i] = pts1[i]

def Main():
    Initialize()
    window = ti.ui.Window("CCD", (1000, 1000))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 0.5, 2.5)
    camera.lookat(0.5, 0.5, 0.5)

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE): break
        camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        Integration()
        for __ in range(collision_pass): Collision()
        UpdatePosition()

        scene.mesh(pts,idx,per_vertex_color = clr)
        canvas.scene(scene)
        window.show()

Main()