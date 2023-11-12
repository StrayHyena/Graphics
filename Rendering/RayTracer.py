import taichi as ti
import taichi.math as tm
import numpy as np
import time

array_max_size = 200
Array   = ti.types.vector( array_max_size  ,ti.i32)
vec3    = ti.types.vector(3,ti.f64)
vec3i   = ti.types.vector(3, ti.i32)
EPSILON = 5e-7
PDF_EPSILON = 0.01
FLT_MAX = 1e10
INT_MAX = 0x3fffffff
PI      = 3.1415926536
LIGHT,MIRROR,GLASS,MICROFACET   = 0,1,2,3

@ti.data_oriented
class ObjLoader:
    def __init__(self,objPath):
        pos,idx,normals,vnidx = [],[],[],[]
        for line in open(objPath,'r').readlines():
            if   line[0:2] == 'v ':  pos += list(map(float,line[1:].split()))[0:3]
            elif   line[0:3] == 'vn ': normals += list(map(float,line[2:].split()))
            elif   line[0:2] == 'f ':
                for pair in line[1:].split():
                    vvn = list(map(int,pair.split('//')))
                    idx.append(vvn[0])
                    vnidx.append(vvn[1])
        for i in range(len(idx)):   idx[i]-=1
        for i in range(len(vnidx)): vnidx[i]-=1
        # for i in range(len(normals)):print(normals[i])

        self.point_num,self.prim_num = len(pos)//3,len(idx)//3
        assert(len(vnidx)==len(idx))
        self.points      = ti.Vector.field(3,ti.f64,self.point_num)
        self.primitives  = ti.Vector.field(3,ti.i32,self.prim_num)
        self.primitives_flattened  = ti.field(ti.i32,self.prim_num*3)
        self.normals         =  np.array(normals).reshape(len(normals)//3,3).astype(np.float64)
        self.prim_normal_idx =  np.array(vnidx).reshape(self.prim_num,3).astype(np.int32)

        self.points.from_numpy(np.array(pos).reshape(self.point_num,3).astype(np.float64))
        self.primitives.from_numpy(np.array(idx).reshape(self.prim_num,3).astype(np.int32))
        self.primitives_flattened.from_numpy(np.array(idx).astype(np.int32))

ti.init(arch=ti.cpu,default_fp  =ti.f64)

Material = ti.types.struct( type = ti.u8,color = vec3,eta = ti.f64,roughness = ti.f64,kd = ti.f64) # type: [0]light [1]diffuse [2]mirror [3]glass
AABB     = ti.types.struct(bmin = vec3,bmax = vec3)
BVHNode  = ti.types.struct(box = AABB,li = ti.i32,ri = ti.i32,start = ti.i32, end = ti.i32)

def AABBGrow(this,other):
    for i in range(3):
        this.bmin[i] = min(this.bmin[i],other.bmin[i])
        this.bmax[i] = max(this.bmax[i],other.bmax[i])
def AABBArea(this):
    extend = this.bmax - this.bmin
    return extend[0]*extend[1]+extend[2]*extend[1]+extend[0]*extend[2]

res = (400,400)
up  = vec3(0,1,0)

img    = ti.Vector.field(3, ti.f64, res)

max_points_num   = 6000
max_triangle_num = 10000
max_object_num  = 100

points    = ti.Vector.field(3,ti.f64, max_points_num)
numpt     = ti.field(ti.i32,())

triangles = ti.Vector.field(3,ti.i32,max_triangle_num)
normals   = ti.Vector.field(3,ti.f64,max_triangle_num)    # face normal
vnormals  = ti.Vector.field(3,ti.f64,max_triangle_num*3)  # vertex normal
vn_idx    = ti.Vector.field(3,ti.i32,max_triangle_num)
centroids = ti.Vector.field(3,ti.f64,max_triangle_num)  
tri2obj   = ti.field(ti.i32,max_triangle_num)
aabbs     = AABB.field(shape = max_triangle_num)

tidx      = ti.field(ti.i32,max_triangle_num)  # triangle index in bvh node's [start,end]
numtri    = ti.field(ti.i32,())

materials = Material.field(shape= max_object_num)  # a object has a material
numobj    = ti.field(ti.i32,())

nodes    = BVHNode.field(shape=max_triangle_num*2)
numnode  = ti.field(ti.i32,())

camera_pos    = vec3(2,0.5,0)
camera_target = vec3(0,.5,0)
camera_dir    = (camera_target - camera_pos).normalized()
camera_right  = camera_dir.cross(up).normalized()
camera_up     = camera_right.cross(camera_dir).normalized()
near_plane,fov = 0.01,0.35
near_plane_center     = camera_pos + near_plane * camera_dir
near_plane_leftbottom = near_plane_center - near_plane*fov*camera_up - near_plane*fov*camera_right
up_stride    =  camera_up*2*near_plane*fov/(res[1]-1)
right_stride =  camera_right*2*near_plane*fov/(res[0]-1)

#  ray hit methods -----------------------------------------------------------------------
# https://zhuanlan.zhihu.com/p/436993075
@ti.func
def HitTriangle(o,d,p0,p1,p2,tmax):
    ret_t,t,beta,gamma = FLT_MAX,-1.0,-1.0,-1.0
    e0,e1,b = p1-p0,p2-p0,o-p0
    detA = d.cross(e1).dot(e0)
    if ti.abs(detA) > EPSILON:
        t,beta,gamma = b.cross(e0).dot(e1)/detA,d.cross(e1).dot(b)/detA,b.cross(e0).dot(d)/detA
    if t>0.0 and t<tmax and beta>0.0 and gamma>0.0 and beta+gamma<1.0: ret_t = t
    return ret_t,vec3(1-beta-gamma,beta,gamma)

@ti.func
def HitAABB(o,d,box,tmax):
    bmin,bmax = box.bmin,box.bmax
    d = d.normalized()
    tnmax ,tfmin, tnmin,is_hit =  0.0,FLT_MAX, FLT_MAX,True
    for j in ti.static(range(3)):
        if d[j]==0.0:
            if bmin[j]>o[j] or o[j]>bmax[j]: is_hit = False
        else:
            tNear,tFar = (bmin[j]-o[j])/d[j],(bmax[j]-o[j])/d[j]
            if tNear>tFar: tNear,tFar = tFar,tNear
            if tFar  < tfmin : tfmin = tFar
            if tNear > tnmax : tnmax = tNear
            if tNear < tnmin : tnmin = tNear
            if tnmax > tfmin : is_hit= False
    return tnmin if is_hit else FLT_MAX

# https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/
@ti.func
def HitScene(o,d):
    t,triidx,weights = FLT_MAX,-1,vec3(0,0,0)
    s,i = Array([0 for _ in range(Array.n)]),0
    node = nodes[0]
    while True:
        if node.start<=node.end:  # leaf node
            for j in range(node.start,node.end+1):
                tri       = triangles[tidx[j]]
                this_t,w  = HitTriangle(o,d, points[tri[0]],points[tri[1]],points[tri[2]],t )
                if this_t < FLT_MAX : t,triidx,weights = this_t,tidx[j],w
            if i==0: break
            i-=1
            node = nodes[s[i]]
            continue
        i0,i1 = node.li,node.ri
        n0,n1 = nodes[i0],nodes[i1]
        t0,t1 = HitAABB(o,d,n0.box,t),HitAABB(o,d,n1.box,t)
        if t0>t1: t0,t1,i0,i1 = t1,t0,i1,i0
        if t0 == FLT_MAX:
            if i==0:break
            i-=1
            node = nodes[s[i]]
        else:
            node = nodes[i0]
            if t1<FLT_MAX:
                s[i] = i1
                i+=1
                if i>=array_max_size:print("ERR ",i)
    return t,triidx,weights

#  brdfs -----------------------------------------------------------------------
# i,n must normalized
@ti.func
def Reflect(i,n):return (i-2*n*i.dot(n)).normalized()

@ti.func
def Refract(i,n,etaI,etaT):
    cosI = -i.dot(n)
    return (etaI/etaT*i + (etaI/etaT*cosI-ti.sqrt(1 - (etaI/etaT)**2*(1-cosI**2)))*n).normalized()

@ti.func
def D_GGX(cos_theta,alpha):return alpha**2/(PI*(cos_theta**2*(alpha**2-1)+1)**2)
@ti.func
def Lambda(cos_theta,alpha): return 0.5*(ti.sqrt(1 + alpha*alpha*(1.0/cos_theta/cos_theta-1))-1)
@ti.func
def G_GGX(cos_thetaI,cos_thetaO,alpha): return 1.0/(1.0+Lambda(cos_thetaI,alpha)+Lambda(cos_thetaO,alpha))
@ti.func
def Fresnel(i,n,etaI,etaT):
    cosI = -i.dot(n)
    cosT = ti.sqrt( ti.max(0.0, 1 - (etaI/etaT)**2*(1-cosI**2)))
    r_parl = (etaT*cosI-etaI*cosT)/(etaT*cosI+etaI*cosT)
    r_perp = (etaI*cosI-etaT*cosT)/(etaI*cosI+etaT*cosT)
    return 0.5*(r_parl**2+r_perp**2)
@ti.func
def TorranceSparrow(i,o,n,etaI,etaT,alpha):  # i: some obj to hit point , o : out ray direction
    h = (o-i).normalized()
    return D_GGX(n.dot(h),alpha)*G_GGX(n.dot(-i),n.dot(o),alpha) * Fresnel(i,h,etaI,etaT) /ti.abs(4*o.dot(n)*i.dot(-n))

@ti.func
def ToWorld(s,n):
    cos_sn,u = up.dot(n),up.cross(n) # rotate angle and rotate axis
    ret = cos_sn*s
    if ti.abs(cos_sn)<1:
        u = u.normalized()
        ret = cos_sn*s+(1-cos_sn)*u.dot(s)*u+ti.sqrt(1-cos_sn**2)*u.cross(s)  # rotate s along u with radian acos(cos_sn)
    return ret.normalized()

@ti.func
def SampleCosineWeighted(n):
    phi,cos_theta = ti.random()*PI*2 ,ti.sqrt(ti.random())  
    sin_theta = ti.sqrt(1-cos_theta**2)
    s = vec3(sin_theta*ti.sin(phi),cos_theta,sin_theta*ti.cos(phi))
    return ToWorld(s,n).normalized(),  cos_theta/PI

@ti.func
def SampleGGX(n,i,alpha):
    random0   = ti.random()
    cos_theta = ti.sqrt((1-random0)/(random0*(alpha*alpha-1)+1))
    sin_theta = ti.sqrt(1-cos_theta*cos_theta)
    phi       = ti.random()*PI*2
    w_h       = ToWorld(vec3(sin_theta*ti.sin(phi),cos_theta,sin_theta*ti.cos(phi)),n)
    s         = Reflect(i,w_h)
    return s.normalized(),D_GGX(cos_theta, alpha)*cos_theta/4/ti.abs(w_h.dot(-i)),cos_theta

@ti.kernel
def Render():
    for i,j in img:
        o = near_plane_leftbottom + (i+ti.random()*0.5-1)*right_stride+(j+ti.random()*0.5-1)*up_stride
        d = (o-camera_pos).normalized()
        accumulator,radiance = vec3(1,1,1),vec3(0,0,0)
        for _ in range(10):
            hit_t,hit_tri_idx,hit_weights = HitScene(o,d)
            if hit_tri_idx < 0 : break
            material = materials[tri2obj[hit_tri_idx]]
            normal_idx = vn_idx[hit_tri_idx]
            normal   =  ( normals[normal_idx[0]]*hit_weights[0] + normals[normal_idx[1]]*hit_weights[1] + normals[normal_idx[2]]*hit_weights[2]  ).normalized()
            if material.type == LIGHT: 
                radiance = accumulator*material.color
                break
            if material.type == MICROFACET:   
                o += (d*hit_t + EPSILON*normal)
                if ti.random()<material.kd: 
                    d,pdf = SampleCosineWeighted(normal)
                    accumulator *= material.color # (albedo/PI [brdf]) * cosTheta / ( cosTheta/PI [pdf] )
                else:
                    di = d
                    d,pdf,cos = SampleGGX(normal, di, material.roughness)
                    if pdf<PDF_EPSILON: break
                    accumulator *= TorranceSparrow(di, d, normal, 1.0, material.eta, material.roughness) * ti.max(0.0,d.dot(normal))/pdf 
            if material.type == MIRROR: 
                o += (d*hit_t + EPSILON*normal)
                d = Reflect(d,normal)
            if material.type == GLASS:
                etaI,etaT = 1.0,material.eta   # default : air to glass
                if normal.dot(d) > 0: normal, etaI, etaT = -normal,material.eta,1.0  # glass to air
                if ti.random() < Fresnel(d,normal,etaI,etaT): 
                    o = o + d*hit_t + EPSILON*normal
                    d = Reflect(d, normal)
                else:
                    o = o + d*hit_t - EPSILON*normal
                    d = Refract(d, normal, etaI,etaT)
        img[i,j]+=radiance

def Test(test_case):
    objs,mats = [],[]
    objs.append( ObjLoader("../assets/Cornell/quad_left.obj") )
    mats.append( Material(MICROFACET,(0,0.6,0),1,1,1.0))
   
    objs.append( ObjLoader("../assets/Cornell/quad_right.obj") )
    mats.append( Material(MICROFACET,(0.6,0,0),1,1,1.0))
   
    objs.append( ObjLoader("../assets/Cornell/quad_bottom.obj") )
    mats.append( Material(MICROFACET,(0.9,0.9,0.9),1,1,1.0))
   
    objs.append( ObjLoader("../assets/Cornell/quad_back.obj"))
    mats.append( Material(MICROFACET,(0.9,0.9,0.9),1,1,1.0))
   
    objs.append( ObjLoader("../assets/Cornell/quad_top.obj"))
    mats.append( Material(MICROFACET,(0.9,0.9,0.9),1,1,1.0))
    if test_case=='MICROFACET':
        objs.append( ObjLoader("../assets/Cornell/sphere.obj"))
        mats.append( Material(MICROFACET,(0.1,0.5,0.8),1.5, 0.01,1.0))
    if test_case=='GLASS':
        objs.append( ObjLoader("../assets/Cornell/bunny.obj"))
        mats.append( Material(GLASS,(0.1,0.5,0.8),1.5, 0.01,1.0))
    if test_case=='MIRROR':
        objs.append( ObjLoader("../assets/Cornell/bunny.obj"))
        mats.append( Material(MIRROR,(0.1,0.5,0.8),1.5, 0.01,1.0))
    objs.append( ObjLoader("../assets/Cornell/lightSmall.obj"))
    mats.append( Material(LIGHT,(50,50,50),1,1))
    return objs,mats

def Initialize():
    objs,mats = Test('GLASS')

    numobj[None] = len(objs)
    for i in range(numobj[None]):materials[i] = mats[i]
    numpt[None] = 0
    for j in range(len(objs)):
        mesh = objs[j]
        for i in range(mesh.point_num):
            points[i + numpt[None] ]  = mesh.points[i]
        for i in range(mesh.prim_num):
            vertices = []
            tri_idx = i+numtri[None]
            bmin,bmax = vec3(1,1,1)*FLT_MAX,-vec3(1,1,1)*FLT_MAX
            for k in range(3): vertices.append( mesh.points[mesh.primitives[i][k]])
            for p in vertices:
                for k in range(3):
                    bmin[k],bmax[k] = min(bmin[k],p[k]),max(bmax[k],p[k])
            triangles[tri_idx] = mesh.primitives[i] + vec3i(1,1,1)*numpt[None]
            
            vn_idx[tri_idx] = mesh.prim_normal_idx[i] + vec3i(1,1,1)*3*numtri[None]
            for k in range(3):
                normals[ tri_idx*3 + k] = mesh.normals[mesh.prim_normal_idx[i][k]]

            centroids[tri_idx]=(vertices[0]+vertices[1]+vertices[2])/3.0
            tri2obj[tri_idx] = j
            aabbs[tri_idx]   = AABB(bmin,bmax)
        numpt[None]  += mesh.point_num
        numtri[None] += mesh.prim_num

    numnode[None] = 1
    for i in range(nodes.shape[0]):
        nodes[i].box = AABB(vec3(1,1,1)*FLT_MAX,-vec3(1,1,1)*FLT_MAX)
        nodes[i].start,nodes[i].end = INT_MAX,0
    for i in range(tidx.shape[0]):tidx[i] = i
    img.fill(0)

bin_cnt = 8
def BVHBuildSAHBin(node_idx,start,end):
    this_node = nodes[node_idx]
    split_axis,split_pos,split_cost = -1,-1,FLT_MAX
    left_count , right_count = 0,0
    for i in range(start,end+1):AABBGrow(this_node.box, aabbs[tidx[i]])
    for j in range(3):
        boundsMin , boundsMax = FLT_MAX,-FLT_MAX
        for i in range(start,end+1):
            boundsMin = min(boundsMin,centroids[tidx[i]][j])
            boundsMax = max(boundsMax,centroids[tidx[i]][j])
        if boundsMin == boundsMax:continue
        stride = (boundsMax - boundsMin) / bin_cnt
        bins = [ [AABB(vec3(1,1,1)*FLT_MAX , -vec3(1,1,1)*FLT_MAX),0] for _ in range(bin_cnt) ]
        for i in range(start,end+1):
            bin_idx = min(bin_cnt-1, int(( centroids[tidx[i]][j] - boundsMin)/stride))
            AABBGrow(bins[bin_idx][0],aabbs[tidx[i]])
            bins[bin_idx][1]+=1

        left_area  = [0.0 for _ in range(bin_cnt-1)]
        left_cnt   = [0 for _ in range(bin_cnt-1)]
        left_box   = AABB(vec3(1,1,1)*FLT_MAX , -vec3(1,1,1)*FLT_MAX)
        left_sum   = 0
        right_area = [0.0 for _ in range(bin_cnt-1)]
        right_cnt  = [0 for _ in range(bin_cnt-1)]
        right_box  = AABB(vec3(1,1,1)*FLT_MAX , -vec3(1,1,1)*FLT_MAX)
        right_sum  = 0
        for i in range(bin_cnt-1):
            AABBGrow(left_box, bins[i][0])
            left_area[i] = AABBArea(left_box)
            left_sum += bins[i][1]
            left_cnt[i]  = left_sum
            AABBGrow(right_box, bins[bin_cnt-i-1][0])
            right_area[bin_cnt-i-2] = AABBArea(right_box)
            right_sum += bins[bin_cnt-i-1][1]
            right_cnt[bin_cnt-i-2]  = right_sum
        for i in range(bin_cnt-1):
            cost = left_area[i]*left_cnt[i] + right_area[i]*right_cnt[i]
            if cost<split_cost: 
                split_axis,split_pos,split_cost=j,boundsMin+stride*(i+1),cost
                left_count,right_count=left_cnt[i],right_cnt[i]
                assert(left_count+right_count==end-start+1)
    if left_count==0 or right_count==0 or AABBArea(this_node.box)*(end-start+1)<=split_cost: 
        this_node.start,this_node.end = start,end
        return 1
    l,r = start,end
    while l<r:
        if centroids[tidx[l]][split_axis] <= split_pos: l+=1
        else:  tidx[l],tidx[r],r = tidx[r],tidx[l],r-1
    this_node.start,this_node.end = INT_MAX,0  # a non-leaf node
    this_node.li  = node_idx+1
    lchildren_num = BVHBuildSAHBin(this_node.li, start, start+left_count-1)
    this_node.ri  = this_node.li + lchildren_num
    rchildren_num = BVHBuildSAHBin(this_node.ri, start+left_count,  end)
    return lchildren_num+rchildren_num+1

def Main():
    Initialize()
    print('num objects ',numobj[None])
    print('num points ',numpt[None])
    print('num primitives ',numtri[None])
    time_start = time.time()
    BVHBuildSAHBin(0, 0, numtri[None]-1)
    print("Build Time:  ", time.time()-time_start)

    window = ti.ui.Window("Ray Tracer", res)
    canvas = window.get_canvas()
    frame = 0
    while not window.is_pressed(ti.ui.ESCAPE):
        Render()
        frame+=1
        canvas.set_image(img.to_numpy().astype(np.float32)/frame)
        window.show()

Main()