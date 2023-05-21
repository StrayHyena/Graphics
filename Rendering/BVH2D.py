import taichi as ti
import random

FLT_MAX = 1e30
INT_MAX = 0xffffffff
EPS     = 1e-8
vec2  = ti.types.vector(2,ti.f32)
vec3  = ti.types.vector(3,ti.f32)
vec2i = ti.types.vector(2,ti.i32)
ti.init(arch=ti.cpu)

n = 20 # how many edges
leaf_prim_count = 2

aabb     = ti.types.struct(bmin = vec2,bmax = vec2)
bvh_node = ti.types.struct(box = aabb, li = ti.u32,ri = ti.u32, start = ti.u32, end = ti.u32)  # end < start indicates a non-leaf node

p     = ti.Vector.field(2,ti.f32,2*n)
e     = ti.Vector.field(2,ti.i32,n)
aabbs = aabb.field(shape=n)
nodes = bvh_node.field(shape = 2*n)  #即使每个叶子节点放一个primitive，整个树的结点也不会超过2*n
eidx = ti.field(ti.u32,n)  # element is edge index for BVH building

ray = ti.Vector.field(2,ti.f32,3)
raye= ti.field(ti.i32,2)

e1      = ti.field(ti.i32,2*n)  # flattened e for canvas.lines()
temp    = [ti.field(ti.i32,n*10+10) for _ in range(2)]  # [0] will be size of this array

def AABBGrow(this,other):
    this.bmax = vec2(max(this.bmax[0],other.bmax[0]),max(this.bmax[1],other.bmax[1]))
    this.bmin = vec2(min(this.bmin[0],other.bmin[0]),min(this.bmin[1],other.bmin[1]))
def AABBCenter(this):
    return (this.bmin+this.bmax)/2.0

# build node at node_idx in nodes(taichi struct field)
#                parent->  node_idx
#           left_child ->  node_idx + 1
#            .               ..                  
#           right_child -> node_idx + 1 + left_node_count
# [start,end]    NOT [start,end)   return how many nodes this tree have
def BVHBuildSAH(node_idx,start,end):
    assert(node_idx<2*n)
    this_node = nodes[node_idx]
    this_node.box = aabb( vec2(FLT_MAX,FLT_MAX) , -vec2(FLT_MAX,FLT_MAX))
    
    split_axis,split_pos,split_cost = -1,-1,FLT_MAX
    left_count , right_count = 0,0

    for i in range(start,end+1):
        curr_split_pos = AABBCenter(aabbs[eidx[i]])
        AABBGrow(this_node.box, aabbs[eidx[i]])
        for j in range(2):
            l_aabb = aabb(vec2(FLT_MAX,FLT_MAX),-vec2(FLT_MAX,FLT_MAX))
            r_aabb = aabb(vec2(FLT_MAX,FLT_MAX),-vec2(FLT_MAX,FLT_MAX))
            lc,rc  = 0,0
            for k in range(start,end+1):
                curr_box = aabbs[eidx[k]]
                if AABBCenter(curr_box)[j] <= curr_split_pos[j]:
                    lc+=1
                    AABBGrow(l_aabb, curr_box)
                else:
                    rc+=1
                    AABBGrow(r_aabb, curr_box)
            cost = (l_aabb.bmax-l_aabb.bmin).sum() * lc + (r_aabb.bmax-r_aabb.bmin).sum()* rc
            if cost < split_cost:
                split_cost,split_axis,split_pos = cost,j,curr_split_pos[j]
                left_count,right_count = lc,rc

    count = end-start+1
    assert(left_count+right_count==count)

    if  count<=leaf_prim_count or left_count==0 or right_count==0 or (this_node.box.bmax-this_node.box.bmin).sum()*count<split_cost: 
        this_node.start,this_node.end = start,end
        return 1

    l,r = start,end
    while l!=r:
        if AABBCenter(aabbs[eidx[l]])[split_axis] <= split_pos: l+=1
        else:  eidx[l],eidx[r],r = eidx[r],eidx[l],r-1

    this_node.start,this_node.end = INT_MAX,0  # a non-leaf node
    this_node.li     = node_idx+1
    left_node_cnt    =  BVHBuildSAH(this_node.li, start, start+left_count-1)
    this_node.ri     = this_node.li + left_node_cnt
    right_node_cnt   =  BVHBuildSAH(this_node.ri, start+left_count,  end)
    assert(left_node_cnt>0 and right_node_cnt>0)
    return left_node_cnt + right_node_cnt + 1

@ti.kernel
def Initialize():
    for i in range(n):
        p[2*i]   = 0.7*vec2(ti.random(),ti.random())+vec2(1,1)*0.2
        p1 = p[2*i] + vec2(ti.sin(ti.random()*10),ti.cos(ti.random()*10)) * (ti.random()*0.2 + 0.05 )
        p[2*i+1] = vec2(min(0.999,max(p1[0],0.001)),min(0.999,max(p1[1],0.001)))
    for i in p: p[i][1]*=0.5

    for i in e:
        i0,i1 = 2*i,2*i+1
        e[i] = (i0,i1)
        aabbs[i].bmin = vec2(min(p[i0][0],p[i1][0]),min(p[i0][1],p[i1][1]))
        aabbs[i].bmax = vec2(max(p[i0][0],p[i1][0]),max(p[i0][1],p[i1][1]))
    for i in e1:   e1[i] = i
    for i in eidx: eidx[i] = i
    ray[0],ray[1] = (0.5,0.9),(0.5,0.0)
    raye[0],raye[1] = 0,1
    for i in nodes: nodes[i].start,nodes[i].end = ti.u32(INT_MAX),0

@ti.func
def HitAABB(o,d,box):
    bmin,bmax = box.bmin,box.bmax
    d = d.normalized()
    tnmax ,tfmin, ret =  0.0,FLT_MAX, 1
    for j in ti.static(range(2)):
        if d[j]==0.0:
            if bmin[j]>o[j] or o[j]>bmax[j]:ret = 0
        else:
            tNear,tFar = (bmin[j]-o[j])/d[j],(bmax[j]-o[j])/d[j]
            if tNear>tFar: tNear,tFar = tFar,tNear
            if tFar  < tfmin : tfmin = tFar
            if tNear > tnmax : tnmax = tNear
            if tnmax > tfmin : ret = 0
    return ret

@ti.func
def HitEdge(o,d,p0,p1):
    ret = vec2(-1,-1)  #  o + ret[0]* d = p0 + ret[1]* (p1-p0)
    A = ti.Matrix.cols([d,p0-p1])
    if ti.abs( A.determinant() )>EPS: ret = A.inverse()@(p0 - o)
    if ret[1]<0.0 or ret[1]>1.0: ret[0] = -1
    return ret[0]

@ti.func
def HitScene(o,d):
    t,is_intersect = FLT_MAX,False
    temp[0][0],temp[0][1] = 1,0  # count , bvh index
    while temp[0][0]>0:
        new_cnt = 0
        for i in range(temp[0][0]):
            node = nodes[temp[0][i+1]]
            t_box = HitAABB(o,d,node.box)
            if t_box<0.0:continue
            if node.start > node.end:
                temp[1][new_cnt+0] = node.li
                temp[1][new_cnt+1] = node.ri
                new_cnt+=2
                continue
            for j in range(node.start,node.end+1):
                edge   = e[eidx[j]]
                this_t = HitEdge(o,d, p[edge[0]],p[edge[1]] )
                if this_t>0.0:
                    is_intersect = True
                    ti.atomic_min(t, this_t)
        temp[0][0] = new_cnt
        for i in range(new_cnt):
            temp[0][i+1] = temp[1][i]
    if not is_intersect: t = -1
    return t

@ti.kernel
def HitScene_BroutForce():
    o,d,t,is_intersect = ray[0],(ray[1]-ray[0]).normalized(),FLT_MAX,False
    for i in e:
        ret = HitEdge(o,d,p[e[i][0]],p[e[i][1]])
        if ret >=0 :
            is_intersect = True 
            ti.atomic_min(t, ret)
    if is_intersect:ray[2] = o+t*d
    else: ray[2] = -vec2(FLT_MAX,FLT_MAX)

@ti.kernel
def HitScene_BVH():
    o,d = ray[0],(ray[1]-ray[0]).normalized()
    t = HitScene(o,d)
    if t > 0.0: ray[2] = o+t*d
    else: ray[2] = o

def Main():
    Initialize()
    BVHBuildSAH(0,0, n-1)
    windows = ti.ui.Window("BVH2D", (1400,1400))
    canvas = windows.get_canvas()

    while not windows.is_pressed(ti.ui.ESCAPE):
        if windows.is_pressed(ti.ui.LMB):
            ray[1] = windows.get_cursor_pos()
            ray[1] = ray[0] + 10*(ray[1]-ray[0]).normalized()
        if windows.is_pressed(ti.ui.RMB):
            ray[0] = windows.get_cursor_pos()
            ray[1] = ray[0] + 10*(ray[1]-ray[0]).normalized()
        canvas.set_background_color((0.1,0.1,0.1))
        canvas.lines( p,  0.001, e1,(236/255,240/255,241/255))
        canvas.circles(p, 0.002,(102/255,187/255,106/255))
        
        # HitScene_BroutForce()
        HitScene_BVH()
        canvas.lines( ray,  0.001, raye,(242/255,202/255,47/255))
        canvas.circles(ray,0.004,(212/255,13/255,18/255))
        windows.show()

Main()