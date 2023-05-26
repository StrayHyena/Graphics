import taichi as ti
import random,time

FLT_MAX = 1e30
INT_MAX = 0xffffffff
EPS     = 1e-8
vec2  = ti.types.vector(2,ti.f32)
vec3  = ti.types.vector(3,ti.f32)
vec2i = ti.types.vector(2,ti.i32)
ti.init(arch=ti.cpu,cpu_max_num_threads=1)

n = 20 # how many edges
leaf_prim_count = 2
bin_cnt = 8

aabb     = ti.types.struct(bmin = vec2,bmax = vec2)
bvh_node = ti.types.struct(box = aabb, li = ti.u32,ri = ti.u32, start = ti.u32, end = ti.u32)  # end < start indicates a non-leaf node

p     = ti.Vector.field(2,ti.f32,2*n)
e     = ti.Vector.field(2,ti.i32,n)
aabbs = aabb.field(shape=n)
nodes = bvh_node.field(shape = 2*n)  #即使每个叶子节点放一个primitive，整个树的结点也不会超过2*n
eidx = ti.field(ti.u32,n)  # element is edge index for BVH building

e1      = ti.field(ti.i32,2*n)  # flattened e for canvas.lines()

boxs_pts = [ ti.Vector.field(2, ti.f32,8) for _ in range(2*n) ]
boxs_idx = ti.field(ti.i32,8)

def AABBGrow(this,other):
    this.bmin = vec2(min(this.bmin[0],other.bmin[0]),min(this.bmin[1],other.bmin[1]))
    this.bmax = vec2(max(this.bmax[0],other.bmax[0]),max(this.bmax[1],other.bmax[1]))
def AABBCenter(this):
    return (this.bmin+this.bmax)/2.0

# build node at node_idx in nodes(taichi struct field)
#                parent->  node_idx
#           left_child ->  node_idx + 1
#            .               ..                  
#           right_child -> node_idx + 1 + left_node_count
# [start,end]    NOT [start,end)   return how many nodes this tree have

def BuildBVH(node_idx,start,end):
    this_node = nodes[node_idx]
    split_axis,split_pos,split_cost = -1,-1,FLT_MAX
    left_count , right_count = 0,0

    for i in range(start,end+1):AABBGrow(this_node.box, aabbs[eidx[i]])
    for j in range(2):
        if this_node.box.bmax[j]<=this_node.box.bmin[j]:continue
        stride = (this_node.box.bmax[j] - this_node.box.bmin[j]) / bin_cnt
        bins = [ [aabb(vec2(FLT_MAX,FLT_MAX) , -vec2(FLT_MAX,FLT_MAX)),0] for _ in range(bin_cnt) ]
        for i in range(start,end+1):
            pos = AABBCenter(aabbs[eidx[i]])
            bin_idx = min(bin_cnt-1, int((pos[j]-this_node.box.bmin[j])/stride))
            AABBGrow(bins[bin_idx][0],aabbs[eidx[i]])
            bins[bin_idx][1]+=1

        left_area  = [0.0 for _ in range(bin_cnt-1)]
        left_cnt   = [0 for _ in range(bin_cnt-1)]
        left_box   = aabb(vec2(1,1)*FLT_MAX,-vec2(1,1)*FLT_MAX)
        left_sum   = 0
        right_area = [0.0 for _ in range(bin_cnt-1)]
        right_cnt  = [0 for _ in range(bin_cnt-1)]
        right_box  = aabb(vec2(1,1)*FLT_MAX,-vec2(1,1)*FLT_MAX)
        right_sum  = 0
        for i in range(bin_cnt-1):
            AABBGrow(left_box, bins[i][0])
            left_area[i] = (left_box.bmax-left_box.bmin).sum()
            left_sum += bins[i][1]
            left_cnt[i]  = left_sum

            AABBGrow(right_box, bins[bin_cnt-i-1][0])
            right_area[bin_cnt-i-2] = (right_box.bmax-right_box.bmin).sum()
            right_sum += bins[bin_cnt-i-1][1]
            right_cnt[bin_cnt-i-2]  = right_sum
        for i in range(bin_cnt-1):
            cost = left_area[i]*left_cnt[i] + right_area[i]*right_cnt[i]
            if cost<split_cost: 
                split_axis,split_pos,split_cost=j,this_node.box.bmin[j]+stride*(i+1),cost
                left_count,right_count=left_cnt[i],right_cnt[i]

    boxs_pts[node_idx][0] = this_node.box.bmin
    boxs_pts[node_idx][2] = vec2(this_node.box.bmin[0],this_node.box.bmax[1])
    boxs_pts[node_idx][4] = this_node.box.bmax
    boxs_pts[node_idx][6] = vec2(this_node.box.bmax[0],this_node.box.bmin[1])
    for j in range(1,8,2):boxs_pts[node_idx][j] = boxs_pts[node_idx][(j+1)%8]

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
    left_node_cnt    =  BuildBVH(this_node.li, start, start+left_count-1)
    this_node.ri     = this_node.li + left_node_cnt
    right_node_cnt   =  BuildBVH(this_node.ri, start+left_count,  end)
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
    for i in nodes:
        nodes[i].box = aabb(FLT_MAX*vec2(1,1),-FLT_MAX*vec2(1,1)) 
        nodes[i].start,nodes[i].end = ti.u32(INT_MAX),0
    for i in boxs_idx: boxs_idx[i] = i

def RenderBVH(canvas,depth):
    s,_ = [(0,(52/255,152/255,219/255))],0
    while len(s)>0 and _<=depth:
        ns = []
        for i,color in s:
            if _==depth: canvas.lines(boxs_pts[i],0.001,boxs_idx,color)
            if nodes[i].start<=nodes[i].end: continue
            ns.append((nodes[i].li, (231/255,76/255,60/255)))
            ns.append((nodes[i].ri,(52/255,152/255,219/255)))
        s,_ = ns,_+1
def GetDepthBVH(i):
    if nodes[i].start<=nodes[i].end:return 1
    else:return 1+max(GetDepthBVH(nodes[i].li),GetDepthBVH(nodes[i].ri))

def Main():
    Initialize()
    BuildBVH(0,0, n-1)
    windows = ti.ui.Window("BVH2D", (1400,1400))
    canvas = windows.get_canvas()

    _,d = 0,GetDepthBVH(0)
    while not windows.is_pressed(ti.ui.ESCAPE):
        canvas.set_background_color((0.1,0.1,0.1))
        canvas.lines( p,  0.001, e1,(236/255,240/255,241/255))
        canvas.circles(p, 0.002,(102/255,187/255,106/255))
        RenderBVH(canvas,_)
        _ = (_+1)%d
        time.sleep(1)
        windows.show()

Main()