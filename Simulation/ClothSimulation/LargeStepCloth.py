import numpy as np
import polyscope as ps
import taichi as ti
import polyscope.imgui as psim
import scipy.sparse as sp
import trimesh,time

ti.init(arch=ti.cpu,default_fp=ti.f64)
EPS,MAX = 1e-8,1.7976931348623157e+308
I3,O3 = ti.math.mat3([[1,0,0],[0,1,0],[0,0,1]]),ti.math.mat3([[0,0,0],[0,0,0],[0,0,0]])
vec3,vec2,mat2,vec3i = ti.math.vec3,ti.math.vec2,ti.math.mat2,ti.math.ivec3
nan,inf = ti.math.nan,ti.math.inf

class Constraint:
    @ti.data_oriented
    class EdgeStrecth:
        def __init__(self,cloth,k,kd):
            self.kd,self.n = kd,len(cloth.edges_unique)  # how many constraints
            self.ij = set()
            self.idx = ti.Vector.field(2, ti.i32, self.n)
            self.idx.from_numpy(cloth.edges_unique.astype(np.int32))
            self.l0 = ti.field(ti.f64, self.n)
            for ei, (i, j) in enumerate(cloth.edges_unique):
                self.ij |= {(i, i),(i, j),(j, i),(j, j)}
                self.l0[ei] = np.linalg.norm(cloth.vertices[i] - cloth.vertices[j])
            self.k = ti.field(ti.f64, self.n)
            self.k.fill(k)

        @ti.func
        def C_DC_DDC(xi,xj,l0): # constraint, derivative of constraint, 2nd derivative of constraint
            xji = xi - xj
            normxji = xji.norm()
            C = normxji - l0
            pC_pxi = xji / normxji
            pC_pxj = -pC_pxi
            ppC_pxipxi = I3 / normxji - xji.outer_product(xji) / normxji ** 3
            ppC_pxipxj = -ppC_pxipxi
            ppC_pxjpxj = ppC_pxipxi
            ppC_pxjpxi = -ppC_pxjpxj
            return C,(pC_pxi,pC_pxj),(ppC_pxipxi,ppC_pxipxj,ppC_pxjpxi,ppC_pxjpxj)

    @ti.data_oriented
    class Pin:
        def __init__(self,cloth,pins,k,kd):
            self.kd,self.n = kd,len(pins)  # how many constraints
            self.ij = set([(i,i)for i in pins])
            self.idx = ti.field(ti.i32, self.n)
            self.idx.from_numpy(np.array(pins).astype(np.int32))
            self.l0 = ti.Vector.field(3,ti.f64, self.n)
            self.l0.from_numpy(cloth.vertices[np.array(pins)])
            self.k = ti.field(ti.f64, self.n)
            self.k.fill(k)

        @ti.func
        def C_DC_DDC(xi,l0): # constraint, derivative of constraint, 2nd derivative of constraint
            xji = xi - l0
            C = xji.norm()
            pC_pxi = xji / C
            ppC_pxipxi = I3 / C - xji.outer_product(xji) / C ** 3
            return C,pC_pxi,ppC_pxipxi

    @ti.data_oriented
    class Bend:
        def __init__(self,cloth,k,kd):
            self.kd,edge_faces = kd, {(min(i,j),max(i,j)):[]for i,j in cloth.edges_unique} # 这个边链接了哪些三角形
            for f in cloth.faces:
                for i in range(3):
                    vi,vj = f[i],f[(i+1)%3]
                    key = (min(vi,vj),max(vi,vj))
                    edge_faces[key].append(f)
            indices = []
            for (i,j),fs in edge_faces.items():
                if len(fs)!=2:continue
                indices.append([i,j])
                for f in fs:
                    for vi in f:
                        if vi==i or vi==j:continue
                        indices[-1].append(vi)
            self.ij = set()
            for idx in indices:
                assert len(idx)==4
                for i in range(4):
                    for j in range(4):
                        self.ij.add((idx[i],idx[j]))
            self.n = len(indices) # how many constraints
            self.idx = ti.Vector.field(4,ti.i32, self.n)
            self.idx.from_numpy(np.array(indices).astype(np.int32))  # if len(indices)!=0:
            self.l0 = ti.field(ti.f64, self.n)
            self.l0.fill(0)
            self.k = ti.field(ti.f64, self.n)
            self.k.fill(k)

        @ti.func     #  x index is conform with the ref paper: Derivation of discrete bending forces and their gradients
        def C_DC_DDC(x0,x1,x2,x3,l0): # constraint, derivative of constraint, 2nd derivative of constraint
            e0,e1,e2,e3,e4 = x1-x0,x2-x0,x3-x0,x2-x1,x3-x1  # Figure 1
            e0l = e0.norm()
            e0_,e1_,e2_,e3_,e4_ = e0/e0l,e1/e1.norm(),e2/e2.norm(),e3/e3.norm(),e4/e4.norm()
            cos1,cos2,cos3,cos4 = e0_.dot(e1_),e0_.dot(e2_),-e0_.dot(e3_),-e0_.dot(e4_) # cos α 1234
            sin1,sin2,sin3,sin4 = e0_.cross(e1_).norm(),e0_.cross(e2_).norm(),e0_.cross(e3_).norm(),e0_.cross(e4_).norm()
            n1,n2 = e0.cross(e3),-e0.cross(e4)  # page 12
            S1,S2 = n1.norm(),n2.norm() # 2 * triangle area
            h01,h1,h3,  h02,h2,h4 = S1/e0l,S1/e1.norm(),S1/e3.norm(),   S2/e0l,S2/e2.norm(),S2/e4.norm()
            n1_,n2_ = n1/S1,n2/S2
            m1_,m01_,m3_, m2_,m02_,m4_ = n1_.cross(e1_), e0_.cross(n1_), e3_.cross(n1_),   e2_.cross(n2_), n2_.cross(e0_), n1_.cross(e4_),
            return ti.atan2(n1_.cross(n2_).dot(e0_),n1_.dot(n2_))-l0,(  # page 14
                    (cos3/sin3*n1_+cos4/sin4*n2_)/e0l,(cos1/sin1*n1_+cos2/sin2*n2_)/e0l,-e0l*n1_/S1,-e0l*n2_/S2
                ),( # page 27
           cos3/h3**2*(m3_.outer_product(n1_)+n1_.outer_product(m3_))-n1_.outer_product(m01_)/e0l**2+cos4/h4**2*(m4_.outer_product(n2_)+n2_.outer_product(m4_))-n2_.outer_product(m02_)/e0l**2,
           (cos3*m1_.outer_product(n1_)+cos1*n1_.outer_product(m3_))/(h3*h1)+n1_.outer_product(m01_)/e0l**2 + (cos4*m2_.outer_product(n2_)+cos2*n2_.outer_product(m4_))/(h2*h4)+n2_.outer_product(m02_)/e0l**2,
           (cos3*m01_.outer_product(n1_)-n1_.outer_product(m3_))/(h3*h01),
           (cos4*m02_.outer_product(n2_)-n2_.outer_product(m4_))/(h4*h02),

           (cos1 * m3_.outer_product(n1_) + cos3 * n1_.outer_product(m1_)) / (h3 * h1) + n1_.outer_product(m01_) / e0l ** 2 + (cos2 * m4_.outer_product(n2_) + cos4 * n2_.outer_product(m2_)) / (h2 * h4) + n2_.outer_product(m02_) / e0l ** 2,
           cos1 / h1 ** 2 * (m1_.outer_product(n1_) + n1_.outer_product(m1_)) - n1_.outer_product(m01_) / e0l ** 2 + cos2 / h2 ** 2 * (m2_.outer_product(n2_) + n2_.outer_product(m2_)) - n2_.outer_product(m02_) / e0l ** 2,
           (cos1 * m01_.outer_product(n1_) - n1_.outer_product(m1_)) / (h1 * h01),
           (cos2 * m02_.outer_product(n2_) - n2_.outer_product(m2_)) / (h2 * h02),

           (cos3*n1_.outer_product(m01_)-m3_.outer_product(n1_))/(h01*h3),(cos1*n1_.outer_product(m01_)-m1_.outer_product(n1_))/(h01*h1),-(m01_.outer_product(n1_)+n1_.outer_product(m01_))/h01**2,0,    # page 21
           (cos4*n2_.outer_product(m02_)-m4_.outer_product(n2_))/(h02*h4),(cos4*n2_.outer_product(m02_)-m2_.outer_product(n2_))/(h02*h2),0,-(m02_.outer_product(n2_)+n2_.outer_product(m02_))/h02**2   # page 21
                 )

    class VF:
        @ti.func
        def C_DC_DDC(xv,x0,x1,x2,w:vec3,thickness):
            n = (x1-x0).cross(x2-x0).normalized()
            if (xv-w[0]*x0-w[1]*x1-w[2]*x2).dot(n)<0:n*=-1  # 假设此时v在f正确的一边，n总是从face指向vertex
            return thickness - (xv-w[0]*x0-w[1]*x1-w[2]*x2).dot(n),(-n,w[0]*n,w[1]*n,w[2]*n),(O3, O3, O3, O3,O3, O3, O3, O3,O3, O3, O3, O3,O3, O3, O3, O3)
    class EE:
        @ti.func
        def C_DC_DDC(x0,x1,x2,x3,w:vec2,thickness):
            a,b = x0+w[0]*(x1-x0),x2+w[1]*(x3-x2)
            n = (x0-x1).cross(x2-x3).normalized()
            if n.dot(a-b)<0:n*=-1 # 假设此时e0在e1正确的一边，总是e1指向e0
            return thickness - (a-b).dot(n),(-(1-w[0])*n,-w[0]*n,(1-w[1])*n,w[1]*n),(O3, O3, O3, O3,O3, O3, O3, O3,O3, O3, O3, O3,O3, O3, O3, O3)

# There are three types of Constraints.
# A. 【Material】Material-related, e.g. bend,edge stretch, which are decided by the cloth geometry
# B. 【CustomConstraint】User-defined constraint, e.g. pin point, stitch points (only support pin point now)
# C. 【Collision】Constraints that generated by Collision
# A,B are always fixed, while C will change dynamically during every iteration

@ti.data_oriented
class Collision:
    @ti.func
    def quadratic_root(a: ti.f64, b: ti.f64, c: ti.f64):
        count, x0, x1, d = 0, nan, nan, b * b - 4 * a * c
        if d >= 0: count, x0, x1 = 2, 0 if c == 0 else -2 * c / (b + ti.math.sign(b) * ti.sqrt(d)), -(b + ti.math.sign(b) * ti.sqrt(d)) / (2 * a)
        return 1 if 0 <= d < EPS or -EPS < a < EPS else count, ti.min(x0, x1), ti.max(x0, x1)

    @ti.func
    def cubic_root(a: ti.f64, b: ti.f64, c: ti.f64, d: ti.f64):
        count, x0, x1, x2, i = 0, nan, nan, nan, 0
        qcount, s0, s1 = Collision.quadratic_root(3 * a, 2 * b, c)
        fs0, fs1, r = a * s0 ** 3 + b * s0 ** 2 + c * s0 + d, a * s1 ** 3 + b * s1 ** 2 + c * s1 + d, 1e-3
        if qcount <= 1:   count, x0 = 1, -b / 3 / a - ti.math.sign(a) * r  # 全实数域严格单调
        if qcount == 2:  # 全实数域三段单调
            if fs0 * fs1 > 0:count, x0 = 1, (s0 - r) if fs0 * a > 0 else (s1 + r)  # 全实数域只有一个实根，要么在 [-∞,s0] 要么在 [s1,+∞]
            else:count, x0 = 3, (s0 + s1) / 2  # 全实数域有仨实根,有可能有俩实根相同
        while i < 15:
            f, i = a * x0 ** 3 + b * x0 ** 2 + c * x0 + d, i + 1
            if abs(f) < 1e-8: break
            df = 3 * a * x0 ** 2 + 2 * b * x0 + c
            x0 -= f / df
        if count == 3:  # 这个值在三个根中是中间的那个
            x1 = x0
            qcount, x0, x2 = Collision.quadratic_root(a, b + a * x1, c + (b + a * x1) * x1)
        return count, vec3(x0, x1, x2)

    @ti.dataclass
    class AABB:
        bmin: vec3
        bmax: vec3
        @ti.func
        def EatPoint(self, pos: vec3):
            self.bmax, self.bmin = ti.math.max(self.bmax, pos), ti.math.min(self.bmin, pos)
            return self
        @ti.func
        def EatAABB(self, box): return self.EatPoint(box.bmin).EatPoint(box.bmax)
        @ti.func
        def Extend(self,value:vec3):return self.EatPoint(self.bmin-value).EatPoint(self.bmax+value)

    @ti.func
    def IJK2TableI(self,ijk):return (73856093*ijk[0]^19349663*ijk[1]^83492791*ijk[2])%self.tablesize

    def __init__(self,cloth,tablesize=2039,element_num_per_cell=512,thickness=0.001):
        self.tablesize,self.d,self.enpc = tablesize,thickness,element_num_per_cell
        self.vn,self.en,self.fn = len(cloth.vertices),len(cloth.edges_unique),len(cloth.faces)
        self.edges = ti.Vector.field(2,ti.i32,self.en)
        self.edges.from_numpy(cloth.edges_unique.astype(np.int32))
        self.faces = ti.Vector.field(3,ti.i32,self.fn)
        self.faces.from_numpy(cloth.faces.astype(np.int32))
        self.vAABBs = self.AABB.field(shape = self.vn)
        self.eAABBs = self.AABB.field(shape = self.en)
        self.fAABBs = self.AABB.field(shape = self.fn)
        # Broad Phase
        self.cellf = ti.field(ti.i32)   #  face(triangle index) in spatial cell, cellf[i]里放的是一些在这个cell里的face indices
        self.cellfnode = ti.root.pointer(ti.i,tablesize).dense(ti.j,element_num_per_cell)
        self.cellfnode.place(self.cellf)
        self.cellfn = ti.field(ti.i32,tablesize)  # how many current faces at each table entry ?
        self.celle = ti.field(ti.i32)  #  edge(edge index) in spatial cell
        self.cellenode = ti.root.pointer(ti.i,tablesize).dense(ti.j,element_num_per_cell)
        self.cellenode.place(self.celle)
        self.cellen = ti.field(ti.i32,tablesize)
        # Narrow Phase
        self.vf = ti.Vector.field(3,ti.f64)           # vf pairs' barycentric coordinate where actual collisions occur
        self.vfnode = ti.root.pointer(ti.i,self.vn).bitmasked(ti.j,self.fn) # use this sparse structure to remove repeated pair from broad phase
        self.vfnode.place(self.vf)
        self.ee = ti.Vector.field(2,ti.f64)
        self.eenode = ti.root.pointer(ti.i,self.en).bitmasked(ti.j,self.en)
        self.eenode.place(self.ee)

        self.f = ti.Vector.field(3, ti.f64, self.vn)
        self.pf_px = ti.Matrix.field(3, 3, ti.f64)
        self.pf_px_node = ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn)
        self.pf_px_node.place(self.pf_px)

    @ti.kernel
    def Print(self):
        print('vf')
        for vi,fi in self.vf:print(vi,self.faces[fi][0],self.faces[fi][1],self.faces[fi][2],self.vf[vi,fi])
        print('ee')
        for ei0,ei1 in self.ee:print(self.edges[ei0][0],self.edges[ei0][1],self.edges[ei1][0],self.edges[ei1][1],self.ee[ei0,ei1])
    @ti.kernel
    def PrintNum(self):
        vfnum,eenum = 0,0
        for vi,fi in self.vf: ti.atomic_add(vfnum,1)
        for ei0,ei1 in self.ee: ti.atomic_add(eenum,1)
        print('vf,ee ',vfnum,eenum)

    @ti.kernel  # X: current position  Y: next position
    def CollectCollisionPairs(self,X:ti.template(),Y:ti.template(),is_continuous:bool):
        # clear cells -----------------------------------------------------------------------------------
        for i in self.vAABBs:self.vAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))
        for i in self.eAABBs:self.eAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))
        for i in self.fAABBs:self.fAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))
        # find max cell size  -----------------------------------------------------------------------------------
        eCellSize,fCellSize = -vec3(MAX),-vec3(MAX)
        for vi in X:self.vAABBs[vi].EatPoint(X[vi]).EatPoint(Y[vi])
        for ei in self.edges:
            vi,vj = self.edges[ei]
            self.eAABBs[ei].EatPoint(X[vi]).EatPoint(X[vj]).EatPoint(Y[vi]).EatPoint(Y[vj]).Extend(vec3(self.d))
            boxsize = self.eAABBs[ei].bmax-self.eAABBs[ei].bmin
            for i in ti.static(range(3)): ti.atomic_max(eCellSize[i],boxsize[i])
        for fi in self.faces:
            vi,vj,vk = self.faces[fi]
            self.fAABBs[fi].EatPoint(X[vi]).EatPoint(X[vj]).EatPoint(X[vk]).EatPoint(Y[vi]).EatPoint(Y[vj]).EatPoint(Y[vk]).Extend(vec3(self.d))
            boxsize = self.fAABBs[fi].bmax-self.fAABBs[fi].bmin
            for i in ti.static(range(3)): ti.atomic_max(fCellSize[i],boxsize[i])
        # fill cell with faces or edges -----------------------------------------------------------------------------------
        self.cellfn.fill(0)
        for i, j in self.cellfnode: ti.deactivate(self.cellfnode, [i, j])
        for fi in self.fAABBs:
            st, ed = ti.floor(self.fAABBs[fi].bmin / fCellSize).cast(ti.i32), ti.floor(self.fAABBs[fi].bmax / fCellSize).cast(ti.i32)
            for i0, i1, i2 in ti.ndrange(ed[0] - st[0] + 1, ed[1] - st[1] + 1, ed[2] - st[2] + 1):
                i = self.IJK2TableI(st + vec3i(i0, i1, i2))
                self.cellf[i, ti.atomic_add(self.cellfn[i], 1)] = fi
                if self.cellfn[i] >= self.enpc: print('cell too many faces ', self.cellfn[i])
        self.cellen.fill(0)
        for i, j in self.cellenode: ti.deactivate(self.cellenode, [i, j])
        for ei in self.eAABBs:
            st, ed = ti.floor(self.eAABBs[ei].bmin / eCellSize).cast(ti.i32), ti.floor(self.eAABBs[ei].bmax / eCellSize).cast(ti.i32)
            for i0, i1, i2 in ti.ndrange(ed[0] - st[0] + 1, ed[1] - st[1] + 1, ed[2] - st[2] + 1):
                i = self.IJK2TableI(st + vec3i(i0, i1, i2))
                self.celle[i, ti.atomic_add(self.cellen[i], 1)] = ei
                if self.cellen[i] >= self.enpc: print('cell too many edges ', self.cellen[i])
        # collect actual pairs -----------------------------------------------------------------------------------
        for i, j in self.vfnode: ti.deactivate(self.vfnode, [i, j])
        for vi in X:
            i, x4 = self.IJK2TableI(ti.floor(X[vi] / fCellSize).cast(ti.i32)), X[vi]
            for fii in range(self.cellfn[i]):
                fi = self.cellf[i, fii]
                vi1,vi2,vi3 = self.faces[fi][0],self.faces[fi][1],self.faces[fi][2]
                if vi == vi1 or vi == vi2 or vi == vi3: continue
                x1, x2, x3 = X[vi1], X[vi2], X[vi3]
                x13, x23, x43, x31, x21 = x1 - x3, x2 - x3, x4 - x3,x3 - x1,x2 - x1
                if is_continuous:
                    y1,y2,y3,y4 = Y[vi1],Y[vi2],Y[vi3],Y[vi]
                    v1,v2,v3,v4 = y1-x1,y2-x2,y3-x3,y4-x4
                    v43,v21,v31 = v4-v3,v2-v1,v3-v1
                    n,r = Collision.cubic_root(v43.dot(v21.cross(v31)),
                                               x43.dot(v21.cross(v31))+v43.dot(v21.cross(x31))+v43.dot(x21.cross(v31)),
                                               x43.dot(v21.cross(x31))+x43.dot(x21.cross(v31))+v43.dot(x21.cross(x31)),
                                               x43.dot(x21.cross(x31)) )
                    for j in range(n):  # 有根只是说明共面，还要算重心坐标才能确定是否真的穿了
                        if r[j]<=0 or r[j]>=1:continue
                        x1t,x2t,x3t,x4t = ti.math.mix(x1,y1,r[j]),ti.math.mix(x2,y2,r[j]),ti.math.mix(x3,y3,r[j]),ti.math.mix(x4,y4,r[j])
                        w = vec3((x4t-x2t).cross(x4t-x3t).norm(),(x4t-x1t).cross(x4t-x3t).norm(),(x4t-x2t).cross(x4t-x1t).norm())/(x2t-x1t).cross(x3t-x1t).norm()
                        if abs(w.sum()-1)<EPS:  self.vf[vi, fi] = w
                        break
                else:
                    n = x13.cross(x23).normalized()
                    if ti.abs(x43.dot(n)) >= self.d: continue
                    w = mat2([(x13.dot(x13), x13.dot(x23)), (x13.dot(x23), x23.dot(x23))]).inverse() @ vec2(x13.dot(x43), x23.dot(x43))  # [Bridson 2002]page4eq1
                    if 0 <= w[0] <= 1 and 0 <= w[1] <= 1 and 0 <= w.sum() <= 1: self.vf[vi, fi] = vec3(w[0], w[1],1 - w.sum())
        for i, j in self.eenode: ti.deactivate(self.eenode, [i, j])
        for i in self.cellen:
            for ei0_ in range(self.cellen[i]):
                ei0 = self.celle[i, ei0_]
                v0i, v0j = self.edges[ei0][0], self.edges[ei0][1]
                x1, x2 = X[v0i], X[v0j]
                for ei1_ in range(ei0_):
                    ei1 = self.celle[i, ei1_]
                    v1i, v1j = self.edges[ei1][0], self.edges[ei1][1]
                    if v0i == v1i or v0i == v1j or v0j == v1i or v0j == v1j: continue
                    x3, x4 = X[v1i], X[v1j]
                    x21, x31, x43, x31, x21 = x2 - x1, x3 - x1, x4 - x3,x3 - x1,x2 - x1
                    if is_continuous:
                        y1,y2,y3,y4 = Y[v0i],Y[v0j],Y[v1i],Y[v1j]
                        v1,v2,v3,v4 = y1-x1,y2-x2,y3-x3,y4-x4
                        v43,v21,v31 = v4-v3,v2-v1,v3-v1
                        n,r = Collision.cubic_root(v43.dot(v21.cross(v31)),
                                                   x43.dot(v21.cross(v31))+v43.dot(v21.cross(x31))+v43.dot(x21.cross(v31)),
                                                   x43.dot(v21.cross(x31))+x43.dot(x21.cross(v31))+v43.dot(x21.cross(x31)),
                                                   x43.dot(x21.cross(x31)) )
                        for j in range(n):
                            if r[j] <= 0 or r[j] >= 1: continue
                            x1t,x2t,x3t,x4t = ti.math.mix(x1,y1,r[j]),ti.math.mix(x2,y2,r[j]),ti.math.mix(x3,y3,r[j]),ti.math.mix(x4,y4,r[j])
                            x21,x43,x31 = x2t-x1t,x4t-x3t,x3t-x1t
                            w = mat2([(x21.dot(x21), -x21.dot(x43)), (-x21.dot(x43), x43.dot(x43))]).inverse() @ vec2(x21.dot(x31), -x43.dot(x31))  # [Bridson 2002]page4eq2
                            if 0 <= w[0] <= 1 and 0 <= w[1] <= 1:  self.ee[ei0, ei1] = w
                            break
                    else:
                        n = x21.cross(x43)
                        if n.norm() < EPS or ti.abs(x31.dot(n.normalized())) >= self.d: continue  # ei0,ei1 parallel or distance > d
                        w = mat2([(x21.dot(x21), -x21.dot(x43)), (-x21.dot(x43), x43.dot(x43))]).inverse() @ vec2(x21.dot(x31), -x43.dot(x31))  # [Bridson 2002]page4eq2
                        if 0 <= w[0] <= 1 and 0 <= w[1] <= 1: self.ee[ei0, ei1] = w

    @ti.kernel
    def Update(self,X:ti.template()):
        self.f.fill(0)
        self.pf_px.fill(0)
        # for vf pair: E = k/2*(d-(xv-(w1*x1+w2*x2+w3*x3).n)^2  [CAMA2016]        里面的那个距离计算见[Bridson 2002] page-5 eq(1)
        for i,j in self.pf_px_node:ti.deactivate(self.pf_px_node,[i,j])
        for vi,fi in self.vf:
            k,idx = 1e3,(vi,self.faces[fi][0],self.faces[fi][1],self.faces[fi][2])
            C, C_jacobi, C_hess = Constraint.VF.C_DC_DDC(X[idx[0]],X[idx[1]],X[idx[2]],X[idx[3]],self.vf[vi,fi],self.d)
            if C<0 or C>self.d:print('[ERR] vf C=',C,idx)
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(4)):
                i, pC_pxi = idx[i_], C_jacobi[i_]
                self.f[i] += -k * pC_pxi * C
                for j_ in ti.static(range(4)):
                    j, pC_pxj = idx[j_], C_jacobi[j_]
                    self.pf_px[i, j] += -k * pC_pxi.outer_product(pC_pxj) - C_hess[4 * i_ + j_] * k * C
        for ei0,ei1 in self.ee:
            k,idx = 1e3,(self.edges[ei0][0],self.edges[ei0][1],self.edges[ei1][0],self.edges[ei1][1])
            C, C_jacobi, C_hess = Constraint.EE.C_DC_DDC(X[idx[0]],X[idx[1]],X[idx[2]],X[idx[3]],self.ee[ei0,ei1],self.d)
            if C<0 or C>self.d:print('[ERR] ee C=',C,idx)
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(4)):
                i, pC_pxi = idx[i_], C_jacobi[i_]
                self.f[i] += -k * pC_pxi * C
                for j_ in ti.static(range(4)):
                    j, pC_pxj = idx[j_], C_jacobi[j_]
                    self.pf_px[i, j] += -k * pC_pxi.outer_product(pC_pxj) - C_hess[4 * i_ + j_] * k * C

@ti.data_oriented
class Material:
    def __init__(self,cloth):
        self.vn = len(cloth.vertices)
        self.stretch = Constraint.EdgeStrecth(cloth, 1e3, 0.0001)
        self.bend = Constraint.Bend(cloth, 0.0001, 0.00001)
        self.ij = sorted(list(  set().union(*[con.ij for con in [self.stretch,self.bend]])     ))  # 一个关键的观察，一旦约束定下来了，稀疏hessian的ij项也就定下来了。
        self.f = ti.Vector.field(3, ti.f64, self.vn)
        self.pf_px = ti.Matrix.field(3, 3, ti.f64)
        self.pf_pv = ti.Matrix.field(3, 3, ti.f64)
        ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn).place(self.pf_px)
        ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn).place(self.pf_pv)

    @ti.kernel
    def Update(self,X:ti.template(),V:ti.template()):
        self.f.fill(0)
        self.pf_px.fill(0)
        self.pf_pv.fill(0)
        for ci in self.stretch.k:
            k, kd = self.stretch.k[ci], self.stretch.kd
            i, j = self.stretch.idx[ci]
            v = (V[i], V[j])
            C, C_jacobi, C_hess = Constraint.EdgeStrecth.C_DC_DDC(X[i], X[j], self.stretch.l0[ci])
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(2)):
                i, pC_pxi, dotC = self.stretch.idx[ci][i_], C_jacobi[i_], C_jacobi[i_].dot(v[i_])
                self.f[i] += -k * pC_pxi * C - kd * pC_pxi * dotC
                for j_ in ti.static(range(2)):
                    j, pC_pxj = self.stretch.idx[ci][j_], C_jacobi[j_]
                    self.pf_px[i, j] += -k * pC_pxi.outer_product(pC_pxj) - C_hess[2 * i_ + j_] * (k * C + kd * dotC)  # ))#)
                    self.pf_pv[i, j] += -kd * pC_pxi.outer_product(pC_pxj)
        for ci in self.bend.k:
            k, kd = self.bend.k[ci], self.bend.kd
            i0, i1, i2, i3 = self.bend.idx[ci]
            v = (V[i0], V[i1], V[i2], V[i3])
            C, C_jacobi, C_hess = Constraint.Bend.C_DC_DDC(X[i0], X[i1], X[i2], X[i3],self.bend.l0[ci])
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(4)):
                i, pC_pxi, dotC = self.bend.idx[ci][i_], C_jacobi[i_], C_jacobi[i_].dot(v[i_])
                self.f[i] += -k * pC_pxi * C - kd * pC_pxi * dotC
                for j_ in ti.static(range(4)):
                    j, pC_pxj = self.bend.idx[ci][j_], C_jacobi[j_]
                    self.pf_px[i, j] += -k * pC_pxi.outer_product(pC_pxj) - C_hess[4 * i_ + j_] * (k * C + kd * dotC)  # ))#)
                    self.pf_pv[i, j] += -kd * pC_pxi.outer_product(pC_pxj)

# Only support pin now
@ti.data_oriented
class CustomConstraint:
    def __init__(self,cloth,pins):
        self.vn= len(cloth.vertices)
        self.pin     = Constraint.Pin(cloth,pins,1e4,0.1)
        self.f = ti.Vector.field(3, ti.f64, self.vn)
        self.pf_px = ti.Matrix.field(3, 3, ti.f64)
        self.pf_pv = ti.Matrix.field(3, 3, ti.f64)
        ti.root.bitmasked(ti.i, self.vn).place(self.pf_px)
        ti.root.bitmasked(ti.i, self.vn).place(self.pf_pv)

    @ti.kernel
    def Update(self,X:ti.template(),V:ti.template()):
        self.f.fill(0)
        self.pf_px.fill(0)
        self.pf_pv.fill(0)
        for ci in self.pin.k:
            i, l0, k, kd = self.pin.idx[ci], self.pin.l0[ci], self.pin.k[ci], self.pin.kd
            C, pC_pxi, ppC_pxipxi = Constraint.Pin.C_DC_DDC(X[i], l0)
            if -EPS < C < EPS: continue
            dotC = pC_pxi.dot(V[i])
            self.f[i] = -k * pC_pxi * C - kd * pC_pxi * dotC
            self.pf_px[i] = -k * (pC_pxi.outer_product(pC_pxi) + ppC_pxipxi * C) - kd * ppC_pxipxi * dotC
            self.pf_pv[i] = -kd * pC_pxi.outer_product(pC_pxi)

@ti.data_oriented
class Simulator:
    def __init__(self, clothobjpath, pins=[]):
        self.h = 1.0 / 150.0
        cloth = trimesh.load(clothobjpath)
        self.cloth = cloth
        self.vn, self.en, self.fn = len(cloth.vertices), len(cloth.edges_unique), len(cloth.faces)
        self.x = ti.Vector.field(3, ti.f64, self.vn)  # position at the start of time step
        self.y = ti.Vector.field(3, ti.f64, self.vn)  # position after cloth internal dynamics (after material before ccd)
        self.x.from_numpy(cloth.vertices)
        self.v = ti.Vector.field(3, ti.f64, self.vn)
        self.v.from_numpy(np.zeros_like(cloth.vertices))
        self.m = ti.field(ti.f64, self.vn)
        self.m.from_numpy(2*np.ones(self.vn)*cloth.area/self.vn)
        self.faces = ti.Vector.field(3, ti.i32, self.fn)
        self.faces.from_numpy(cloth.faces.astype(np.int32))
        self.external_force = ti.Vector.field(3,ti.f64,self.vn)

        self.material    = Material(cloth)
        self.constraints = CustomConstraint(cloth,pins)
        self.collision   = Collision(cloth)
        self.b = ti.Vector.field(3, ti.f64, self.vn)
        self.A = ti.Matrix.field(3, 3, ti.f64)
        self.ANode = ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn)
        self.ANode.place(self.A)

        hessian_entry_num = len(self.material.ij)
        self.triplet_i2ij = ti.Vector.field(2, ti.i32, hessian_entry_num)
        self.ij2triplet_i = ti.field(ti.i32)
        self.fixed_ij_node = ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn)
        self.fixed_ij_node.place(self.ij2triplet_i)
        self.hess_i = np.zeros(9 * hessian_entry_num).astype(np.int32)
        self.hess_j = np.zeros(9 * hessian_entry_num).astype(np.int32)
        self.hess_value = ti.field(ti.f64, 9 * hessian_entry_num)
        for entry_idx, (i, j) in enumerate(self.material.ij):
            self.ij2triplet_i[i, j] = entry_idx
            self.triplet_i2ij[entry_idx] = ti.math.ivec2(i, j)
            for i_, j_ in ((i, j) for i in range(3) for j in range(3)):
                self.hess_i[9 * entry_idx + 3 * i_ + j_] = 3 * i + i_
                self.hess_j[9 * entry_idx + 3 * i_ + j_] = 3 * j + j_
        # for collision
        self.mven = 2048 # max variadic entry num
        self.curr_variadic_entry_num = ti.field(ti.i32,())
        self.hess_i_variadic = ti.field(ti.i32,9*self.mven)
        self.hess_j_variadic = ti.field(ti.i32,9*self.mven)
        self.hess_value_variadic = ti.field(ti.f64,9*self.mven)

    @ti.kernel
    def AssembleMatrix(self):
        h = ti.cast(self.h,ti.f64)
        for i in self.b:
            self.b[i] = self.m[i] * vec3(0, -1.81, 0) + self.external_force[i] + self.constraints.f[i] + self.material.f[i]+ self.collision.f[i] #
        for i,j in self.material.pf_px:     self.b[i] += h * self.material.pf_px[i, j] @ self.v[j]
        for i,j in self.collision.pf_px:     self.b[i] += h * self.collision.pf_px[i, j] @ self.v[j]
        for i in self.constraints.pf_px:    self.b[i] += h * self.constraints.pf_px[i] @ self.v[i]

        self.A.fill(0)
        for i,j in self.A:ti.deactivate(self.ANode,[i,j])
        for i in self.m:                    self.A[i,i] = I3*self.m[i]
        for i,j in self.material.pf_px:     self.A[i,j] += -h*h*self.material.pf_px[i,j]
        for i in self.constraints.pf_px:    self.A[i,i] += -h*h*self.constraints.pf_px[i]
        for i,j in self.collision.pf_px:    self.A[i,j] += -h*h*self.collision.pf_px[i,j]
        for i,j in self.material.pf_pv:     self.A[i,j] += -h*self.material.pf_pv[i,j]
        for i in self.constraints.pf_pv:    self.A[i,i] += -h*self.constraints.pf_pv[i]

        self.curr_variadic_entry_num[None] = 0
        for i,j in self.A:
            if ti.is_active(self.fixed_ij_node,(i,j)):
                triplet_i = self.ij2triplet_i[i, j]
                for i_, j_ in ti.static(ti.ndrange(3, 3)): self.hess_value[9 * triplet_i + 3 * i_ + j_] = self.A[i,j][i_,j_]
            else:
                triplet_i = ti.atomic_add(self.curr_variadic_entry_num[None],1)
                if triplet_i >= self.mven:print('error: too many variadic entries')
                for i_, j_ in ti.static(ti.ndrange(3, 3)):
                    self.hess_i_variadic[9 * triplet_i + 3 * i_ + j_] = 3 * i + i_
                    self.hess_j_variadic[9 * triplet_i + 3 * i_ + j_] = 3 * j + j_
                    self.hess_value_variadic[9 * triplet_i + 3 * i_ + j_] = self.A[i,j][i_,j_]

    @ti.kernel
    def UpdateY(self, dv: ti.types.ndarray(dtype=vec3,ndim=1)):
        for I in ti.grouped(dv): self.v[I] += dv[I]
        for i in self.y: self.y[i] = self.x[i] + self.h * self.v[i]

    @ti.kernel
    def UpdateX(self):
        for i in self.v:self.v[i] = (self.y[i]-self.x[i])/self.h
        for i in self.x :self.x[i] = self.y[i]

    def Run(self):
        self.material.Update(self.x,self.v)
        self.constraints.Update(self.x,self.v)
        self.collision.CollectCollisionPairs(self.x,self.x,False)
        self.collision.Update(self.x)
        # print(self.curr_variadic_entry_num[None])
        self.AssembleMatrix()
        rhs = self.h * self.b.to_numpy().reshape(-1)
        hessI = np.r_[self.hess_i,self.hess_i_variadic.to_numpy()[:self.curr_variadic_entry_num[None]]]
        hessJ = np.r_[self.hess_j,self.hess_j_variadic.to_numpy()[:self.curr_variadic_entry_num[None]]]
        hessV = np.r_[self.hess_value.to_numpy(),self.hess_value_variadic.to_numpy()[:self.curr_variadic_entry_num[None]]]
        lhs = sp.coo_matrix((hessV, (hessI,hessJ)), shape=(3 * self.vn, 3 * self.vn)).tocsr()
        self.UpdateY(sp.linalg.spsolve(lhs, rhs).reshape(self.vn, 3))
        # self.collision.CollectCollisionPairs(self.x,self.y,True)
        self.UpdateX()
        self.external_force.fill(0)
        return self

def Main(testcase):
    simulator = Simulator(testcase, [1,125,190,62 ]) # ,5,7,9
    ps.init()
    ps.set_warn_for_invalid_values(True)
    ps.set_ground_plane_mode('none')
    ps.set_background_color((0.5, 0.5, 0.5))
    ps.set_shadow_darkness(0.75)
    ps.look_at((1.5, 1., 1.5), (0., 0, 0.0))
    ps_mesh = ps.register_surface_mesh('cloth', simulator.x.to_numpy(), simulator.faces.to_numpy(),color=(255/255, 140/255, 238/255))
    ps_mesh.set_back_face_color((242/255, 220/255, 107/255))
    ps_mesh.set_back_face_policy('custom')
    ps_mesh.set_edge_width(1.0)
    io,frameid,stepmode = psim.GetIO(),0,False
    while not io.KeyCtrl:
        print('FRAME ',frameid,'-----------------------------------------------')
        if not stepmode or (stepmode and  io.MouseDoubleClicked[0]):ps_mesh.update_vertex_positions(simulator.Run().x.to_numpy())
        simulator.cloth.vertices = simulator.x.to_numpy()
        simulator.cloth.export('assets/seq/' + str(frameid)+'.obj')
        # if simulator.curr_variadic_entry_num[None] >= 1:
        #     simulator.collision.Print()
        #     print('max v ',simulator.v.to_numpy().max())
        # if frameid==109:    exit()
        ps.frame_tick()
        frameid += 1

Main('./assets/quad01_2.obj')