import numpy as np
import polyscope as ps
import taichi as ti
import polyscope.imgui as psim
import scipy.sparse as sp
import trimesh,time

ti.init(arch=ti.cpu,default_fp=ti.f64)
EPS,MAX = 1e-8,1.7976931348623157e+308
vec3,vec2,mat2,vec3i = ti.math.vec3,ti.math.vec2,ti.math.mat2,ti.math.ivec3

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
            ppC_pxipxi = ti.math.eye(3) / normxji - xji.outer_product(xji) / normxji ** 3
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
            ppC_pxipxi = ti.math.eye(3) / C - xji.outer_product(xji) / C ** 3
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
            self.idx.from_numpy(np.array(indices).astype(np.int32))
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

# There are three types of Constraints.
# A. 【Material】Material-related, e.g. bend,edge stretch, which are decided by the cloth geometry
# B. 【CustomConstraint】User-defined constraint, e.g. pin point, stitch points
# C. 【Collision】Constraints that generated by Collision
# A,B are always fixed, while C will change dynamically during every iteration

@ti.data_oriented
class Collision:
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
    def IJK2TableI(self,ijk):return (73856093*ijk[0]^19349663*ijk[1]^83492791*ijk[2])%self.tablesize

    @ti.kernel
    def ResetAABBs(self):
        for i in self.vAABBs:self.vAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))
        for i in self.eAABBs:self.eAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))
        for i in self.fAABBs:self.fAABBs[i] = self.AABB(bmin = vec3(MAX),bmax = -vec3(MAX))

    def __init__(self,cloth,tablesize=2039,element_num_per_cell=256,thickness=0.01):
        self.tablesize,self.h,self.enpc = tablesize,thickness,element_num_per_cell
        self.vn,self.en,self.fn = len(cloth.vertices),len(cloth.edges_unique),len(cloth.faces)
        self.edges = ti.Vector.field(2,ti.i32,self.en)
        self.edges.from_numpy(cloth.edges_unique.astype(np.int32))
        self.faces = ti.Vector.field(3,ti.i32,self.fn)
        self.faces.from_numpy(cloth.faces.astype(np.int32))
        self.vAABBs = self.AABB.field(shape = self.vn)
        self.eAABBs = self.AABB.field(shape = self.en)
        self.fAABBs = self.AABB.field(shape = self.fn)

        self.cellf = ti.field(ti.i32)   #  face(triangle index) in spatial cell
        self.cellfnode = ti.root.pointer(ti.i,tablesize).dense(ti.j,element_num_per_cell)
        self.cellfnode.place(self.cellf)
        self.cellfn = ti.field(ti.i32,tablesize)  # how many current faces at each table entry ?
        self.celle = ti.field(ti.i32)  #  edge(edge index) in spatial cell
        self.cellenode = ti.root.pointer(ti.i,tablesize).dense(ti.j,element_num_per_cell)
        self.cellenode.place(self.celle)
        self.cellen = ti.field(ti.i32,tablesize)

        self.vf = ti.Vector.field(3,ti.f64)           # vf pairs' barycentric coordinate where actual collisions occur
        self.vfnode = ti.root.pointer(ti.i,self.vn).bitmasked(ti.j,self.fn) # use this sparse structure to remove repeated pair from broad phase
        self.vfnode.place(self.vf)
        self.ee = ti.Vector.field(2,ti.f64)
        self.eenode = ti.root.pointer(ti.i,self.en).bitmasked(ti.j,self.en)
        self.eenode.place(self.ee)

    def HandleDiscrete(self,x,v):
        self.ResetAABBs()
        self.cellfn.fill(0)
        self.cellen.fill(0)
        self.cellfnode.deactivate_all()
        self.cellenode.deactivate_all()
        self.vfnode.deactivate_all()
        self.eenode.deactivate_all()
        self.HandleDiscrete_(x,v)

    @ti.kernel
    def HandleDiscrete_(self,x:ti.template(),v:ti.template()):
        eCellSize,fCellSize = -vec3(MAX),-vec3(MAX)
        for vi in x:self.vAABBs[vi].EatPoint(x[vi]) # result in : bmin == bmax = self.x[vi]
        for ei in self.edges:
            vi,vj = self.edges[ei]
            self.eAABBs[ei].EatPoint(x[vi]).EatPoint(x[vj])
            boxsize = self.eAABBs[ei].bmax-self.eAABBs[ei].bmin
            for i in ti.static(range(3)): ti.atomic_max(eCellSize[i],boxsize[i])
        for fi in self.faces:
            vi,vj,vk = self.faces[fi]
            self.fAABBs[fi].EatPoint(x[vi]).EatPoint(x[vj]).EatPoint(x[vk])
            boxsize = self.fAABBs[fi].bmax-self.fAABBs[fi].bmin
            for i in ti.static(range(3)): ti.atomic_max(fCellSize[i],boxsize[i])
        for fi in self.fAABBs:
            st,ed = ti.floor(self.fAABBs[fi].bmin/fCellSize).cast(ti.i32),ti.floor(self.fAABBs[fi].bmax/fCellSize).cast(ti.i32)
            for i0,i1,i2 in ti.ndrange(st[0]-ed[0]+1,st[1]-ed[1]+1,st[2]-ed[2]+1):
                i = self.IJK2TableI(st+vec3i(i0,i1,i2))
                self.cellf[i,ti.atomic_add(self.cellfn[i],1)] = fi
                if self.cellfn[i]>=self.enpc:print('cell too many faces ',self.cellfn[i])
        for vi in x:
            i,x4 = self.IJK2TableI(ti.floor(x[vi]/fCellSize).cast(ti.i32)),x[vi]
            for fii in range(self.cellfn[i]):
                fi = self.cellf[i,fii]
                x1,x2,x3 = x[self.faces[fi][0]],x[self.faces[fi][1]],x[self.faces[fi][2]]
                x13,x23,x43 = x1-x3,x2-x3,x4-x3
                n = x13.cross(x23).normalized()
                if x43.dot(n)>self.h: continue
                w = mat2([(x13.dot(x13),x13.dot(x23)),(x13.dot(x23),x23.dot(x23))]).inverse()@vec2(x13.dot(x43),x23.dot(x43))  # page(4) eq(1)
                d = self.h/ti.math.max(x13.norm(),x23.norm(),(x1-x2).norm())
                if -d<w[0]<1+d and -d<w[1]<1+d and -d<w.sum()<1+d : self.vf[vi,fi] = vec3(w[0],w[1],1-w.sum())
        for ei in self.eAABBs:
            st,ed = ti.floor(self.eAABBs[ei].bmin/eCellSize).cast(ti.i32),ti.floor(self.eAABBs[ei].bmax/eCellSize).cast(ti.i32)
            for i0,i1,i2 in ti.ndrange(st[0]-ed[0]+1,st[1]-ed[1]+1,st[2]-ed[2]+1):
                i = self.IJK2TableI(st+vec3i(i0,i1,i2))
                self.celle[i,ti.atomic_add(self.cellen[i],1)] = ei
                if self.cellen[i] >= self.enpc: print('cell too many edges ', self.cellen[i])
        for i in self.cellen:
            for ei0_ in range(self.cellen[i]):
                ei0 = self.celle[i,ei0_]
                x1,x2 = x[self.edges[ei0][0]],x[self.edges[ei0][1]]
                for ei1_ in range(ei0_):
                    ei1 = self.celle[i, ei1_]
                    x3, x4 = x[self.edges[ei1][0]], x[self.edges[ei1][1]]
                    x21,x31,x43 = x2-x1,x3-x1,x4-x3
                    if x21.cross(x43).norm()<EPS  : # ei0,ei1 parallel
                        if x31.cross(x21).norm()/x21.norm()<self.h: self.ee[ei0,ei1] = vec2(0.5)  # 平行四边形的面积等于底乘高
                        continue
                    w = mat2([(x21.dot(x21),-x21.dot(x43)),(-x21.dot(x43),x43.dot(x43))]).inverse()@vec2(x21.dot(x31),-x43.dot(x31))  # page(4) eq(2)
                    if 0<=w[0]<=1 and 0<=w[1]<=1: self.ee[ei0,ei1] = w
        # Now Apply Repulsion
        for vi,fi in self.vf:
            x1,x2,x3,x4 = x[self.faces[fi][0]],x[self.faces[fi][1]],x[self.faces[fi][2]],x[vi]
            w = self.vf[vi,fi]
            xp,vp = vec3(0),vec3(0)
            for i in ti.static(range(3)):
                xp+= x[self.faces[fi][i]]*w[i]
                vp+= v[self.faces[fi][i]]*w[i]
            n = (x4-xp).normalized()

@ti.data_oriented
class Material:
    def __init__(self,cloth):
        self.vn = len(cloth.vertices)
        self.stretch = Constraint.EdgeStrecth(cloth, 1e4, 0.0000)
        self.bend = Constraint.Bend(cloth, 0.001, 0.00000)
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
        self.vn, self.en, self.fn = len(cloth.vertices), len(cloth.edges_unique), len(cloth.faces)
        self.x = ti.Vector.field(3, ti.f64, self.vn)
        self.x.from_numpy(cloth.vertices)
        self.v = ti.Vector.field(3, ti.f64, self.vn)
        self.v.from_numpy(np.zeros_like(cloth.vertices))
        self.m = ti.field(ti.f64, self.vn)
        self.m.from_numpy(2*np.ones(self.vn)*cloth.area/self.vn)
        self.faces = ti.Vector.field(3, ti.i32, self.fn)
        self.faces.from_numpy(cloth.faces.astype(np.int32))
        self.external_force = ti.Vector.field(3,ti.f64,self.vn)

        self.material = Material(cloth)
        self.constraints = CustomConstraint(cloth,pins)
        # self.collision = Collision(cloth)
        self.b = ti.Vector.field(3, ti.f64, self.vn)
        self.A = ti.Matrix.field(3, 3, ti.f64)
        ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn).place(self.A)

        hessian_entry_num = len(self.material.ij)
        self.triplet_i2ij = ti.Vector.field(2, ti.i32, hessian_entry_num)
        self.ij2triplet_i = ti.field(ti.i32)
        ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn).place(self.ij2triplet_i)
        self.hess_i = np.zeros(9 * hessian_entry_num).astype(np.int32)
        self.hess_j = np.zeros(9 * hessian_entry_num).astype(np.int32)
        self.hess_value = ti.field(ti.f64, 9 * hessian_entry_num)
        for entry_idx, (i, j) in enumerate(self.material.ij):
            self.ij2triplet_i[i, j] = entry_idx
            self.triplet_i2ij[entry_idx] = ti.math.ivec2(i, j)
            for i_, j_ in ((i, j) for i in range(3) for j in range(3)):
                self.hess_i[9 * entry_idx + 3 * i_ + j_] = 3 * i + i_
                self.hess_j[9 * entry_idx + 3 * i_ + j_] = 3 * j + j_

    @ti.kernel
    def AssembleMatrix(self):
        h = ti.cast(self.h,ti.f64)
        self.A.fill(0)
        for i in self.b:self.b[i] = self.m[i] * vec3(0, -9.81, 0) + self.external_force[i] + self.constraints.f[i] + self.material.f[i] #
        for i,j in self.material.pf_px:     self.b[i] += h * self.material.pf_px[i, j] @ self.v[j]
        for i in self.constraints.pf_px:    self.b[i] += h * self.constraints.pf_px[i] @ self.v[i]

        for i in self.m:                    self.A[i,i] = ti.math.eye(3)*self.m[i]
        for i,j in self.material.pf_px:     self.A[i,j] += -h*h*self.material.pf_px[i,j]
        for i in self.constraints.pf_px:    self.A[i,i] += -h*h*self.constraints.pf_px[i]
        for i,j in self.material.pf_pv:     self.A[i,j] += -h*self.material.pf_pv[i,j]
        for i in self.constraints.pf_pv:    self.A[i,i] += -h*self.constraints.pf_pv[i]

        for i,j in self.A:
            triplet_i = self.ij2triplet_i[i, j]
            for i_, j_ in ti.static(ti.ndrange(3, 3)):
                self.hess_value[9 * triplet_i + 3 * i_ + j_] = self.A[i,j][i_,j_]

    @ti.kernel
    def Update(self, dv: ti.types.ndarray(dtype=vec3,ndim=1)):
        for I in ti.grouped(dv): self.v[I] += dv[I]
        for i in self.x: self.x[i] += self.h * self.v[i]
    def Run(self):
        self.material.Update(self.x,self.v)
        self.constraints.Update(self.x,self.v)
        self.AssembleMatrix()
        rhs = self.h * self.b.to_numpy().reshape(-1)
        lhs = sp.coo_matrix((self.hess_value.to_numpy(), (self.hess_i, self.hess_j)), shape=(3 * self.vn, 3 * self.vn)).tocsr()
        self.Update(sp.linalg.spsolve(lhs, rhs).reshape(self.vn, 3))
        self.external_force.fill(0)
        return self

def Main(testcase):
    simulator = Simulator(testcase, [1, ])
    ps.init()
    ps.set_warn_for_invalid_values(True)
    ps.set_ground_plane_mode('none')
    ps.set_background_color((0.5, 0.5, 0.5))
    ps.set_shadow_darkness(0.75)
    ps.look_at((1.5, 0.2, 1.5), (0., -0.5, 0.5))
    ps_mesh = ps.register_surface_mesh("cloth", simulator.x.to_numpy(), simulator.faces.to_numpy(),color=(0.13333333, 0.44705882, 0.76470588))
    ps_mesh.set_back_face_color((84/255, 120/255, 161/255))
    ps_mesh.set_back_face_policy('custom')
    # for k in dir(type(ps_mesh)):print(k)
    # exit(0)
    ps_mesh.set_edge_width(1.0)
    io = psim.GetIO()
    while not io.KeyCtrl:
        # if io.MouseClicked[2]:
        #     picker =  ps.pick(screen_coords=io.MousePos)
        #     if picker.is_hit:
        #         print(picker.structure_data['index'],picker.local_index )
        #         for k,v in picker.structure_data.items():print(k,v)
        #         simulator.external_force[picker.structure_data['index']] = vec3(0,1,0)
        ps_mesh.update_vertex_positions(simulator.Run().x.to_numpy())
        ps.frame_tick()

Main('./assets/quad01.obj')