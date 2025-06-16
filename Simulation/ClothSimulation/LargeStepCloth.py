import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import scipy.sparse as sp
import time
import taichi as ti

ti.init(arch=ti.cpu)
EPS = 1e-8

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

        @ti.func
        #  x index is conform with the ref paper: Derivation of discrete bending forces and their gradients
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

@ti.data_oriented
class Simulator:
    def __init__(self, clothobjpath, pins=[]):
        self.h = 1 / 150
        cloth = trimesh.load(clothobjpath)
        self.vn, self.en, self.fn = len(cloth.vertices), len(cloth.edges_unique), len(cloth.faces)
        self.uv = ti.Vector.field(2, ti.f64, self.vn)
        self.uv.from_numpy(cloth.visual.uv)
        self.x0 = ti.Vector.field(3, ti.f64, self.vn)
        self.x0.from_numpy(cloth.vertices)
        self.x = ti.Vector.field(3, ti.f64, self.vn)
        self.x.from_numpy(cloth.vertices)
        self.v = ti.Vector.field(3, ti.f64, self.vn)
        self.v.from_numpy(np.zeros_like(cloth.vertices))
        self.m = ti.field(ti.f64, self.vn)
        self.faces = ti.Vector.field(3, ti.i32, self.fn)
        self.faces.from_numpy(cloth.faces.astype(np.int32))
        self.external_force = ti.Vector.field(3,ti.f64,self.vn)

        self.stretch = Constraint.EdgeStrecth(cloth,1e2,0)
        self.bend = Constraint.Bend(cloth,0.01,0)
        self.pin = Constraint.Pin(cloth,pins,1e4,0)
        # 一个关键的观察，一旦约束定下来了，稀疏hessian的ij项也就定下来了。
        self.ij = sorted(list(  set().union(*[con.ij for con in [self.stretch,self.bend, self.pin]])     ))  # 一个关键的观察，一旦约束定下来了，稀疏hessian的ij项也就定下来了。
        hessian_entry_num = len(self.ij)
        self.b = ti.Vector.field(3, ti.f64, self.vn)
        self.hess_entry_ij    = ti.Vector.field(2, ti.i32, hessian_entry_num)
        self.hess_entry_value = ti.Matrix.field(3, 3, ti.f64, hessian_entry_num)
        self.ij2entry_idx = ti.field(ti.i32)
        ti.root.dense(ti.i, self.vn).bitmasked(ti.j, self.vn).place(self.ij2entry_idx)
        self.hess_i_flatten = np.zeros(9*hessian_entry_num).astype(np.int32)
        self.hess_j_flatten = np.zeros(9*hessian_entry_num).astype(np.int32)
        self.hess_value_flatten = ti.field(ti.f64, 9*hessian_entry_num)
        for entry_idx, (i, j) in enumerate(self.ij):
            self.ij2entry_idx[i, j] = entry_idx
            self.hess_entry_ij[entry_idx] = ti.math.ivec2(i, j)
            for i_, j_ in ((i, j) for i in range(3) for j in range(3)):
                self.hess_i_flatten[9 * entry_idx + 3 * i_ + j_] = 3 * i + i_
                self.hess_j_flatten[9 * entry_idx + 3 * i_ + j_] = 3 * j + j_

        vm = np.zeros(self.vn)  # 点的质量是incident的三角形面积和
        for fi, f in enumerate(cloth.faces):
            for vi in f: vm[vi] += cloth.area_faces[fi]#/len(cloth.vertex_faces[vi])
        self.m.from_numpy(vm)

    @ti.kernel
    def BuildLinearSystem(self):
        h = ti.cast(self.h,ti.f64)
        self.b.fill(0)
        self.hess_entry_value.fill(0)
        # external force --------------------------------------------------------------------------
        for i in range(self.vn):self.b[i] = self.m[i] * ti.math.vec3(0, -9.81, 0) + self.external_force[i]
        # ------------------------------- PIN POINT -------------------------------------------
        for ci in self.pin.k:
            i, l0, k = self.pin.idx[ci], self.pin.l0[ci], self.pin.k[ci]
            C,pC_pxi,ppC_pxipxi = Constraint.Pin.C_DC_DDC(self.x[i],l0)
            if -EPS < C < EPS: continue
            Kii = -k * (pC_pxi.outer_product(pC_pxi) + ppC_pxipxi * C)
            self.b[i] += -k * pC_pxi * C
            self.hess_entry_value[self.ij2entry_idx[i, i]] += Kii
        #  ----------------------------- EDGE STRETCH ---------------------------------------------
        for ci in self.stretch.k:
            k = self.stretch.k[ci]
            i, j = self.stretch.idx[ci]
            v = (self.v[i], self.v[j])
            C,C_jacobi,C_hess = Constraint.EdgeStrecth.C_DC_DDC(self.x[i], self.x[j],self.stretch.l0[ci])
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(2)):
                i,pC_pxi = self.stretch.idx[ci][i_],C_jacobi[i_]
                ti.atomic_add(self.b[i], -k * pC_pxi * C)
                for j_ in ti.static(range(2)):
                    j,pC_pxj = self.stretch.idx[ci][j_],C_jacobi[j_]
                    ti.atomic_add(self.hess_entry_value[self.ij2entry_idx[i, j]], -k * (pC_pxi.outer_product(pC_pxj) + C_hess[2*i_ + j_] * C))
        for ci in self.bend.k:
            k = self.bend.k[ci]
            i0,i1,i2,i3 = self.bend.idx[ci]
            v = (self.v[i0], self.v[i1],self.v[i2], self.v[i3])
            C,C_jacobi,C_hess = Constraint.Bend.C_DC_DDC(self.x[i0], self.x[i1],self.x[i2],self.x[i3],self.bend.l0[ci])
            if -EPS < C < EPS: continue
            for i_ in ti.static(range(4)):
                i,pC_pxi = self.bend.idx[ci][i_],C_jacobi[i_]
                ti.atomic_add(self.b[i], -k * pC_pxi * C)
                for j_ in ti.static(range(4)):
                    j,pC_pxj = self.bend.idx[ci][j_],C_jacobi[j_]
                    ti.atomic_add(self.hess_entry_value[self.ij2entry_idx[i, j]], -k * (pC_pxi.outer_product(pC_pxj) + C_hess[ 4 * i_ + j_] * C))
        #  -------------------------- add b with h*pf_px*v --------------------------
        for i,j in self.ij2entry_idx: ti.atomic_add(self.b[i], h * self.hess_entry_value[self.ij2entry_idx[i, j]] @ self.v[j])
        #  -----------------------------  Assemble Flatten Hessian  ---------------------------------------------
        for i, j in self.ij2entry_idx:
            entry_idx = self.ij2entry_idx[i, j]
            for i_, j_ in ti.static(ti.ndrange(3, 3)):
                self.hess_value_flatten[9*entry_idx + 3*i_ + j_] = -h*self.hess_entry_value[entry_idx][i_,j_] + (self.m[i]/h if i==j and i_==j_ else 0)

    @ti.kernel
    def UpdateXAndV(self, dv: ti.types.ndarray()):
        for i in range(self.vn):
            self.v[i] += ti.math.vec3(dv[i, 0], dv[i, 1], dv[i, 2])
            self.x[i] += self.h * self.v[i]
        self.external_force.fill(0)

    def Run(self):
        self.BuildLinearSystem()
        lhs = sp.coo_matrix((self.hess_value_flatten.to_numpy(), (self.hess_i_flatten, self.hess_j_flatten)), shape=(3 * self.vn, 3 * self.vn)).tocsr()
        rhs = self.b.to_numpy().reshape(-1)   # 对于eq(6) 相当于左右同时除以h
        self.UpdateXAndV(sp.linalg.spsolve(lhs, rhs).reshape(self.vn, 3))
        return self

def Main(testcase):
    simulator = Simulator(testcase, [1, ])
    ps.set_warn_for_invalid_values(True)
    ps.set_ground_plane_mode('none')
    ps.set_background_color((0.5, 0.5, 0.5))
    ps.set_shadow_darkness(0.75)
    ps.init()
    ps.look_at((1.5, 0.2, 1.5), (0., -0.5, 0.5))
    ps_mesh = ps.register_surface_mesh("cloth", simulator.x.to_numpy(), simulator.faces.to_numpy(),color=(0.13333333, 0.44705882, 0.76470588))
    ps_mesh.set_edge_width(1.0)
    i = 0
    io = psim.GetIO()
    while not io.KeyCtrl:
        # if io.MouseClicked[2]:
        #     picker =  ps.pick(screen_coords=io.MousePos)
        #     if picker.is_hit:
        #         print(picker.structure_data['index'],picker.local_index )
        #         for k,v in picker.structure_data.items():print(k,v)
        #         simulator.external_force[picker.structure_data['index']] = ti.math.vec3(0,1,0)
        ps_mesh.update_vertex_positions(simulator.Run().x.to_numpy())
        ps.frame_tick()
        i += 1

Main('./assets/quad01.obj')