# ref:
# 1. https://www.cs.cmu.edu/~kmcrane/Projects/DDG/
# 2. Discrete Differential Forms for Computational Modeling

import math
import scipy.sparse.linalg
import trimesh, os
import numpy as np
import scipy.sparse as sp
import scipy.linalg as scila
from collections import defaultdict

class MatrixBuilder:
    def __init__(self, row_cnt,col_cnt):
        self.rn,self.cn = row_cnt,col_cnt
        self.triplets = defaultdict(np.float64)
    def AddTriplet(self,ri,ci,value):
        self.triplets[(ri,ci)] += value
        return self
    def SetTriplet(self,ri,ci,value):
        self.triplets[(ri,ci)] = value
        return self
    @property
    def scipy_coo_matrix(self):
        row,col,val = [],[],[]
        for (ri,ci),v in self.triplets.items():
            row.append(ri)
            col.append(ci)
            val.append(v)
        return sp.coo_matrix((val, (row, col)), shape=(self.rn, self.cn))

class Halfedge:
    def __init__(self):
        self.twin:  Halfedge = None
        self.next:  Halfedge = None
        self.prev:  Halfedge = None
        self.v0:    Vertex = None
        self.v1:    Vertex = None
        self.fi:    int    = None
        self.boundary = False

    def __str__(self):return f'({self.v0.i},{self.v1.i})'

    @property
    def debug_str(self):
        return (str(self)+' prev'+str(self.prev)+' twin'+ str(self.twin)+' next'+str(self.next)+
                ' boundary '+str(self.boundary)+ ' fi '+str(self.fi))

    @property
    def vector(self):return self.v1.p - self.v0.p

    @property
    def length(self):return np.linalg.norm(self.vector)

    @property
    def length2(self):return np.dot(self.vector, self.vector)

    @property
    def normalized_vector(self):return self.vector / self.length

    @property
    def normal(self):
        n = np.cross(-self.vector, self.next.vector)
        return n/np.linalg.norm(n)

    @property
    def cotan(self):
        if  self.boundary: return 0.0
        a, b = -self.next.vector, self.prev.vector
        return np.dot(a, b) / np.linalg.norm(np.cross(a, b))

class Vertex:
    def __init__(self,i, p):
        self.i = i # index number
        self.p = p # position
        self.m_area = None
        self.halfedges = []

    def __str__(self):return str(self.i)

    @property
    def str_halfedges(self):
        ret = ''
        for e in self.halfedges:ret+=str(e)
        return ret

    @property
    def he(self):return self.halfedges[0]

    # CircumcentricDualArea : http://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleAreasCheatSheet.pdf
    @property
    def area(self):
        if self.m_area is not None: return self.m_area
        self.m_area = sum([e.length2*e.cotan+e.prev.length2*e.prev.cotan for e in self.halfedges])/8
        return self.m_area

class Mesh(trimesh.Trimesh):
    def __init__(self, obj_path):
        mesh = trimesh.load(obj_path,maintain_order=True, process=False)
        super().__init__(mesh.vertices, mesh.faces)

        self.Vertexs = [Vertex(i, v) for i, v in enumerate(self.vertices)]
        self.vn = len(self.vertices)
        self.fn = len(self.faces)
        self.en = len(self.edges_unique)
        self.chi = self.vn-self.en+self.fn
        self.bn = max(0,2-self.chi)
        self.IJ2UniqueEdgeIdx = {}
        for ei,(i,j) in enumerate(self.edges_unique):self.IJ2UniqueEdgeIdx[(i,j)] = ei

        self.m_d0,self.m_d1,self.m_x0,self.m_x1,self.m_x2 = None,None,None,None,None
        self.m__x0,self.m__x1,self.m__x2 = None,None,None
        self.m_Lc = None
        self.m_tree,self.m_cotree = None,None
        self.m_generators = None
        self.m_harmonic_bases = None

        IJ2EdgeIdx = {(vi, vj): i for i, (vi, vj) in enumerate(self.edges)}
        boundaryEdgeIdx = set()
        for (vi, vj) in self.edges:
            if (vj, vi) in IJ2EdgeIdx: continue
            boundaryEdgeIdx.add((vj, vi))
            IJ2EdgeIdx[(vj, vi)] = len(IJ2EdgeIdx)

        self.Halfedges = [Halfedge() for _ in range(len(IJ2EdgeIdx))]
        for (vi, vj),ei in IJ2EdgeIdx.items():
            he = self.Halfedges[ei]
            he.v0, he.v1 = self.Vertexs[vi],self.Vertexs[vj]
            he.v0.halfedges.append(he)
            # print(ei)
            he.twin = self.Halfedges[IJ2EdgeIdx[(vj,vi)]]
            if (vi, vj) in boundaryEdgeIdx:he.boundary = True
        for fi,vidxs in enumerate(self.faces):
            for i in range(3):
                vi,vj,vk = vidxs[i],vidxs[(i+1)%3],vidxs[(i+2)%3]
                self.Halfedges[IJ2EdgeIdx[(vi,vj)]].next = self.Halfedges[IJ2EdgeIdx[(vj,vk)]]
                self.Halfedges[IJ2EdgeIdx[(vi,vj)]].prev = self.Halfedges[IJ2EdgeIdx[(vk,vi)]]
                self.Halfedges[IJ2EdgeIdx[(vi, vj)]].fi  = fi
        self.IJ2HalfEdgeIdx = IJ2EdgeIdx
        self.IJ2HalfEdge = {(vi,vj):self.Halfedges[heIdx] for (vi,vj),heIdx in self.IJ2HalfEdgeIdx.items()}
        # boundary halfedge 也有prev next
        for (vi,vj) in boundaryEdgeIdx:
            he = self.Halfedges[IJ2EdgeIdx[(vi,vj)]]
            for (vj0,vk) in boundaryEdgeIdx:
                if vj!=vj0:continue
                he.next = self.Halfedges[IJ2EdgeIdx[(vj,vk)]]
        for he in self.Halfedges:
            if he.next.prev is None: he.next.prev = he

        for vi, v in enumerate(self.Vertexs):
            h = v.halfedges[0]
            halfedges = []
            while True:
                h = h.prev.twin
                halfedges.append(h)
                if h is v.he: break
            v.halfedges = halfedges
            isBoundary = [e.boundary for e in v.halfedges]
            if isBoundary[0] or isBoundary[-1] or not any(isBoundary): continue
            # 处理边缘点
            for e in v.halfedges:
                if not e.boundary: continue
                idx = v.halfedges.index(e)
                v.halfedges = v.halfedges[idx:] + v.halfedges[:idx]
                if v.halfedges[0].boundary or v.halfedges[-1].boundary: break
            assert v.halfedges[0].boundary or v.halfedges[-1].boundary

        # sanity check
        for he in self.Halfedges:
            if not he.boundary: continue
            assert he.fi is None
        for i,j in self.edges:
            if (i,j) not in self.IJ2HalfEdgeIdx:
                assert (j,i) in self.IJ2HalfEdgeIdx

    def Print(self):
        print('halfedges-------------------------')
        for he in self.Halfedges: print(he.debug_str)
        print('vertex edges-------------------------')
        for i, v in enumerate(self.Vertexs):
            print(i)
            for e in v.halfedges: print(e,end='')
            print()

    # ---------------------------------DISCRETE DIFFERENTIAL OPERATOR START-------------------------------------------
    # discrete operator d apply to PRIMAL 0-form, a |E|×|V| matrix
    @property
    def d0(self):
        if self.m_d0 is not None:return self.m_d0
        matrix_builder = MatrixBuilder(self.en,self.vn)
        for ei,(vi,vj) in enumerate(self.edges_unique):
            matrix_builder.SetTriplet(ei,vj,1).SetTriplet(ei, vi, -1)
        self.m_d0 =  matrix_builder.scipy_coo_matrix
        return self.m_d0

    # discrete operator d apply to PRIMAL 1-form, a |F|×|E| matrix
    @property
    def d1(self):
        if self.m_d1 is not None:return self.m_d1
        def Dir(vi,vj):
            if (vi,vj) in self.IJ2UniqueEdgeIdx:return 1,self.IJ2UniqueEdgeIdx[(vi,vj)]
            assert(vj,vi) in self.IJ2UniqueEdgeIdx
            return -1,self.IJ2UniqueEdgeIdx[(vj,vi)]
        row, col, data = [], [], []
        for fi, idxs in enumerate(self.faces):
            for i in range(3):
                vi,vj = idxs[i],idxs[(i+1)%3]
                direction,ei = Dir(vi,vj)
                row.append(fi)
                col.append(ei)
                data.append(direction)
        self.m_d1 = sp.coo_matrix((data, (row, col)), shape=(self.fn, self.en))
        return self.m_d1

    # discrete operator d apply to DUAL 1-form, a |V|×|E| matrix
    @property
    def _d1(self):
        return -self.d0.T  # ref[2] eq(6)
    # discrete operator d apply to DUAL 0-form, a |E|×|F| matrix
    @property
    def _d0(self):
        return self.d1.T

    # discrete Hodge star * apply to PRIMAL 0-form, a |V|×|V| diagonal matrix
    @property
    def x0(self):
        if self.m_x0 is not None:return self.m_x0
        row, col = list(range(self.vn)), list(range(self.vn))
        data = [v.area() for v in self.Vertexs]
        self.m_x0 = sp.coo_matrix((data,(row,col)),shape=(self.vn,self.vn))
        return self.m_x0

    # discrete Hodge star * apply to PRIMAL 1-form, a |E|×|E| diagonal matrix
    @property
    def x1(self):
        if self.m_x1 is not None:return self.m_x1
        row, col ,data = list(range(self.en)), list(range(self.en)),[]
        for vi,vj in self.edges_unique:
            he = self.IJ2HalfEdge[(vi,vj)]
            data.append( 0.5*(he.cotan+he.twin.cotan)) # page 110
        self.m_x1 = sp.coo_matrix((data, (row, col)), shape=(self.en, self.en))
        return self.m_x1

    # discrete Hodge star * apply to PRIMAL 2-form, a |F|×|F| diagonal matrix
    @property
    def x2(self):
        if self.m_x2 is not None:return self.m_x2
        row, col,data = list(range(self.fn)), list(range(self.fn)),[]
        for area in self.area_faces:data.append(1.0/area)
        self.m_x2 = sp.coo_matrix((data, (row, col)), shape=(self.fn, self.fn))
        return self.m_x2

    # discrete Hodge star * apply to DUAL 0-form, a |F|×|F| diagonal matrix
    @property
    def _x0(self):
        if self.m__x0 is not None:return self.m__x0
        row, col = list(range(self.fn)), list(range(self.fn))
        self.m__x0 = sp.coo_matrix((self.area_faces, (row, col)), shape=(self.fn, self.fn))
        return self.m__x0
    # discrete Hodge star * apply to DUAL 1-form, a |E|×|E| diagonal matrix
    @property
    def _x1(self):
        if self.m__x1 is not None:return self.m__x1
        row, col, data = list(range(self.en)), list(range(self.en)), []
        for vi, vj in self.edges_unique:
            he = self.IJ2HalfEdge[(vi, vj)]
            data.append(2 / (he.cotan + he.twin.cotan))  # ref[1] page 110
        self.m__x1 =  sp.coo_matrix((data, (row, col)), shape=(self.en, self.en))
        return self.m__x1

    # discrete Hodge star * apply to DUAL 2-form, a |V|×|V| diagonal matrix
    @property
    def _x2(self):
        if self.m__x2 is not None:return self.m__x2
        row, col = list(range(self.vn)), list(range(self.vn))
        data = [1 / v.area for v in self.Vertexs]
        self.m__x2 = sp.coo_matrix((data, (row, col)), shape=(self.vn, self.vn))
        return self.m__x2

    @property
    def Lc(self): # d*d
        if self.m_Lc is not None:return self.m_Lc
        matrix_builder = MatrixBuilder(self.vn,self.vn)
        for  v in self.Vertexs:
            w_sum = 1e-8
            for h in v.halfedges:
                w = (h.cotan + h.twin.cotan) / 2
                matrix_builder.SetTriplet(v.i,h.v1.i,w)
                w_sum += w
            matrix_builder.SetTriplet(v.i,v.i,-w_sum)
        self.m_Lc = matrix_builder.scipy_coo_matrix
        return self.m_Lc

    @property
    def L(self): # *d*d
        return sp.coo_matrix( self._x2 @ self.Lc )
    # ---------------------------------DISCRETE DIFFERENTIAL OPERATOR END-------------------------------------------

    @property
    def K(self): # Gaussian Curvature
        K = np.array([2 * np.pi for _ in self.Vertexs])
        for vi, v in enumerate(self.Vertexs):
            for i in range(len(v.halfedges)):
                e0, e1 = v.halfedges[i], v.halfedges[(i + 1) % len(v.halfedges)]
                if (e0.boundary or e0.twin.boundary) and (e1.boundary or e1.twin.boundary): continue
                K[vi] -= np.arccos(np.dot(e0.vector, e1.vector) / e0.length / e1.length)
        return K

    @property
    def H(self):  # Mean Curvature
        H = np.array([0.0 for _ in self.Vertexs])
        for vi, v in enumerate(self.Vertexs):
            for h in v.halfedges:
                if h.boundary or h.twin.boundary: continue
                H[vi] += np.arccos(np.dot(h.normal, h.twin.normal)) * np.sign(
                    np.dot(np.cross(h.normal, h.twin.normal), h.vector))
        H = H * 0.25
        return H

    # tree: 以vertex 0为根的由edge组成的tree, cotree:以face 0为根的由dual edge组成的tree
    @property
    def tree(self):
        if self.m_tree is not None:return self.m_tree
        tree = [i for i in range(self.vn)]
        root = [0]
        while root != []:
            parentId = root.pop(0)
            for he in self.Vertexs[parentId].halfedges:
                childId = he.v1.i
                if childId == 0 or tree[childId] != childId : continue  # already processed
                tree[childId] = parentId
                root.append(childId)
        self.m_tree = tree
        tree_lines = [] # for visualize
        for i, j in enumerate(tree): tree_lines.extend([self.vertices[i], self.vertices[j]])
        self.tree_lines = np.array(tree_lines)
        return self.m_tree

    @property
    def cotree(self):
        if self.m_cotree is not None:return self.m_cotree
        cotree = [i for i in range(self.fn)]
        root = [0]
        while root != []:
            parentId = root.pop(0)
            for _ in range(3):
                i0, i1 = self.faces[parentId][_], self.faces[parentId][(_ + 1) % 3]
                he = self.IJ2HalfEdge[(i0, i1)]
                if he.twin.boundary: continue
                childId = he.twin.fi
                if cotree[childId] != childId or childId==0 or self.tree[he.v0.i]==he.v1.i or self.tree[he.v1.i]==he.v0.i: continue  # already processed or cross tree
                cotree[childId] = parentId
                root.append(childId)
        self.m_cotree = cotree
        cotree_lines = []  # for visualize
        for i, j in enumerate(self.cotree): cotree_lines.extend([self.triangles_center[i], self.triangles_center[j]])
        self.cotree_lines = np.array(cotree_lines)
        return self.m_cotree

    # each generator is a closed path that starts with x(face idx) and ends with x
    @property
    def generators(self):
        if self.m_generators is not None:return self.m_generators
        def TraceToRoot(tree, i):
            ret = [i]
            while tree[ret[-1]] != ret[-1]: ret.append(tree[ret[-1]])
            return ret
        generators = []
        for edge in self.edges_unique:
            he = self.IJ2HalfEdge[(edge[0], edge[1])]
            if he.boundary or he.twin.boundary: continue
            fi, fj = he.fi, he.twin.fi
            if self.cotree[fi] == fj or self.cotree[fj] == fi or self.tree[he.v0.i] == he.v1.i or self.tree[
                he.v1.i] == he.v0.i: continue
            tracei, tracej = TraceToRoot(self.cotree, fi), TraceToRoot(self.cotree, fj)
            lastCommon = -1
            while tracei[-1] == tracej[-1]:
                lastCommon = tracei.pop()
                lastCommon = tracej.pop()
            generators.append([lastCommon] + tracei[::-1] + tracej + [lastCommon])
        self.m_generators = generators
        lines = []
        for generator in generators:
            for _i in range(len(generator) - 1):
                i, j = generator[_i], generator[_i + 1]
                lines.extend([self.triangles_center[i], self.triangles_center[j]])
        self.generators_lines = np.array(lines)  # for visualize
        return self.m_generators

    @property
    def harmonic_bases(self):
        if self.m_harmonic_bases is not None:return self.m_harmonic_bases
        def CommonEdge(fi,fj): #用一个来自fi的halfedge(有向边)来表示这个有向的generator
            for i in range(3):
                for j in range(3):
                    if min(self.faces[fi][i],self.faces[fi][(i+1)%3])==min(self.faces[fj][j],self.faces[fj][(j+1)%3]) and max(self.faces[fi][i],self.faces[fi][(i+1)%3])==max(self.faces[fj][j],self.faces[fj][(j+1)%3]) :
                        vi,vj = min(self.faces[fi][i],self.faces[fi][(i+1)%3]),max(self.faces[fi][i],self.faces[fi][(i+1)%3])
                        if self.IJ2HalfEdge[(vi,vj)].fi==fi:return (vi,vj)
                        else:                               return (vj,vi)
        bases = []
        for generator in self.generators:
            oneform = np.zeros(self.en)
            for i in range(len(generator)-1):
                fi,fj = generator[i],generator[i+1]
                vi0,vi1 = CommonEdge(fi,fj)
                if (vi0,vi1) not in self.IJ2UniqueEdgeIdx: oneform[self.IJ2UniqueEdgeIdx[(vi1,vi0)]]=-1
                else:oneform[self.IJ2UniqueEdgeIdx[(vi0,vi1)]]=1
            bases.append(oneform-self.d0@ HodgeDecomposition(self,oneform).alpha)
        self.m_harmonic_bases = bases
        self.harmonic_bases_on_face = [self.Whitney1Form(base) for base in bases] # for visualize
        return self.m_harmonic_bases

    def BT(self,T):return np.array([np.cross(T[i],self.vertex_normals[i]) for i in range(self.vn)])

    # get 1-form on circumcenter of triangle face
    def Whitney1Form(self,oneform):
        face1form = np.array([np.zeros(3) for _ in range(self.fn)])
        for fi,idxs in enumerate(self.faces):
            N = self.face_normals[fi]
            A = self.area_faces[fi]
            for _ in range(3):
                i,j,k = idxs[_],idxs[(_+1)%3],idxs[(_+2)%3]
                vi,vj,vk = self.vertices[i],self.vertices[j],self.vertices[k]
                phi_ij = np.cross(N,(vi-vk) + (vj-vk))/(6*A)
                if (i,j) in self.IJ2UniqueEdgeIdx: phi_ij *= oneform[self.IJ2UniqueEdgeIdx[(i,j)]]
                else: phi_ij *= -oneform[self.IJ2UniqueEdgeIdx[(j,i)]]
                face1form[fi] += phi_ij
        return face1form

    def Random1Form(self):
        scalar_potential = np.zeros(self.vn)
        vector_potential = np.zeros(self.vn)
        for _ in range(self.vn//500):
            scalar_potential[np.random.choice(self.vn)] = 5000*np.random.rand()-2500
            vector_potential[np.random.choice(self.vn)] = 5000 * np.random.rand() - 2500
        scalar_potential = sp.linalg.spsolve(self.L,scalar_potential)
        vector_potential = sp.linalg.spsolve(self.L,vector_potential)
        face_omega = np.array([np.zeros(3) for _ in range(self.fn)])
        for he in self.Halfedges:
            if he.boundary:continue
            A = self.area_faces[he.fi]
            N = self.face_normals[he.fi]
            counter_idx = he.next.v1.i
            face_omega[he.fi] += scalar_potential[counter_idx]*np.cross(N,he.vector)/(2*A)
            face_omega[he.fi] += vector_potential[counter_idx]*np.cross(N,np.cross(N,he.vector))/(2*A)
        omega = np.zeros(self.en)
        for i,(ei,ej) in enumerate(self.edges_unique):
            he = self.IJ2HalfEdge[(ei,ej)]
            if not he.boundary:         omega[i] += np.dot(he.vector,face_omega[he.fi])
            if not he.twin.boundary:    omega[i] += np.dot(he.vector,face_omega[he.twin.fi])
        return omega

    def ScalarPoissonProblem(self,idx=0):
        rho = np.array( [(0 if i!=idx else 1) for i in range(self.vn) ])
        diagM = np.array([v.area() for v in self.Vertexs])
        rhs =  rho - np.sum(diagM*rho)/self.area
        u = sp.linalg.spsolve(self.L, np.array(rhs))
        return u

    def VertexTangent(self):
        # project vertex's halfedge into vertex's tangent plane
        T = []
        for v in self.Vertexs:
            temp = np.cross(v.he.normalized_vector,self.vertex_normals[v.i])
            T.append(np.cross(self.vertex_normals[v.i],temp))
        return (np.array([1,0]*self.vn).reshape(self.vn,2) , np.array(T), self.BT(np.array(T)))

# ------------------------------ Some Operation Class  注意下面这些类的成员变量名称最好不要有重复 -----------------------------------------
class SpectralConformalParameterization(Mesh):
    def __init__(self,mesh):
        self.__dict__ = mesh.__dict__
        # 对mesh上的每一个点都求得了复平面上对应的一个点
        def I(i,j):
            return sp.coo_matrix(([1],([i],[j])),shape=(self.vn, self.vn))
        ED = -0.5*self.Lc

        # boundary的方向应该和三角形的vertex缠绕方向相同。但就这个问题来说，无所谓。因为不影响特征值
        A = sp.coo_matrix(([],([],[])),shape=(self.vn, self.vn))
        for he in self.Halfedges:
            if not he.boundary: continue
            he = he.twin
            vi,vj = he.v0.i,he.v1.i
            A = A - 0.25j*(I(vi,vj)-I(vj,vi))

        EC = ED - A
        # c is a constant map
        c = np.ones(self.vn)/np.sqrt(self.vn)

        # Inverse Power Method
        x,r = np.random.rand(self.vn)+1j*np.random.rand(self.vn),np.inf
        for _ in range(100):
            x = sp.linalg.spsolve(EC, x)
            x = x - (x.conj().T @ c).conj() * c
            x = x / np.linalg.norm(x)
            l = x.conj().T@EC@x
            r = np.linalg.norm(EC@x-l*x)
            if r <= 1e-10: break
        if r>1e-10:print('\033[33mWARNING: Spectral Conformal Parameterization may not converge\033[0m')
        self.uv = x # uv as one complex number
        u, v = np.real(x), np.imag(x)
        u ,v = u - min(u),v - min(v)
        self.uv_matrix = np.column_stack((u,v))*min(1/max(u),1/max(v))

    def VisualizeParameterization3D(self):
        u, v = np.real(self.uv), np.imag(self.uv)
        u1,v1 = np.array([u[vj]-u[vi] for vi,vj in self.edges_unique]),np.array([v[vj]-v[vi] for vi,vj in self.edges_unique])
        fu1,fv1 = self.Whitney1Form(u1),self.Whitney1Form(v1)
        return fu1,fv1
        def vertex1f(f):
            ret = np.zeros((self.vn,3))
            vtx_area = np.zeros(self.vn)
            for fi,value in enumerate(f):
                for vi in self.faces[fi]:
                    ret[vi]+=value*self.area_faces[fi]
            return ret
        vu1,vv1 = vertex1f(fu1),vertex1f(fv1)
        return vu1,vv1

    def VisualizeParameterization(self):
        from matplotlib import pyplot as plt
        u,v = np.real(self.uv),np.imag(self.uv)
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        for vi,vj in self.edges_unique:ax.plot([u[vi],u[vj]],[v[vi],v[vj]],linewidth=0.5)
        plt.show()

class HeatMethodGeodesic(Mesh):
    # [@param] pid: which point do you want to compute geodesics from
    def __init__(self,mesh,pid):
        self.__dict__ = mesh.__dict__
        u0 = np.zeros(self.vn)
        u0[pid] = 1
        t = np.mean(self.edges_unique_length)**2
        I = sp.identity(self.vn, format='csr')
        u = sp.linalg.spsolve( I - t*self.L, u0)

        grad_u = np.zeros_like(self.faces).astype(float)
        for i,v in  enumerate(self.Vertexs):
            for he in v.halfedges:
                if he.boundary:continue
                grad_u[he.fi] += u[i]/(2*self.area_faces[he.fi])*np.cross(self.face_normals[he.fi],he.next.vector)

        X = np.array([ -grad_u_/np.linalg.norm(grad_u_) for grad_u_ in grad_u ])
        div_X = np.zeros(self.vn)
        for v in self.Vertexs:
            for he in v.halfedges:
                if he.boundary:continue
                x  = X[he.fi]
                div_X[v.i]+=0.5*( np.dot(x,he.vector)*he.cotan + np.dot(x,-he.prev.vector)*he.prev.cotan)

        self.geodesics =  sp.linalg.spsolve(self.Lc,div_X )
        self.geodesics -= np.min(self.geodesics)

    def ISOLines(self,stride=0.5):
        import collections
        from math import ceil,floor
        h = np.mean(self.edges_unique_length)*stride
        ret = []
        for f in self.faces:
            valuemap = collections.defaultdict(list)
            vtxpos = self.vertices[f]
            vtxvalue = self.geodesics[f]
            for i in range(3):
                j = (i+1)%3
                if vtxvalue[i]>vtxvalue[j]:i,j = j,i
                xi,xj = vtxpos[i],vtxpos[j]
                vi,vj = vtxvalue[i],vtxvalue[j]
                for k in range(ceil(vi/h),floor(vj/h)+1):
                    pos_interpolate = (xi*(vj-k*h)+xj*(k*h-vi))/(vj-vi)
                    valuemap[k].append(pos_interpolate)
            for poss in valuemap.values():
                assert(len(poss)==2)
                ret.extend(poss[:])
        return np.array(ret)

class HodgeDecomposition(Mesh):
    def __init__(self,mesh,omega = None):
        self.__dict__ = mesh.__dict__
        if omega is None: omega = self.Random1Form()
        assert len(omega)==self.en
        temp = self._x2@self._d1@self.x1
        self.alpha = sp.linalg.spsolve(temp@self.d0,temp@omega)
        self.beta  = sp.linalg.spsolve(self.d1@self._x1@self._d0 ,self.d1@omega)
        self.beta  = sp.linalg.spsolve(self.x2,self.beta)
        self.face_omega = self.Whitney1Form(omega)
        self.face_alpha = self.Whitney1Form(self.d0@self.alpha)
        self.face_beta  = self.Whitney1Form(self._x1@self._d0@self.x2@self.beta)

class TrivialConnection(Mesh):
    def __init__(self,mesh):
        self.__dict__ = mesh.__dict__
        k = np.zeros(self.vn)
        for _ in range(self.euler_number): k[np.random.choice(self.vn)] += 1
        #注意,beta是dual 2-form
        beta = sp.linalg.spsolve(self.Lc@self._x2,2 * np.pi * k - self.vertex_defects)
        def IntegrateAlongGenerator(oneform,generator):
            ret = 0
            for he in generator:
                if (he.v0.i,he.v1.i) in self.IJ2UniqueEdgeIdx:ret+=oneform[self.IJ2UniqueEdgeIdx[(he.v0.i,he.v1.i)]]
                else: ret -= oneform[self.IJ2UniqueEdgeIdx[(he.v1.i,he.v0.i)]]
            return ret
        def Convert(generator): #把 (i,j)形式表示的generator转换成halfedge表示的
            def ConvertPair2HalfEdge(fi, fj):  # 用一个来自fi的halfedge(有向边)来表示这个有向的generator
                for i in range(3):
                    for j in range(3):
                        if min(self.faces[fi][i], self.faces[fi][(i + 1) % 3]) == min(self.faces[fj][j], self.faces[fj][
                            (j + 1) % 3]) and max(self.faces[fi][i], self.faces[fi][(i + 1) % 3]) == max(
                                self.faces[fj][j], self.faces[fj][(j + 1) % 3]):
                            vi, vj = min(self.faces[fi][i], self.faces[fi][(i + 1) % 3]), max(self.faces[fi][i],self.faces[fi][(i + 1) % 3])
                            if self.IJ2HalfEdge[(vi, vj)].fi == fi:return self.IJ2HalfEdge[(vi, vj)]
                            else:return self.IJ2HalfEdge[(vj, vi)]
            ret = []
            for i in range(len(generator)-1):ret.append(ConvertPair2HalfEdge(generator[i],generator[i+1]))
            return ret
        generators = [Convert(generator)for generator in self.generators]
        P = np.zeros((len(self.harmonic_bases),len(self.generators)))
        for i,base in enumerate(self.harmonic_bases):
            for j,gen in enumerate(generators):
                P[i,j] = IntegrateAlongGenerator(base,gen)
        delta_beta = self.x1@self.d0@self._x2@beta
        z = np.linalg.solve(P,-np.array([IntegrateAlongGenerator(delta_beta,gen)for gen in generators]))
        assert len(z) == len(generators) ==len(self.harmonic_bases)
        self.phi = delta_beta+np.sum(np.array([z[i]*self.harmonic_bases[i]for i in range(len(self.harmonic_bases))]),axis=0)
    @property
    def parallel_vector(self):
        def FaceFrameField(fi):
            e = self.IJ2HalfEdge[(self.faces[fi][2], self.faces[fi][0])].vector
            e/=np.linalg.norm(e)
            return (e,np.cross(self.face_normals[fi],e))
        face_frame_fields = [FaceFrameField(fi) for fi in range(self.fn)]
        fid2angle = {0:0}
        cotree = [i for i in range(self.fn)]
        fidInProcess = [0]
        while fidInProcess!=[]:
            fi = fidInProcess.pop(0)
            for _ in range(3):
                i, j = self.faces[fi][_], self.faces[fi][(_ + 1) % 3]
                he = self.IJ2HalfEdge[(i,j)]
                if he.twin.boundary :continue
                if cotree[he.twin.fi]!=he.twin.fi or he.twin.fi==0:continue
                cotree[he.twin.fi] = fi
                fidInProcess.append(he.twin.fi)
                if (i,j) in self.IJ2UniqueEdgeIdx: phi_ = self.phi[self.IJ2UniqueEdgeIdx[(i,j)]]
                else: phi_ = -self.phi[self.IJ2UniqueEdgeIdx[(j,i)]]
                e0,e1 = face_frame_fields[fi]
                d0,d1 = face_frame_fields[he.twin.fi]
                fid2angle[he.twin.fi] = -phi_ + fid2angle[fi] - np.arctan2(np.dot(e1,he.vector),np.dot(e0,he.vector)) + np.arctan2(np.dot(d1,he.vector),np.dot(d0,he.vector))
        face_vector = []
        for fi in range(self.fn):
            e0,e1 = face_frame_fields[fi]
            angle = fid2angle[fi]
            face_vector.append( e0*np.cos(angle)+e1*np.sin(angle))
        return np.array(face_vector)

# ============= Globally Optimal Direction Fields =============
class NDirectionField(Mesh):
    mach = [2.2250738585072014e-308,1.7976931348623157e+308,1.1102230246251565e-16,2.2204460492503131e-16,3.0102999566398120e-01]
    s11r = [0.0448875760891932036595562553276, 0.0278480909574822965157922173757,0.00394490790249120295818107628687, -0.00157697939158619172562804651751,-0.0000886578217796691901712579357311, 0.0000301708056772263120428135787035,9.521839632337438230089618156e-7, -3.00028307455805582080773625835e-7,-6.14917009583473496433650831019e-9, 1.85133588988085286010092653662e-9,2.67848449041765751590373973224e-11, -7.82394575359355297437491915705e-12,-8.44240072511090922609176843848e-14, 2.41333276776166240844516922196e-14,2.02015531985181413114834031833e-16, -5.68171271075270422851146478874e-17,-3.80082421064644521052871349836e-19, 1.05551739229841670238163200361e-19,5.7758422925275435667221605993e-22, -1.58774695838716531303310462626e-22,-7.24181766014636685673730787292e-25]
    s11i = [0.100116671557942715638078149123, 0.0429600096728215971268270800599,-0.00799014859477407505770275088389, -0.000664114111384495427035329182866,0.000240714510952202000864758517061, 9.89085259369337382687437812294e-6,-3.22040860178194578481012477174e-6, -8.08401148192350365282200249069e-8,2.48351290049260966544658921605e-8, 4.24154988067028660399867468349e-10,-1.25611378629704490237955971836e-10, -1.56053077919196502557674988724e-12,4.50565044006801278137904597946e-13, 4.2641179237225098728291226479e-15,-1.2084245714879456268965803807e-15, -9.01338537885038989528688031325e-18,2.5180796700698002962991581923e-18, 1.51955263898294940481729370636e-20,-4.19737873024216866691628952458e-21, -2.092488792285595339755624521e-23,5.72708467031136321701747126611e-24]
    s12r = [ -0.376145877558191778393359413441, 0.0775244431850198578126067647425, 0.0120396593748540634695397747695, -0.00385683684390247509721340352427, -0.000232359275790231209370627606991, 0.0000697318379146209092637310696007, 2.32354473986257272021507575389e-6, -6.71692140309360615694979580992e-7, -1.43946361256617673523038166877e-8, 4.06087820907414336567714443732e-9, 6.10183339004616075548375321861e-11, -1.69196418769523832825063863136e-11, -1.88669746820541798989965091628e-13, 5.16473095452962111184823547686e-14, 4.45066881692009291504139737861e-16, -1.20625107617859803735741992452e-16, -8.28193837331508300767103116139e-19, 2.22680015825230528892642524445e-19, 1.24755889505424049389100515561e-21, -3.33254971913153176741833960484e-22, -1.55307002839777371508497520751e-24]
    s12i = [0.0527472790869782317601048210983, 0.00823962722148093961886198320927,-0.0205185842051817330153151013327, -0.00184683218270819613487368071941,0.000569681886932212757533488372406, 0.0000248774530818801164177266528608,-7.31121019876580624171992432347e-6, -1.92744564223806538367454388776e-7,5.49794278719049727550379096876e-8, 9.78237385539447442446850072421e-10,-2.7341624177723508216430132999e-10, -3.51839815887772323640101921381e-12,9.68934411607055794052256859665e-13, 9.45703963505047353201918875825e-15,-2.57516976113400217760868402425e-15, -1.97419921753098238455550504742e-17,5.32820017906655555903355375475e-18, 3.29581793797656865402793252539e-20,-8.83137325823594007269279476114e-21, -4.50279718100548728336329365981e-23,1.19941679774924468309434420379e-23]
    m12r = [0.148523151773238914750879360089, -0.0117856118001224048185631301904,-0.00248887208039014371691400683052, 0.000250045060357076469386198883676,0.0000227217776065076434637230864113, -2.48764935230787745662127026799e-6,-1.32138506847814502856384193414e-7, 1.50966754393693942843767293542e-8,5.3472999553162661403204445045e-10, -6.26136041009708550772228055719e-11,-1.59574066624737000616598104732e-12, 1.89788785691219687197167013023e-13,3.66030609080549274006207730375e-15, -4.39955659500182569051978906011e-16,-6.65848768159000092224193226014e-18, 8.06343127453005031535923212263e-19,9.84397490339224661524630997726e-21, -1.19869887155210161836484730378e-21,-1.20634550494837590549640883469e-23, 1.47512193662595435067359954287e-24,1.24549093756962710863096766634e-26]
    m12i = [-0.0454399665519585306943416687117, -0.0210517666740874019203591488894,0.00194647501081621201871675259482, 0.000253466068123907163346571754613,-0.0000268083453427538717591876419304, -1.82138740336918117478832696004e-6,2.04357511048425337951376869602e-7, 8.75944656915074206478854298947e-9,-1.01466837126303146739791005703e-9, -3.02573132377805421636557302451e-11,3.57358222114420372764650037191e-12, 7.88121312149152771558608913996e-14,-9.42758576193708862552405242331e-15, -1.60439904050827900099939709069e-16,1.93624791035947590366500765061e-17, 2.62394448214143482490534256935e-19,-3.18700789496399461681365308408e-20, -3.52400207248027768109209530864e-22,4.30074555255053206057921088056e-23, 3.95655079023456015736315286131e-25,-4.84642137915095135859812028886e-26]
    @staticmethod
    def inits(series: list[float], n: int, eta: float = mach[2]/10 ) -> int:
        err,current_n = 0.0,n
        while err <= eta and current_n > 0:
            current_n -= 1
            err += abs(series[current_n])
        return current_n + 1
    @staticmethod
    def csevl(x: float, cs: list[float] ) -> float:
        b2,b1,b0,twox = 0.0,0,0,2*x
        for i in range(NDirectionField.inits(cs,len(cs)) - 1, -1, -1):
            b2,b1 = b1,b0
            b0 = twox * b1 - b2 + cs[i]
        return (b0 - b2) / 2.0
    @staticmethod
    def s11(t): return NDirectionField.csevl(t, NDirectionField.s11r) + 1j*NDirectionField.csevl(t, NDirectionField.s11i)
    @staticmethod
    def s12(t): return NDirectionField.csevl(t, NDirectionField.s12r) + 1j*NDirectionField.csevl(t, NDirectionField.s12i)
    @staticmethod
    def m12(t): return NDirectionField.csevl(t, NDirectionField.m12r) + 1j*NDirectionField.csevl(t, NDirectionField.m12i)
    @staticmethod
    def DirichletIJDirect(s, gii, gij, gjj):
        s2,si,is3,s4 = s * s,s*1j,s**3*1j,s**4
        term1 = (3 * gii + 4 * gij + 3 * gjj) + si * (gii + gij + gjj) - is3 * gij / 6
        term2 = np.exp(1j*s) * (-(3 * gii + 4 * gij + 3 * gjj) + si * (2 * gii + 3 * gij + 2 * gjj) + s2 * (gii + 2 * gij + gjj) / 2)
        term3 = (gii - 2 * gij + gjj) / 24 - si * (gii - 2 * gij + gjj) / 60
        return (term1+term2)/s4 + term3
    @staticmethod
    def DirichletII(s, gjj, gjk, gkk):  return 0.25 * ((gjj - 2 * gjk + gkk) + s * s * (gjj + gjk + gkk) / 90)
    @staticmethod
    def DirichletIJ(s, gii, gij, gjj):
        if abs(s) > math.pi:return NDirectionField.DirichletIJDirect(s, gii, gij, gjj)
        elif s > 0:         return ((gii + gjj) * NDirectionField.s11(s * 2 / math.pi - 1) + gij * NDirectionField.s12(s * 2 / math.pi - 1)).conj()
        else:               return (gii + gjj) * NDirectionField.s11(-s * 2 / math.pi - 1) + gij * NDirectionField.s12(-s * 2 / math.pi - 1)
    @staticmethod
    def MassII():return 1.0 / 6.0
    @staticmethod
    def MassIJ(s):
        if abs(s) > math.pi:return (6*np.exp(1j*s)-6-6j*s+3*s**2+1j*s**3)/3/s**4
        elif s > 0:         return NDirectionField.m12(s / math.pi * 2 - 1).conj()
        else:               return NDirectionField.m12(-s / math.pi * 2 - 1)

    def __init__(self,mesh,n=1,Senergy=0):
        self.__dict__ = mesh.__dict__
        self.n = n

        s = np.pi*2/(2*np.pi-self.vertex_defects)
        # calculate face angle. (Trimesh里的face_angle缺少和点的对应关系)
        face_angle = {} # key: (face id,vertex id), value: vertex angle in the face
        face_K = np.zeros(self.fn)
        for he in self.Halfedges:
            if he.boundary :continue
            angle = np.arctan2(2*self.area_faces[he.fi],np.dot(-he.prev.vector,he.vector))
            face_angle[(he.fi,he.v0.i)] = angle
            face_K[he.fi] += s[he.v0.i]*angle
        face_K -= np.pi
        # 默认Vertex的halfedges[0]是basis section(Xi), thetas里存放着每个点对每个incident edge的夹角 (eij与Xi的夹角)
        thetas = {}
        for vi,v in enumerate( self.Vertexs ):
            for hei,he in enumerate(v.halfedges): # 注意，face里的he是和face vertex permutation一致的，这里假设是逆时针 而vertex的he也是逆时针的(see Mesh.__init__,这也是角度的定义)
                if hei==0:  thetas[(vi,he.v1.i)] = 0
                else:       thetas[(vi,he.v1.i)] = thetas[(vi,v.halfedges[hei-1].v1.i)] + face_angle[(he.twin.fi,he.v0.i)]
        rho = {}
        for vi,vj in self.edges: rho[(vi,vj)] = (s[vj]*thetas[(vj,vi)]-np.pi) -s[vi]*thetas[(vi,vj)] # angle between vj.he and eij (angle between vj.he and eji - pi)  -  angle between vi.he and eij.

        Mbuilder,Abuilder = MatrixBuilder(self.vn,self.vn),MatrixBuilder(self.vn,self.vn)
        for fi,vis in enumerate(self.faces):
            om,A = n*face_K[fi],self.area_faces[fi]
            Mij = A*NDirectionField.MassIJ(om)
            for i in range(3):
                vi,vj,vk = vis[i],vis[(i+1)%3],vis[(i+2)%3]
                r_ij = np.exp(1j*n*rho[(vi,vj)])
                M_ii = A/6
                M_ij = r_ij.conj()*Mij
                Mbuilder.AddTriplet(vi,vi,M_ii).AddTriplet(vi,vj,M_ij).AddTriplet(vj,vi,M_ij.conj())

                xi,xj,xk = self.vertices[vi],self.vertices[vj],self.vertices[vk]
                pij,pjk,pki = xj-xi,xk-xj,xi-xk
                N_ii = NDirectionField.DirichletII(om,np.dot(pij,pij),-np.dot(pij,pki),np.dot(pki,pki))/A
                N_ij = NDirectionField.DirichletIJ(om,np.dot(pki,pki),-np.dot(pki,pjk),np.dot(pjk,pjk))/A
                A_ii = N_ii - Senergy*om/A*M_ii
                A_ij = N_ij - Senergy*(om/A*Mij-0.5*1j)*r_ij.conj()
                Abuilder.AddTriplet(vi,vi,A_ii).AddTriplet(vi,vj,A_ij).AddTriplet(vj,vi,A_ij.conj())

        M = Mbuilder.scipy_coo_matrix
        A = Abuilder.scipy_coo_matrix
        x = np.random.rand(self.vn) + 1j * np.random.rand(self.vn)
        for _ in range(100):
            x = sp.linalg.spsolve(A, M@x)
            x = x / np.sqrt(x.conj().T @ M @ x )
            r = np.linalg.norm(A @ x - (x.conj().T@A@x)*(M@x) )
            if r < 1e-10: break
        if r>1e-10:print('\033[33mWARNING: N-Direction-Field may not converge\033[0m')
        arg = np.arctan2(np.real(x),np.imag(x))/n
        self.uv = np.column_stack((np.cos(arg),np.sin(arg)))
        # project vertex's halfedge into vertex's tangent plane
        T = []
        for v in self.Vertexs:
            T.append(np.cross(self.vertex_normals[v.i], np.cross(v.he.normalized_vector, self.vertex_normals[v.i])))
        self.T = np.array(T)
        self.BT = self.BT(self.T)

# this class validates several identities or equations numerically
class Validator:
    @staticmethod
    def CompareMatrix(A,B):
        A,B = A.todense(),B.todense()
        assert A.shape == B.shape
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if not np.isclose(A[i,j]-B[i,j],0) :
                    print(i,j,A[i,j],B[i,j])

    # check d*d is equal to Lc(calculate using cotan formula)
    @staticmethod
    def dxd_Lc(m:Mesh):
        Validator.CompareMatrix(m._d1@m.x1@m.d0,m.Lc)

    @staticmethod
    def IsHermitian(M):
        M = M.todense()
        delta = M-np.conj(M.T)
        if np.allclose(np.max(np.abs(delta)),0):return
        print('not Hermitian')

    # 无论m是否是有边界的(sphere,disk)  ∫∇u•∇vdA+∫v∧*Δu=0
    # 这主要是在于 discrete laplacian-beltrami 在构建的时候对边界情况的处理(cot的一边是0)。
    @staticmethod
    def Green1st(m:Mesh,type=0):
        def Grad(u):
            grad_u = np.zeros_like(m.faces).astype(float)
            for i, v in enumerate(m.Vertexs):
                for he in v.halfedges:
                    if he.boundary: continue
                    grad_u[he.fi] += u[i] / (2 * m.area_faces[he.fi]) * np.cross(m.face_normals[he.fi], he.next.vector)
            return grad_u
        if type==0:    # 【u】random 【v】random
            u,v = np.random.rand(m.vn),np.random.rand(m.vn)
        elif type==1:  # 【u】radial 【v】costant 1
            u = np.array([np.linalg.norm(v)**2 for v in m.vertices])
            v = np.array([1.0 for v in m.vertices]) # np.sin(v[0])*10+3*v[1]+v[2]**2+10
        elif type==2:  # 【u】【v】 linear x+z   ∫∇u•∇vdA should be 2A
            u = np.array([v[0]+v[2] for v in m.vertices])
            v = np.array([v[0] + v[2] for v in m.vertices])
        du,dv = Grad(u),Grad(v)
        integral0 = sum([du[i].T@dv[i]*m.area_faces[i] for i in range(m.fn)])
        integral1 = u.T@m.Lc@v
        print('∫∇u•∇vdA  '  ,integral0)
        print('∫v∧*Δu    '  ,integral1)
        print('sum    '  ,integral0+integral1,end='\n\n')

#             chi    |     bn
# sphere      2      |     0
# torus       0      |     2
# disk        1      |     1
# quad-circle 0      |     2

if __name__ == '__main__':
    np.set_printoptions(suppress=True,precision=3)
    print()
    # # Validator.Green_1st(Mesh(os.path.join(__file__, '..', 'input', 'quadx.obj')),2)
    m = Mesh(os.path.join(__file__, '..', 'input', 'sphere.obj'))
    NDirectionField(m)