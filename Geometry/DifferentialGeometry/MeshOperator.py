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

    def __str__(self):
        return f'({self.v0.i},{self.v1.i})'

    @property
    def debug_str(self):
        return (str(self)+' prev'+str(self.prev)+' twin'+ str(self.twin)+' next'+str(self.next)+
                ' boundary '+str(self.boundary)+ ' fi '+str(self.fi))

    @property
    def vector(self):
        return self.v1.p - self.v0.p

    @property
    def length(self):
        return np.linalg.norm(self.vector)

    @property
    def length2(self):
        return np.dot(self.vector, self.vector)

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
        self.halfedges = []

    def __str__(self):
        return str(self.i)

    @property
    def str_halfedges(self):
        ret = ''
        for e in self.halfedges:ret+=str(e)
        return ret

    @property
    def he(self):
        return self.halfedges[0]

    # http://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleAreasCheatSheet.pdf
    def CircumcentricDualArea(self):
        return sum([e.length2*e.cotan+e.prev.length2*e.prev.cotan for e in self.halfedges])/8

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
                h = h.twin.next
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

    # ---------------------------------DISCRETE DIFFERENTIAL OPERATOR-------------------------------------------
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
        data = [v.CircumcentricDualArea() for v in self.Vertexs]
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
        data = [1 / v.CircumcentricDualArea() for v in self.Vertexs]
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
    # ---------------------------------DISCRETE DIFFERENTIAL OPERATOR-------------------------------------------

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

    def ScalarPoissonProblem(self,idx=0):
        rho = np.array( [(0 if i!=idx else 1) for i in range(self.vn) ])
        diagM = np.array([v.CircumcentricDualArea() for v in self.Vertexs])
        rhs =  rho - np.sum(diagM*rho)/self.area
        u = sp.linalg.spsolve(self.L, np.array(rhs))
        return u

    #------------------------------------------------- SCP -------------------------------------------------
    # 对mesh上的每一个点都求得了复平面上对应的一个点
    def SpectralConformalParameterization(self,ComplexAsVec2d=False):
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
        x = np.random.rand(self.vn)+1j*np.random.rand(self.vn)
        for _ in range(100):
            x = sp.linalg.spsolve(EC, x)
            x = x - (x.conj().T @ c).conj() * c
            x = x / np.linalg.norm(x)
            l = x.conj().T@EC@x
            r = np.linalg.norm(EC@x-l*x)
            # print(_,r)
            if r < 1e-10: break
        print('WARNING: Spectral Conformal Parameterization may not converge')
        if ComplexAsVec2d: # translate leftlower to (0,0); rescale to (1,1)
            u, v = np.real(x), np.imag(x)
            u ,v = u - min(u),v - min(v)
            return np.column_stack((u,v))*min(1/max(u),1/max(v))
        return x

    def VisualizeParameterization3D(self, uv):
        u, v = np.real(uv), np.imag(uv)
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

    def VisualizeParameterization(self,uv):
        from matplotlib import pyplot as plt
        u,v = np.real(uv),np.imag(uv)
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        for vi,vj in self.edges_unique:ax.plot([u[vi],u[vj]],[v[vi],v[vj]],linewidth=0.5)
        plt.show()
    #------------------------------------------------- SCP -------------------------------------------------

    #------------------------------------------------- Heat Method -------------------------------------------------
    def HeatMethodGeodesics(self,pid):
        u0 = np.zeros(self.vn)
        u0[pid] = 1
        t = np.mean(self.edges_unique_length)**2
        I = sp.identity(self.vn, format='csr')
        u = sp.linalg.spsolve( I - t*self.L, u0)
        # for i,g in enumerate(u):print(i,g)

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

        geodesics =  sp.linalg.spsolve(self.Lc,div_X )
        geodesics -= np.min(geodesics)
        return geodesics

    def ISOLines(self,center_idx,stride=0.5):
        import collections
        from math import ceil,floor
        geodesics = self.HeatMethodGeodesics(center_idx)
        h = np.mean(self.edges_unique_length)*stride
        ret = []
        for f in self.faces:
            valuemap = collections.defaultdict(list)
            vtxpos = self.vertices[f]
            vtxvalue = geodesics[f]
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
    #------------------------------------------------- Heat Method -------------------------------------------------

    #------------------------------------------------- Hodge Decomposition -------------------------------------------------
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
    def HodgeDecomposition(self,omega):
        assert len(omega)==self.en
        temp = self._x2@self._d1@self.x1
        alpha = sp.linalg.spsolve(temp@self.d0,temp@omega)
        beta  = sp.linalg.spsolve(self.d1@self._x1@self._d0 ,self.d1@omega)
        beta  = sp.linalg.spsolve(self.x2,beta)
        return alpha,beta
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
    def HodgeDecompositionTest(self):
        omega = self.Random1Form()
        alpha,beta = self.HodgeDecomposition(omega)
        face_alpha = self.Whitney1Form(self.d0@alpha)
        face_beta = self.Whitney1Form( self._x1@self._d0@self.x2@beta)
        return self.Whitney1Form(omega) ,face_alpha, face_beta
    #------------------------------------------------- Hodge Decomposition -------------------------------------------------

    #------------------------------------------------- Generators /Harmonic Bases -------------------------------------------------------------
    def TreeCotree(self):
        tree = [i for i in range(self.vn)]
        root = [0]
        while root != []:
            parentId = root.pop(0)
            for he in self.Vertexs[parentId].halfedges:
                childId = he.v1.i
                if childId == 0 or tree[childId] != childId : continue  # already processed
                tree[childId] = parentId
                root.append(childId)
        cotree = [i for i in range(self.fn)]
        root = [0]
        while root != []:
            parentId = root.pop(0)
            for _ in range(3):
                i0, i1 = self.faces[parentId][_], self.faces[parentId][(_ + 1) % 3]
                he = self.IJ2HalfEdge[(i0, i1)]
                if he.twin.boundary: continue
                childId = he.twin.fi
                if cotree[childId] != childId or childId==0 or tree[he.v0.i]==he.v1.i or tree[he.v1.i]==he.v0.i: continue  # already processed or cross tree
                cotree[childId] = parentId
                root.append(childId)
        return tree,cotree
    # each generator is a closed path that starts with x(face idx) and ends with x
    def Generators(self):
        def TraceToRoot(tree,i):
            ret = [i]
            while tree[ret[-1]]!=ret[-1]: ret.append(tree[ret[-1]])
            return ret
        tree,cotree = self.TreeCotree()
        generators = []
        for edge in self.edges_unique:
            he = self.IJ2HalfEdge[(edge[0],edge[1])]
            if he.boundary or he.twin.boundary: continue
            fi,fj = he.fi,he.twin.fi
            if cotree[fi] == fj or cotree[fj] == fi or tree[he.v0.i] == he.v1.i or tree[he.v1.i] == he.v0.i: continue
            tracei,tracej = TraceToRoot(cotree,fi),TraceToRoot(cotree,fj)
            lastCommon = -1
            while tracei[-1]==tracej[-1]:
                lastCommon = tracei.pop()
                lastCommon = tracej.pop()
            generators.append([lastCommon]+tracei[::-1]+tracej+[lastCommon])
        return generators
    def HarmonicBases(self):
        def CommonEdge(fi,fj): #用一个来自fi的halfedge(有向边)来表示这个有向的generator
            for i in range(3):
                for j in range(3):
                    if min(self.faces[fi][i],self.faces[fi][(i+1)%3])==min(self.faces[fj][j],self.faces[fj][(j+1)%3]) and max(self.faces[fi][i],self.faces[fi][(i+1)%3])==max(self.faces[fj][j],self.faces[fj][(j+1)%3]) :
                        vi,vj = min(self.faces[fi][i],self.faces[fi][(i+1)%3]),max(self.faces[fi][i],self.faces[fi][(i+1)%3])
                        if self.IJ2HalfEdge[(vi,vj)].fi==fi:return (vi,vj)
                        else:                               return (vj,vi)
        bases = []
        for generator in self.Generators():
            oneform = np.zeros(self.en)
            for i in range(len(generator)-1):
                fi,fj = generator[i],generator[i+1]
                vi0,vi1 = CommonEdge(fi,fj)
                if (vi0,vi1) not in self.IJ2UniqueEdgeIdx: oneform[self.IJ2UniqueEdgeIdx[(vi1,vi0)]]=-1
                else:oneform[self.IJ2UniqueEdgeIdx[(vi0,vi1)]]=1
            bases.append(oneform-self.d0@self.HodgeDecomposition(oneform)[0])
        return bases

    def VisualizeTreeCoTree(self):
        tree,cotree = self.TreeCotree()
        tree_lines = []
        for i,j in enumerate(tree):tree_lines.extend([self.vertices[i],self.vertices[j]])
        cotree_lines = []
        for i,j in enumerate(cotree):cotree_lines.extend([self.triangles_center[i],self.triangles_center[j]])
        return np.array(tree_lines+cotree_lines)
    def VisualizeGenerators(self):
        generators = self.Generators()
        lines = []
        for generator in generators:
            for _i in range(len(generator)-1):
                i,j  = generator[_i],generator[_i+1]
                lines.extend([self.triangles_center[i],self.triangles_center[j]])
        return np.array(lines)
    def VisualizeHarmonicBases(self):
        return [self.Whitney1Form(base) for base in self.HarmonicBases()]
    #------------------------------------------------- Generators -------------------------------------------------------------

    #------------------------------------------------- Vector Field Design -------------------------------------------------
    def TrivialConnection(self):
        k = np.zeros(self.vn)
        # k[2]=k[3]=1
        for _ in range(self.euler_number): k[np.random.choice(self.vn)] += 1
        #注意,beta是dual 2-form
        beta = sp.linalg.spsolve(self.Lc@self._x2,2 * np.pi * k - self.vertex_defects)
        generators = self.Generators()
        bases = self.HarmonicBases()
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
        generators = [Convert(generator)for generator in generators]
        P = np.zeros((len(bases),len(generators)))
        for i,base in enumerate(bases):
            for j,gen in enumerate(generators):
                P[i,j] = IntegrateAlongGenerator(base,gen)
        delta_beta = self.x1@self.d0@self._x2@beta
        z = np.linalg.solve(P,-np.array([IntegrateAlongGenerator(delta_beta,gen)for gen in generators]))
        assert len(z) == len(generators) ==len(bases)
        return delta_beta+np.sum(np.array([z[i]*bases[i]for i in range(len(bases))]),axis=0)
    def VisualizeConnection(self,phi):
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
                if (i,j) in self.IJ2UniqueEdgeIdx: phi_ = phi[self.IJ2UniqueEdgeIdx[(i,j)]]
                else: phi_ = -phi[self.IJ2UniqueEdgeIdx[(j,i)]]
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
    def NDirectionField(self,n=1,Senergy=0):
        s = np.pi*2/(2*np.pi-self.vertex_defects)
        # 默认Vertex的halfedges[0]是basis section(Xi), thetas里存放着每个点对每个incident edge的夹角 (eij与Xi的夹角)
        # thetas = [[0]]*self.vn
        thetas = {}
        for vi,v in enumerate( self.Vertexs ):
            for hei,he in enumerate(v.halfedges):
                if hei==0:
                    thetas[(vi,he.v1.i)] = 0
                else:
                    vj,vj_prev = he.v1.i,v.halfedges[hei-1].v1.i
                    xi,xj,xj_prev = he.v0.p,he.v1.p,self.vertices[vj_prev]
                    thetas[(vi,vj)] = thetas[(vi,vj_prev)] + np.arccos(np.dot(xj-xi,xj_prev-xi)/np.linalg.norm(xj-xi)/np.linalg.norm(xj_prev-xi))
        rho = {}
        for vi,vj in self.edges:
            rho[(vi,vj)] = n*(s[vj]*thetas[(vj,vi)]-s[vi]*thetas[(vi,vj)])
        e_iOmega = [1]*self.fn
        for fi,vis in enumerate(self.faces):
            for i in range(3):
                e_iOmega[fi]*= np.exp(1j*rho[(vi,vj)])
        Omega = np.angle(e_iOmega)
        f1 = lambda s: (3+1j*s+s**4/24-1j*s**5/60+(-3+2j*s+s*s/2)*np.exp(1j*s))/s**4
        f2 = lambda s: (4+1j*s-1j*s**3/6-s**4/12+1j*s**5/30+(-4+3j*s+s*s)*np.exp(1j*s))/s**4
        Mbuilder = MatrixBuilder(self.vn,self.vn)
        Abuilder = MatrixBuilder(self.vn,self.vn)
        for fi,vis in enumerate(self.faces):
            for i in range(3):
                vi,vj,vk = vis[i],vis[(i+1)%3],vis[(i+2)%3]
                S,omega,r_jk = self.area_faces[fi],Omega[fi],np.exp(1j*rho[(vj,vk)])
                M_ii = S/6
                M_jk = np.conj(r_jk)*S*(6*np.exp(1j*omega)-6-6j*omega+3*omega**2+1j*omega**3)/3/omega**4

                xi,xj,xk = self.vertices[vi],self.vertices[vj],self.vertices[vk]
                pjk,pij,pki = xk-xj,xj-xi,xi-xk
                pjk2,pij2,pki2 = np.dot(pjk,pjk),np.dot(pij,pij),np.dot(pki,pki)
                pijpik = np.dot(pij,-pki)
                N_ii = (pjk2+omega**2*(pij2+pijpik+pki2)/90)/S/4
                N_jk = np.conj(r_jk)*((pij2+pki2)*f1(omega)+pijpik*f2(omega))/S

                A_ii = N_ii - Senergy*omega/S*M_ii
                A_jk = N_jk - Senergy*(omega/S*M_jk-(1 if (vj,vk) in self.edges_unique else -1)*1j*np.conj(r_jk)/2)

                Mbuilder.AddTriplet(vi,vi,M_ii).AddTriplet(vj,vk,M_jk)
                Abuilder.AddTriplet(vi,vi,A_ii).AddTriplet(vj,vk,A_jk)

        M = Mbuilder.scipy_coo_matrix
        A = Abuilder.scipy_coo_matrix
        Validator.IsHermitian(M)
        exit(0)
        x = np.random.rand(self.vn) + 1j * np.random.rand(self.vn)
        for _ in range(100):
            x = sp.linalg.spsolve(A, M@x)
            x = x / np.sqrt(x.conj().T @ M @ x )
            r = np.linalg.norm(A @ x - (x.conj().T@A@x)*(M@x) )
            print(_,r)
            if r < 1e-10: break
        return x

    #------------------------------------------------- Vector Field Design -------------------------------------------------

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
        print(np.max(np.abs(delta)))

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
    # np.set_printoptions(suppress=True,precision=3)
    # print()
    # # Validator.Green_1st(Mesh(os.path.join(__file__, '..', 'input', 'quadx.obj')),2)
    Mesh(os.path.join(__file__, '..', 'input', 'quad-circle.obj'))

