import taichi as ti
import taichi.math as tm
import numpy as np
import trimesh,enum,json,time

ti.init(arch=ti.cpu,default_fp  =ti.f64)
Array   = ti.types.vector(200,ti.i32)
vec2,vec3,vec4     = ti.types.vector(2,ti.f64),ti.types.vector(3,ti.f64),ti.types.vector(4,ti.f64)
vec3i   = ti.types.vector(3, ti.i32)
Medium   = ti.types.struct(eta=vec3,k=vec3)
Material = ti.types.struct(albedo=vec3,Le = vec3,mdm=Medium,type=ti.i32,ax=ti.f64,ay=ti.f64)

nan,inf = ti.math.nan,ti.math.inf
WIDTH,HEIGHT = 400,400
EPS,NOHIT = 1e-8,inf
inf3,nan3,INone = vec3(inf),vec3(nan),-1
# https://refractiveindex.info/  R 630 nm ,G 532 nm ,B 465 nm
Air,Glass,Gold = Medium(eta=vec3(1),k=vec3(0)),Medium(eta=vec3(1.5),k=vec3(0)),Medium(eta=vec3(0.18836,0.54386,1.3319),k=vec3(3.4034,2.2309,1.8693))
MdmNone = Medium(eta=vec3(0))
stack_n = 11

@ti.dataclass
class NodeStack:  # node stack for bvh tree traversal
    node_idxs:  ti.types.vector(200,ti.i32)
    ts:         ti.types.vector(200,ti.f64)  # aabb ray hit time
    n:ti.i32
    @ti.func
    def Push(self,idx,t):
        self.node_idxs[self.n],self.ts[self.n] = idx,t
        self.n += 1
        if self.n >= 200:print('too many nodes in stack')
    @ti.func
    def Pop(self):
        self.n-=1
        if self.n<0:print('node stack is already empty but pop')
        return self.node_idxs[self.n],self.ts[self.n]

@ti.dataclass
class BezierStack:                            # 0   1   2  ...  9    10  11  12 13 14
    stack:ti.types.vector(15*stack_n,ti.f64)  # p0x p0y p0z ... p3x p3y p3z  u0 u1 depth   // u0,u1是 p0,p3处的(在深度为0的那个bezier的)参数坐标
    n:ti.i32
    @ti.func
    def Push(self,p0,p1,p2,p3,u,depth):
        stride = 15
        for j in range(3):
            self.stack[self.n*stride+3*0+j] = p0[j]
            self.stack[self.n*stride+3*1+j] = p1[j]
            self.stack[self.n*stride+3*2+j] = p2[j]
            self.stack[self.n*stride+3*3+j] = p3[j]
        self.stack[self.n*stride+12] = u[0]
        self.stack[self.n*stride+13] = u[1]
        self.stack[self.n*stride+stride-1] = ti.cast(depth,ti.f64)
        self.n+=1
        if self.n>stack_n:print('ERROR: bezier stack too big :',self.n)
    @ti.func
    def Pop(self):
        stride = 15
        self.n-=1
        if self.n<0:print('ERROR: bezier stack can not pop')
        p0,p1,p2,p3,u,depth = nan3,nan3,nan3,nan3,vec2(self.stack[self.n*stride+12],self.stack[self.n*stride+13]),ti.cast(self.stack[self.n*stride+stride-1],ti.i32)
        for j in range(3):
            p0[j]=self.stack[self.n*stride+3*0+j]
            p1[j]=self.stack[self.n*stride+3*1+j]
            p2[j]=self.stack[self.n*stride+3*2+j]
            p3[j]=self.stack[self.n*stride+3*3+j]
        return p0,p1,p2,p3,u,depth

@ti.dataclass
class Bezier:
    p0:vec3
    p1:vec3
    p2:vec3
    p3:vec3
    w:vec2
    bmin:vec3
    bmax:vec3
    sub_n:ti.i32  # 细分深度，最大深度时，认为bezier≈直线。 0表示不用细分，已是直线
    @ti.func
    def At(self,t:ti.f64):return (1-t)**3*self.p0+3*(1-t)**2*t*self.p1+3*(1-t)*t**2*self.p2+t**3*self.p3
    @ti.func
    def TangentAt(self,t:ti.f64):return 3*(1-t)**2*(self.p1-self.p0)+6*(1-t)*t*(self.p2-self.p1)+3*t**2*(self.p3-self.p2)
    @ti.func
    def Init(self):
        L0,self.sub_n = 0.0,0
        for j in ti.static(range(3)):
            L0 = ti.max(L0, ti.abs(self.p0[j] - 2 * self.p1[j] + self.p2[j]),ti.abs(self.p1[j] - 2 * self.p2[j] + self.p3[j]))
        if L0 > 0:
            eps = ti.max(self.w[0], self.w[1]) * 0.05
            value = 1.41421356237 * 6.0 * L0 / (8.0 * eps)
            if value > 0: self.sub_n = int(ti.math.clamp(ti.math.log2(value) / 2, 0, 10))
        self.bmin,self.bmax = ti.min(self.p0,self.p1,self.p2,self.p3),ti.max(self.p0,self.p1,self.p2,self.p3)
        self.bmin -= vec3(self.w.max())
        self.bmax += vec3(self.w.max())
        return self
    @ti.func
    def HitBy(self,o,d):  # return ray's t
        ret = NOHIT
        dx = d.cross(self.p3 - self.p0)
        if dx.norm() < 1e-20: dx, _ = Utils.CoordinateSystem(d)
        w2v = Utils.LookAt(o, o + d, dx)
        # camera space control points
        p0, p1, p2, p3 = (w2v @ vec4(self.p0, 1)).xyz, (w2v @ vec4(self.p1, 1)).xyz, (w2v @ vec4(self.p2, 1)).xyz, (w2v @ vec4(self.p3, 1)).xyz
        stack,found_intersect = BezierStack(),False
        stack.Push(p0, p1, p2, p3,vec2(0,1),0)
        while stack.n > 0 and not found_intersect:
            p0, p1, p2, p3,u, depth = stack.Pop()
            width = vec2(ti.math.mix(self.w[0],self.w[1],u[0]),ti.math.mix(self.w[0],self.w[1],u[1]))
            bezier = Bezier(p0=p0,p1=p1,p2=p2,p3=p3,w=width).Init()  # ray space bezier
            if depth < self.sub_n:  # check box intersect & possibly subdivide
                if bezier.bmin.x <= 0 <= bezier.bmax.x and bezier.bmin.y <= 0 <= bezier.bmax.y and bezier.bmax.z >= 0: # ray INTERSECT with box
                    stack.Push(p0,0.5*(p0+p1),0.25*(p0+2*p1+p2),0.125*(p0+3*p1+3*p2+p3),vec2(u[0],u.sum()/2),depth+1)
                    stack.Push(0.125*(p0+3*p1+3*p2+p3),0.25*(p1+2*p2+p3),0.5*(p2+p3),p3,vec2(u.sum()/2,u[1]),depth+1)
            else:  # check curve intersect
                if (p1.y - p0.y)*-p0.y + p0.x * (p0.x - p1.x) < 0 or (p2.y - p3.y) * -p3.y + p3.x * (p3.x - p2.x)<0:continue
                w = (p3.xy-p0.xy).dot(-p0.xy)/(p3.xy-p0.xy).norm_sqr()  # w ∈ [0,1]
                if bezier.At(w).xy.norm_sqr() > ti.math.mix(width[0],width[1],w)**2: continue
                root_u = ti.math.mix(u[0],u[1],w)
                ret,found_intersect = bezier.At(w).z,True
        return ret

@ti.dataclass
class Ray:
    o:vec3
    d:vec3
    mdm:Medium
    @ti.func
    def At(self, t):return self.o+self.d*t
    @ti.func
    def HitTriangle(self,v0,v1,v2):
        ret = NOHIT
        e0,e1 = v1-v0,v2-v0
        h = self.d.cross(e1)
        a = e0.dot(h)
        if ti.abs(a) > EPS:
            f,s = 1/a,self.o-v0
            u = f*s.dot(h)
            q = s.cross(e0)
            v = f*q.dot(self.d)
            t = f*e1.dot(q)
            if 0<=u<=1 and 0<=v<=1 and u+v<=1 and t>EPS: ret = t
        return ret
    @ti.func
    def HitAABB(self,bmin,bmax):
        t_enter,t_exit,t,isHit,originInsideBox = -NOHIT,NOHIT,NOHIT,True,True
        for i in ti.static(range(3)):
            if self.o[i]<bmin[i] or self.o[i]>bmax[i]: originInsideBox = False
            if self.d[i]==0:
                if bmin[i]>self.o[i] or self.o[i]>bmax[i] : isHit = False
            else:
                t0,t1 = (bmin[i]-self.o[i])/self.d[i],(bmax[i]-self.o[i])/self.d[i]
                if self.d[i]<0:t0,t1=t1,t0
                t_enter,t_exit = max(t_enter,t0),min(t_exit,t1)
        if t_enter<=t_exit and t_exit>=0: t = t_enter if t_enter>=0 else t_exit
        return -1.0 if originInsideBox else (t if isHit else NOHIT)

Sample   = ti.types.struct(pdf=ti.f64, ray=Ray, value=vec3) # bxdf sample or phase function sample. value is bxdf value or phase function value

@ti.dataclass
class Interaction:
    t:ti.f64
    fi:ti.i32
    ray:Ray
    normal:vec3
    tangent:vec3
    pos:vec3
    mat:Material
    inside_mesh:ti.u1
    mdmT:Medium
    @ti.func
    def Valid(self):return self.t!=NOHIT
    @ti.func
    def FetchInfo(self,scene):
        if self.Valid():
            self.pos = self.ray.At(self.t)
            face,area = scene.faces[self.fi],scene.face_areas[self.fi]
            ws = vec3(0)
            for i in ti.static(range(3)):
                j,k = (i+1)%3,(i+2)%3
                ws[i] = (self.pos-scene.vertices[face[j]]).cross(self.pos-scene.vertices[face[k]]).norm()/2/area
            self.normal = (ws[0]*scene.vertex_normals[face[0]]+ws[1]*scene.vertex_normals[face[1]]+ws[2]*scene.vertex_normals[face[2]]).normalized()
            t = (ws[0]*scene.vertex_tangents[face[0]]+ws[1]*scene.vertex_tangents[face[1]]+ws[2]*scene.vertex_tangents[face[2]]).normalized()
            self.tangent = self.normal.cross(t.cross(self.normal)).normalized() # force orthogonal because  t is interpolated tangent , that may not perpendicular to self.normal
            if FLAT: self.normal,self.tangent = scene.face_normals[self.fi],scene.face_tangents[self.fi]
            self.mat = scene.face_materials[self.fi]
            self.inside_mesh = False if (self.ray.mdm.eta - Air.eta).norm()==0 else True  #注意，不能用法线判断。因为如果不是平坦着色，插值结果可能不对。当然，这么做的前提是每次生成光线时的mdm要给的是对的
            self.mdmT = self.mat.mdm  # transmission medium
            if self.inside_mesh:self.mdmT = Air
            if scene.face_doubleside[self.fi]:
                self.inside_mesh = False
                if self.ray.d.dot(self.normal)>0:self.normal *= -1  #对于双面的面片，法线总是朝向光线的起点。
        return self

class BxDF:
    class Type(enum.IntEnum):
        Lambertian = enum.auto() # reflection_diffuse
        Specular = enum.auto()   # reflection_specular
        Transmission = enum.auto() # reflection_specular  transmission_specular
        Microfacet  = enum.auto()  # reflection_glossy
    Sample = ti.types.struct(pdf=ti.f64,ray=Ray,value=vec3)
    @ti.func
    def CosTheta(w):return w.y
    @ti.func
    def CosTheta2(w): return w.y**2
    @ti.func
    def CosPhi2(w): return w.x**2/(1-BxDF.CosTheta2(w))
    @ti.func
    def SinPhi2(w): return 1 - BxDF.CosPhi2(w)

    @ti.func
    def SampleUniformDiskPolar(u):
        r,theta = ti.sqrt(u[0]),2*tm.pi*u[1]
        return r*ti.cos(theta),r*ti.sin(theta)
    @ti.func
    def Sample_wm(w,u,ax,az):
        wh = vec3(ax*w.x,w.y,az*w.z).normalized()
        if wh.y<0:wh.y=-wh.y
        t1 = vec3(0,1,0).cross(wh).normalized() if wh.y<0.999999 else vec3(1,0,0)
        t2 = wh.cross(t1)
        p = BxDF.SampleUniformDiskPolar(u)
        h = ti.sqrt(1-p[0]**2)
        p[1] = (1-(1 + wh.y) / 2) * h + (1 + wh.y) / 2 * p[1]
        # p[1] = (1+wh.y)*0.5*(1-p[1])+h*p[1]
        pz = ti.sqrt(ti.max(0,1-p[0]**2-p[1]**2))
        nh = p[0]*t1+pz*wh+p[1]*t2
        return vec3(ax*nh[0],ti.max(EPS,nh[1]),az*nh[2]).normalized()
    @ti.func
    def CosineSampleHemisphere():
        phi,cos_theta = 2*tm.pi*ti.random(),ti.sqrt(ti.random())
        sin_theta = ti.sqrt(1-cos_theta**2)
        return vec3(sin_theta*ti.sin(phi),cos_theta,sin_theta*ti.cos(phi))
    @ti.func
    def ToWorld(localCoord:vec3,worldNormal:vec3,worldTangent:vec3):
        binormal = worldNormal.cross(worldTangent)
        return localCoord[0]*worldTangent+localCoord[1]*worldNormal+localCoord[2]*binormal
    @ti.func
    def ToLocal(v:vec3,normal:vec3,tangent:vec3):
        binormal = normal.cross(tangent)
        return vec3(v.dot(tangent),v.dot(normal),v.dot(binormal))
    @ti.func
    def Reflect(wi, n):return (2 * n * wi.dot(n)-wi).normalized()
    @ti.func #https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/ eq([1])
    def Fresnel(i, n, mdmI, mdmT):  # only consider Dielectric-Conductor Dielectric-Dielectric
        eta,etak,cos_theta = mdmT.eta/mdmI.eta,mdmT.k/mdmI.eta,ti.abs(i.dot(n))
        cos_theta2 = cos_theta**2
        sin_theta2 = 1 - cos_theta2
        eta2,etak2 = eta**2,etak**2
        t0 = eta2-etak2-sin_theta2
        A2plusB2 = ti.sqrt(t0**2+4*etak**2)
        t1,a = A2plusB2+cos_theta2,ti.sqrt(0.5*(A2plusB2+t0))
        t2 = 2*a*cos_theta
        Rs = (t1-t2) /(t1+t2)
        t3,t4 = cos_theta2*A2plusB2+sin_theta2**2,t2*sin_theta2
        Rp = Rs*(t3-t4)/(t3+t4)
        return 0.5 * (Rs+Rp)
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.16)
    def D(w,ax,ay):
        ct2,cp2,sp2 = BxDF.CosTheta2(w),BxDF.CosPhi2(w),BxDF.SinPhi2(w)
        return 1.0/(tm.pi*ax*ay*ct2**2*(1+(1/ct2-1)*(cp2/ax**2+sp2/ay**2))**2)
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.20)
    def Lambda(w,ax,ay):
        ct2,cp2,sp2 = BxDF.CosTheta2(w),BxDF.CosPhi2(w),BxDF.SinPhi2(w)
        return ti.sqrt(1+(ax**2*cp2+ay**2*sp2)*(1/ct2-1))/2-0.5
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.22)
    def G(wi,wo,ax,ay):return 1.0/(1.0+BxDF.Lambda(wi,ax,ay)+BxDF.Lambda(wo,ax,ay))
    @ti.func
    def G1(w,ax,ay): return 1.0 / (1.0 + BxDF.Lambda(w, ax,ay) )
    @ti.func
    def D_PDF(w,wm,ax,ay):return BxDF.G1(w,ax,ay)/ti.abs(BxDF.CosTheta(w))*BxDF.D(wm,ax,ay)*ti.abs(w.dot(wm))
    @ti.func
    def Sample(ix:Interaction):
        assert ix.ray.mdm.eta.sum() != 0
        N,T,n = ix.normal,ix.tangent,vec3(0,1,0)  # N: world normal, T: world tangent, n: local normal
        wo,wi,pdf,f = BxDF.ToLocal(-ix.ray.d,N,T).normalized(),vec3(0),0.,vec3(inf,0,0) # use inf RED to expose problem
        next_ix_inside_mesh = ix.inside_mesh # next intersection inside mesh : default is same with current
        if ix.mat.type==BxDF.Type.Lambertian:
            wi = BxDF.CosineSampleHemisphere()
            pdf = wi.y/tm.pi
            f = ix.mat.albedo/tm.pi
        elif ix.mat.type==BxDF.Type.Specular:
            wi,pdf = BxDF.Reflect(wo,n),1
            f = vec3(1)/ti.abs(BxDF.CosTheta(wi)) # https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF  eq(9.9)
        elif ix.mat.type==BxDF.Type.Transmission:
            cosI,eta = wo.dot(n), ix.mdmT.eta/ix.ray.mdm.eta
            if cosI<0:cosI,n = -cosI,-n
            wi =  (-wo/eta + (cosI/eta - ti.sqrt(1 - (1/eta)**2 * (1 -cosI**2))) * n).normalized()
            if ti.random()<BxDF.Fresnel(wo,n,ix.ray.mdm,ix.mdmT)[0]: # for Dielectric to Dielectric fresnel's rgb are identical
                wi = BxDF.Reflect(wo,n)
            else: next_ix_inside_mesh = not next_ix_inside_mesh # refraction happens: mesh to air or air to mesh
            pdf,f = 1,vec3(1)/ti.abs(BxDF.CosTheta(wi))
        elif ix.mat.type==BxDF.Type.Microfacet:
            ax,ay = ix.mat.ax,ix.mat.ay
            wm = BxDF.Sample_wm(wo,[ti.random(),ti.random()],ax,ay)
            wi = BxDF.Reflect(wo, wm)
            # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.33)
            pdf = BxDF.D_PDF(wo,wm,ax,ay)/4/ti.abs(wo.dot(wm))
            f   = BxDF.D(wm,ax,ay)*BxDF.G(wi,wo,ax,ay)*BxDF.Fresnel(wo, wm, ix.ray.mdm, ix.mdmT) / ti.abs(4 * BxDF.CosTheta(wi) * BxDF.CosTheta(wo))
        nextRayO,nextRayMdm = ix.pos+ix.normal*EPS, Air
        if next_ix_inside_mesh:nextRayO,nextRayMdm = ix.pos - ix.normal*EPS,   ix.mat.mdm
        # assert nextRayMdm.eta.norm()>EPS
        return Sample(pdf=pdf,  ray=Ray(o=nextRayO,d=BxDF.ToWorld(wi,N,T).normalized(),mdm=nextRayMdm), value=f)

class Utils:
    @staticmethod
    def DiffuseLike(*arg):
        if len(arg)==1:return Material(albedo=vec3(arg[0],arg[0],arg[0]),Le=vec3(0),mdm = MdmNone,type=BxDF.Type.Lambertian)
        if len(arg)==3:return Material(albedo=vec3(arg[0],arg[1],arg[2]),Le=vec3(0),mdm = MdmNone,type=BxDF.Type.Lambertian)
        if len(arg)==6:return Material(albedo=vec3(arg[0],arg[1],arg[2]),Le=vec3(arg[3],arg[4],arg[5]),mdm = MdmNone,type=BxDF.Type.Lambertian)
    @staticmethod
    def MetalLike(m,rx,ry):return Material(Le=vec3(0),mdm=m,type=BxDF.Type.Microfacet,ax=rx,ay=ry)
    @staticmethod
    def GlassLike(eta): return Material(Le=vec3(0),mdm=Medium(eta = vec3(eta)),type=BxDF.Type.Transmission)
    @staticmethod
    def MirrorLike():return Material(Le=vec3(0),mdm = MdmNone,type=BxDF.Type.Specular)
    # make (v2, v3, v) 一个右手坐标系.要求v是normalized的
    #https://www.pbr-book.org/4ed/Geometry_and_Transformations/Vectors#CoordinateSystemfromaVector
    @ti.func
    def CoordinateSystem(v):
        sign = ti.math.sign(v.z) if v.z!=0 else 1
        a = -1/(sign+v.z)
        b = v.x*v.y*a
        return vec3(1+sign*v.x**2*a,sign*b,-sign*v.x),vec3(b,sign+v.y**2*a,-v.y)
    @ti.func
    def LookAt(pos,target,up):
        d = (target - pos).normalized()
        r = (up.normalized().cross(d)).normalized()
        newup,m = d.cross(r).normalized(),ti.math.eye(4)
        for i in ti.static(range(3)): m[i,3],m[i,0],m[i,1],m[i,2] = pos[i],r[i],newup[i],d[i]
        return m.inverse()

class Mesh(trimesh.Trimesh):
    def __init__(self,objpath,material):
        mesh = trimesh.load(objpath,process=False)
        super().__init__(vertices=mesh.vertices,faces=mesh.faces,visual=mesh.visual)
        self.material = material
        def Perpendicular(n):
            axis = 0
            while n[axis] == 0: axis += 1
            assert n[axis] != 0
            t = np.zeros(3)
            t[axis] = n[(axis + 1) % 3]
            t[(axis + 1) % 3] = - n[axis]
            return t
        vt,ft = [],[]
        for n in self.vertex_normals:vt.append(Perpendicular(n))
        for n in self.face_normals:ft.append(Perpendicular(n))
        self.vertex_tangents,self.face_tangents = np.array(vt),np.array(ft)
        if not hasattr(self.visual,'uv'):return
        ft = np.zeros_like(self.faces).astype(np.float64)
        for fi,(vi,vj,vk) in  enumerate(self.faces):
            uv,n = self.visual.uv,self.face_normals[fi]
            ft[fi] += (uv[vj][0]-uv[vi][0])*np.cross(n,self.vertices[vi]-self.vertices[vk])
            ft[fi] += (uv[vk][0]-uv[vi][0])*np.cross(n,self.vertices[vj]-self.vertices[vi])
            ft[fi] /= np.linalg.norm(ft[fi])
        self.face_tangents = ft
        vt = np.zeros_like(self.vertex_tangents)
        for fi,face in enumerate(self.faces):
            for vi in face: vt[vi]+=ft[fi]*self.area_faces[fi]
        for vi in range(self.vertices.shape[0]):self.vertex_tangents[vi] = vt[vi]/np.linalg.norm(vt[vi])

class Camera:
    def __init__(self,pos,target,near_plane=0.01,fov = 0.35):
        self.pos = pos
        front = (target-pos).normalized()
        right = front.cross(vec3(0,1,0))
        up = right.cross(front)
        nearplane_center = pos+near_plane*front
        self.origin = nearplane_center-near_plane*fov*up-near_plane*fov*right
        self.unit_x = right * 2. * near_plane * fov / (WIDTH - 1);
        self.unit_y = up * 2. * near_plane * fov / (HEIGHT - 1);

class BVH:
    @ti.dataclass
    class Node:
        min:vec3             # aabb's min for this node
        max:vec3             # aabb's max for this node
        li:ti.i32            # left child index   (similar to  a pointer to node)
        ri:ti.i32            # right child index  (similar to  a pointer to node)
        start:ti.i32         # begin of triangle indices
        end:ti.i32           # end of triangle indices   P.S. triangle range [start,end]
        @ti.func
        def Leaf(self): return self.li==INone and self.ri==INone
    # https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    def __init__(self,aabbs):  # aabbs : (box mins(ndarray),box maxs(ndarray))
        n = len(aabbs[0])
        self.nodes = self.Node.field(shape = n*2)
        self.idx = ti.field(ti.i32,n) # element idx . (triangle index)
        # 归一化vertices到[0,1] 为了后面的Morton code
        centroids = 0.5*(aabbs[0]+aabbs[1])
        centroids -= np.min(centroids,axis=0)
        centroids /= np.max(centroids,axis=0)
        def ExpandBits(v: int) -> int:
            v = (v * 0x00010001) & 0xFF0000FF
            v = (v * 0x00000101) & 0x0F00F00F
            v = (v * 0x00000011) & 0xC30C30C3
            v = (v * 0x00000005) & 0x49249249
            return v
        def Morton3D(pos):
            x, y, z = pos
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 <= z <= 1
            x = min(max(x * 1024.0, 0.0), 1023.0)
            y = min(max(y * 1024.0, 0.0), 1023.0)
            z = min(max(z * 1024.0, 0.0), 1023.0)
            xx = ExpandBits(int(x))
            yy = ExpandBits(int(y))
            zz = ExpandBits(int(z))
            return (xx << 2) | (yy << 1) | zz
        mortons = np.array([Morton3D(pos) for pos in centroids])
        sorted_idx    = np.argsort(mortons)
        sorted_mortons = mortons[sorted_idx]
        self.idx.from_numpy(sorted_idx)
        node_type =  np.dtype([('min', np.float64, (3,)),('max', np.float64, (3,)),('li',np.int32),('ri',np.int32),('start',np.int32),('end',np.int32)])
        nodes = np.empty(n*2, dtype=node_type)
        nodes[:] = ([inf,inf,inf],[-inf,-inf,-inf],INone,INone,INone,INone)
        #  now build tree  --------------------------------------------------------------------------
        def LeadingZeros(x):return 32 if x==0 else (31 - int(x).bit_length())
        def FindSplit(sorted_morton_codes, first, last):
            if sorted_morton_codes[first] == sorted_morton_codes[last]:return (first + last) // 2
            common_prefix = LeadingZeros(sorted_morton_codes[first] ^ sorted_morton_codes[last])
            split = first
            step = last - first
            while step > 1:
                step = (step + 1) // 2
                new_split = split + step
                if new_split < last:
                    split_code = sorted_morton_codes[new_split]
                    split_prefix = LeadingZeros(sorted_morton_codes[first] ^ split_code)
                    if split_prefix > common_prefix:split = new_split
            return split
        def BuildTree(node_idx, start_idx, end_idx):  # return nodes number(childs + this), 并且会在node_idx处初始化node
            nonlocal nodes,sorted_idx,sorted_mortons,aabbs
            if start_idx == end_idx:
                nodes[node_idx] =  (aabbs[0][sorted_idx[start_idx]],aabbs[1][sorted_idx[start_idx]],INone,INone,start_idx,end_idx)
                return 1
            split = FindSplit(sorted_mortons, start_idx, end_idx)
            li = node_idx+1
            lcnt = BuildTree(li,start_idx,split)
            ri = li + lcnt
            rcnt = BuildTree(ri,split+1,end_idx)
            nodes[node_idx] = (np.minimum(nodes[li]['min'],nodes[ri]['min']),np.maximum(nodes[li]['max'],nodes[ri]['max']),li,ri,start_idx,end_idx)
            return lcnt+rcnt+1
        BuildTree(0,0,n-1)
        self.nodes.from_numpy(nodes)

@ti.data_oriented
class Scene:
    def __init__(self,hair_json_path,furnace_test=False,camera = Camera(pos=vec3(0,10,-20),target=(0,10,0))):
        self.furnace,self.maxdepth = furnace_test,1 if not furnace_test else 100
        self.env_Le = vec3(0.5) if furnace_test else vec3(0)
        self.img = ti.Vector.field(3,ti.f64,(WIDTH,HEIGHT))
        self.camera = camera
        hair_data = []
        with open(hair_json_path,'r') as f:
            json_data = json.load(f)
            self.cn = len(json_data['shapes'])
            for curve in json_data['shapes']:
                hair_data.extend(curve['points'])
                hair_data.extend(curve['width'])
        self.hair = Bezier.field(shape = self.cn)
        self.curve_bmin,self.curve_bmax = ti.Vector.field(3,ti.f64,self.cn),ti.Vector.field(3,ti.f64,self.cn)
        self.HairInit(np.array(hair_data,dtype=np.float64).reshape(-1,14))  # 3*4+2  (4 control points + width at start and end )
        st = time.time()
        self.bvh = BVH((self.curve_bmin.to_numpy(),self.curve_bmax.to_numpy()))
        print('bvh build cost ',time.time()-st,' n is ',self.cn)

    @ti.kernel
    def HairInit(self,curves:ti.types.ndarray(dtype = ti.types.vector(14,ti.f64),ndim=1)):
        for i in ti.grouped(curves):
            self.hair[i] = Bezier(p0 = vec3(curves[i][0],curves[i][1],curves[i][2]),
                                  p1 = vec3(curves[i][3],curves[i][4],curves[i][5]),
                                  p2 = vec3(curves[i][6],curves[i][7],curves[i][8]),
                                  p3 = vec3(curves[i][9],curves[i][10],curves[i][11]),
                                  w  = vec2(curves[i][12],curves[i][13])).Init()
            self.curve_bmin[i] = self.hair[i].bmin
            self.curve_bmax[i] = self.hair[i].bmax

    # 光线与AABB的相交，有两种情况：a.光线起点就在AABB内部,这时会返回-1.0  b.光线起点在AABB外部，这时返回相交时的time  (see Ray.HitAABB)
    # 函数里的t是光线与某个curve(不是AABB)的交点t。如果stack里有任何 node_t >= t，此时都可以提前拒绝(即continue)
    # 进一步解释，node_t>=t 说明ray.o在此node之外,任何node之内的元素与ray的交点都比t要大，所以可以提前拒绝
    # 对于else语句，想法是，总是加入先被ray hit的node. (以期待后续的early reject)
    # 当然，如果ray.o本身就在某个node之内，要先Push此node (其实这也是为啥ray.HitAABB的a情况要返回一个负数)
    @ti.func
    def HitBy(self,ray):
        t, triidx = NOHIT, INone
        stack = NodeStack()
        stack.Push(0,ray.HitAABB(self.bvh.nodes[0].min,self.bvh.nodes[0].max))
        while stack.n>0:
            node_idx, node_t = stack.Pop()
            if node_t >= t: continue
            node = self.bvh.nodes[node_idx]
            if node.Leaf():
                bezier = self.hair[self.bvh.idx[node.start]]
                this_t = bezier.HitBy(ray.o,ray.d)
                if this_t < t: t, curve_idx = this_t, self.bvh.idx[node.start]
            else:
                i0, i1 = node.li, node.ri
                n0, n1 = self.bvh.nodes[i0], self.bvh.nodes[i1]
                t0, t1 = ray.HitAABB(n0.min,n0.max), ray.HitAABB(n1.min,n1.max)
                if t0!=NOHIT and t1!=NOHIT:
                    if t1<t0: i0,i1,t0,t1 = i1,i0,t1,t0
                if t0!=NOHIT: stack.Push(i0,t0)
                if t1!=NOHIT: stack.Push(i1,t1)
        return  t #Interaction(t,triidx,ray).FetchInfo(self)

    @ti.kernel
    def Draw(self):
        for i,j in self.img:
            o = self.camera.origin + (i+ti.random()/2-1)*self.camera.unit_x + (j+ti.random()/2-1)*self.camera.unit_y
            ray = Ray(o=o,d = (o-self.camera.pos).normalized(),mdm=Air)
            #https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation eq13.4  可以参考13.1.2举的例子, Le + rho_hh( Le + rho_hh( Le +...
            L,beta = vec3(0),vec3(1)
            for _ in range(1):#self.maxdepth):
                ix = self.HitBy(ray)
                if ix==NOHIT :
                    L += beta*self.env_Le
                    break
                # L += beta* ix.mat.Le
                # sample = BxDF.Sample(ix)
                # beta *= sample.value*ti.abs(sample.ray.d.dot(ix.normal))/sample.pdf
                # ray = sample.ray
                L = vec3(1)
            self.img[i,j] = L

class Film:
    def __init__(self,scene):
        self.img = ti.Vector.field(3,ti.f64,(WIDTH,HEIGHT))
        self.scene = scene

    def Show(self):
        window = ti.ui.Window("", (WIDTH,HEIGHT))
        canvas = window.get_canvas()
        frame = 0
        while not window.is_pressed(ti.ui.ESCAPE):
            self.scene.Draw()
            self.img.from_numpy(self.img.to_numpy()+self.scene.img.to_numpy())
            frame += 1
            canvas.set_image(self.img.to_numpy().astype(np.float32) / frame)
            window.show()

Film(Scene('./assets/hair/straight-hair.json')).Show()