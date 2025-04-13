import taichi as ti
import taichi.math as tm
import numpy as np
import trimesh,enum

ti.init(arch=ti.cpu,default_fp  =ti.f64,debug=True)
Array   = ti.types.vector( 200  ,ti.i32)
vec3    = ti.types.vector(3,ti.f64)
vec3i   = ti.types.vector(3, ti.i32)
# Assume there is no emitter in Medium and no absorption and scattering is constant
Medium   = ti.types.struct(eta=vec3,k=vec3,ss=ti.f64,g = ti.f64)
Material = ti.types.struct(albedo=vec3,Le = vec3,mdm=Medium,type=ti.i32,ax=ti.f64,ay=ti.f64)

FLAT,ThickAir = False,False # flat shading
WIDTH,HEIGHT = 400,400
SURFACE,VOLUME = 0,1
EPS = 1e-8
MAX = 1.7976931348623157e+308
NOHIT = MAX
MAX3 = vec3(MAX,MAX,MAX)
FNone,  INone,   = -1.0,  -1
# https://refractiveindex.info/  R 630 nm ,G 532 nm ,B 465 nm
Air,Glass,Gold = Medium(eta=vec3(1),k=vec3(0),ss=0.0,g=0),Medium(eta=vec3(1.5),k=vec3(0)),Medium(eta=vec3(0.18836,0.54386,1.3319),k=vec3(3.4034,2.2309,1.8693))
Jade,Jade1 = Medium(eta=vec3(1.1),k=vec3(0),ss=1.5,g=0.9),Medium(eta=vec3(1.1),k=vec3(0),ss=1.5,g=-0.9),
MdmNone = Medium(eta=vec3(0))
if ThickAir: Air.ss = 1.5

@ti.dataclass
class Ray:
    o:vec3
    d:vec3
    mdm:Medium

    @ti.func
    def At(self, t):
        return self.o+self.d*t

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
        t_enter,t_exit,t,isHit = -NOHIT,NOHIT,NOHIT,True
        for i in ti.static(range(3)):
            if self.d[i]==0:
                if bmin[i]>self.o[i] or self.o[i]>bmax[i] : isHit = False
            else:
                t0,t1 = (bmin[i]-self.o[i])/self.d[i],(bmax[i]-self.o[i])/self.d[i]
                if self.d[i]<0:t0,t1=t1,t0
                t_enter,t_exit = max(t_enter,t0),min(t_exit,t1)
        if t_enter<=t_exit and t_exit>=0: t = t_enter if t_enter>=0 else t_exit
        return t if isHit else NOHIT

Sample   = ti.types.struct(pdf=ti.f64, ray=Ray, value=vec3) # bxdf sample or phase function sample. value is bxdf value or phase function value

@ti.dataclass
class Interaction:
    type:ti.i32  # 0: surface,  1: volume
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
class PF:
    @ti.func
    def F(ix:Interaction,Wi):
        Wo, g = -ix.ray.d, ix.ray.mdm.g
        costheta = Wo.dot(Wi)
        return (1-g*g)/(1+g*g+2*g*costheta)**1.5/4/tm.pi
    @ti.func
    def PDF(ix:Interaction,Wi):return PF.F(ix,Wi)
    @ti.func
    def Sample(ix:Interaction):
        assert ix.type==VOLUME
        Wo,g = -ix.ray.d,ix.ray.mdm.g
        #https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions  (last eq)
        phi = tm.pi*2*ti.random()
        costheta = 1-2*ti.random()
        if ti.abs(g)>EPS:  costheta = -(1+g*g-((1-g*g)/(1+g-2*g*ti.random()))**2)/(2*g)
        sintheta = ti.sqrt(1-costheta**2)
        wi = vec3(ti.cos(phi)*sintheta,costheta,ti.sin(phi)*sintheta)
        Y = Wo
        X = Y.cross(vec3(0,1,0)) if ti.abs(Y.y)<1-EPS else Y.cross(vec3(1,0,0))
        Z = X.cross(Y)
        Wi = X*wi.x+Y*wi.y+Z*wi.z
        # phi = ti.random() * 2 * tm.pi
        # cos_theta = ti.random() * 2 - 1
        # sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
        # Wi = vec3(sin_theta * ti.sin(phi), cos_theta, sin_theta * ti.cos(phi))
        # infact, pdf == value == phase function.  set them to 1 is same for the  【pf/pdf * Li】
        pdf = PF.F(ix,Wi)
        return Sample(pdf = pdf ,ray = Ray(o=ix.pos,d = Wi.normalized(),mdm=ix.ray.mdm),value =pdf )

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
        wo,wi,pdf,f = BxDF.ToLocal(-ix.ray.d,N,T).normalized(),vec3(0),0.,vec3(MAX,0,0) # use MAX RED to expose problem
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
        min:vec3
        max:vec3
        li:ti.i32
        ri:ti.i32
        start:ti.i32
        end:ti.i32
        @ti.func
        def Leaf(self):
            return self.li==INone and self.ri==INone

    def __init__(self,vertices,faces):
        vn,fn = len(vertices),len(faces)
        self.nodes = self.Node.field(shape = fn*2)
        self.tidx = ti.field(ti.i32,fn)
        self.aabbs = []
        self.centroids = []
        for i in range(fn*2):
            self.nodes[i] = self.Node(min=MAX3,max=-MAX3,li=INone,ri=INone,start=INone,end=INone)
        for i in range(fn):
            self.tidx[i] = i
            self.aabbs.append((np.min(vertices[faces[i]],axis=0),np.max(vertices[faces[i]],axis=0)))
            self.centroids.append(np.mean(vertices[faces[i]],axis=0))
        self.Build(0,0,fn)

    def Build(self,node_idx,start,end):  #[start,end)
        def AABBArea(box):
            extend = box[1] - box[0]
            return extend[0] * extend[1] + extend[2] * extend[1] + extend[0] * extend[2]
        bin_cnt = 8
        node = self.nodes[node_idx]
        split_axis, split_pos, split_cost = INone, INone, MAX
        left_count, right_count = 0, 0
        for i in range(start, end):
            node.min = ti.min(node.min,self.aabbs[self.tidx[i]][0])
            node.max = ti.max(node.max,self.aabbs[self.tidx[i]][1])
        for j in range(3):
            boundsMin, boundsMax = MAX, -MAX
            for i in range(start, end):
                boundsMin = min(boundsMin, self.centroids[self.tidx[i]][j])
                boundsMax = max(boundsMax, self.centroids[self.tidx[i]][j])
            if boundsMin == boundsMax: continue
            stride = (boundsMax - boundsMin) / bin_cnt
            bins = [[MAX3, -MAX3, 0] for _ in range(bin_cnt)]
            for i in range(start, end):
                bin_idx = min(bin_cnt - 1, int((self.centroids[self.tidx[i]][j] - boundsMin) / stride))
                bins[bin_idx][0] = ti.min(bins[bin_idx][0],self.aabbs[self.tidx[i]][0])
                bins[bin_idx][1] = ti.max(bins[bin_idx][1],self.aabbs[self.tidx[i]][1])
                bins[bin_idx][2] += 1

            left_area = [0.0 for _ in range(bin_cnt - 1)]
            left_cnt = [0 for _ in range(bin_cnt - 1)]
            left_box = [MAX3, -MAX3]
            left_sum = 0
            right_area = [0.0 for _ in range(bin_cnt - 1)]
            right_cnt = [0 for _ in range(bin_cnt - 1)]
            right_box = [MAX3, -MAX3]
            right_sum = 0
            for i in range(bin_cnt - 1):
                left_box[0] = ti.min(left_box[0],bins[i][0])
                left_box[1] = ti.max(left_box[1],bins[i][1])
                left_area[i] = AABBArea(left_box)
                left_sum += bins[i][2]
                left_cnt[i] = left_sum

                right_box[0] = ti.min(right_box[0],bins[bin_cnt - i - 1][0])
                right_box[1] = ti.max(right_box[1],bins[bin_cnt - i - 1][1])
                right_area[bin_cnt - i - 2] = AABBArea(right_box)
                right_sum += bins[bin_cnt - i - 1][2]
                right_cnt[bin_cnt - i - 2] = right_sum
            for i in range(bin_cnt - 1):
                cost = left_area[i] * left_cnt[i] + right_area[i] * right_cnt[i]
                if cost < split_cost:
                    split_axis, split_pos, split_cost = j, boundsMin + stride * (i + 1), cost
                    left_count, right_count = left_cnt[i], right_cnt[i]
                    assert (left_count + right_count == end - start )
        if left_count == 0 or right_count == 0 or AABBArea([node.min,node.max]) * (end - start ) <= split_cost:
            node.start, node.end = start, end
            return 1
        l, r = start, end-1
        while l < r:
            if self.centroids[self.tidx[l]][split_axis] <= split_pos:   l += 1
            else:  self.tidx[l], self.tidx[r], r = self.tidx[r], self.tidx[l], r - 1
        node.li = node_idx + 1
        lchildren_num = self.Build(node.li, start, start + left_count)
        node.ri = node.li + lchildren_num
        rchildren_num = self.Build(node.ri, start + left_count, end)
        return lchildren_num + rchildren_num + 1

@ti.data_oriented
class Scene:
    def __init__(self,meshes,furnace_test=False,camera = Camera(pos=vec3(2,0.5,0),target=(0,0.5,0))):
        self.furnace,self.maxdepth = furnace_test,11 if not furnace_test else 100
        self.env_Le = vec3(0.5) if furnace_test else vec3(0)
        self.vn = sum([len(m.vertices) for m in meshes])
        self.fn = sum([len(m.faces) for m in meshes])
        self.vertices = ti.Vector.field(3,ti.f64,self.vn)
        self.vertex_normals  = ti.Vector.field(3,ti.f64,self.vn)
        self.vertex_tangents = ti.Vector.field(3,ti.f64,self.vn)
        self.faces = ti.Vector.field(3,ti.i32,self.fn)
        self.face_normals    = ti.Vector.field(3,ti.f64,self.fn) # normal always points to outside
        self.face_tangents   = ti.Vector.field(3,ti.f64,self.fn) # normal always points to outside
        self.face_areas     = ti.field(ti.f64,self.fn)
        self.face_materials = Material.field(shape=self.fn)
        self.face_doubleside  = ti.field(ti.u1,shape=self.fn)
        self.img = ti.Vector.field(3,ti.f64,(WIDTH,HEIGHT))
        self.camera = camera
        voffset,foffset = 0,0
        for mi,m in enumerate(meshes):
            if self.furnace:m.material.albedo,m.material.Le = vec3(1),vec3(0)
            for vi,v in enumerate(m.vertices):
                self.vertices[voffset+vi] = m.vertices[vi]
                self.vertex_normals[voffset + vi] = m.vertex_normals[vi]
                self.vertex_tangents[voffset + vi] = m.vertex_tangents[vi]
            for fi,f in enumerate(m.faces):
                self.faces[foffset+fi] = f+voffset
                self.face_areas[foffset+fi] = m.area_faces[fi]
                self.face_normals[foffset+fi] = m.face_normals[fi]
                self.face_tangents[foffset+fi] = m.face_tangents[fi]
                self.face_materials[foffset+fi] = m.material
                self.face_doubleside[foffset + fi] = not m.is_watertight
            voffset += len(m.vertices)
            foffset += len(m.faces)
        self.bvh = BVH(self.vertices.to_numpy(),self.faces.to_numpy())

    @ti.func
    def HitBy(self,ray):
        t, triidx = NOHIT, INone
        s, i = Array([INone for _ in range(Array.n)]), 0
        s[0] = 0
        while i>=0:
            if i >= Array.n: print("Exceed stack's max size  ")
            node = self.bvh.nodes[s[i]]
            i -= 1
            if node.Leaf():
                for j in range(node.start, node.end ):
                    tri = self.faces[self.bvh.tidx[j]]
                    this_t = ray.HitTriangle(self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]])
                    if this_t < t: t, triidx = this_t, self.bvh.tidx[j]
            else:
                i0, i1 = node.li, node.ri
                n0, n1 = self.bvh.nodes[i0], self.bvh.nodes[i1]
                t0, t1 = ray.HitAABB(n0.min,n0.max), ray.HitAABB(n1.min,n1.max)
                if t0!=NOHIT: s[i+1],i = i0,i+1
                if t1!=NOHIT: s[i+1],i = i1,i+1
        return Interaction(SURFACE,t,triidx,ray).FetchInfo(self)

    @ti.func
    def Scatter(self, ray):
        ret = self.HitBy(ray)
        sigma_maj = ray.mdm.ss
        # sample t ~ sigma_maj * e**(-sigma_maj * t )
        t = -ti.log(1 - ti.random()) / sigma_maj
        if t < ret.t: ret.type, ret.t, ret.fi = VOLUME, t, 0  # 把fi置为0去通过FectchInfo的代码。
        return ret.FetchInfo(self)

    @ti.kernel
    def Draw(self):
        for i,j in self.img:
            o = self.camera.origin + (i+ti.random()/2-1)*self.camera.unit_x + (j+ti.random()/2-1)*self.camera.unit_y
            ray = Ray(o=o,d = (o-self.camera.pos).normalized(),mdm=Air)
            #https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation eq13.4  可以参考13.1.2举的例子, Le + rho_hh( Le + rho_hh( Le +...
            L,beta = vec3(0),vec3(1)
            for _ in range(self.maxdepth):
                ix = self.Scatter(ray)
                if not ix.Valid() :
                    L += beta*self.env_Le
                    break
                if ix.type==SURFACE:
                    L += beta* ix.mat.Le
                    sample = BxDF.Sample(ix)
                    beta *= sample.value*ti.abs(sample.ray.d.dot(ix.normal))/sample.pdf
                    ray = sample.ray
                else:
                    sample = PF.Sample(ix)
                    beta *= sample.value/sample.pdf
                    ray = sample.ray
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

def TestCase(name=''):
    if name=='JadeBunny':
        return Scene([
                    Mesh('./assets/medium/floor.obj', Utils.DiffuseLike(0.3, 0.6, 0.9)),
                    Mesh('./assets/medium/light.obj', Utils.DiffuseLike(0.9, 0.9, 0.9, 1, 1, 1)),
                    Mesh('./assets/medium/bunny.obj', Material(Le=vec3(0),mdm=Jade,type=BxDF.Type.Transmission)),
                    Mesh('./assets/medium/bunny1.obj', Material(Le=vec3(0), mdm=Jade1, type=BxDF.Type.Transmission)),
                ], False,Camera(pos=vec3(0, 2, 15),target=(0, -2, 2.5)))
    else: return Scene([
                    Mesh('./assets/cornell/quad_top.obj', Utils.DiffuseLike(0.9, 0.9, 0.9)),
                    Mesh('./assets/cornell/quad_bottom.obj', Utils.DiffuseLike(0.9)),
                    Mesh('./assets/cornell/quad_left.obj', Utils.DiffuseLike(0.6, 0, 0)),
                    Mesh('./assets/cornell/quad_right.obj', Utils.DiffuseLike(0., 0.6, 0.)),
                    Mesh('./assets/cornell/quad_back.obj', Utils.DiffuseLike(0.9, 0.9, 0.9)),
                    Mesh('./assets/cornell/sphere1.obj', Utils.GlassLike(2.)),
                    Mesh('./assets/cornell/lightSmall.obj', Utils.DiffuseLike(0.9, 0.9, 0.9, 50, 50, 50)),
                ], False)

Film(TestCase('JadeBunny')).Show()