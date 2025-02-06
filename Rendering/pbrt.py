import taichi as ti
import taichi.math as tm
import numpy as np
import trimesh,enum

# ,debug=True,cpu_max_num_threads = 1
ti.init(arch=ti.cpu,default_fp  =ti.f64,debug=True)
Array   = ti.types.vector( 200  ,ti.i32)
vec3    = ti.types.vector(3,ti.f64)
vec3i   = ti.types.vector(3, ti.i32)
Medium   = ti.types.struct(eta=ti.f64,k=ti.f64)
Material = ti.types.struct( albedo=vec3,mdm=Medium,type=ti.i32,alpha=ti.f64)

MAXDEPTH = 100
WIDTH,HEIGHT = 400,400
EPS = 1e-8
MAX = 1.7976931348623157e+308
NOHIT = MAX
MAX3 = vec3(MAX,MAX,MAX)
FNone,  INone,  RR,     ENUM_LIGHT = -1.0,  -1,     1.0,       0
FURNACETEST = False
# https://refractiveindex.info/  R 630 nm ,G 532 nm ,B 465 nm
Air,Glass,Gold = Medium(eta=1,k=0),Medium(eta=1.5,k=0),Medium(eta=0.47,k=2.83)

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

@ti.dataclass
class Intersection:
    t:ti.f64
    fi:ti.i32
    ray:Ray
    normal:vec3
    tangent:vec3
    pos:vec3
    mat:Material
    isRayInAir:ti.u1
    mdmT:Medium
    @ti.func
    def HasValue(self):
        return self.fi!=INone or self.t!=NOHIT
    @ti.func
    def FetchInfo(self,scene):
        if self.HasValue():
            self.pos = self.ray.At(self.t)
            face,area = scene.faces[self.fi],scene.face_areas[self.fi]
            ws = vec3(0)
            for i in ti.static(range(3)):
                j,k = (i+1)%3,(i+2)%3
                ws[i] = (self.pos-scene.vertices[face[j]]).cross(self.pos-scene.vertices[face[k]]).norm()/area
            self.normal = (ws[0]*scene.vertex_normals[face[0]]+ws[1]*scene.vertex_normals[face[1]]+ws[2]*scene.vertex_normals[face[2]]).normalized()
            # self.normal = scene.face_normals[self.fi]
            self.mat = scene.face_materials[self.fi]
            axis = 0
            while self.normal[axis] == 0: axis += 1
            assert self.normal[axis] != 0
            tangent = vec3(0)
            tangent[axis] = self.normal[(axis + 1) % 3]
            tangent[(axis + 1) % 3] = - self.normal[axis]
            self.tangent = tangent.normalized()
            self.isRayInAir = True if self.ray.d.dot(-self.normal) > 0 else False # NOTE: mesh's normal always points outside
            self.mdmT = self.mat.mdm  # transmission medium
            if not self.isRayInAir:self.mdmT = Air
        return self

class BxDF:
    class Type(enum.IntEnum):
        Lambertian = enum.auto() # reflection_diffuse
        Specular = enum.auto()   # reflection_specular
        Transmission = enum.auto() # reflection_specular  transmission_specular
        Microfacet  = enum.auto()  # reflection_glossy
    Sample = ti.types.struct(pdf=ti.f64,ray=Ray,bxdf_value=vec3)
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
        p[1] = (1+wh.y)*0.5*(1-p[1])+h*p[1]
        pz = ti.sqrt(ti.max(0,1-p[0]**2+p[1]**2))
        nh = p[0]*t1+pz*wh+p[1]*t2
        return vec3(ax*nh[0],ti.max(0,nh[1]),az*nh[2]).normalized()
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
    @ti.func
    def Fresnel(i, n, etaI, etaT):
        cosI = ti.max(0,0,ti.min(1,i.dot(n)))
        cosT = ti.sqrt(ti.max(0.0, 1 - (etaI / etaT) ** 2 * (1 - cosI ** 2)))
        r_parl = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT)
        r_perp = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT)
        return 0.5 * (r_parl ** 2 + r_perp ** 2)
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.16)
    def D_GGX(cos_theta, alpha):return alpha ** 2 / (tm.pi * (cos_theta ** 2 * (alpha ** 2 - 1) + 1) ** 2)
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.20)
    def Lambda(cos_theta, alpha):return 0.5 * (ti.sqrt(1 + alpha * alpha * (1.0 / cos_theta / cos_theta - 1)) - 1)
    @ti.func  # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.22)
    def G_GGX(cos_thetaI, cos_thetaO, alpha):return 1.0 / (1.0 + BxDF.Lambda(cos_thetaI, alpha) + BxDF.Lambda(cos_thetaO, alpha))
    @ti.func
    def G1_GGX(cos_theta, alpha):return 1.0 / (1.0 + BxDF.Lambda(cos_theta, alpha) )
    @ti.func
    def D(w,wm,a):return BxDF.G1_GGX(w.y,a)/ti.abs(w.y)*BxDF.D_GGX(wm.y,a)*ti.abs(w.dot(wm))
    @ti.func
    def f(wi:vec3,wo:vec3, ix:Intersection):
        ret,mat,mdmI,mdmT = vec3(0),ix.mat,ix.ray.mdm,ix.mdmT
        if FURNACETEST: ix.mat.albedo = vec3(1)
        if ix.mat.type==BxDF.Type.Lambertian:ret = ix.mat.albedo/tm.pi
        elif ix.mat.type==BxDF.Type.Specular:
            ret = vec3(1)/ti.abs(wo.y)
        elif ix.mat.type==BxDF.Type.Transmission:ret = vec3(1)/ti.abs(wo.y)
        elif ix.mat.type==BxDF.Type.Microfacet:
            h = (wo + wi).normalized()
            ret = BxDF.D_GGX(h.y, mat.alpha) * BxDF.G_GGX(wi.y, wo.y, mat.alpha) * BxDF.Fresnel(wi, h, mdmI.eta, mdmT.eta) / ti.abs(4 * wi.y * wo.y)
        return ret
    @ti.func
    def SampleF(ix:Intersection):
        N,T,n = ix.normal,ix.tangent,vec3(0,1,0)  # N: world normal, T: world tangent, n: local normal
        wi,wo,pdf,f = BxDF.ToLocal(-ix.ray.d,N,T).normalized(),vec3(0),0.,MAX3 # use MAX3 to expose problem
        isNextRayInAir = ix.isRayInAir
        if FURNACETEST: ix.mat.albedo = vec3(1)
        if ix.mat.type==BxDF.Type.Lambertian:
            wo = BxDF.CosineSampleHemisphere()
            pdf = wo.y/tm.pi
            f = ix.mat.albedo/tm.pi
        elif ix.mat.type==BxDF.Type.Specular:
            wo,pdf = BxDF.Reflect(wi,n),1
            f = vec3(1)/ti.abs(wo.y) # https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF  eq(9.9)
        elif ix.mat.type==BxDF.Type.Transmission:
            cosI,eta = wi.dot(n), ix.mdmT.eta/ix.ray.mdm.eta
            if cosI<0:cosI,n = -cosI,-n
            wo =  (-wi/eta + (cosI/eta - ti.sqrt(1 - (1/eta)**2 * (1 -cosI**2))) * n).normalized()
            if ti.random()<BxDF.Fresnel(wi,n,ix.ray.mdm.eta,ix.mdmT.eta):
                wo = BxDF.Reflect(wi,n)
            else: isNextRayInAir = not isNextRayInAir
            pdf,f = 1,vec3(1)/ti.abs(wo.y)
        elif ix.mat.type==BxDF.Type.Microfacet:
            wh = BxDF.Sample_wm(wi,[ti.random(),ti.random()],ix.mat.alpha,ix.mat.alpha)
            wo = BxDF.Reflect(wi, wh)
            pdf = BxDF.D(wo,wh,ix.mat.alpha)/4/wo.dot(wh)
            # https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory eq(9.33)
            f = BxDF.D_GGX(wh.y, ix.mat.alpha) * BxDF.G_GGX(wi.y, wo.y, ix.mat.alpha) * BxDF.Fresnel(wi, wh, ix.ray.mdm.eta, ix.mdmT.eta) / ti.abs(4 * wi.y * wo.y)
            assert pdf>0
            # print(wh,pdf,f)
            # print(pdf)
        nextRayO,nextRayMdm = ix.pos+ix.normal*EPS, Air
        if not isNextRayInAir:  nextRayO,nextRayMdm = ix.pos - ix.normal*EPS,   ix.mat.mdm
        return BxDF.Sample(pdf=pdf,  ray=Ray(o=nextRayO,d=BxDF.ToWorld(wo,N,T).normalized(),mdm=nextRayMdm), bxdf_value=f)

class Mesh(trimesh.Trimesh):
    def __init__(self,objpath,material):
        mesh = trimesh.load(objpath)
        super().__init__(vertices=mesh.vertices,faces=mesh.faces)
        self.material = material

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
    def __init__(self,meshes):
        self.vn = sum([len(m.vertices) for m in meshes])
        self.fn = sum([len(m.faces) for m in meshes])
        self.vertices = ti.Vector.field(3,ti.f64,self.vn)
        self.vertex_normals = ti.Vector.field(3,ti.f64,self.vn)
        self.faces = ti.Vector.field(3,ti.i32,self.fn)
        self.face_normals   = ti.Vector.field(3,ti.f64,self.fn) # normal always points to outside
        self.face_areas     = ti.field(ti.f64,self.fn)
        self.face_materials = Material.field(shape=self.fn)
        self.img = ti.Vector.field(3,ti.f64,(WIDTH,HEIGHT))
        self.camera = Camera(pos=vec3(2,0.5,0),target=(0,0.5,0))
        voffset,foffset = 0,0
        for mi,m in enumerate(meshes):
            for vi,v in enumerate(m.vertices):
                self.vertices[voffset+vi] = m.vertices[vi]
                self.vertex_normals[voffset + vi] = m.vertex_normals[vi]
            for fi,f in enumerate(m.faces):
                self.faces[foffset+fi] = f+voffset
                self.face_areas[foffset+fi] = m.area_faces[fi]
                self.face_normals[foffset+fi] = m.face_normals[fi]
                self.face_materials[foffset+fi] = m.material
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
        return Intersection(t,triidx,ray).FetchInfo(self)

    @ti.kernel
    def Draw(self):
        for i,j in self.img:
            o = self.camera.origin + (i+ti.random()/2-1)*self.camera.unit_x + (j+ti.random()/2-1)*self.camera.unit_y
            ray = Ray(o=o,d = (o-self.camera.pos).normalized(),mdm=Air)
            ret = vec3(1)
            for k in range(MAXDEPTH+1):
                ix = self.HitBy(ray)
                if ti.random() > RR or not ix.HasValue() or ix.mat.type==ENUM_LIGHT or k==MAXDEPTH:
                    if not ix.HasValue(): ret*= (vec3(0.3,0.6,0.9) if FURNACETEST else vec3(0))
                    elif ix.mat.type==ENUM_LIGHT: ret*=(ix.mat.albedo/RR)
                    else: ret*=vec3(0)
                    break
                sample = BxDF.SampleF(ix)
                ray = sample.ray
                ret *= sample.bxdf_value*ti.abs(ray.d.dot(ix.normal))/sample.pdf/RR
            self.img[i,j] = ret

            # self.img[i,j] = ti.Vector([0,0,0])
            # ix = self.HitBy(ray)
            # t,triidx = MAX,INone
            # for fi in range(self.fn):
            #     tri = self.faces[fi]
            #     t1 = ray.HitTriangle(self.vertices[tri[0]],self.vertices[tri[1]],self.vertices[tri[2]])
            #     if t1<t:t,triidx = t1,fi
            # if t!=ix.t:print(i,j,t,ix.t)
            # assert t==ix.t
            # if ix.HasValue() : self.img[i,j] = ix.mat.albedo

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

Film(Scene([
    Mesh('./assets/Cornell/quad_top.obj',       Material(albedo=vec3(0.9),mdm=Air,type=BxDF.Type.Lambertian)),
    Mesh('./assets/Cornell/quad_bottom.obj',    Material(albedo=vec3(0.9),mdm=Gold,type=BxDF.Type.Microfacet,alpha = 1)),
    Mesh('./assets/Cornell/quad_left.obj',      Material(albedo=vec3(0.6, 0, 0),mdm=Air,type=BxDF.Type.Lambertian)),
    Mesh('./assets/Cornell/quad_right.obj',     Material(albedo=vec3(0., 0.6, 0.),mdm=Air,type=BxDF.Type.Lambertian)),
    Mesh('./assets/Cornell/quad_back.obj',      Material(albedo=vec3(0.9),mdm=Air,type=BxDF.Type.Lambertian)),
    Mesh('./assets/Cornell/lightSmall.obj',     Material(albedo=vec3(50),mdm=Air,type=ENUM_LIGHT)),
    Mesh('./assets/Cornell/sphere.obj',          Material(albedo=vec3(0.3,0.6,0.9),mdm=Glass,type=BxDF.Type.Lambertian)),
])).Show()