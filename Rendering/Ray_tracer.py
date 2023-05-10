import taichi as ti
import math

ti.init(arch = ti.gpu)

#------------helper function-------------------------------------------
# theta = rand(0,2pi)  fpi= arccos(rand(-1,1)) 
@ti.func
def RandomOnUnitSphere():
    theta = 2*math.pi*ti.random()
    cosPhi = 1-2*ti.random()
    sinPhi = ti.sqrt(1-cosPhi*cosPhi)
    return ti.Vector([ti.cos(theta)*sinPhi,ti.sin(theta)*sinPhi,cosPhi])
    
# i,v must be normalized
@ti.func
def Reflect(i,n):return i-2*n*i.dot(n)

@ti.func
def Refract(i,n,ratio):
    cosTheta = -i.dot(n)
    return ratio*i+(ratio*cosTheta-ti.sqrt(1-ratio**2*(1-cosTheta**2)))*n

@ti.func
def Reflectance(cosine,ratio):
    r0 = (1-ratio)/(1+ratio)
    r0 = r0*r0
    return r0+(1-r0)*pow((1-cosine),5)

#------------CONSTANTS-------------------------------------------
LIGHT,DIFFUSE,GLASS,METAL = 0,1,2,3
RES = (800,800)
P_RR,N_GLASS,N_GLASS_INV = 0.8,3/2,2/3

@ti.data_oriented
class Ray:
    def __init__(self,o,d) :
        self.origin = o
        self.direction = d
    def At(self,t):
        return self.origin+t*self.direction

@ti.data_oriented
class Sphere:
    def __init__(self,center,radius,material,color):
        self.position = center
        self.radius   = radius
        self.material = material
        self.color    = color

    @ti.func
    def Hit(self,ray,tmin,tmax,):
        isHit,t,hitNormal,hitPoint = False,0.0,ti.Vector([0.0,0.0,0.0]),ti.Vector([0.0,0.0,0.0])
        co    = ray.origin-self.position
        a     = ray.direction.dot(ray.direction)
        b     = 2.0 * ray.direction.dot(co)
        c     = co.dot(co) - self.radius**2
        delta = b**2-4*a*c
        if delta>0:
            t = (-b-ti.sqrt(delta))/(2*a)
            if t<tmin or t>tmax:
                t = (-b+ti.sqrt(delta))/(2*a)
                if tmin<=t<=tmax: isHit = True
            else:isHit = True
        if isHit:
            hitPoint  = ray.At(t)
            hitNormal = (hitPoint-self.position)/self.radius
        return isHit,t,hitNormal,hitPoint,self.material,self.color

@ti.data_oriented
class Scene:
    def __init__(self):self.objects = []

    def Add(self,o):self.objects.append(o)
    
    @ti.func
    def Hit(self,ray,tmin = 1e-4,tmax = 10e8):
        isHit,tCloest,hitMat = False,tmax,-1
        hitColor,hitPoint,hitNormal  = ti.Vector([0.0,0.0,0.0]),ti.Vector([0.0,0.0,0.0]),ti.Vector([0.0,0.0,0.0])
        for index in ti.static(range(len(self.objects))):
            is_hit,t,hit_normal,hit_point,hit_mat,hit_color = self.objects[index].Hit(ray,tmin,tCloest)
            if is_hit: isHit,hitColor,hitNormal,hitPoint,hitMat,tCloest = True,hit_color,hit_normal,hit_point,hit_mat,t
        return isHit,hitNormal,hitPoint,hitMat,hitColor

@ti.data_oriented
class Camera:
    def __init__(self,scene,res = (400,400),fov = 60):
        self.resolution=res
        self.image      = ti.Vector.field(3,dtype = ti.f32,shape = res)
        self.position   = ti.Vector.field(3,dtype = ti.f32,shape = ())
        self.aimTarget  = ti.Vector.field(3,dtype = ti.f32,shape = ())
        self.lowerLeft  = ti.Vector.field(3,dtype = ti.f32,shape = ())
        self.verticle   = ti.Vector.field(3,dtype = ti.f32,shape = ())
        self.horizontal = ti.Vector.field(3,dtype = ti.f32,shape = ())
        self.renderCnt  = 0
        self.fov = fov*math.pi/180
        self.scene = scene
        self.Initialize()

    @ti.kernel
    def Initialize(self):
        self.position[None] = [0.0,1.0,-5.0]
        self.aimTarget[None] = [0.0,1.0,-1.0]
        up = ti.Vector([0.0,1.0,0.0])
        direction = (self.aimTarget[None] - self.position[None]).normalized()
        u = direction.cross(up).normalized()
        v = u.cross(direction).normalized()
        halfH = ti.tan(self.fov/2)
        halfW = halfH*self.resolution[0]/self.resolution[1]
        self.lowerLeft[None] = self.position[None]+direction-u*halfW-v*halfH
        self.horizontal[None] = u*halfW*2/self.resolution[0]
        self.verticle[None]   = v*halfH*2/self.resolution[1]

    @ti.kernel
    def Render(self):
        for i,j in self.image:
            rayOrigin    = self.position[None]
            rayDirection = self.lowerLeft[None]+(i+ti.random()*0.5-1)*self.horizontal[None]+(j+ti.random()*0.5-1)*self.verticle[None]-self.position[None]
            colorAccumulator = ti.Vector([1.0,1.0,1.0])
            while True:
                if ti.random()>P_RR:
                    colorAccumulator = ti.Vector([0.0,0.0,0.0])
                    break
                isHit,hitNormal,hitPoint,hitMat,hitColor = self.scene.Hit(Ray(rayOrigin,rayDirection))
                if not isHit:
                    colorAccumulator = ti.Vector([0.0,0.0,0.0])
                    break
                colorAccumulator*=(hitColor/P_RR)
                rayOrigin    = hitPoint
                if hitMat == LIGHT:     break
                elif hitMat == DIFFUSE: rayDirection = hitNormal+RandomOnUnitSphere() 
                elif hitMat == METAL:   rayDirection = Reflect(rayDirection,hitNormal)
                elif hitMat == GLASS:
                    n = N_GLASS_INV
                    rayDirection = rayDirection.normalized()
                    cosThetaIn = hitNormal.dot(rayDirection)
                    if cosThetaIn > 0: hitNormal,n = -hitNormal,N_GLASS
                    if ti.sqrt(1-cosThetaIn**2)*n >= 1.0 or ti.random()< Reflectance(ti.abs(cosThetaIn),n): 
                        rayDirection = Reflect(rayDirection,hitNormal)
                    else:
                        rayDirection = Refract(rayDirection,hitNormal,n)
            self.image[i,j] += colorAccumulator

def Main():
    gui = ti.GUI('Ray Tracer',res = RES)
    scene = Scene()
    scene.Add(Sphere(center=ti.Vector([0, 5.4, -1]),    radius=3.0,   material=LIGHT,   color=ti.Vector([10.0, 10.0, 10.0])))
    scene.Add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=DIFFUSE, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.Add(Sphere(center=ti.Vector([0, 102.5, -1]),  radius=100.0, material=DIFFUSE, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.Add(Sphere(center=ti.Vector([0, 1, 101]),     radius=100.0, material=DIFFUSE, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.Add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=DIFFUSE, color=ti.Vector([0.6, 0.0, 0.0])))
    scene.Add(Sphere(center=ti.Vector([101.5, 0, -1]),  radius=100.0, material=DIFFUSE, color=ti.Vector([0.0, 0.6, 0.0])))
    scene.Add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3,   material=DIFFUSE, color=ti.Vector([0.8, 0.3, 0.3])))
    scene.Add(Sphere(center=ti.Vector([0.7, 0, -0.5]),  radius=0.5,   material=GLASS,   color=ti.Vector([1.0, 1.0, 1.0])))
    scene.Add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7,   material=METAL,   color=ti.Vector([0.6, 0.8, 0.8])))
    camera = Camera(scene,res = RES)
    while not gui.get_event(ti.GUI.ESCAPE):
        camera.renderCnt+=4
        for _ in range(4):camera.Render()
        gui.set_image(camera.image.to_numpy()/camera.renderCnt)
        gui.show()

Main()