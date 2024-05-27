import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

res = (1000,1000)
n = 4
eps = 1e-7
vecn = ti.types.vector(n,ti.f32)
vec2 = ti.types.vector(2,ti.f32)
vec3 = ti.types.vector(3,ti.f32)

v = ti.Vector.field(2,ti.f32,n)
c = ti.Vector.field(3,ti.f32,n)
idx = ti.Vector.field(2,ti.i32,n)
img = ti.Vector.field(3,ti.f32,res)


def Initialize():
    v.from_numpy(np.array([[0.,0],[0.5,0.5],[1.,0.],[0.5,1.]]))
    c.from_numpy(np.array([[0.,0.,1.],[0.,1,0.],[1.,0.,0.],[1.,1.,0.]]))
    idx.from_numpy(np.array([[0,1],[1,2],[2,3],[3,0]]))

@ti.func
def MeanValueCoordinate(p):
    alpha = vecn([0 for _ in range(n)])
    for i in range(n):
        j = (i+1) if i+1<n else 0
        pi = v[i]-p
        pj = v[j]-p
        alpha[i] = ti.acos(pi.dot(pj)/pi.norm(eps)/pj.norm(eps))
    weights =  vecn([0 for _ in range(n)])
    weightSum = 0.0
    for i in range(n):
        j = (i-1) if i-1>=0 else n-1
        pi = v[i]-p
        weights[i] = (ti.tan(alpha[j]/2)+ti.tan(alpha[i]/2))/pi.norm(eps)
        weightSum += weights[i]
    return weights/weightSum

# use color to visualize the smoothness of MVC interpolation
@ti.kernel
def Visualize():
    for i,j in img:
        x,y = i/res[0],j/res[1]
        coords = MeanValueCoordinate(vec2(x,y))
        img[i,j] = vec3(0,0,0)
        for k in range(n):img[i,j]+=c[k]*coords[k]

def Main():
    Initialize()
    window = ti.ui.Window(name='Mean Value Coordinate', res = res)
    canvas = window.get_canvas()
    Visualize()
    while not window.is_pressed(ti.ui.ESCAPE):
        canvas.set_image(img)
        canvas.lines(v, 0.001, idx, (255,255,255))
        window.show()

Main()