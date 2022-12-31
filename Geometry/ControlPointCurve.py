import taichi as ti
from taichi.math import vec2,vec3
from time import sleep

ti.init(arch = ti.cpu,dynamic_index=True)

bg_color = (0.8,0.8,0.8)
max_ctrl_n,max_n = 100,1000
radius = 0.01
K = 3  # BSpline K
veck = ti.types.vector(K+1,ti.f32)
pylist = [0.0 for _ in range(K+1)]

N = ti.field(ti.u32,1)  # current control point number
p = ti.Vector.field(2,ti.f32,max_ctrl_n) #control point position
knots = ti.field(ti.f32,max_ctrl_n+K*10)    # knots
indices = ti.field(ti.u32,max_ctrl_n*2)
curve       = ti.Vector.field(2,ti.f32,max_n+1)

@ti.func
def Bernstein(n,i,x):
    c0,c1 = 1,1
    for j in range(n-i+1,n+1):c0*=j
    for j in range(1,i+1):    c1*=j
    return c0/c1*ti.pow(x,i)*ti.pow(1.0-x,n-i)

@ti.kernel
def Bezier():
    for j in range(max_n+1):
        pos,x = vec2(0,0),j/max_n
        for i in range(N[0]):
            pos += p[i]* Bernstein(N[0]-1.0,i,x)
        curve[j] = pos

@ti.func
def Basic(i,x):
    ret = 0.0
    if knots[i]<=x<knots[i+K+1]:
        temp = [veck(pylist),veck(pylist)]
        for j in ti.static(range(K+1)):
            if knots[j+i]<=x<knots[j+1+i]: temp[0][j] = 1.0
        for _k in range(1,K+1):
            for j in range(K-_k+1):
                c0 = ( 0.0 if knots[j+_k+i]==knots[j+i]     else (x-knots[j+i])/(knots[j+_k+i]-knots[j+i]) )
                c1 = ( 0.0 if knots[j+_k+i+1]==knots[j+1+i] else (knots[j+_k+i+1]-x)/(knots[j+_k+i+1]-knots[j+1+i]) )
                temp[1][j] = c0 * temp[0][j] + c1 * temp[0][j+1]
            for j in range(K-_k+1): temp[0][j] = temp[1][j]
        ret = temp[0][0]
        if K==0: ret = 1.0
    return ret

# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/
@ti.kernel
def BSpline():
    n = N[0] - 1
    m = n+K+1
    for j in knots:knots[j] = 0.0
    for j in range(n-K+1):knots[j+K] = j/(n-K+1)
    for j in range(m-K,m+1):knots[j] = 1.0

    for j in range(max_n+1):
        pos,x = vec2(0,0),j/max_n
        for i in range(N[0]):
            pos += p[i]* Basic(i,x)
        curve[j] = pos

@ti.kernel
def Initialize():
    N[0] = 2
    for i in p:            p[i] = vec2(-1,-1)
    for i in range(N[0]):  p[i] = vec2(ti.random(),ti.random())

@ti.kernel
def Update():
    for i in indices:indices[i] = max_ctrl_n-1
    for i in range(N[0]-1):
        indices[2*i]   = i
        indices[2*i+1] = i+1

def Main():
    method_idx,methods = 0,{'Bezier':Bezier,'BSpline':BSpline}
    method_names = list(methods.keys())
   
    Initialize()

    window = ti.ui.Window('Curve',res=(1000,1000))
    gui = window.get_gui()
    canvas = window.get_canvas()
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):break
        if window.is_pressed(ti.ui.SPACE) and window.get_event(ti.ui.RELEASE): method_idx = (method_idx+1)%len(method_names)
        pos,selected = window.get_cursor_pos(),None
        for i in range(N[0]):
            if (pos[0]-p[i][0])**2+(pos[1]-p[i][1])**2 < radius**2:
                selected = i
                break
        if window.is_pressed(ti.ui.LMB) :
            if selected!=None: p[selected]=window.get_cursor_pos()
        elif window.is_pressed(ti.ui.RMB):
            p[N[0]] = window.get_cursor_pos()
            N[0] = min(N[0]+1,max_ctrl_n)
            sleep(1/2)
        elif window.is_pressed(ti.ui.MMB):
            if selected!=None:
                N[0] = max(0,N[0]-1)
                for i in range(selected,N[0]):p[i] = p[i+1]
                for i in range(N[0],max_ctrl_n): p[i] = vec2(-1,-1)

        with gui.sub_window(" ", 0, 0, 0.5, 0.1) as w:
            w.text("[Left mouse]- Drag; [Middle mouse]- Delete; [Right mouse]- Add")
            w.text("[Space] - switch curve type.  Current curve: " + method_names[method_idx] + ('' if method_idx==0 else ' degree='+str(K)))
       
        canvas.set_background_color(bg_color)
        if (method_idx==1 and N[0]>=K) or (method_idx==0 and N[0]>1):
            methods[method_names[method_idx]]()
            Update()
            canvas.lines(p,0.005,indices = indices)
        canvas.circles(p,radius,(0,0,0))
        canvas.circles(curve,0.002,color = (37/255, 150/255, 190/255))
        window.show()

Main()