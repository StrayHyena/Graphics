import taichi as ti
from taichi.math import vec2,vec3
from time import sleep

ti.init(arch = ti.cpu,dynamic_index=True)

bg_color = (0.8,0.8,0.8)
max_ctrl_n,max_n = 100,10*2**10
radius = 0.01
CHAIKIN,BSPLINE3,POINT4=0,1,2
detail_names = ['Chaikin','3-order uniform BSpline','4-point interpolatory']

N       = ti.field(ti.i32,1)  # current control point number
p       = ti.Vector.field(2,ti.f32,max_ctrl_n) #control point position
colors  = ti.Vector.field(3,ti.f32,max_ctrl_n)
indices = ti.field(ti.u32,max_ctrl_n*2)

temp0   = ti.Vector.field(2,ti.f32,max_n)
temp1   = ti.Vector.field(2,ti.f32,max_n)
curve   = ti.Vector.field(2,ti.f32,max_n)

@ti.kernel
def Initialize():
    N[0] = 4
    for i in p:            p[i] = vec2(-1,-1)
    for i in range(N[0]):  p[i] = vec2(ti.random(),ti.random())


@ti.kernel
def Prologue():
    for i in temp0: temp0[i] = vec2(-1,-1)
    for i in p:     temp0[i] = p[i]

@ti.kernel
def Subdivision(n:ti.i32, method:ti.u32,alpha:ti.f32):
    a_1,a0,a1, b_1,b0,b1,b2 = 0.25,0.75,0.0,  0.0,0.75,0.25,0.0
    if method==BSPLINE3: a_1,a1, b0,b1        = 0.125,0.125, 0.5,0.5
    if method==POINT4:   a_1,a0, b_1,b0,b1,b2 = 0.0,1.0,     -alpha/2,(1+alpha)/2,(1+alpha)/2,-alpha/2 
    for i in range(n):temp1[i] = temp0[i]
    for i in range(n):
        temp0[2*i]   = a_1*temp1[(i-1+n)%n] + a0*temp1[i] + a1*temp1[(i+1)%n]
        temp0[2*i+1] = b_1*temp1[(i-1+n)%n] + b0*temp1[i] + b1*temp1[(i+1)%n]+ b2*temp1[(i+2)%n]
       
@ti.kernel
def Epilogue():
    for i in curve:        curve[i] = temp0[i]
    for i in indices:indices[i] = max_ctrl_n-1
    for i in range(N[0]):
        indices[2*i]   = i
        indices[2*i+1] = i+1
    indices[2*N[0]-1] = 0
    for i in colors:       colors[i] = vec3(0,0,0)
    colors[0] =colors[N[0]-1]  = vec3(0.9,0.4,0.3)

def Main():
    Initialize()
    method = BSPLINE3
    alpha  = 1/16
    window = ti.ui.Window('Curve',res=(1000,1000))
    gui = window.get_gui()
    canvas = window.get_canvas()
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):break
        if window.is_pressed(ti.ui.SPACE) and window.get_event(ti.ui.RELEASE): method = (method+1)%3
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
                N[0] = max(1,N[0]-1)
                for i in range(selected,N[0]):p[i] = p[i+1]
                for i in range(N[0],max_ctrl_n): p[i] = vec2(-1,-1)

        with gui.sub_window(" ", 0, 0, 0.5, 0.1) as w:
            w.text("[Left mouse]- Drag; [Middle mouse]- Delete; [Right mouse]- Add")
            w.text("[Space]- switch subdivision type. Current: "+detail_names[method] )
            alpha = w.slider_float("alpha ",alpha, 0.0, 0.8)
        
        Prologue()
        for n in [N[0]*2**i for i in range(100)]:
            if 2*n >= max_n:break
            Subdivision(n,method,alpha)
        Epilogue()

        canvas.set_background_color(bg_color)
        canvas.lines(p,0.005,indices = indices)
        canvas.circles(p,radius,per_vertex_color=colors)
        canvas.circles(curve,0.0025,color = (37/255, 150/255, 190/255))
        window.show()

Main()