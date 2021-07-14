import taichi as ti
import random,math

from taichi.lang.impl import call_internal

ti.init(arch=ti.cpu)

poly6_factor      = 315.0/(64.0*math.pi)
spiky_grad_factor = -45.0/math.pi
spiky_lap_factor  = -90.0/math.pi

particle_radius = 0.3 
support_radius = 1.1
neighbor_radius = 1.05*support_radius

iteration_num = 5
particle_num_max = 2000
max_neighbor_num = 40
max_particle_in_cell = 100
rho_0 = 1.0
delta_t = 1.0 / 50.0
corrK = 0.001
lambda_epsilon = 100.0
epsilon = 1e-5

temp_conduct_rate = 0.3
melting_point = 70.0

gravity = (0,-9.8)

resolution = (500,500)      
world_to_screen_ratio = 10
boundary = (resolution[0]/10,resolution[1]/10)
grid_cell_size  = 2.51
grid_total_size = (int(boundary[0]/grid_cell_size)+1,int(boundary[1]/grid_cell_size)+1)


pos                = ti.Vector.field(2,dtype= ti.f32, shape=particle_num_max)
last_pos           = ti.Vector.field(2,dtype= ti.f32,shape=particle_num_max)
velocities         = ti.Vector.field(2,dtype= ti.f32,shape=particle_num_max)
pos_deltas         = ti.Vector.field(2,dtype= ti.f32,shape=particle_num_max)
lambdas            = ti.field(dtype = ti.f32,shape=particle_num_max)
state              = ti.field(dtype  = ti.i32,shape = particle_num_max) # 0:FLUID ; 1:SOLID
T                  = ti.field(dtype = ti.f32,shape = particle_num_max)
last_T             = ti.field(dtype = ti.f32,shape = particle_num_max)
rhos               = ti.field(dtype  = ti.f32,shape = particle_num_max)
colors             = ti.field(dtype = ti.i32,shape = particle_num_max)

grid_particle_num       = ti.field(dtype  = ti.i32,shape = grid_total_size )
grid_to_particle        = ti.field(dtype  = ti.i32,shape = (grid_total_size[0],grid_total_size[1],max_particle_in_cell))

neighbor_num      = ti.field(dtype = ti.i32,shape = particle_num_max)
neighbors         = ti.field(dtype  = ti.i32,shape=(particle_num_max,max_neighbor_num))

grid_index_offset = ti.Vector.field(2,dtype = ti.i32,shape = 9)
particle_num      = ti.field(dtype = ti.i32,shape = ())

@ti.func
def HSV2RGB(h, s, v):
    ret = 0
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        ret = int(v*255)*2**16+int(t*255)*2**8+int(p*255)
    elif i == 1:
        ret = int(q*255)*2**16+int(v*255)*2**8+int(p*255)
    elif i == 2:
        ret = int(p*255)*2**16+int(v*255)*2**8+int(t*255)
    elif i == 3:
        ret = int(p*255)*2**16+int(q*255)*2**8+int(v*255)
    elif i == 4:
        ret = int(t*255)*2**16+int(p*255)*2**8+int(v*255)
    elif i == 5:
        ret = int(v*255)*2**16+int(p*255)*2**8+int(q*255)
    if s == 0.0:
        ret = int(v*255)*2**16+int(v*255)*2**8+int(v*255)
    return ret

@ti.func
def GenerateSolidParticle(x, y, delta):
    rx = (random.random()+1)/4
    ry = (random.random()+1)/4
    start_x = rx*particle_radius
    start_y = ry*particle_radius
    for i in range(int((x-start_x)/delta)+1):
        for j in range(int((y-start_y)/delta)+1):
            k           =  ti.atomic_add(particle_num[None],1)
            pos[k]      =  ti.Vector([40+start_x+delta*i,start_y+delta*j])
            last_pos[k] = pos[k]
            T[k]        =  -50.0
            last_T[k]   =  T[k]
            state[k]    =  1
            velocities[k] = ti.Vector([0.0,0.0])
            pos_deltas[k] = ti.Vector([0.0,0.0])
    

@ti.func
def ConfinePositionToBoundary(p):
    right = boundary[0] - particle_radius
    top   = boundary[1] - particle_radius
    left  = 0 + particle_radius
    bot   = 0 + particle_radius
    if p[0] < left:
        p[0] = left + ti.random()*epsilon
    elif p[0] >right :
        p[0] = right - ti.random()*epsilon
    if p[1]<bot:
        p[1] = bot + ti.random()*epsilon
    elif p[1]>top:
        p[1] = top - ti.random()*epsilon
    return p

@ti.func
def Poly6(r,h):
    ret = 0.0
    if 0.0 < r < h:
        x = (h*h - r*r)/(h*h*h)
        ret = poly6_factor*x*x*x
    return ret
    
@ti.func
def SpikyGradient(r,h):
    ret = ti.Vector([0.0,0.0])
    rlen = r.norm()
    if 0.0 < rlen < h:
        x = (h-rlen)/(h*h*h)
        ret = spiky_grad_factor*x*x*r/rlen
    return ret

@ti.func
def SpikyLaplace(r,h):
    ret = 0.0
    if 0.0 < r < h:
        ret = spiky_lap_factor*(r/h-1)/(h*h*h*h*h)
    return ret

@ti.func
def ComputeScorr(pos_ji):
    x = Poly6(pos_ji.norm(),support_radius)/Poly6(0.3*support_radius,support_radius)
    x = x*x
    x = x*x
    return (-corrK)*x

@ti.kernel
def Initialize():
    particle_num[None] = 0
    GenerateSolidParticle(2,40,particle_radius*2)
    for i in range(9):
        grid_index_offset[i] = ti.Vector([i//3-1,i%3-1])

@ti.kernel
def PopFluid():
    init_pos = (particle_radius,40)
    for i in range(5):
        j = ti.atomic_add(particle_num[None],1)
        velocities[j] = ti.Vector([20,0])
        pos_deltas[j] = ti.Vector([0.0,0.0])
        pos[j]        = ti.Vector([init_pos[0],init_pos[1]+i*particle_radius*2.5])
        last_pos[j]   = pos[j]
        T[j]          = 100.0  
        last_T[j]     = T[j]
        lambdas[j]    = 0.0
        state[j]      = 0

@ti.kernel
def Prologue():
    # melting
    for i in range(particle_num[None]):
        if T[i] > melting_point and state[i] == 1:
            state[i] = 0

    #predict fluid position
    for i in range(particle_num[None]):
        if state[i] == 1: continue
        pos_i,vel_i = pos[i],velocities[i]
        vel_i += delta_t*ti.Vector([1.0,1.0])*gravity
        pos_i += delta_t*vel_i
        pos[i] = ConfinePositionToBoundary(pos_i)
    
    # update grid
    for I in ti.grouped(grid_particle_num):
        grid_particle_num[I] = 0
    for i in range(particle_num[None]):
        cell_i = int(pos[i]/grid_cell_size)
        oldValue = ti.atomic_add(grid_particle_num[cell_i],1)
        grid_to_particle[cell_i,oldValue] = i
    
    #find neighbors for both fluid and solid particle
    for i in range(particle_num[None]):
        pos_i = pos[i]
        cell_i = int(pos_i/grid_cell_size)
        neighbor_cnt = 0
        for offseti in range(9):
            neighborCell = cell_i + grid_index_offset[offseti]
            if neighborCell[0]<0 or neighborCell[0]>=grid_total_size[0] or neighborCell[1]<0 or neighborCell[1]>=grid_total_size[1]:
                continue
            for j_index in range(grid_particle_num[neighborCell]):
                j = grid_to_particle[neighborCell,j_index]
                pos_j = pos[j]
                if (pos_i-pos_j).norm() < neighbor_radius and i!=j and neighbor_cnt<max_neighbor_num:
                    neighbors[i,neighbor_cnt]=j
                    neighbor_cnt+=1
        neighbor_num[i] = neighbor_cnt                  
    
    # advect attribute i.e.temperature
    for i in range(particle_num[None]):
        lap = 0.0
        pos_i = pos[i]
        for nj in range(neighbor_num[i]):
            j = neighbors[i,nj]
            lap += SpikyLaplace((pos_i-pos[j]).norm(),support_radius)*(last_T[j]-last_T[i])
        T[i] = temp_conduct_rate*delta_t* lap/rho_0 + last_T[i]
    
@ti.kernel
def ComputeLambdas():
    for i in range(particle_num[None]):
        if state[i]==1:
            continue
        pos_i = pos[i]
        grad_sqr_sum = 0.0
        grad_i = ti.Vector([0.0,0.0])
        rho_i = 0.0
        for j in range(neighbor_num[i]):
            pos_j = pos[neighbors[i,j]]
            pos_ji = pos_i - pos_j
            grad_j = SpikyGradient(pos_ji,support_radius)/rho_0
            grad_sqr_sum+=grad_j.dot(grad_j)
            grad_i += grad_j
            rho_i  += Poly6(pos_ji.norm(),support_radius)
        C_i = rho_i/rho_0 - 1.0
        if C_i < 0.0: C_i = 0.0
        grad_sqr_sum += grad_i.dot(grad_i)
        lambdas[i] = (-C_i)/(grad_sqr_sum + lambda_epsilon)
        rhos[i] = rho_i

@ti.kernel
def ComputeDeltaPosition():
    for p_i in range(particle_num[None]):
        if state[p_i]==1:
            continue
        pos_i       = pos[p_i]
        lambda_i    = lambdas[p_i]
        delta_pos_i = ti.Vector([0.0,0.0])
        for j in range(neighbor_num[p_i]):
            p_j = neighbors[p_i,j]
            pos_ji = pos_i-pos[p_j]
            factor = lambda_i
            if state[p_j]==0:
                factor += lambdas[p_j]+ComputeScorr(pos_ji)
            delta_pos_i+=  factor*SpikyGradient(pos_ji,support_radius)
        pos_deltas[p_i] = delta_pos_i/rho_0

@ti.kernel
def SolidFluidCollision():
    for i in range(particle_num[None]):
        if state[i] == 1: continue
        pos_i = pos[i]
        delta_pos_i = ti.Vector([0.0,0.0])
        length = 2*particle_radius
        for pj in range(neighbor_num[i]):
            j      = neighbors[i,pj]
            if state[j]==0: continue
            pos_ji = pos_i - pos[j]
            norm = pos_ji.norm()
            if norm<length:
                delta_pos_i+=pos_ji*(length/norm-1)*0.5
        pos_deltas[i]+=delta_pos_i

@ti.kernel
def ApplyDeltaPosition():
    for i in range(particle_num[None]):
        pos[i]+=pos_deltas[i]

@ti.kernel
def Epilogue():
    for i in range(particle_num[None]):
        pos[i]        = ConfinePositionToBoundary(pos[i])
        velocities[i] = (pos[i]-last_pos[i])/delta_t
        last_pos[i]   = pos[i]
        colors[i]     = HSV2RGB(0.5-0.5*T[i]*0.01,1,1)
        last_T[i]     = T[i]

def LogInfo():
    import numpy as np
    print(np.max(T.to_numpy()[:particle_num[None]]),np.min(T.to_numpy()[:particle_num[None]]))

def Simulate():
    Prologue()
    for _ in range(iteration_num):
        ComputeLambdas()
        ComputeDeltaPosition()
        SolidFluidCollision()
        ApplyDeltaPosition()
    Epilogue()
    # LogInfo()

def Main():
    Initialize()
    gui = ti.GUI('Position Based Fluid - Melting',res = resolution,background_color= 0xffffff)
    while not gui.get_event(gui.SPACE) :
        if gui.frame<=(1500*2/5) and gui.frame%2==0:
            PopFluid()
        Simulate()
        gui.circles(pos.to_numpy()[:particle_num[None]]/boundary,color=colors.to_numpy()[:particle_num[None]], radius=particle_radius*world_to_screen_ratio)
        gui.show()

Main()