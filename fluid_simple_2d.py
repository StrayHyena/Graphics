import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

poly6_factor = 315.0/(64.0*math.pi)
spiky_grad_factor = -45.0/math.pi

particle_radius = 0.3 #0.3m
support_radius = 1.1
neighbor_radius = 1.05*support_radius

iteration_num = 5
particle_num_x = 60  
particle_num =particle_num_x*20
max_neighbor_num = 40
max_particle_in_cell = 100
rho_0 = 1.0
delta_t = 1.0 / 20.0
corrK = 0.001
lambda_epsilon = 100.0
epsilon = 1e-5

gravity = (0,-9.8)
resolution = (800,400) # 80m*40m
world_to_screen_ratio = 10
boundary = (resolution[0]/10,resolution[1]/10)
grid_cell_size  = 2.51
grid_total_size = (int(boundary[0]/grid_cell_size)+1,int(boundary[1]/grid_cell_size)+1)

pos       = ti.Vector.field(2,dtype= ti.f32, shape=particle_num)
last_pos  = ti.Vector.field(2,dtype= ti.f32,shape=particle_num)
velocities = ti.Vector.field(2,dtype= ti.f32,shape=particle_num)
pos_deltas = ti.Vector.field(2,dtype= ti.f32,shape=particle_num)
lambdas    = ti.field(dtype = ti.f32,shape=particle_num)

neighbors  = ti.field(dtype  = ti.i32,shape=(particle_num,max_neighbor_num))
neighbor_num  = ti.field(dtype = ti.i32,shape = particle_num)
grid_particle_num = ti.field(dtype  = ti.i32,shape = grid_total_size )
grid_to_particle  = ti.field(dtype  = ti.i32,shape = (grid_total_size[0],grid_total_size[1],max_particle_in_cell))

grid_index_offset = ti.Vector.field(2,dtype = ti.i32,shape = 9)
timeCnt    = ti.field(dtype  = ti.f32,shape = ())
rhos       = ti.field(dtype  = ti.f32,shape = particle_num)

@ti.func
def ConfinePositionToBoundary(p,tc):   
    right = 65.0 + 10.0*ti.sin(0.25*math.pi*tc) #tc : time count
    left = 0 + particle_radius
    bot  = 0 + particle_radius
    top  = boundary[1] - particle_radius
    if p[0] < left:
        p[0] = left + ti.random()*epsilon
    elif p[0] >right :
        p[0] = right - ti.random()*epsilon
    if p[1]<bot:
        p[1] = bot+ti.random()*epsilon
    elif p[1] > top:
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

@ti.kernel
def Initialize():
    for I in ti.grouped(grid_particle_num):
        grid_particle_num[I] = 0
    for I in ti.grouped(grid_to_particle):
        grid_to_particle[I] = 0
    for i in range(9):
        grid_index_offset[i] = ti.Vector([i//3-1,i%3-1])

    timeCnt[None] = 0.0
    delta = support_radius  *0.8
    for i in range(particle_num):
        x = 10.0+delta*(i%particle_num_x)
        y = 2.8 +delta*(i//particle_num_x)
        pos[i] = ti.Vector([x,y])
        velocities[i]   = ti.Vector([0.0,0.0])
        last_pos[i]     = pos[i]
        lambdas[i]      = 0.0
        pos_deltas[i]   = ti.Vector([0.0,0.0])

@ti.kernel
def PredictPosition():
    for i in pos:
        pos_i,vel_i = pos[i],velocities[i]
        vel_i += delta_t*ti.Vector([1.0,1.0])*gravity
        pos_i += delta_t*vel_i
        pos[i] = ConfinePositionToBoundary(pos_i,timeCnt[None])

@ti.kernel
def UpdateGrid():
    for I in ti.grouped(grid_particle_num):
        grid_particle_num[I] = 0
    for i in pos:
        cell_i = int(pos[i]/grid_cell_size)
        oldValue = ti.atomic_add(grid_particle_num[cell_i],1)
        grid_to_particle[cell_i,oldValue] = i


@ti.kernel
def FindNeighbors():
    for i in pos:
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

@ti.kernel
def ComputeLambdas():
    for i in pos:
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
        grad_sqr_sum += grad_i.dot(grad_i)
        lambdas[i] = (-C_i)/(grad_sqr_sum + lambda_epsilon)
        rhos[i] = rho_i

@ti.func
def ComputeScorr(pos_ji):
    x = Poly6(pos_ji.norm(),support_radius)/Poly6(0.3*support_radius,support_radius)
    x = x*x
    x = x*x
    return (-corrK)*x

@ti.kernel
def ComputeDeltaPosition():
    for p_i in pos:
        pos_i       = pos[p_i]
        lambda_i    = lambdas[p_i]
        delta_pos_i = ti.Vector([0.0,0.0])
        for j in range(neighbor_num[p_i]):
            p_j = neighbors[p_i,j]
            pos_ji = pos_i-pos[p_j]
            delta_pos_i+=(lambda_i+lambdas[p_j]+ComputeScorr(pos_ji))*SpikyGradient(pos_ji,support_radius)
        pos_deltas[p_i] = delta_pos_i/rho_0
        
@ti.kernel
def ApplyDeltaPosition():
    for i in pos:
        pos[i]+=pos_deltas[i]

@ti.kernel
def UpdateVelocities():
    for i in velocities:
        pos[i] = ConfinePositionToBoundary(pos[i],timeCnt[None])
        velocities[i] = (pos[i]-last_pos[i])/delta_t
        last_pos[i]   = pos[i]
    timeCnt[None] += delta_t

def LogInfo():
    rho_num = rhos.to_numpy()
    print(np.mean(rho_num),np.max(rho_num))

def run_pbf():
    PredictPosition()
    UpdateGrid()
    FindNeighbors()
    for _ in range(iteration_num):
        ComputeLambdas()
        ComputeDeltaPosition()
        ApplyDeltaPosition()
    UpdateVelocities()
    # debug_info()

def Main():
    Initialize()
    gui = ti.GUI('Position Based Fluid',res = resolution,background_color= 0xffffff)
    while not gui.get_event(gui.SPACE) :
        run_pbf()
        gui.circles(pos.to_numpy()/boundary,color=0x3098d9, radius=particle_radius*world_to_screen_ratio)
        gui.show()

Main()