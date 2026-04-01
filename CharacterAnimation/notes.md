# IK
### ref 
- Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods
- https://games-105.github.io/ppt/04%20-%20Keyframe%20Animation.pdf Page 85

考虑牛顿法求根 有 $ x_{i+1} = x_i - (\frac{df}{dx})^{-1} f(x_i) $

设 f(θ)是 forward kinematics, t是target positions, s是current positions, J是Jacobian    
**Q0: 如何求θ使 f(θ)=t**    
我们同样可以使用牛顿法,有  
$ \theta_{i+1} = \theta_i + J^{-1}(t- f(\theta_i)) $    

OK,直接一步到位上 Damped Least Squares 公式:  
$ \boxed{ \theta_{i+1} = \theta_i + J^T(JJ^T+\lambda^2I)^{-1}(t- f(\theta_i)) }$    
λ ≈ 1e-3 ~ 1e-1  (maybe 1e-2 is a good choice)  
现在我们来看一下矩阵的形状，假设旋转有n个,末端效应器有m个 即θ(n × 1), t(m × 1), f(θ) (m × 1)  
则 J (m × n).可以容易验证形状是正确的 Jθ=f(θ)     
所以 $ JJ^T+\lambda^2I \quad $  (m × m)

**Q1: 如何求J**  
If a ball joint is parameterized as Euler angles:𝑅𝑖 =𝑅𝑖𝑥𝑅𝑖𝑦𝑅𝑖z 则  
$\boxed{ \frac{\partial f_j}{\partial \theta_{i*}} = \mathbf{a_{i*}} \times \mathbf{(s_j-s_i)} }$  
rotation axes are  
$ 𝒂_{ix} =𝑄_{𝑖−1}𝒆_𝑥 $    
$ 𝒂_{iy} =𝑄_{𝑖−1}𝑅_{ix}𝒆_y $    
$ 𝒂_{iz} =𝑄_{𝑖−1}𝑅_{ix}𝑅_{iy}𝒆_z $    
$ 𝑄_{𝑖−1}$  is the orientation of i's parent joint  


# MOTION MATCHING
### ref
- https://theorangeduck.com/page/code-vs-data-driven-displacement Siggraph Learned Motion Matching作者的博客
下面是这篇博文的一些笔记  

Simulation Object:是角色运动的代理,比如一个圆柱体,它可以和环境有碰撞。 控制技术:critically damped spring  
Character Entity: 即角色本身。   
Simulation Bone: 原始的Character Entity数据里只有动捕数据,为了能让Simulation Object去控制Character Entity,我们需要一个虚拟的
根骨骼。这个根骨骼应该和Simulation Object有这相同的速度，角速度，加速度。 
那我们如何计算这些物理量呢？可以用动画数据里upper spine的位置在地面上的投影计算位置与速度，对hip使用相同的操作计算角速度。  滤波技术:Savitzky-Golay filter   
注意，上述方法针对那些从站立姿势开始站立姿势结束的动画，对于loop animation不适用。  

- **Learned Motion Matching**  

feature vector $ x = \{t^t \: t^d \: f^t \: \dot{f}^t \: \dot{h}^t\}  $  
**t**:trajectory **f**:foot **h**:hip **上标t**:translation **上标d**:direction ·代表对于时间的一阶导数  
$t^t \in R^6$: 未来20,40,60帧在地面的投影轨迹的位置(默认60fps)      
$t^d \in R^6$: 未来20,40,60帧在地面的投影轨迹的朝向  
$f^t \in R^6$: 当前的两个脚关节的位置        
$\dot{f}^t \in R^6$: 当前的两个脚关节的速度           
$\dot{h}^t \in R^6$: hip关节的速度           

normalize feature vector. how?

pose vector $ y = \{y^t \: y^r \: \dot{y}^t \: \dot{y}^r \: \dot{r}^t \: \dot{r}^r \: o^* \}  $  
$y^t \: y^r$: 分别表示各关节的局部(相对于父joint)平移和局部旋转      
$\dot{y}^t \: \dot{y}^r$: 分别表示各关节的局部平移速度和局部旋转速度       
$\dot{r}^t \: \dot{r}^r$: 分别表示角色根节点的平移速度和旋转速度，这些速度是相对于角色前向朝向的局部坐标系定义的？？？；   
$ o^*$: 表示所有其他与具体任务相关的附加输出，例如脚部接触信息、场景中其他角色的位置或轨迹，或者角色某些关节的未来位置等。  

对于每一帧,该骨骼都对应一个特征向量,这些所有特征向量的拼接叫做**X**,即Matching Database.  
所有pose vector的拼接叫做**Y**,即Animation Database.

在运行时，每N帧或者用户的输入被极大的改变(摇滚突然从右推向左),我们就构建一个查询向量$\hat{x}$  
我们的目标是从Matching Database里找到一个entry. $k^* = \arg\min_k \left\lVert \hat{x} - x_k \right\rVert^2$

# MISC TOPICS  
### The Spring Damper
有时我们需要从一个目标变化到另一个目标(比如从位置A到位置B),使用线性插值会是一个不错的选择,即  
lerp(x,y,a) = (1-a)x+ay  
假设我们需要从位置x0变化到位置g(oal),每一帧都变化一点儿  
$ x_{i+1} = lerp(x_i \:,g\:,factor) $  
举个例子,若x0=1,g=0,factor=0.5.则x1 = 0.5;  x2 = 0.25;  x3 = 0.125  
如果我想让变化更光滑呢？我可以让这个lerp不直接去更新位置，而是去更新速度。 
首先把lerp展开, $x_{t+dt}=x_t+a(g-x_t)$,这实际上是  
$
v_{t+dt} = damping⋅(g−x_t)  \\
x_{t+dt} = x_t + dt \cdot v_t
$  
现在令 $v_{t+dt} = v_t+dt\cdot stiffness\cdot(g−x_t)+dt\cdot damping\cdot(q−v_t)  $ 

### The Critical Spring Damper
直接上公式,x应该有如下的形式 $x_t = j \cdot e^{-y t} \cdot \cos(w \cdot t + p) + c $  
然后我们根据 $a_t = s(g-x_t)+d(q-v_t)$ 推导其余的未知变量，有  
$c = g+\frac{d \cdot q}{s} \\  y = \frac{d}{2}  \\  \omega = \sqrt{s-\frac{d^2}{4}} \\ j=\sqrt{\frac{\left(v_0 + y(x_0 - c)\right)^2}{w^2}+ (x_0 - c)^2} \\ p = \arctan\left( \frac{v_0 + (x_0 - c) \cdot y}{-(x_0 - c) \cdot w} \right)
$    
实际上,对于$\omega$我们需要讨论s和d的关系。现在直接给出$\omega$为0时(Critical)的方程  
$
x_t = j_0 \cdot e^{-y \cdot t} + t \cdot j_1 \cdot e^{-y \cdot t} + c \\
v_t = -y \cdot j_0 \cdot e^{-y \cdot t} - y \cdot t \cdot j_1 \cdot e^{-y \cdot t} + j_1 \cdot e^{-y \cdot t} \\
a_t = y^2 \cdot j_0 \cdot e^{-y \cdot t} + y^2 \cdot t \cdot j_1 \cdot e^{-y \cdot t} - 2 \cdot y \cdot j_1 \cdot e^{-y \cdot t}\\
其中\\
j_0=x_0-c \\ 
j_1 = v_0+j_0 \cdot y
$  
我们可以把这个运用到摇杆上,其中q=0。注意！！这里的$x_{goal}$是**速度**  
为了求得真正的移动，做积分有  
$x_t = \frac{-j_1}{y^2} \cdot e^{-y \cdot t} + \frac{-j_0 - j_1 \cdot t}{y} \cdot e^{-y \cdot t} + \frac{j_1}{y^2} + \frac{j_0}{y} + c \cdot t + x_0$  
另外,damping这个参数不直观。一般有 $ damping = \frac{4ln(2)}{halflife+\epsilon}$ 可以把halflife理解成衰减的速率

