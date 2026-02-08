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
则 J (m × n).可以容易形状验证 Jθ=f(θ)     
所以 $ JJ^T+\lambda^2I \quad $  (m × m)

**Q1: 如何求J**  
If a ball joint is parameterized as Euler angles:𝑅𝑖 =𝑅𝑖𝑥𝑅𝑖𝑦𝑅𝑖z 则  
$\boxed{ \frac{\partial f_j}{\partial \theta_{i*}} = \mathbf{a_{i*}} \times \mathbf{(s_j-s_i)} }$  
rotation axes are  
$ 𝒂_{ix} =𝑄_{𝑖−1}𝒆_𝑥 $    
$ 𝒂_{iy} =𝑄_{𝑖−1}𝑅_{ix}𝒆_y $    
$ 𝒂_{iz} =𝑄_{𝑖−1}𝑅_{ix}𝑅_{iy}𝒆_z $    
$ 𝑄_{𝑖−1}$  is the orientation of i's parent joint  