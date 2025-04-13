Python -- 3.8.10  
Taichi -- 1.6.0  

<font color=#288FD4  size=5 >Linear Solver</font>
- Jacobi
- Jacobi with Chebyshev acceleration
- Gauss-Seidel with Color Graphing
- Conjugate Gradient  
<img src="./results/Linear_Solver.png" alt="show" /> 

<font color=#288FD4  size=5 >Optimization (ref-book:最优化：建模、算法与理论) </font>
## Unconstrained Optimization
- Line search:
  - Armijo
  - Wolfe
- Gradient Descent with line search
<img src="./results/GradientDescent.gif" alt="show" /> 
- Barzilar-Borwein with line search
<img src="./results/BarzilarBorwein.gif" alt="show" /> 
- BFGS
<img src="./results/BFGS.gif" alt="show" /> 
- LBFGS
<img src="./results/LBFGS.gif" alt="show" /> 
- Trust Region
<img src="./results/TrustRegion.gif" alt="show" /> 
- Gradient Descent with momentum and fixed learning rate
<img src="./results/Momentum.gif" alt="show" /> 
- Adam
<img src="./results/ADAM.gif" alt="show" /> 
## Constrained Optimization 
- Penalty (not visualize)
- Augmented Lagrangian (not visualize)

<font color=#288FD4  size=5 >Linear Elastic FEM</font>  
<img src="./results/Linera_Elastic_FEM.gif" alt="show" />  

<font color=#288FD4  size=5 >Implicit Mass Spring</font>
- One Step Newton 
  - Baraff, D., & Witkin, A. (1998, July). Large steps in cloth simulation.
- Gradient descent with line search
- Newton's Method with fixed Positive Definite Hessian  
<img src="./results/Implicit Mass Spring.gif" alt="show" />  
  
<font color=#288FD4  size=5 >Fast Mass Spring</font>
- Liu, T., Bargteil, A. W., O'Brien, J. F., & Kavan, L. (2013). Fast simulation of mass-spring systems.  
<img src="./results/Fast_Mass_Spring.gif" alt="show" />  

<font color=#288FD4  size=5 >Pendulum</font>  
<img src="./results/Pendulum.gif" alt="show" />    
  
<font color=#288FD4  size=5 >Position Based Fluid with thermal conduction</font>  
- Macklin, M., & Müller, M. (2013). Position based fluids.  
 <img src="./results/Fluid_melting.gif" alt="show" />  
  
<font color=#288FD4  size=5 >Material point method sand</font>   
- Klár,G., Gast, T., Pradhana, A., Fu, C., Schroeder, C., Jiang, C., & Teran, J. (2016). Drucker-prager elastoplasticity for sand animation.  
(不确定这个仿真结果是否正确)  
<img src="./results/mpm_sand0.gif" alt="show" />  
<img src="./results/mpm_sand1.gif" alt="show" />  
  
<font color=#288FD4  size=5 >Material point method snow</font>  
- Stomakhin, A., Schroeder, C., Chai, L., Teran, J., & Selle, A. (2013). A material point method for snow simulation.  
- Jiang, Chenfanfu, et al. "The affine particle-in-cell method." ACM Transactions on Graphics (TOG) 34.4 (2015): 1-10.
<img src="./results/mpm_snow.gif" alt="show" />  
  
<font color=#288FD4  size=5 >XPBD Chain</font>  
- Macklin, M., Müller, M., & Chentanez, N. (2016, October). XPBD: position-based simulation of compliant constrained dynamics.
- Müller, M., Bender, J., Chentanez, N., & Macklin, M. (2016, October). A robust method to extract the rotational part of deformations. 
<img src="./results/XPBD_Chain.gif" alt="show" />  

<font color=#288FD4  size=5 >Position Based Dynamics Rope</font>    
- Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007). Position based dynamics.   
<img src="./results/rope.gif" alt="show" />  
  
<font color=#288FD4  size=5 >Continuous Collision</font>
- Robust Treatment of Collisions, Contact and Friction for Cloth Animation  
- Cubic equation solver (bisect)
- Vertex-Face Edge-Edge Collision 
<img src="./results/continuous_collision.gif" alt="show" />  

<font color=#288FD4  size=5 >Ray Tracer</font>  
- SAH BVH
- Brdfs:
  -   Specular Model:  GGX Microfacet
  -   Diffuse  Model:  Lambert
  -   Mirror
  -   Glass
- Volume Path Tracer
- Importance Sampline:
  -   sampling cosine term
  -   sampling brdf term

<img src="./results/bunny_mf.png"   alt="show" />  
<img src="./results/sphere_mf.png"  alt="show" />  
<img src="./results/glass.png"      alt="show" />  
<img src="./results/mirror.png"     alt="show" />  
<img src="./results/volume.png"     alt="show" />  

<font color=#288FD4  size=5 >2D surface area heuristic BVH</font>  
<img src="./results/bvh2d.gif" alt="show" />  

# <a href="https://pbr-book.org/4ed/contents" target="_blank"> Physically Based Rendering </a>

## path tracer
| Cornell  | Anisotropic Microfacet | Multiple Importance Sampling  |
|:------:|:-----:|:-----:|
|  <img src="./results/render/cornell.png" alt="show" />   | <img src="./results/render/anisotropic_microfacet_x.png" alt="show" />   | <img src="./results/render/mis.png" alt="show" />  

## volume path tracer
| fog  | Jade Bunny | Explosion  |
|:------:|:-----:|:-----:|
|  <img src="./results/render/fog.png" alt="show" />   | <img src="./results/render/homogeneous_medium.png" alt="show" />   | <img src="./results/render/aerial.png" alt="show" />  

<font color=#288FD4  size=5 ><a href="http://www.cse.yorku.ca/~amana/research/grid.pdf" target="_blank">A Fast Voxel Traversal Algorithm for Ray Tracing</a></font>  
| TestCase0  | TestCase1 |
|:------:|:-----:|
|  <img src="./results/render/raydda0.png" alt="show" />   | <img src="./results/render/raydda1.png" alt="show" />  |



# Discrete Differential Geometry
| Tree-Cotree  | Generators | HarmonicBasis  |
|:------:|:-----:|:-----:|
|  <img src="./results/torus_treecotree.png" alt="show" />   | <img src="./results/torus_generators.png" alt="show" />   | <img src="./results/torus_harmonicbases0.png" alt="show" />  |
|   |    | <img src="./results/torus_harmonicbases1.png" alt="show" /> |

## Hodge Decomposition
|  ω | dα | δβ  |
|:------:|:-----:|:-----:|
|  <img src="./results/bunny_omega.png" alt="show" />   | <img src="./results/bunny_d_alpha.png" alt="show" />   | <img src="./results/bunny_delta_beta.png" alt="show" />  |

|  parallel vector | geodesics  |
|:------:|:-----:|
|  <img src="./results/bunny_parallelvector.png" alt="show" />   | <img src="./results/bunny_geodesics.png" alt="show" />   | 

## SpectralConformalParameterization
|  uv in 2d | uv encoded in vertex | uv encoded on face  |
|:------:|:-----:|:-----:|
|  <img src="./results/SpectralConformalParameterization.png" alt="show" />   | <img src="./results/vertex_u.png" alt="show" />  | <img src="./results/face_u.png" alt="show" /> |
|     | <img src="./results/vertex_v.png" alt="show" />  | <img src="./results/face_v.png" alt="show" /> |


<font color=#288FD4  size=5 >Bezier and BSpline Curve</font>  
<img src="./results/control_point_curve.gif" alt="show" />  

<font color=#288FD4  size=5 >Subdivision Curve</font>  
<img src="./results/subdivision_curve.gif" alt="show" />  

<font color=#288FD4  size=5 >Delaunay Triangluation</font>  
<img src="./results/triangluation.gif" alt="show" />  

<font color=#288FD4  size=5 >Mean Value Coordinate</font>  
<img src="./results/MVC.png" alt="show" />  

<font color=#288FD4  size=5 >Deformation Transfer</font> 
- Deformation Transfer for Triangle Meshes 
<img src="./results/DeformationTransfer.png" alt="show" />  