Python -- 3.10
Taichi -- 1.7.3  


# <span style="color: #6657bd;"> Numerical Methods </span>

##  <font color=#288FD4  size=5 >Linear Solver</font>
- Jacobi
- Jacobi with Chebyshev acceleration
- Gauss-Seidel with Color Graphing
- Conjugate Gradient  
<img src="./results/Linear_Solver.png" alt="show" /> 

##  <font color=#288FD4  size=5 >Optimization (ref-book:最优化：建模、算法与理论) </font>
## Unconstrained Optimization
- Line search:
  - Armijo
  - Wolfe

| Gradient Descent with line search  | Barzilar-Borwein with line search | 
|:------:|:-----:|
|  <img src="./results/GradientDescent.gif" alt="show" />   | <img src="./results/BarzilarBorwein.gif" alt="show" />   

| BFGS  | LBFGS | 
|:------:|:-----:|
|  <img src="./results/BFGS.gif" alt="show" />   | <img src="./results/LBFGS.gif" alt="show" />   

| Trust Region  | Gradient Descent with momentum and fixed learning rate | Adam |
|:------:|:-----:|:-----:|
|  <img src="./results/TrustRegion.gif" alt="show" />   | <img src="./results/Momentum.gif" alt="show" />    | <img src="./results/ADAM.gif" alt="show" />   

## Constrained Optimization 
- Penalty (not visualize)
- Augmented Lagrangian (not visualize)  
  

  
    
  
   
  
# <span style="color: #6657bd;"> <br><br><br><br>Simulation </span>

| Linear Elastic FEM  | Implicit Mass Spring | Fast Mass Spring |
|:------:|:-----:|:-----:|
||- One Step Newton <br>- Baraff, D., & Witkin, A. (1998, July). Large steps in cloth simulation. <br>- Gradient descent with line search <br>- Newton's Method with fixed Positive Definite Hessian|- Liu, T., Bargteil, A. W., O'Brien, J. F., & Kavan, L. (2013). Fast simulation of mass-spring systems.  |
|  <img src="./results/Linera_Elastic_FEM.gif" alt="show" />   | <img src="./results/Implicit Mass Spring.gif" alt="show" />    | <img src="./results/Fast_Mass_Spring.gif" alt="show" />   

| Pendulum  | Position Based Fluid with thermal conduction | Material point method sand |
|:------:|:-----:|:-----:|
||- Macklin, M., & Müller, M. (2013). Position based fluids|- Klár,G., Gast, T., Pradhana, A., Fu, C., Schroeder, C., Jiang, C., & Teran, J. (2016). Drucker-prager elastoplasticity for sand animation.   |
|  <img src="./results/Pendulum.gif" alt="show" />   | <img src="./results/Fluid_melting.gif" alt="show" />    | <img src="./results/mpm_sand0.gif" alt="show" />  

| Material point method snow  | XPBD Chain | Position Based Dynamics Rope |
|:------:|:-----:|:-----:|
|- Stomakhin, A., Schroeder, C., Chai, L., Teran, J., & Selle, A. (2013). A material point method for snow simulation. <br> - Jiang, Chenfanfu, et al. "The affine particle-in-cell method." ACM Transactions on Graphics (TOG) 34.4 (2015): 1-10.|- Macklin, M., Müller, M., & Chentanez, N. (2016, October). XPBD: position-based simulation of compliant constrained dynamics.<br> - Müller, M., Bender, J., Chentanez, N., & Macklin, M. (2016, October). A robust method to extract the rotational part of deformations.|- Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007). Position based dynamics.  |
|  <img src="./results/mpm_snow.gif" alt="show" />   | <img src="./results/XPBD_Chain.gif" alt="show" />    | <img src="./results/rope.gif" alt="show" />  


<font color=#288FD4  size=5 >Cloth Simulation</font>
- Robust Treatment of Collisions, Contact and Friction for Cloth Animation
- Derivation of discrete bending forces and their gradients
- Optimized Spatial Hashing for Collision Detection of Deformable Objects  
<img src="./results/quad2_collision.gif" alt="show" />  






# <span style="color: #6657bd;"> <br><br><br><br>Rendering </span>


## <a href="https://pbr-book.org/4ed/contents" target="_blank"> Physically Based Rendering </a>
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






# <span style="color: #6657bd;"> <br><br><br><br>Geometry </span>
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


|  Bezier and BSpline Curve | Subdivision Curve | Delaunay Triangluation |
|:------:|:-----:|:-----:|
|  <img src="./results/control_point_curve.gif" alt="show" />   | <img src="./results/subdivision_curve.gif" alt="show" />   | <img src="./results/triangluation.gif" alt="show" />  |

|  Mean Value Coordinate | Deformation Transfer |  |
|:------:|:-----:|:-----:|
|  <img src="./results/MVC.png" alt="show" />   | <img src="./results/DeformationTransfer.png" alt="show" />   |  |
