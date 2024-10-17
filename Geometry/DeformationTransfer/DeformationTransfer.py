# Deformation Transfer for Triangle Meshes
# https://blog.scubot.com/article/4581/

import trimesh
import numpy as np
from scipy.sparse import coo_matrix,vstack
from scipy.sparse.linalg import spsolve

def IsSameTopology(a,b):
    if len(a.vertices)!=len(b.vertices) or len(a.faces) != len(b.faces):
        return False
    for ta,tb in zip(a.faces,b.faces):
        if set(ta)!=set(tb):return False
    return True

def GetMatrixV(v0,v1,v2):
    n = np.cross(v1-v0,v2-v0)
    v3 = v1+n/np.sqrt(np.linalg.norm(n))
    return (np.array([v1,v2,v3])-v0).T  # eq(3) #注意这种构造ndarray的方法把v1当作行向量了。最后要转置一下

def Main(src_path = 'src.obj',
         src_deform_path = 'src_deform.obj',
         tgt_path = 'tgt.obj',
         tgt_deform_path='tgt_deform.obj'):
    src = trimesh.load(src_path)
    src_deform = trimesh.load(src_deform_path)
    tgt = trimesh.load(tgt_path)
    tgt_deform = tgt.copy()

    Qs,ts = [],[]
    assert IsSameTopology(src,src_deform)
    for tri_s,tri_d in zip(src.faces,src_deform.faces):
        v = np.array(  [src.vertices[vidx] for vidx in tri_s])
        v_tilde = np.array(  [src_deform.vertices[vidx] for vidx in tri_d])
        V = GetMatrixV(v[0],v[1],v[2])
        V_tilde = GetMatrixV(v_tilde[0],v_tilde[1],v_tilde[2])
        Q = V_tilde@np.linalg.inv(V)   #  eq(4)
        t = v_tilde[0]-Q@v[0]
        Qs.append(Q)
        ts.append(t)

    # eq(7).
    # 优化目标: tgt三角形的非平移部分的形变要尽量和src的相同。
    # 约束条件: tgt的三角形j,k有公共的顶点vi,则从三角形j,三角形k这两个视角下变换后的vi要一致
    # T是由V和V_tilde计算得到的，对于tgt mesh,T是已知的. T其实是V_tilde的线性组合
    # 决策变量: tgt三角形的affine matrix(T) & 位移(d) ===>  其实可以把tgt_deform vertice(包括added vertex)作为决策变量，消去约束条件

    n,m = len(tgt.vertices),len(tgt.faces)
    V_hat_invs = []  # 意义参见第二行的博客
    for j,tri in enumerate(tgt.faces):
        v = np.array(  [tgt.vertices[vidx] for vidx in tri])
        V = GetMatrixV(v[0],v[1],v[2])
        sparse_data = np.array([1,     1,      1,      -1,  -1,     -1])
        sparse_row = np.array([tri[1], tri[2], n+j, tri[0], tri[0],tri[0]])
        sparse_col = np.array([0,      1,      2,      0,   1,      2])
        IdxMatrix = coo_matrix((sparse_data,(sparse_row,sparse_col)),shape=(n+m,3))
        V_hat_invs.append(coo_matrix(IdxMatrix@np.linalg.inv(V)))  #注意，它已经不是sparse的了
    A = vstack([V_hat_inv.T for V_hat_inv in V_hat_invs])
    b = np.vstack([Q.T for Q in Qs])
    x = spsolve(A.T@A,A.T@b)
    tgt_deform.vertices = x[:n]
    tgt_deform.export(tgt_deform_path)

Main()