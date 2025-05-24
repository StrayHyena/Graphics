import numpy as np
import polyscope as ps
from MeshOperator import *

def Main(testcase):
    ObjPath = lambda objname: os.path.join(__file__, '..', 'input', objname+'.obj')
    if testcase == 'face':
        mesh = Mesh(os.path.join(__file__, '..', 'input', 'face.obj'))
        data = {'Spectral Conformal Parameterization':SpectralConformalParameterization(mesh).uv_matrix}
    if testcase == 'bunny':
        mesh = Mesh(os.path.join(__file__,'..','input','bunny.obj'))
        hd = HodgeDecomposition(mesh)
        omega,d_alpha,delta_beta = hd.face_omega/np.linalg.norm(hd.face_omega,axis=1).reshape(-1,1),hd.face_alpha/np.linalg.norm(hd.face_alpha,axis=1).reshape(-1,1),hd.face_beta/np.linalg.norm(hd.face_beta,axis=1).reshape(-1,1)
        data = {
            'ParallelVector ': TrivialConnection(mesh).parallel_vector,
            'HodgeDecomposition omega': omega,
            'HodgeDecomposition d_alpha': d_alpha,
            'HodgeDecomposition delta_beta': delta_beta,
            'lines:geodesics contour':HeatMethodGeodesic(mesh,0).ISOLines(1.0),
        }
    if testcase == 'torus':
        mesh = Mesh(os.path.join(__file__,'..','input','torus.obj'))
        mesh.harmonic_bases # this will trigger to compute tree,cotree and generators
        data = {
            'lines:tree':mesh.tree_lines,
            'lines:cotree':mesh.cotree_lines,
            'lines:generators':mesh.generators_lines
        }
        for i,basis in enumerate(mesh.harmonic_bases_on_face): data['HarmonicBasis'+str(i)]=basis
    # if testcase == QMC:
    #     mesh = Mesh(os.path.join(__file__, '..', 'input', 'quad-circle.obj'))
    #     data = {'':mesh.CrossField()}
    ps.set_warn_for_invalid_values(True)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_background_color((0.5,0.5,0.5))
    ps.set_shadow_darkness(0.75)
    ps.init()
    ps_mesh = ps.register_surface_mesh(testcase, mesh.vertices, mesh.faces,color=(0.13333333,0.44705882,0.76470588))
    for dataname,datavalue in data.items():
        if dataname.startswith('lines:'):
            ps.register_curve_network(dataname[6:],datavalue, np.arange(len(datavalue)).reshape(-1,2),radius=0.0005,color=(1,1,1),enabled=False)
        else:
            if datavalue.shape == (mesh.vn,2): # assume in this case, vertex holds uv attribute
                ps_mesh.add_parameterization_quantity(dataname, datavalue)
            elif datavalue.shape == (mesh.fn,3):
                ps_mesh.add_vector_quantity(dataname, datavalue, defined_on='faces', radius=0.001,length=0.01, color=(0.97,0.9,0.18))
            elif datavalue.shape == (mesh.vn,3):
                ps_mesh.add_vector_quantity(dataname, datavalue, radius=0.001,length=0.01, color=(0.97,0.9,0.18))
    ps.show()

Main('bunny')