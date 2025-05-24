import numpy as np
import polyscope as ps
from MeshOperator import *

def Main(testcase):
    if testcase == 'face':
        mesh = Mesh(os.path.join(__file__, '..', 'input', 'face.obj'))
        data = {'Spectral Conformal Parameterization':mesh.SpectralConformalParameterization(ComplexAsVec2d=True)}
    if testcase == 'bunny':
        mesh = Mesh(os.path.join(__file__,'..','input','bunny.obj'))
        omega,d_alpha,delta_beta = mesh.HodgeDecompositionTest()
        omega,d_alpha,delta_beta = omega/np.linalg.norm(omega,axis=1).reshape(-1,1),d_alpha/np.linalg.norm(d_alpha,axis=1).reshape(-1,1),delta_beta/np.linalg.norm(delta_beta,axis=1).reshape(-1,1)
        data = {
            'ParallelVector ': mesh.VisualizeConnection(mesh.TrivialConnection()),
            'HodgeDecomposition omega': omega,
            'HodgeDecomposition d_alpha': d_alpha,
            'HodgeDecomposition delta_beta': delta_beta,
            'lines:geodesics contour':mesh.ISOLines(0,1.0),
        }
    if testcase == 'torus':
        mesh = Mesh(os.path.join(__file__,'..','input','torus.obj'))
        data = {
            'lines:tree-cotree':mesh.VisualizeTreeCoTree(),
            'lines:generators':mesh.VisualizeGenerators()
        }
        harmonicBases = mesh.VisualizeHarmonicBases()
        for i,basis in enumerate(harmonicBases): data['HarmonicBasis'+str(i)]=basis
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