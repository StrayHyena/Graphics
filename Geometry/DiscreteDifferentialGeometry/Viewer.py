import numpy as np
from mayavi import mlab
from MeshOperator import *

class Viewer:
    def __init__(self,mesh,data):
        colormap = "gist_rainbow" # red small, purple big
        el = mesh.edges_unique_length.mean()
        self.vn ,self.en, self.fn = len(mesh.vertices), len(mesh.edges_unique), len(mesh.faces)
        self.default_point_scalar = np.zeros(self.vn)
        self.default_face_vector  = np.zeros_like(mesh.faces)
        self.lineSegmentNum = self.fn*10 # 画线段，最大的线段数量。随便设的，但是要设大一些
        self.line_start = np.zeros((self.lineSegmentNum,3))
        self.line_stretch = np.zeros((self.lineSegmentNum,3))

        vxs,vys,vzs = mesh.vertices[:, 0], mesh.vertices[:, 1],mesh.vertices[:, 2]
        self.solid_mesh     = mlab.triangular_mesh(vxs,vys,vzs,mesh.faces,color=(0.7,0.7,0.7))
        self.wireframe_mesh = mlab.triangular_mesh(vxs,vys,vzs,mesh.faces,representation='wireframe',color=(0,0,0))

        self.point_scalar = mlab.points3d(vxs,vys,vzs,self.default_point_scalar, scale_factor=el*0.2, scale_mode = 'none', colormap=colormap)
        self.face_vector  = mlab.quiver3d(mesh.triangles_center[:, 0], mesh.triangles_center[:, 1], mesh.triangles_center[:, 2],
                      self.default_face_vector[:, 0], self.default_face_vector[:, 1], self.default_face_vector[:, 2],
                      scale_factor=el*0.5, scale_mode='none',mode='arrow', colormap=colormap)
        self.lines = mlab.quiver3d(self.line_start[:, 0],self.line_start[:, 1],self.line_start[:, 2],
                                   self.line_stretch[:, 0], self.line_stretch[:, 1], self.line_stretch[:, 2],
                                   scale_factor=1, mode='2ddash', color=(1,1,1))

        scene = mlab.gcf().scene
        scene.interactor.add_observer('KeyPressEvent', self.OnKeyPressed)

        self.show_wireframe = True
        self.attrib_idx = 0
        self.data = data
        self.attrib_names = list(self.data.keys())
        self.title = mlab.title('                       ',height=0.95,opacity=0.5,size=0.1,color=(1,0,0))
        self.Clear()

    def Clear(self):
        self.point_scalar.visible = self.face_vector.visible = self.lines.visible  = False
        self.line_start.fill(0)
        self.line_stretch.fill(0)

    def OnKeyPressed(self,obj,event):
        if obj.GetKeySym() == 'space': self.show_wireframe = not self.show_wireframe
        if obj.GetKeySym() == 'x': self.attrib_idx = (self.attrib_idx-1)%len(self.attrib_names)
        if obj.GetKeySym() == 'c': self.attrib_idx = (self.attrib_idx+1)%len(self.attrib_names)

    def Update(self):
        self.Clear()
        self.wireframe_mesh.visible   = self.show_wireframe
        attrib_name = self.attrib_names[self.attrib_idx]
        self.title.text = attrib_name
        data = self.data[attrib_name]
        if attrib_name.startswith('lines'):
            for line_idx in range(len(data)//2):
                self.line_start[line_idx] = data[2*line_idx]
                self.line_stretch[line_idx] = data[2 * line_idx+1] - self.line_start[line_idx]
            self.lines.mlab_source.trait_set(
                x=self.line_start[:, 0],y=self.line_start[:, 1],z=self.line_start[:, 2],
                u=self.line_stretch[:, 0],v=self.line_stretch[:, 1],w=self.line_stretch[:, 2],)
            self.lines.visible = True
        elif data.shape == (self.vn,):
            self.point_scalar.mlab_source.scalars = data
            self.point_scalar.visible = True
        elif data.shape == (self.fn,3):
            self.face_vector.mlab_source.vectors = data
            self.face_vector.visible = True

@mlab.animate(delay=500)
def Draw(viewer):
    while True:
        viewer.Update()
        yield

BUNNY,TORUS,FACE = 0,1,2

def Main(testcase=TORUS):
    if testcase == FACE:
        mesh = Mesh(os.path.join(__file__, '..', 'input', 'face.obj'))
        mesh.VisualizeParameterization(mesh.SpectralConformalParameterization())
        return
    if testcase == BUNNY:
        mesh = Mesh(os.path.join(__file__,'..','input','bunny.obj'))
        omega,d_alpha,delta_beta = mesh.HodgeDecompositionTest()
        data = {
            'ParallelVector ': mesh.VisualizeConnection(mesh.TrivialConnection()),
            'HodgeDecomposition-omega': omega,
            'HodgeDecomposition-d_alpha': d_alpha,
            'HodgeDecomposition-delta_beta': delta_beta,
            'lines_ISO:Geodesics':mesh.ISOLines(0,1.0),
                }
    if testcase == TORUS:
        mesh = Mesh(os.path.join(__file__,'..','input','torus.obj'))
        data = {
            'lines:tree-cotree':mesh.VisualizeTreeCoTree(),
            'lines:generators':mesh.VisualizeGenerators()
        }
        harmonicBases = mesh.VisualizeHarmonicBases()
        for i,basis in enumerate(harmonicBases): data['HarmonicBasis'+str(i)]=basis
    mlab.figure(size=(1500, 1500))
    Draw(Viewer(mesh,data))
    mlab.show()

Main(FACE)