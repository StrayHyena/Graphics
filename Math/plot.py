import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,Colormap

class Ploter:
    def __init__(self):
        self.ax = plt.figure().add_subplot(projection='3d')

    def DrawFunction(self,f,xrange=(-1,1),yrange=(-1,1),step=0.01,draw_type='sc'):
        xs = np.arange(xrange[0],xrange[1]+step,step)
        ys = np.arange(yrange[0],yrange[1]+step,step)
        X,Y = np.meshgrid(xs,ys)
        Z = np.zeros(X.shape)
        for i,_ in np.ndenumerate(Z):Z[i] = f((X[i],Y[i]))
        self.ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Y)))
        if 's' in draw_type: self.ax.plot_surface(X,Y,Z, edgecolor='royalblue', lw=0.1, rstride=16, cstride=16, alpha=0.1)
        if 'c' in draw_type: self.ax.contour(X, Y, Z,10, cmap='coolwarm')  

    def DrawPoint(self,pts,interplate_num=100,show_speed=False):
        minz,maxz = np.min(pts[:,2]),np.max(pts[:,2])
        draw_pts = []
        n = max(1,interplate_num)
        for i in range(1,len(pts)):
            p0,p1 = pts[i-1],pts[i]
            for j in range(n):draw_pts.append(p1*j/n+p0*(1-j/n))
        draw_pts.append(pts[-1])
        draw_pts = np.array(draw_pts)
        length = np.arange(len(draw_pts))[::-1]
        if show_speed==False:
            length = np.array([0.0])
            for i in range(1,len(draw_pts)):length = np.append(length, np.linalg.norm(draw_pts[i]-draw_pts[i-1]))
            for i in range(1,len(length)):length[i]+=length[i-1]
            length/=length[-1]
        self.ax.scatter(draw_pts[:,0],draw_pts[:,1],draw_pts[:,2], c = length ,s=1, cmap ='gist_rainbow')

    def DrawLines(self,pts):
        for i in range(len(pts)-1):
            color = np.random.rand(3)
            if i==0: color = (1,0,0)
            elif i==len(pts)-2: color = (0,0,1)
            self.ax.plot([pts[i][0],pts[i+1][0]],[pts[i][1],pts[i+1][1]],[pts[i][2],pts[i+1][2]],c = color)

    def Show(self):
        plt.show()

def Test():
    plter = Ploter()
    # plter.DrawFunction( lambda x:2*x[0]*x[0]+x[1]*x[1] )
    plter.DrawLines(np.random.rand(10,3))
    # plter.DrawPoint( np.random.rand(10,3) )
    # plter.ax.plot([0,1],[0,1],[0,1],c = [1,0,0])
    plter.Show()

if __name__ == '__main__':
    Test()

