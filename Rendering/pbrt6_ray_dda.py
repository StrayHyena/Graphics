import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('TkAgg')

def draw_grid_and_line(grid_base,a, p0, p1,rect_coords=[],ts = []):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')  # 确保比例相同

    p0-=grid_base
    p1-=grid_base
    x0, y0 = p0
    x1, y1 = p1

    border = 0
    adjusted_x_min = (min(x0, x1) // a[0]-  border) * a[0]
    adjusted_x_max = ((max(x0, x1) // a[0]) + 1+ border) * a[0]
    adjusted_y_min = (min(y0, y1) // a[1]-  border) * a[1]
    adjusted_y_max = ((max(y0, y1) // a[1]) + 1+ border) * a[1]

    x_ticks = [grid_base[0] + i * a[0] for i in range(int(adjusted_x_min / a[0]),int(adjusted_x_max / a[0]) + 1)]
    y_ticks = [grid_base[1] + i * a[1] for i in range(int(adjusted_y_min / a[1]),int(adjusted_y_max / a[1]) + 1)]
    ax.clear()
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    cmap = plt.get_cmap('viridis')
    rect_colors = [cmap(pos) for pos in  np.linspace(0, 1, len(rect_coords))]
    for i,rect_coord in enumerate( rect_coords ):
        rect_x, rect_y = rect_coord
        rect = Rectangle((rect_x*a[0]+grid_base[0], rect_y*a[1]+grid_base[1]), a[0], a[1],facecolor=rect_colors[i], edgecolor='none', zorder=0)
        ax.add_patch(rect)
    pts = []
    pts_colors = [cmap(pos) for pos in  np.linspace(0, 1, len(ts))]
    for i,t in enumerate(ts):pts.append( grid_base + (1-t)*p0 + t*p1)
    pts = np.array(pts)
    ax.scatter(pts[:,0],pts[:,1],c=pts_colors[::-1],s=10,zorder=3)

    ax.scatter([grid_base[0] +x0, grid_base[0] +x1], [grid_base[1] +y0, grid_base[1] +y1], color=['red','blue'], zorder=2)
    ax.plot([grid_base[0] +x0, grid_base[0] +x1], [grid_base[1] +y0, grid_base[1] +y1], color='black', linewidth=2, zorder=1)

    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.ylim(y_ticks[0], y_ticks[-1])
    fig.canvas.draw()

MAX = 1e100
EPS = 1e-5

class DDAGrid:
    def __init__(self,grid_base,a, p0, p1):
        d = p1 - p0
        self.grid_base = grid_base
        self.a = a
        self.d = d
        p0 -= grid_base
        p1 -= grid_base
        if np.floor(p0[0] / a[0]) == np.ceil(p0[0] / a[0]) or np.floor(p0[1] / a[1]) == np.ceil(p0[1] / a[1]): p0 += EPS * d
        if np.floor(p1[0] / a[0]) == np.ceil(p1[0] / a[0]) or np.floor(p1[1] / a[1]) == np.ceil(p1[1] / a[1]): p1 -= EPS * d
        fc0 = p0 / a
        fc1 = p1 / a
        ic0 = np.floor(fc0)
        ic1 = np.floor(fc1)
        ts = [MAX, MAX]
        for i in range(2):
            if d[i] == 0: continue
            if d[i] > 0:ts[i] = (np.ceil(fc0[i]) - fc0[i]) * a[i] / d[i]
            else:       ts[i] = (fc0[i] - np.floor(fc0[i])) * a[i] / abs(d[i])
        tx, ty = ts[0], ts[1]
        dx = MAX if d[0] == 0 else abs(a[0] / d[0])
        dy = MAX if d[1] == 0 else abs(a[1] / d[1])
        assert tx > 0 and ty > 0 and dx > 0 and dy > 0
        self.tx = tx
        self.ty = ty
        self.dx = dx
        self.dy = dy
        self.curr = ic0  #　当前体素的坐标
        self.end = ic1
        self.prev_t = 0    # 记录光线射(进)入当前体素的前一个体素时的t值
        self.curr_t = 0     #记录光线射(进)入当前体素时的t值

    def Next(self):
        if np.linalg.norm(self.curr-self.end) ==0 :return None
        self.prev_t = self.curr_t
        if self.tx<self.ty:
            self.curr_t = self.tx
            self.tx+=self.dx
            self.curr[0]+=np.sign(self.d[0])
        else:
            self.curr_t = self.ty
            self.ty+=self.dy
            self.curr[1]+=np.sign(self.d[1])
        return 1

    def GetRects(self):
        ret = [self.curr.copy()]
        ts = []
        while self.Next():
            ret.append(self.curr.copy())
            ts.append(self.curr_t)
        return ret,ts

def Test(cnt=1):
    for _ in range(cnt):
        grid_base, a, = np.random.rand(2) * 100, (np.random.rand(2) - 0.5) * 2 + 2.1,
        p0, p1 = np.random.rand(2) * 50+grid_base, np.random.rand(2) * 50+grid_base
        rects,ts = DDAGrid(grid_base.copy(), a.copy(), p0.copy(), p1.copy()).GetRects()
        draw_grid_and_line(grid_base.copy(), a.copy(), p0.copy(), p1.copy(),rects,ts )
        plt.show()

Test(5)
# Main()