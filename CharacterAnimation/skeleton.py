import os,scipy,copy
import polyscope as ps, numpy as np, polyscope.imgui as psim, scipy.sparse as sp
from scipy.spatial.transform import Rotation as R
from typing import Dict

ps.init()
io = psim.GetIO()

class Visualizer:
    class Dragger:  # use middle mouse button to drag
        def __init__(self):
            self.pick_result,self.pick_hit_pos,self.pick_hit_normal,self.pick_mouse_pos = None,None,None,None
        # return joint idx and translation
        def DragJoint(self):
            if io.MouseDown[2]:
                if self.pick_result is None:
                    self.pick_result = ps.pick(screen_coords=io.MousePos)
                    if self.pick_result.is_hit:
                        self.pick_hit_pos,self.pick_mouse_pos = self.pick_result.position,io.MousePos
                        self.pick_hit_normal = ps.get_view_camera_parameters().get_position()-self.pick_hit_pos
                    else: self.pick_result = None
                elif np.linalg.norm(np.array(io.MousePos)-np.array(self.pick_mouse_pos))>2:
                    d = ps.screen_coords_to_world_ray(io.MousePos)
                    o = ps.get_view_camera_parameters().get_position()
                    n = self.pick_hit_normal
                    hit_pos = o+np.dot(n,self.pick_hit_pos-o)/np.dot(n,d)*d
                    translate = hit_pos-self.pick_hit_pos
                    drag_seg_i = self.pick_result.structure_data['index']
                    if self.pick_result.structure_data['element_type'] == 'node': drag_seg_i //= 2
                    return drag_seg_i+1,translate  # see Skeleton's Segments() for why 1 is added.
            else:   self.pick_result,self.pick_hit_pos,self.pick_hit_normal,self.pick_mouse_pos = None,None,None,None
            return None,None

    def __init__(self,skeleton):
        self.skeleton = skeleton
        self.segs = skeleton.segments.copy()
        self.ps_segs = ps.register_curve_network('skeleton', self.segs[0], np.arange(len(self.segs[0])).reshape(-1,2))
        self.ps_segs.add_scalar_quantity('index', np.arange(len(self.segs[0])), enabled=True)
        self.dragger = Visualizer.Dragger()

    def ShowAnimation(self):
        ps.set_enable_vsync(True)
        ps.set_max_fps(int(1/self.skeleton.dt))
        while not psim.IsKeyDown(psim.ImGuiKey_Escape):
            for fi in range(self.skeleton.frame_cnt):
                self.ps_segs.update_node_positions(self.segs[fi])
                ps.frame_tick()
    # only control frame 0
    def RigControl(self):
        rotations,positions,root_pos = copy.deepcopy(self.skeleton.rotations[0]),self.skeleton.positions[0].copy(),self.skeleton.root_pos[0]
        while not psim.IsKeyDownDown(psim.ImGuiKey_Escape):
            jidx,translate = self.dragger.DragJoint()
            if jidx is not None: 
                temp_positions = self.skeleton.FK(self.skeleton.IK(jidx,positions[jidx]+translate,rotations,root_pos),root_pos)[0]
                self.ps_segs.update_node_positions(self.skeleton.Segments(temp_positions))
            ps.frame_tick()

class Skeleton:
    def __init__(self,filepath):
        names,parents,offsets,endsites,parent_idx_stack,root_positions,rotations,i=[],{},[],[],[-1],[],[],0
        with open(filepath, "r") as f:
            line_splits = [line.split() for line in f.readlines()]
            while i<len(line_splits):
                strs = line_splits[i]
                if strs[0]=='End' and strs[1]=='Site':
                    assert line_splits[i+1][0] == '{' and line_splits[i+2][0] == 'OFFSET' and line_splits[i+3][0] == '}'
                    endsites.append( ('EndSite'+str(len(endsites)),parent_idx_stack[-1],(float(line_splits[i+2][1]),float(line_splits[i+2][2]),float(line_splits[i+2][3]))  ))
                    i+=4
                    continue
                if strs[0] == 'ROOT' or strs[0]=='JOINT': names.append(strs[1])
                elif strs[0]=='OFFSET':offsets.append((float(strs[1]),float(strs[2]),float(strs[3])))
                elif strs[0]=='{':parent_idx_stack.append(len(names)-1)
                elif strs[0]=='}':
                    curridx =  parent_idx_stack.pop()
                    parents[curridx] = parent_idx_stack[-1]
                elif strs[0]=='Frames:':self.frame_cnt = int(strs[1])
                elif strs[0]=='Frame' and strs[1]=='Time:':self.dt = float(strs[2])
                elif strs[0]=='HIERARCHY'or strs[0]=='MOTION':pass
                elif strs[0]=='CHANNELS':
                    if strs[1]=='3': assert strs[2]=='Xrotation' and strs[3]=='Yrotation' and strs[4]=='Zrotation' ,'we assume bvh rotation channel order is XYZ'
                    elif strs[1]=='6': assert strs[2][0]=='X' and strs[3][0]=='Y' and strs[4][0]=='Z' and strs[5]=='Xrotation' and strs[6]=='Yrotation' and strs[7]=='Zrotation' ,'we assume bvh rotation channel order is XYZ'
                else: # read motion data
                    root_positions.append((float(strs[0]),float(strs[1]),float(strs[2]))) # equal to Rx@Ry@Rz
                    rotations.append([R.from_euler('XYZ',(float(strs[j]),float(strs[j+1]),float(strs[j+2])),True) for j in range(3,len(strs),3)])
                i+=1
        # note joint_cnt is not include endsites。注意到所有的end-effectors都排在list的最后
        self.joint_cnt,self.names,self.parents,self.offsets = len(names),names+[s[0] for s in endsites],np.array([parents[i] for i in sorted(parents.keys())]+[s[1] for s in endsites]),np.array(offsets+[s[2] for s in endsites])
        self.children = [[] for _ in range(len(self.names))]
        for jidx,pidx in enumerate(self.parents):
            if pidx!=-1: self.children[pidx].append(jidx)
        self.root_pos,self.rotations = np.array(root_positions),rotations
        self.positions = [self.FK(rot,pos)[0] for rot,pos in zip(self.rotations,self.root_pos)]  
        assert len(self.positions) == self.frame_cnt  and len(self.rotations)==self.frame_cnt
        self.segments = np.array([self.Segments(poss) for poss in self.positions] )
        # self.Print()
   
    # Forward Kinematics : this method computes joint's global rotation/position. Positions Size : (M,3) M is joint(include endsite) number
    def FK(self,rotations,root_pos = np.zeros(3)):
        assert len(rotations) == self.joint_cnt
        orientations,positions = copy.deepcopy(rotations),np.zeros((len(self.names),3))
        def forward(jidx):
            if jidx!=0:
                pidx = self.parents[jidx]
                positions[jidx] = positions[pidx]+orientations[pidx].apply(self.offsets[jidx])
                if jidx<self.joint_cnt:  orientations[jidx] = orientations[pidx]*rotations[jidx]  
            for child in self.children[jidx]: forward(child)
        forward(0)
        return positions+root_pos,orientations

    # Inverse Kinematics. @param[rotations] joints current rotations  @[return] new rotations that positions the end-effector at the specified location.
    def IK(self,target_idx:int,target_pos:np.ndarray,rotations,root_pos = np.zeros(3)):
        assert len(rotations)==self.joint_cnt
        ret = copy.deepcopy(rotations)
        if target_idx<self.joint_cnt: print('IK target is not end effector.Ignore.'); return ret
        path2root = [target_idx]  
        while self.parents[path2root[-1]]!=0:path2root.append(self.parents[path2root[-1]])
        path2root.pop(0) # end-effector's rotation can't affects its position
        ax,ay,az,lambda_ = np.array((1,0,0)),np.array((0,1,0)),np.array((0,0,1)),0.05   # local axis
        for _ in range(50):
            xs,os = self.FK(ret,root_pos)  # 这里其实可以优化，只需要FK这一个骨骼链(path2root)就行了,而不用整个skeleton
            theta = np.array([rot.as_euler('XYZ')for rot in ret]).reshape(-1)  # 参数化旋转为一系列欧拉角,注意这里的theta是所有joint而不是path2root里的joint
            if np.linalg.norm(xs[target_idx]-target_pos)<1e-3: break
            J,curr_target_pos = [],xs[target_idx]
            for jidx in path2root:
                pidx,dp = self.parents[jidx],curr_target_pos-xs[jidx]
                J.append(np.cross(os[pidx].apply(ax),dp))
                J.append(np.cross((os[pidx]*R.from_euler('X',theta[3*jidx+0])).apply(ay),dp))
                J.append(np.cross((os[pidx]*R.from_euler('XY',(theta[3*jidx+0],theta[3*jidx+1]))).apply(az),dp))
            J = np.array(J).T
            assert J.shape == (3,3*len(path2root))
            d_theta = J.T@scipy.linalg.solve(J@J.T + lambda_*np.eye(3),target_pos - curr_target_pos,assume_a = 'pos')
            for i,jidx in enumerate(path2root): ret[jidx] = R.from_euler('XYZ',(theta[jidx*3+0]+d_theta[3*i+0],theta[jidx*3+1]+d_theta[3*i+1],theta[jidx*3+2]+d_theta[3*i+2]))
        return ret

    # Collect skeleton segments,i.e. sticks for drawing.Note that segments[i] corresponds to joints[i+1],e.g. seg0 is joint0<==>joint1,which corresponds to joint1
    def Segments(self,positions):
        assert  len(positions)==len(self.names),f'{len(self.positions)} vs {len(self.names)}'
        return np.array([p for jidx,pidx in enumerate(self.parents) if pidx!=-1 for p in (positions[pidx],positions[jidx])])
    def Print(self):
        print('joint cnt is (exculde end-site) ',self.joint_cnt)
        for i,name_parent in enumerate(zip(self.names,self.parents)):print(i,name_parent)
        print('children-----------------------------------------------')
        for i,children in enumerate(self.children):print(i,children)
       
Visualizer( Skeleton(os.path.join(os.path.dirname(__file__),'assets','walk60.bvh')) ).RigControl()
