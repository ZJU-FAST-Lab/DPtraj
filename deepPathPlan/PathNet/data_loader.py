import torch
import numpy as np
import math
from torch.utils.data import Dataset
import cv2
import os

class  pDataset(Dataset):
    def __init__(self):
        self.indexDict = []
        self.startidx = 1
        self.endidx = 29000
        self.pstart = 1
        self.pend = 51
        self.envlist = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dpath = os.path.dirname(os.path.dirname(current_dir)) +"/totalData/"
        self.max_count = 500000
    
        self.envlist = ['forest1', 'office1', 'ruins1']
        for env in self.envlist:
            count = 0 
            for eidx in range(self.startidx,self.endidx):
                if(count > self.max_count):
                    break
                for pathidx in range(self.pstart,self.pend):
                    self.path = self.dpath + env
                    if(os.path.exists(self.path+'/e'+str(eidx)+'/path'+str(pathidx)+'.dat')):
                        self.indexDict.append((eidx, pathidx, self.path))
                        count += 1
                        
        self.envlist = ['forest2', 'office2', 'ruins2']
        self.max_count = int(self.max_count/5)
        for env in self.envlist:
            count = 0 
            for eidx in range(self.startidx,self.endidx):
                if(count > self.max_count):
                    break
                for pathidx in range(self.pstart,self.pend):
                    self.path = self.dpath + env
                    if(os.path.exists(self.path+'/e'+str(eidx)+'/path'+str(pathidx)+'.dat')):
                        self.indexDict.append((eidx, pathidx, self.path))
                        count += 1
            
        self.length_ = len(self.indexDict)
        
    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        (eidx, pathidx, datapath) = self.indexDict[idx]
        fs = cv2.FileStorage(datapath+'/esdfmaps/'+str(eidx)+'.xml', cv2.FILE_STORAGE_READ)
        path = np.fromfile(datapath+'/e'+str(eidx)+'/path'+str(pathidx)+'.dat')
        
        fn = fs.getNode("instance")
        image = fn.mat()
        raw_env = image #H*W
        raw_env = np.expand_dims(raw_env, 0) #H*W
        raw_env = np.where(raw_env > 0.2, 1, 0)

        label = path[10:].reshape(200, 6)
        env = np.expand_dims(image, 0) #H*W
        
        startpos = geom2pix(label[0][1:3])
        goalpos = geom2pix(label[-1][1:3])
        
        data = get_encoder_input(env, goalpos, label[-1][3], startpos, label[0][3])
        opState = label[:, 1:3]
        labelRot = np.zeros((200,2))
        labelRot[:,0] = np.cos(label[:, 3])
        labelRot[:,1] = np.sin(label[:, 3])

        grid = np.floor((opState+10.0)/1.0).astype(int)
        anchors = np.zeros((200,20,20))
        index = [i for i in range(200)]
        anchors[index, grid[:,0], grid[:,1]] = 1
        return  torch.as_tensor(data.copy()).float().contiguous(),torch.as_tensor(raw_env.copy()).float().contiguous(),\
             torch.as_tensor(opState.copy()).float().contiguous(), torch.as_tensor(labelRot.copy()).float().contiguous(),\
             torch.as_tensor(anchors.copy()).float().contiguous()
    
def geom2pix(pos, res=0.1, size=(200, 200)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """

    return (int(np.floor((pos[0] + 10.0) / res)), int(np.floor((pos[1] + 10.0) / res)))
def pix2geom(grid, res=0.1, size=(200, 200)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """

    return (np.float((grid[0]+0.5)*res-10.0), np.float((grid[1]+0.5)*res-10.0))

def get_encoder_input(InputMap, goal_pos, goal_yaw, start_pos, start_yaw):
    '''
    Returns the input map appended with the goal, and start position encoded.
    :param InputMap: The grayscale map
    :param goal_pos: The goal pos of the robot on the costmap.
    :param start_pos: The start pos of the robot on the costmap.
    :returns np.array: The map concatentated with the encoded start and goal pose.
    '''

    receptive_field = 16
    map_size = (200,200)
    context_map = np.zeros(map_size)
    context_cos_map = np.zeros(map_size)
    context_sin_map = np.zeros(map_size)

    goal_start_x = max(0, goal_pos[0]- receptive_field//2)
    goal_start_y = max(0, goal_pos[1]- receptive_field//2)

    goal_end_x = min( map_size[0], goal_pos[0]+ receptive_field//2)
    goal_end_y = min( map_size[1], goal_pos[1]+ receptive_field//2)

    context_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0

    context_cos_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = math.cos(goal_yaw)
    context_sin_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = math.sin(goal_yaw)

    # Mark start region
    start_start_x = max(0, start_pos[0]- receptive_field//2)
    start_start_y = max(0, start_pos[1]- receptive_field//2)
    start_end_x = min( map_size[0], start_pos[0]+ receptive_field//2)
    start_end_y = min( map_size[1], start_pos[1]+ receptive_field//2)

    context_map[start_start_x:start_end_x, start_start_y:start_end_y] = -1.0
    context_cos_map[start_start_x:start_end_x, start_start_y:start_end_y] = math.cos(start_yaw)
    context_sin_map[start_start_x:start_end_x, start_start_y:start_end_y] = math.sin(start_yaw)


    context_map = np.expand_dims(context_map, 0)
    context_cos_map = np.expand_dims(context_cos_map, 0)
    context_sin_map = np.expand_dims(context_sin_map, 0)
    return np.concatenate((InputMap, context_map, context_cos_map, context_sin_map), axis = 0)