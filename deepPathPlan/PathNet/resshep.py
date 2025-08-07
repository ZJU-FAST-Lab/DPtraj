import numpy as np
import reeds_shepp
import math
rho = 1 / 1.67 # turning radius
step_size = 0.1
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
def reedshep_process(opState, opRot, env):
    #opState 100*2 env 200*200
    endPx = opState[-1,0]
    endPy = opState[-1,1]
    endCos = opRot[-1,0]
    endSin = opRot[-1,1]
    endYaw = math.atan2(endSin, endCos)
    endQ = (endPx, endPy, endYaw)
    for i in range(170,opState.shape[0]-1):
        px = opState[i,0]
        py = opState[i,1]
        cos = opRot[i,0]
        sin = opRot[i,1]
        yaw = math.atan2(sin, cos)
        q0 = (px, py, yaw)
        qs = reeds_shepp.path_sample(q0, endQ, rho, step_size)
        xs = [q[0] for q in qs]
        ys = [q[1] for q in qs]
        angles = [q[2] for q in qs]
        collision = False
        for j in range(len(xs)):
            gridx = int((xs[j] + 10.0) / 0.1)
            gridy = int((ys[j] + 10.0) / 0.1)

            if env[gridx][gridy] < 0.2:
                collision = True
                break
        if(collision==False):
            newOpState=  np.zeros((i+len(xs),2))
            newOpRot=  np.zeros((i+len(xs),2))
            newOpState[:i,:] = opState[:i,:]
            newOpRot[:i,:] = opRot[:i,:]
            for j in range(len(xs)):
                rpx = xs[j]
                rpy = ys[j]
                ryaw = angles[j]
                newOpState[i+j][0] = rpx
                newOpState[i+j][1] = rpy
                newOpRot[i+j][0] = math.cos(ryaw)
                newOpRot[i+j][1] = math.sin(ryaw)
            return newOpState, newOpRot
    return opState, opRot