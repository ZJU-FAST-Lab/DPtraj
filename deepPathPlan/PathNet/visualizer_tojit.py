import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from resshep import reedshep_process
import numpy as np
from network import  trajFCNet
from data_loader import geom2pix,get_encoder_input,geom2pix
import cv2
import os
    


def main(args):


    
    mpnet = trajFCNet(4,200,7,l=1.2,use_groundTruth=False)
    model_path='model.pkl'
    mpnet.load_state_dict(torch.load("./models/"+model_path))

    if torch.cuda.is_available():
        mpnet.cuda()


    mpnet.eval()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "totalData/ruins1")
    for eidx in range(29000,30001):
        if(not os.path.exists(filepath+'/e'+str(eidx)+'/path1.dat')):
            continue
        temp=np.fromfile(filepath+'/obcs/obc'+str(eidx)+'.dat')
        env=temp.reshape(15000,2)
        path = np.fromfile(filepath+'/e'+str(eidx)+'/path1.dat')
    
        plt.figure(figsize=(19.2, 10.8))

        plt.scatter(env[:,0], env[:,1], c='black', marker='o', label="ground truth")
        ax = plt.gca()


        raw_path = np.zeros([200,2], float)
        raw_theta = np.zeros(200, float)
        raw_t = np.zeros(200, float)
        raw_v = np.zeros(200, float)
        raw_c = np.zeros(200, float)
        

        for j in range(200):
            raw_t[j] = path[10+j*6]
            raw_path[j,0] = path[10+j*6+1]
            raw_path[j,1] = path[10+j*6+2]
            raw_theta[j] = path[10+j*6+3]
            raw_v[j] = path[10+j*6+4]
            raw_c[j] = path[10+j*6+5] 
        # plt.quiver(raw_path[:,0], raw_path[:,1], 0.1 * np.cos(raw_theta), 0.1*np.sin(raw_theta),color='b',width=0.001, scale=5.0)
        plt.scatter(raw_path[:,0], raw_path[:,1], c='blue', marker='o',s=5, label="ground truth")
        # plt.plot(raw_path[:,0], raw_path[:,1],color='blue', marker='o', linestyle=':', linewidth=2, markersize=1)
        
        fs = cv2.FileStorage(filepath+'/esdfmaps/'+str(eidx)+'.xml', cv2.FILE_STORAGE_READ)
        fn = fs.getNode("instance")
        image = fn.mat()
        raw_env = image#H*W
        raw_env = np.expand_dims(raw_env, 0)#H*W
        raw_env = np.where(raw_env > 0.2, 1, 0)
        # #free is 1, obs = 0
        path = np.fromfile(filepath+'/e'+str(eidx)+'/path1.dat')
        input = path[:10].reshape(2,5)
        label = path[10:].reshape(200, 6)
        env = np.expand_dims(image, 0)#H*W

        
        goalpos = geom2pix(input[1][0:2])
        startpos = geom2pix(input[0][0:2])
        data = get_encoder_input(env, goalpos, input[1][2], startpos, input[0][2])

        label_opDir = np.zeros((200,2))
        label_opDir[:,0] = -np.sin(label[:, 3])
        label_opDir[:,1] = np.cos(label[:, 3])
        label_opState = label[:, 1:3]
        label_Rot = np.zeros((200,2))
        label_Rot[:,0] = np.cos(label[:, 3])
        label_Rot[:,1] = np.sin(label[:, 3])

        label_grid = np.floor((label_opState+10.0)/1.0).astype(int)#200*2
        labelanchors = np.zeros((200,20,20))
        index = [i for i in range(200)]
        labelanchors[index, label_grid[:,0], label_grid[:,1]] = 1

        data = torch.as_tensor(data.copy()).float().unsqueeze(0).contiguous().cuda()
        label_opState = torch.as_tensor(label_opState.copy()).float().contiguous().unsqueeze(0).cuda()
        label_Rot = torch.as_tensor(label_Rot.copy()).float().unsqueeze(0).contiguous().cuda()
        labelanchors = torch.as_tensor(labelanchors.copy()).float().unsqueeze(0).contiguous().cuda()
        
        
        #save model
        mpnet.half()
        data = data.half()
        label_opState = label_opState.half()
        label_Rot = label_Rot.half()
        labelanchors = labelanchors.half()
        opState, opRot,anchors,_= mpnet(data, label_opState,label_Rot,  labelanchors)
        CAB_traced_script_module = torch.jit.trace(mpnet, (data, label_opState,label_Rot, labelanchors))
        CAB_traced_script_module.save("./models/model.pt")
    



        opState=opState.data.cpu().numpy()[0]
        opRot = opRot.data.cpu().numpy()[0]
        env = env[0]
        opState, opRot = reedshep_process(opState, opRot, env)





        for i in range(opState.shape[0]):
            plt.arrow(opState[i,0], opState[i,1],0.05*opRot[i,0] ,0.05*opRot[i,1], width=0.001, color='red')
        plt.plot(opState[:,0], opState[:,1],color='green', marker='o', linestyle=':', linewidth=2, markersize=1)
        plt.scatter(opState[:,0], opState[:,1], c='red', marker='o',s=5, label="planner")

        plt.xlim((-11, 11))
        plt.ylim((-11, 11))
    
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_file','-p', type=str, default='./data/freeEnv/path1002.dat')
    args = parser.parse_args()
    print(args)
    main(args)