#include <path_searching/nnPath.h>
#include <tools/tic_toc.hpp>
#include <torch/torch.h>
#include "torch/script.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <fstream>
using namespace cv;
namespace path_searching{
    void NnPathSearch::init(ConfigPtr config, ros::NodeHandle& nh){
        vis_tool_.reset(new visualization::Visualization(nh));
        vis_tool_->registe<visualization_msgs::MarkerArray>("/visualization/nnarrowTraj");
        vis_tool_->registe<nav_msgs::Path>("/visualization/nnPath");
        model_path_ = config->model_path;
        std::cout << "Check1" << std::endl;
        mpnet_ = torch::jit::load(model_path_,torch::kCUDA);
        std::cout << "Check2" << std::endl;
        if(use_half){
            mpnet_.to(torch::kHalf);
        }
        mpnet_.eval();
        torch::NoGradGuard no_grad_;
        enable_shot_ = config->enable_shot;
        //warm start

        for(int i = 0; i < 30; i++){
           warm();
        }
        std::cout << "neural network finish warm start" << std::endl;

        max_forward_vel = config->vmax-0.2;
        // max_forward_vel = 1.01;//hzc?
        max_forward_acc = config->lonAccmax;
        max_backward_vel = max_forward_vel;
        max_backward_acc = max_forward_acc;
        max_cur = config->kmax-0.3;
        shotptr =std::make_shared<ompl::base::ReedsSheppStateSpace>(1.0/max_cur);
    }
    void NnPathSearch::warm(){
        std::vector<torch::IValue> ws_inputs;
        torch::Tensor ws0 = torch::zeros({1,4,200,200});
        torch::Tensor ws1 = torch::zeros({1,trajLen,2});
        torch::Tensor ws2 = torch::zeros({1,trajLen,2});        
        torch::Tensor ws3 = torch::zeros({1,trajLen,20,20});
        // torch::Tensor ws3 = torch::zeros({1,trajLen,25,25});
        if(use_half){
            ws_inputs.push_back(ws0.to(at::kCUDA).to(torch::kHalf));
            ws_inputs.push_back(ws1.to(at::kCUDA).to(torch::kHalf));
            ws_inputs.push_back(ws2.to(at::kCUDA).to(torch::kHalf));
            ws_inputs.push_back(ws3.to(at::kCUDA).to(torch::kHalf));    
        }
        else{
            ws_inputs.push_back(ws0.to(at::kCUDA));
            ws_inputs.push_back(ws1.to(at::kCUDA));
            ws_inputs.push_back(ws2.to(at::kCUDA));
            ws_inputs.push_back(ws3.to(at::kCUDA));
        }
        auto outputs = mpnet_.forward(ws_inputs).toTuple();
        torch::Tensor r1 = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();
        torch::Tensor r2 = outputs->elements()[1].toTensor().to(at::kCPU).squeeze();
        return;
    }
    void NnPathSearch::intialMap(map_util::OccMapUtil *map_itf){
        frontend_map_itf_ = map_itf;
        Eigen::Vector2i dims = frontend_map_itf_->getDim();
        double startT = ros::Time::now().toSec();
        // for(int i = 0; i < dims[0]; i++){
        //     for(int j = 0; j < dims[1]; j++){
        //         env_tensor[0][0][i][j] =  frontend_map_itf_->getDistance(Eigen::Vector2i(i,j));
        //     }
        // }

        env_tensor = torch::from_blob(frontend_map_itf_->getEsdfData(), {1,1,200,200}, torch::kFloat64);
        env_tensor = torch::transpose(env_tensor, 2, 3);
        // std::cout <<"env_tensor.sizes: "<<env_tensor.sizes()<<"\n";
        // std::cout <<"esdf: "<<frontend_map_itf_->getDistance(Eigen::Vector2i(50,160))<<"\n";
        // std::cout <<"env_tensor: "<<env_tensor[0][0][50][160]<<"\n";
        double endT = ros::Time::now().toSec();
        // ROS_WARN_STREAM("map T: "<<1000.0*(endT-startT) << " ms");



    }
    void NnPathSearch::search(Eigen::Vector3d start_state, Eigen::Vector3d end_state){

        torch::NoGradGuard no_grad_;
        int receptive_field = 16;
        Eigen::Vector2i goalpos, startpos;
        startpos = frontend_map_itf_->floatToInt(start_state.head(2));
        goalpos = frontend_map_itf_->floatToInt(end_state.head(2));

        int goal_start_x = std::max(0, goalpos[0]- receptive_field/2);
        int goal_start_y = std::max(0, goalpos[1]- receptive_field/2);
        int goal_end_x = std::min(200, goalpos[0]+ receptive_field/2);
        int goal_end_y = std::min(200, goalpos[1]+ receptive_field/2);

        context_map.index_put_({Slice(None), Slice(None), Slice(goal_start_x,goal_end_x),Slice(goal_start_y, goal_end_y)},1);
        context_cos_map.index_put_({Slice(None), Slice(None), Slice(goal_start_x,goal_end_x), Slice(goal_start_y,goal_end_y)}, cos(end_state[2]));
        context_sin_map.index_put_({Slice(None), Slice(None), Slice(goal_start_x,goal_end_x), Slice(goal_start_y,goal_end_y)}, sin(end_state[2]));


        int start_start_x = std::max(0, startpos[0]- receptive_field/2);
        int start_start_y = std::max(0, startpos[1]- receptive_field/2);
        int start_end_x = std::min(200, startpos[0]+ receptive_field/2);
        int start_end_y = std::min(200, startpos[1]+ receptive_field/2);

        context_map.index_put_({Slice(None), Slice(None), Slice(start_start_x,start_end_x),Slice(start_start_y, start_end_y)},-1);
        context_cos_map.index_put_({Slice(None), Slice(None), Slice(start_start_x,start_end_x), Slice(start_start_y,start_end_y)}, cos(start_state[2]));
        context_sin_map.index_put_({Slice(None), Slice(None), Slice(start_start_x,start_end_x), Slice(start_start_y,start_end_y)}, sin(start_state[2]));






        input = torch::cat({env_tensor,context_map,context_cos_map,context_sin_map },1);


        label_opState[0][0][0] = start_state[0];
        label_opState[0][0][1] = start_state[1];
        label_opState[0][trajLen-1][0] = end_state[0];
        label_opState[0][trajLen-1][1] = end_state[1];
        label_opRot[0][0][0] = cos(start_state[2]);
        label_opRot[0][0][1] = sin(start_state[2]);
        label_opRot[0][trajLen-1][0] = cos(end_state[2]);
        label_opRot[0][trajLen-1][1] = sin(end_state[2]);
        {
            // int sgx = floor((start_state[0]+10.0)/0.8);
            // int sgy = floor((start_state[1]+10.0)/0.8);
            // int egx = floor((end_state[0]+10.0)/0.8);
            // int egy = floor((end_state[1]+10.0)/0.8);
            int sgx = floor((start_state[0]+10.0)/1.0);
            int sgy = floor((start_state[1]+10.0)/1.0);
            int egx = floor((end_state[0]+10.0)/1.0);
            int egy = floor((end_state[1]+10.0)/1.0);
            label_anchors[0][0][sgx][sgy] = 1;
            label_anchors[0][trajLen-1][egx][egy] = 1;
        }
        
                
        std::vector<torch::IValue> all_input;


        if(use_half){
            all_input.push_back(input.to(at::kCUDA).to(torch::kHalf));
            all_input.push_back(label_opState.to(at::kCUDA).to(torch::kHalf));
            all_input.push_back(label_opRot.to(at::kCUDA).to(torch::kHalf));
            all_input.push_back(label_anchors.to(at::kCUDA).to(torch::kHalf));
        }
        else{
            all_input.push_back(input.to(at::kCUDA));   
            all_input.push_back(label_opState.to(at::kCUDA));
            all_input.push_back(label_opRot.to(at::kCUDA));
            all_input.push_back(label_anchors.to(at::kCUDA));
        }
        auto outputs = mpnet_.forward(all_input).toTuple();
        opState = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();//100*2
        opRot = outputs->elements()[1].toTensor().to(at::kCPU).squeeze();//100*2
        has_path_ = true;
        if(enable_shot_){
            // double t1 = ros::Time::now().toSec();
            reedshep_process(opState, opRot);
            // double t2 = ros::Time::now().toSec();
            // std::cout << "rs time: "<<(t2-t1)*1000.0 << " ms \n";
        }

        return;
    }
    void NnPathSearch::reedshep_process(torch::Tensor& opState, torch::Tensor& opRot){
        int startIdx = 0.70 * trajLen;
        namespace ob = ompl::base;
        namespace og = ompl::geometric;
        ob::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
        std::vector<double> reals;
        to[0] = opState[trajLen-1][0].item<double>();
        to[1] = opState[trajLen-1][1].item<double>();
        to[2] = atan2(opRot[trajLen-1][1].item<double>(), opRot[trajLen-1][0].item<double>());
        for(unsigned int i = startIdx; i < trajLen - 1; i++){
            from[0] = opState[i-1][0].item<double>();
            from[1] = opState[i-1][1].item<double>();
            from[2] = atan2(opRot[i-1][1].item<double>(), opRot[i-1][0].item<double>()); 
            double len = shotptr->distance(from(), to()); 
            bool isocc =false;
            std::vector<Eigen::Vector3d> shot_path; 
            for (double l = 0.0; l <=len; l += frontend_map_itf_->getRes()*2.0)
            {
                shotptr->interpolate(from(), to(), l/len, s());
                reals = s.reals();
                frontend_map_itf_->CheckIfCollisionUsingPosAndYaw(Eigen::Vector3d(reals[0], reals[1], reals[2]), &isocc, 0.0);
                if(isocc)
                    break;
                shot_path.emplace_back(reals[0], reals[1], reals[2]);
            }
            shot_path.emplace_back(to[0], to[1], to[2]);            
            if(isocc==false){
                //collision-free
                torch::Tensor newOpState = torch::zeros({i+shot_path.size(),2});
                torch::Tensor newOpRot = torch::zeros({i+shot_path.size(),2});
                newOpState.index({Slice(0,i),Slice(None)}) = opState.index({Slice(0,i),Slice(None)});
                newOpRot.index({Slice(0,i),Slice(None)}) = opRot.index({Slice(0,i),Slice(None)});
                for(int j = 0; j < shot_path.size(); j++){
                    newOpState[i+j][0] = shot_path[j][0];
                    newOpState[i+j][1] = shot_path[j][1];
                    newOpRot[i+j][0] = cos(shot_path[j][2]);
                    newOpRot[i+j][1] = sin(shot_path[j][2]);
                }
                opState = newOpState;
                opRot = newOpRot;
                // ROS_WARN_STREAM("neural shot successfully i: "<<i);
                return;
            }

        }

        return;
    }
    void NnPathSearch::reset(){
        context_map = torch::zeros({1,1,200,200});
        context_cos_map = torch::zeros({1,1,200,200});
        context_sin_map = torch::zeros({1,1,200,200});//reset
        label_opState =  torch::zeros({1,trajLen,2});
        label_opRot =  torch::zeros({1,trajLen,2});
        label_anchors =  torch::zeros({1,trajLen,20,20});
        // label_anchors =  torch::zeros({1,trajLen,25,25});
        has_path_ = false;
        return;
    }
    void NnPathSearch::display(){
        if(has_path_){
            std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> nn_arrows;
            std::vector<Eigen::Vector3d> nn_points;
            for(int i = 0; i < opState.sizes()[0]; i++){
                double px = opState[i][0].item<double>();
                double py = opState[i][1].item<double>();
                Eigen::Vector2d pos1, pos2;
                pos1 << px, py;
                pos2 =  pos1 + 0.30*Eigen::Vector2d(opRot[i][0].item<double>(), opRot[i][1].item<double>());
                nn_arrows.push_back(std::pair<Eigen::Vector2d, Eigen::Vector2d>(pos1, pos2));
                nn_points.push_back(Eigen::Vector3d(px, py, 0.2));
            }
            vis_tool_->visualize_arrows(nn_arrows, "/visualization/nnarrowTraj", visualization::green);
            vis_tool_->visualize_path(nn_points, "/visualization/nnPath");
            

        }
        return;
    }
    void NnPathSearch::getKinoNode(KinoTrajData &flat_trajs){
        flat_trajs.clear();
        sampleTraj.clear();
        for(int i = 0; i < opState.sizes()[0]; i++){
            double px = opState[i][0].item<double>();
            double py = opState[i][1].item<double>();
            double c = opRot[i][0].item<double>();
            double s = opRot[i][1].item<double>();
            double yaw = atan2(s, c);
            sampleTraj.emplace_back(px,py,yaw);
        }
        std::vector<Eigen::Vector3d> traj_pts;  // 3, N
        std::vector<double> thetas;
        /*divide the whole shot traj into different segments*/   
        shot_lengthList.clear();
        shot_timeList.clear();
        shotindex.clear();
        shot_SList.clear(); 
        int lastS = (sampleTraj[1]-sampleTraj[0]).head(2).dot(Eigen::Vector2d(cos(sampleTraj[1][2]),sin(sampleTraj[1][2])))>=0?1:-1;
        shotindex.push_back(0);
        double tmpl = 0.0;
        //hzchzc attention please!!! max_v must = min_v  max_acc must = min_acc
        for(int i = 0; i<sampleTraj.size()-1; i++){
            Eigen::Vector3d state1 = sampleTraj[i];
            Eigen::Vector3d state2 = sampleTraj[i+1];
            int curS = (state2-state1).head(2).dot(Eigen::Vector2d(cos(state2[2]),sin(state2[2]))) >=0 ? 1:-1;
            if(curS*lastS >= 0){
                tmpl += (state2-state1).head(2).norm();
            }
            else{  
                shotindex.push_back(i);
                shot_SList.push_back(lastS);
                shot_lengthList.push_back(tmpl);
                if(lastS>0)
                shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc));
                else
                shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc));
                tmpl = (state2-state1).head(2).norm();
            }       
            lastS = curS;
        }
        shot_SList.push_back(lastS);
        shot_lengthList.push_back(tmpl);
        if(lastS>0)
            shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc));
        else
            shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc));
        shotindex.push_back(sampleTraj.size()-1);

        /*extract flat traj  
        the flat traj include the end point but not the first point*/
        for(int i=0;i<shot_lengthList.size();i++){
            // double locallength = shot_lengthList[i];
            int sig = shot_SList[i];
            std::vector<Eigen::Vector3d> localTraj;localTraj.assign(sampleTraj.begin()+shotindex[i],sampleTraj.begin()+shotindex[i+1]+1);
            traj_pts.clear();
            thetas.clear();        
            for(const auto pt:localTraj){
                traj_pts.emplace_back(pt[0],pt[1],0.0);
                thetas.push_back(pt[2]);
            }
            FlatTrajData flat_traj;
            flat_traj.traj_pts = traj_pts;
            flat_traj.thetas = thetas;
            flat_traj.singul = sig;
            flat_traj.duration = shot_timeList[i];
            flat_trajs.push_back(flat_traj);
        }
        totalTrajTime = 0.0;
        for(const auto dt : shot_timeList){
            totalTrajTime += dt; 
        }
    }
    double NnPathSearch::evaluateDuration(double length, double max_vel, double max_acc, double startV, double endV){
        double critical_len; //the critical length of two-order optimal control, acc is the input
        double startv2 = pow(startV,2);
        double endv2 = pow(endV,2);
        double maxv2 = pow(max_vel,2);
        critical_len = (maxv2-startv2)/(2*max_acc)+(maxv2-endv2)/(2*max_acc);
        if(length>=critical_len){
            return (max_vel-startV)/max_acc+(max_vel-endV)/max_acc+(length-critical_len)/max_vel;
        }
        else{
            double tmpv = sqrt(0.5*(startv2+endv2+2*max_acc*length));
            return (tmpv-startV)/max_acc + (tmpv-endV)/max_acc;
        }
    }

    Eigen::Vector3d NnPathSearch::evaluatePos(double t){
        t = std::min<double>(std::max<double>(0,t),totalTrajTime);

        int index = -1;
        double tmpT = 0;
        double CutTime;
        //locate the local traj
        for(int i = 0;i<shot_timeList.size();i++){
        tmpT+=shot_timeList[i];
        if(tmpT>=t) {
            index = i; 
            CutTime = t-tmpT+shot_timeList[i];
            break;
            }
        }
        double localtime = shot_timeList[index];
        double locallength = shot_lengthList[index];
        int front = shotindex[index]; int back =  shotindex[index+1];
        std::vector<Eigen::Vector3d> localTraj;localTraj.assign(sampleTraj.begin()+front,sampleTraj.begin()+back+1);
        //find the nearest point
        double arclength;
        if(shot_SList[index] > 0)
            arclength= evaluateLength(CutTime,locallength,localtime,max_forward_vel,max_forward_acc);
        else
            arclength= evaluateLength(CutTime,locallength,localtime,max_backward_vel, max_backward_acc);

        double tmparc = 0;
        for(int i = 0; i < localTraj.size()-1;i++){
            tmparc += (localTraj[i+1]-localTraj[i]).head(2).norm();
            if(tmparc>=arclength){
                double l1 = tmparc-arclength;
                double l = (localTraj[i+1]-localTraj[i]).head(2).norm();
                double l2 = l-l1;//l2
                Eigen::Vector3d state = l1/l*localTraj[i]+l2/l*localTraj[i+1];
                if(fabs(localTraj[i+1][2]-localTraj[i][2])>=M_PI){   
                double normalize_yaw;
                if(localTraj[i+1][2]<=0){
                    normalize_yaw = l1/l*localTraj[i][2]+l2/l*(localTraj[i+1][2]+2*M_PI);
                }
                else if(localTraj[i][2]<=0){
                    normalize_yaw = l1/l*(localTraj[i][2]+2*M_PI)+l2/l*localTraj[i+1][2];
                }
                state[2] = normalize_yaw;
                }
                return state;
            }
        }
        return localTraj.back();
    }

    double NnPathSearch::evaluateLength(double curt,double locallength,double localtime,double max_vel, double max_acc, double startV, double endV){
        double critical_len; //the critical length of two-order optimal control, acc is the input
        if(startV>max_vel||endV>max_vel){
        ROS_ERROR("kinoAstar:evaluateLength:start or end vel is larger that the limit!");
        }
        double startv2 = pow(startV,2);
        double endv2 = pow(endV,2);
        double maxv2 = pow(max_vel,2);
        critical_len = (maxv2-startv2)/(2*max_acc)+(maxv2-endv2)/(2*max_acc);
        if(locallength>=critical_len){
        double t1 = (max_vel-startV)/max_acc;
        double t2 = t1+(locallength-critical_len)/max_vel;
        if(curt<=t1){
            return startV*curt + 0.5*max_acc*pow(curt,2);
        }
        else if(curt<=t2){
            return startV*t1 + 0.5*max_acc*pow(t1,2)+(curt-t1)*max_vel;
        }
        else{
            return startV*t1 + 0.5*max_acc*pow(t1,2) + (t2-t1)*max_vel + max_vel*(curt-t2)-0.5*max_acc*pow(curt-t2,2);
        }
        }
        else{
        double tmpv = sqrt(0.5*(startv2+endv2+2*max_acc*locallength));
        double tmpt = (tmpv-startV)/max_acc;
        if(curt<=tmpt){
            return startV*curt+0.5*max_acc*pow(curt,2);
        }
        else{
            return startV*tmpt+0.5*max_acc*pow(tmpt,2) + tmpv*(curt-tmpt)-0.5*max_acc*pow(curt-tmpt,2);
        }
        }
    }


}