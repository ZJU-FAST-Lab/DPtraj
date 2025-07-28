#ifndef _NNPATH_H
#define _NNPATH_H
#include <string>
#include <iostream>
#include <sstream>
#include <ros/ros.h>
#include <tools/visualization.hpp>
#include <torch/script.h>
#include <tools/config.hpp>
#include "kino_model.hpp"
#include <tools/gridmap.hpp>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
using namespace torch::indexing;
//3.4
namespace path_searching{


    class NnPathSearch {
    private:
        std::string model_path_;
        torch::jit::script::Module mpnet_;
        std::shared_ptr<visualization::Visualization> vis_tool_;
        map_util::OccMapUtil* frontend_map_itf_;
        torch::Tensor input;
        int trajLen = 200;
        torch::Tensor env_tensor = torch::zeros({1,1,200,200});
        bool use_half = true;
        torch::Tensor context_map = torch::zeros({1,1,200,200});
        torch::Tensor context_cos_map = torch::zeros({1,1,200,200});
        torch::Tensor context_sin_map = torch::zeros({1,1,200,200});//reset
        torch::Tensor label_opState =  torch::zeros({1,trajLen,2});
        torch::Tensor label_opRot =  torch::zeros({1,trajLen,2});
        // torch::Tensor label_anchors =  torch::zeros({1,trajLen,25,25});
        torch::Tensor label_anchors =  torch::zeros({1,trajLen,20,20});
        // std::vector<torch::IValue> all_input;
        torch::Tensor opState,opRot;
        bool has_path_ = false;
        std::vector<Eigen::Vector3d> sampleTraj;
        std::vector<double>  shot_timeList;
        std::vector<double>  shot_lengthList;
        std::vector<int>     shotindex;
        std::vector<int>     shot_SList;
        double totalTrajTime; 

        double max_forward_vel;
        double max_forward_acc;
        double max_backward_vel;
        double max_backward_acc;
        double max_cur;
        bool enable_shot_ = true;
        ompl::base::StateSpacePtr shotptr;


    public:
        NnPathSearch(){};
        ~NnPathSearch(){};
        void init(ConfigPtr config, ros::NodeHandle& nh);
        void intialMap(map_util::OccMapUtil *map_itf);
        void search(Eigen::Vector3d start_state, Eigen::Vector3d end_state);
        void reset();
        void display();
        void warm();

        void getKinoNode(KinoTrajData &flat_trajs);
        double evaluateLength(double curt,double locallength,double localtime, double max_vel, double max_acc, double startV = 0.0, double endV = 0.0);
        double evaluateDuration(double length, double max_vel, double max_acc, double startV = 0.0, double endV = 0.0);
        Eigen::Vector3d evaluatePos(double t);
        void reedshep_process(torch::Tensor& opState, torch::Tensor& opRot);


        
        // std::vector<Eigen::Vector4d> SamplePosList(int N); //px py yaw t 
        // std::vector<Eigen::Vector3d> getSampleTraj();

        // double totalTrajTime;
        // double checkl = 0.2;

        // std::vector<Eigen::Vector4d>  state_list;
        // std::vector<Eigen::Vector3d> acc_list;

        typedef std::shared_ptr<NnPathSearch> NnPathSearchPtr;
        
    };



}





#endif