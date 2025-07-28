#ifndef PLAN_MANAGE_HPP
#define PLAN_MANAGE_HPP
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <iostream>
#include <sstream>
#include <ros/ros.h>
#include <path_searching/kino_astar.h>
#include <tools/visualization.hpp>
#include <tools/CorridorBuilder2d.hpp>
#include <torch/script.h>
#include <path_searching/nnPath.h>
#include <nav_msgs/Odometry.h>

#define BUDGET 0.1
namespace plan_manage{
    class PlanManager{
        public:
            PlanManager(){};
            void init(ros::NodeHandle & nh);
        private:
            bool hasTarget = false, hasOdom = false;
            Eigen::Vector3d targetPose, odom;
            std::shared_ptr<visualization::Visualization> vis_tool;
            std::shared_ptr<Config> config_;
            std::unique_ptr<path_searching::KinoAstar> kino_path_finder_,kino_path_finder_2;
            std::unique_ptr<path_searching::NnPathSearch> neural_path_finder_;

            map_util::OccMapUtil gridMap;
            Eigen::Matrix<double, 2, 3> iniState2d, finState2d;
            Eigen::MatrixXd initInnerPts2d;
            std::vector<Eigen::MatrixXd> hPolys;
            void process(const ros::TimerEvent &);
            void warm(const ros::TimerEvent &);
            void targetCallback(const geometry_msgs::PoseStamped &msg);
            void odomCallback(const nav_msgs::OdometryPtr &msg);
            double pieceTime;

            /*ros related*/
            ros::Timer processTimer, cudaWarmTimer;
            ros::Subscriber targetSub, odomSub;
            ros::Publisher trajCmdPub;
            int startid = 1, pathid = 1;
            std::string scene;




    };
}

#endif
