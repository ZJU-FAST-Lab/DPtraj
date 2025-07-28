#include <plan_manage/plan_manage.h>
#include <tf/tf.h>
#include <tools/tic_toc.hpp>
#include <arcPlan/Trajopt_alm.hpp>
#include <fstream>
#include <random>
using namespace plan_manage;
using namespace std;
void PlanManager::init(ros::NodeHandle& nh){
    vis_tool.reset(new visualization::Visualization(nh));
    config_.reset(new Config(nh));
    gridMap.setParam(config_, nh);
    //read map






    hasTarget = false;
    hasOdom = false;
    processTimer = nh.createTimer(ros::Duration(0.02), &PlanManager::process, this);
    cudaWarmTimer = nh.createTimer(ros::Duration(0.01), &PlanManager::warm,this);

    targetSub = nh.subscribe("/move_base_simple/goal", 1, &PlanManager::targetCallback, this);
    odomSub = nh.subscribe("/ugv/odometry", 1, &PlanManager::odomCallback, this);
    vis_tool->registe<nav_msgs::Path>("/visualization/kinoPath");
    vis_tool->registe<nav_msgs::Path>("/visualization/kinoPathNn");
    vis_tool->registe<nav_msgs::Path>("/visualization/optTraj");
    vis_tool->registe<nav_msgs::Path>("/visualization/optTraj_kino");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/debugTraj");
    vis_tool->registe<nav_msgs::Path>("/visualization/NewTraj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/waittoRefine");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/optArrowTraj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/optArrowTraj2");
    vis_tool->registe<decomp_ros_msgs::PolyhedronArray>("/visualization/sfc");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj2");
    vis_tool->registe<nav_msgs::Path>("/visualization/debugRefinedTraj2");
    vis_tool->registe<nav_msgs::Path>("/visualization/debugSe2Traj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/arrowTraj");

    vis_tool->registe<sensor_msgs::PointCloud2>("/visualization/wapoints_nn");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_nn");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/fullshapeTraj_nn");

    vis_tool->registe<sensor_msgs::PointCloud2>("/visualization/wapoints_kinoastar");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_kinoastar");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/fullshapeTraj_kinoastar");


    pieceTime = config_->pieceTime;
    ROS_INFO("Begin to read map");

    double flatObcs[30000] = {0};
    nh.getParam("scene", scene);
    
    std::ifstream fin(ros::package::getPath("plan_manage") + "/testdata/"+scene+"/obcs/obc"+std::to_string(29500)+std::string(".dat"), std::ios::binary);
    fin.read((char*)flatObcs, sizeof(double) * 30000);
    fin.close();
    pcl::PointCloud<pcl::PointXYZ> cloud_map;
    cloud_map.points.clear();
    for(int j = 0; j < 30000; j+=2){
        pcl::PointXYZ pt;
        pt.x = flatObcs[j];
        pt.y = flatObcs[j+1];
        pt.z = 0.0;
        cloud_map.points.push_back(pt);
    }
    cloud_map.width = cloud_map.points.size();
    cloud_map.height = 1;
    cloud_map.is_dense = true;
    sensor_msgs::PointCloud2 global_msg;
    pcl::toROSMsg(cloud_map, global_msg);
    global_msg.header.frame_id = "world";
    gridMap.setEnv(global_msg);

    std::cout << "Check Point1" << std::endl;
    neural_path_finder_.reset(new path_searching::NnPathSearch);
    std::cout << "Check Point2" << std::endl;
    neural_path_finder_->init(config_,nh);
    std::cout << "Check Point3" << std::endl;
    neural_path_finder_->intialMap(&gridMap);
    ROS_INFO("Finish Reading Map");

    kino_path_finder_.reset(new path_searching::KinoAstar);
    kino_path_finder_->init(config_, nh, true);  
    kino_path_finder_->intialMap(&gridMap);
    
    kino_path_finder_2.reset(new path_searching::KinoAstar);
    kino_path_finder_2->init(config_, nh);  
    kino_path_finder_2->intialMap(&gridMap);



}
void PlanManager::warm(const ros::TimerEvent &){
    neural_path_finder_->warm();
    if(kino_path_finder_->use_network){
        kino_path_finder_->warm();
    }
    return;
}


void PlanManager::odomCallback(const nav_msgs::OdometryPtr &msg)
{
    odom[0] = msg->pose.pose.position.x;
    odom[1] = msg->pose.pose.position.y;
    Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z);
    Eigen::Matrix3d R(q);
    odom[2] = atan2(R.col(0)[1],R.col(0)[0]);
    hasOdom = true;
    return;
}
void PlanManager::targetCallback(const geometry_msgs::PoseStamped &msg){
    ROS_INFO("Recieved target!");
    targetPose <<  msg.pose.position.x, msg.pose.position.y,
                tf::getYaw(msg.pose.orientation);
    std::cout<<"targetPose: "<<targetPose.transpose()<<std::endl;
    hasTarget = true;
    return;
}
void PlanManager::process(const ros::TimerEvent &){
    if(!gridMap.has_map_()||pathid>=51) return;
    // neural_path_finder_->intialMap(&gridMap);
    gridMap.publishPCL();
    // std::cout << "pcl number: "<<gridMap.getCloud().size()<<std::endl;
    if(!hasTarget || !hasOdom) return;
    ROS_WARN("Triggering------------------------------------ we begin to plan a trajectory!");
    hasTarget = false;
    for(startid=29551;startid<29600;startid++){
        double flatObcs[30000] = {0};
        std::ifstream fin(
            ros::package::getPath("plan_manage") + "/testdata/"+scene+"/obcs/obc"+std::to_string(startid)+std::string(".dat"), std::ios::binary);
        fin.read((char*)flatObcs, sizeof(double) * 30000);
        fin.close();
        pcl::PointCloud<pcl::PointXYZ> cloud_map;
        cloud_map.points.clear();
        for(int j = 0; j < 30000; j+=2){
            pcl::PointXYZ pt;
            pt.x = flatObcs[j];
            pt.y = flatObcs[j+1];
            pt.z = 0.0;
            cloud_map.points.push_back(pt);
        }
        cloud_map.width = cloud_map.points.size();
        cloud_map.height = 1;
        cloud_map.is_dense = true;
        sensor_msgs::PointCloud2 global_msg;
        pcl::toROSMsg(cloud_map, global_msg);
        global_msg.header.frame_id = "world";
        gridMap.setEnv(global_msg);
        neural_path_finder_->intialMap(&gridMap);
        kino_path_finder_->intialMap(&gridMap);
        kino_path_finder_2->intialMap(&gridMap);
        
//
        for(pathid=5;pathid<51; pathid++){
            double data[1210] = {0};
            std::ifstream fin(ros::package::getPath("plan_manage") + "/testdata/"+scene + "/e"+std::to_string(startid)+"/path"+std::to_string(pathid)+".dat", std::ios::binary);
            if (!fin.good()){
                continue;
            }//hzchzcasds
            fin.read((char*)data, sizeof(double) * 1210);
            fin.close();
            targetPose << data[1205], data[1206], data[1207];
            odom << data[11], data[12], data[13];
            bool checkOcc = false;
            gridMap.CheckIfCollisionUsingPosAndYaw(targetPose, &checkOcc, 0.3);
            if(checkOcc)
                continue;

            gridMap.CheckIfCollisionUsingPosAndYaw(odom, &checkOcc, 0.3);
            if(checkOcc)
                continue;

            if((odom-targetPose).head(2).norm() < 15)
                continue;
            gridMap.publishPCL();
            for(int i = 0; i < 30; i++){
                neural_path_finder_->warm();    
            }
            // std::cout << "mapid: "<<startid << "pathid: "<<pathid <<std::endl;

            {
                //neural path planning
                neural_path_finder_->reset();
                double t1 = ros::Time::now().toSec();
                neural_path_finder_->search(odom, targetPose);
                double t2 = ros::Time::now().toSec();
                // ROS_INFO_STREAM("neural_path_finder time: "<<1000.0*(t2-t1)<<" ms");
                
                neural_path_finder_->display();  
                //use neural network to get initial guess
                path_searching::KinoTrajData nn_trajs_;
                neural_path_finder_->getKinoNode(nn_trajs_);
                int segnum = nn_trajs_.size();
                std::vector<int> refined_singuals; refined_singuals.resize(segnum);
                Eigen::VectorXd refined_rt; refined_rt.resize(segnum);//uniform piece-wise polynomial
                std::vector<Eigen::MatrixXd> refined_inPs_container; refined_inPs_container.resize(segnum);
                std::vector<Eigen::Vector2d> refined_gearPos;refined_gearPos.resize(segnum - 1);
                std::vector<double> refined_angles; refined_angles.resize(segnum - 1);
                double basetime = 0.0;
                for(int i = 0; i < segnum; i++){
                    double timePerPiece = pieceTime;
                    path_searching::FlatTrajData nn_traj = nn_trajs_.at(i);
                    refined_singuals[i] = nn_traj.singul;
                    int piece_nums;
                    double initTotalduration = nn_traj.duration;//hzchzc
                    piece_nums = std::max(int(initTotalduration / timePerPiece + 0.5),1);
                    double dt = initTotalduration / piece_nums; 
                    refined_rt[i] = (initTotalduration / piece_nums);
                    refined_inPs_container[i].resize(2, piece_nums - 1);
                    for(int j = 0; j < piece_nums - 1; j++ ){
                        double t = basetime + (j+1)*dt;
                        Eigen::Vector3d pos = neural_path_finder_->evaluatePos(t);
                        refined_inPs_container[i].col(j) = pos.head(2);
                    }
                    if(i >=1){
                        Eigen::Vector3d pos = neural_path_finder_->evaluatePos(basetime);
                        refined_gearPos[i-1] = pos.head(2);
                        refined_angles[i-1] = pos[2];
                    }
                    basetime += initTotalduration;
                }
                iniState2d << odom[0], refined_singuals[0] * cos(odom[2]), 0.0,
                            odom[1], refined_singuals[0] * sin(odom[2]), 0.0;

                finState2d << targetPose[0], refined_singuals[segnum-1] * cos(targetPose[2]), 0.0,
                            targetPose[1], refined_singuals[segnum-1] * sin(targetPose[2]), 0.0;


                // ROS_INFO("begin to refine");
                PolyTrajOpt::UgvTrajectory optTraj;
                PolyTrajOpt::TrajOpt refinedOpt;

                double t3 = ros::Time::now().toSec();
                int flagSs = refinedOpt.OptimizeSe2Trajectory(
                iniState2d, finState2d, refined_rt,
                refined_inPs_container, refined_gearPos,
                refined_angles, &gridMap,  config_, refined_singuals, vis_tool);
                double t4 = ros::Time::now().toSec();

                printf("\033[32mNeural Network path finder time!,time(ms)=%5.3f \n\033[0m", (t2-t1) * 1000.0-1.0);
                printf("\033[32mBackend trajectory optimizer time!,time(ms)=%5.3f \n\033[0m", (t4-t3) * 1000.0);
                
            }
            /*for benchmark hybridAstar+transformer*/
            {
                kino_path_finder_->reset();
                Eigen::Vector4d iniFs, finFs;
                Eigen::Vector2d initCtrl;
                iniFs << odom, 1.0;
                initCtrl.setZero();
                finFs << targetPose, 1.0;
                TicToc time_profile_tool_;
                time_profile_tool_.tic();
                path_searching::KinoTrajData kino_trajs_;
                double model_time;
                for(int i = 0; i < 20; i++){
                    kino_path_finder_->warm();    
                }
                double t1 = ros::Time::now().toSec();
                int status = kino_path_finder_->search(iniFs, initCtrl, finFs, model_time, true);
                double t2 = ros::Time::now().toSec();
                if(status!=0){
                    kino_path_finder_->getKinoNode(kino_trajs_);
                    std::vector<Eigen::Vector3d> visKinoPath;
                    for(double t = 0.0; t < kino_path_finder_-> totalTrajTime; t += 0.01){
                        Eigen::Vector3d pos;
                        pos = kino_path_finder_->evaluatePos(t);
                        pos[2] = 0.2;
                        visKinoPath.push_back(pos);
                    }
                    vis_tool->visualize_path(visKinoPath, "/visualization/kinoPathNn");
                    //use neural network to get initial guess？？？
                    int segnum = kino_trajs_.size();
                    std::vector<int> refined_singuals; refined_singuals.resize(segnum);
                    Eigen::VectorXd refined_rt;
                    refined_rt.resize(segnum);
                    std::vector<Eigen::MatrixXd> refined_inPs_container;
                    refined_inPs_container.resize(segnum);
                    std::vector<Eigen::Vector2d> refined_gearPos;
                    std::vector<double> refined_angles; 
                    refined_gearPos.resize(segnum - 1); refined_angles.resize(segnum - 1);
                    double basetime = 0.0;
                    for(int i = 0; i < segnum; i++){
                        double timePerPiece = pieceTime;
                        path_searching::FlatTrajData kino_traj = kino_trajs_.at(i);
                        refined_singuals[i] = kino_traj.singul;
                        std::vector<Eigen::Vector3d> pts = kino_traj.traj_pts;
                        int piece_nums;
                        double initTotalduration = 0.0;
                        for(const auto pt : pts){
                            initTotalduration += pt[2];
                        }
                        piece_nums = std::max(int(initTotalduration / timePerPiece + 0.5),1);
                        double dt = initTotalduration / piece_nums; 
                        refined_rt[i] = (initTotalduration / piece_nums);

                        refined_inPs_container[i].resize(2, piece_nums - 1);
                        for(int j = 0; j < piece_nums - 1; j++ ){
                            double t = basetime + (j+1)*dt;
                            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(t);
                            refined_inPs_container[i].col(j) = pos.head(2);
                        }
                        if(i >=1){
                            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(basetime);
                            refined_gearPos[i-1] = pos.head(2);
                            refined_angles[i-1] = pos[2];
                        }
                        basetime += initTotalduration;
                    }
                    iniState2d << odom[0], refined_singuals[0] * cos(odom[2]), 0.0,
                                odom[1], refined_singuals[0] * sin(odom[2]), 0.0;

                    finState2d << targetPose[0], refined_singuals[segnum-1] * cos(targetPose[2]), 0.0,
                                targetPose[1], refined_singuals[segnum-1] * sin(targetPose[2]), 0.0;
                    // ROS_INFO("begin to refine");
                    PolyTrajOpt::UgvTrajectory optTraj;
                    PolyTrajOpt::TrajOpt refinedOpt;
                    double t3 = ros::Time::now().toSec();
                    int flagSs = refinedOpt.OptimizeSe2Trajectory(
                    iniState2d, finState2d, refined_rt,
                    refined_inPs_container, refined_gearPos,
                    refined_angles, &gridMap,  config_, refined_singuals, vis_tool,"kinoastar");
                    double t4 = ros::Time::now().toSec();
                    printf("\033[32mKinoastar+transformer path finder time!,time(ms)=%5.3f \n\033[0m", (t2-t1) * 1000.0);
                    printf("\033[32mBackend trajectory optimizer time!,time(ms)=%5.3f \n\033[0m", (t4-t3) * 1000.0);
                 
                }
                else{
                   ROS_ERROR("kinoastar+transformer failed!!!");
                }
            }
            /*for benchmark hybridAstar*/
            {
                kino_path_finder_2->reset();
                Eigen::Vector4d iniFs, finFs;
                Eigen::Vector2d initCtrl;
                iniFs << odom, 1.0;
                initCtrl.setZero();
                finFs << targetPose, 1.0;
                TicToc time_profile_tool_;
                time_profile_tool_.tic();
                path_searching::KinoTrajData kino_trajs_;
                double model_time;
                double t1 = ros::Time::now().toSec();
                int status = kino_path_finder_2->search(iniFs, initCtrl, finFs, model_time, true);
                double t2 = ros::Time::now().toSec();
                if(status!=0){
                    kino_path_finder_2->getKinoNode(kino_trajs_);
                    std::vector<Eigen::Vector3d> visKinoPath;
                    for(double t = 0.0; t < kino_path_finder_2-> totalTrajTime; t += 0.01){
                        Eigen::Vector3d pos;
                        pos = kino_path_finder_2->evaluatePos(t);
                        pos[2] = 0.2;
                        visKinoPath.push_back(pos);
                    }
                    vis_tool->visualize_path(visKinoPath, "/visualization/kinoPath");
                    //use neural network to get initial guess？？？
                    int segnum = kino_trajs_.size();
                    std::vector<int> refined_singuals; refined_singuals.resize(segnum);
                    Eigen::VectorXd refined_rt;
                    refined_rt.resize(segnum);
                    std::vector<Eigen::MatrixXd> refined_inPs_container;
                    refined_inPs_container.resize(segnum);
                    std::vector<Eigen::Vector2d> refined_gearPos;
                    std::vector<double> refined_angles; 
                    refined_gearPos.resize(segnum - 1); refined_angles.resize(segnum - 1);
                    double basetime = 0.0;
                    for(int i = 0; i < segnum; i++){
                        double timePerPiece = pieceTime;
                        path_searching::FlatTrajData kino_traj = kino_trajs_.at(i);
                        refined_singuals[i] = kino_traj.singul;
                        std::vector<Eigen::Vector3d> pts = kino_traj.traj_pts;
                        int piece_nums;
                        double initTotalduration = 0.0;
                        for(const auto pt : pts){
                            initTotalduration += pt[2];
                        }
                        piece_nums = std::max(int(initTotalduration / timePerPiece + 0.5),1);
                        double dt = initTotalduration / piece_nums; 
                        refined_rt[i] = (initTotalduration / piece_nums);

                        refined_inPs_container[i].resize(2, piece_nums - 1);
                        for(int j = 0; j < piece_nums - 1; j++ ){
                            double t = basetime + (j+1)*dt;
                            Eigen::Vector3d pos = kino_path_finder_2->evaluatePos(t);
                            refined_inPs_container[i].col(j) = pos.head(2);
                        }
                        if(i >=1){
                            Eigen::Vector3d pos = kino_path_finder_2->evaluatePos(basetime);
                            refined_gearPos[i-1] = pos.head(2);
                            refined_angles[i-1] = pos[2];
                        }
                        basetime += initTotalduration;
                    }
                    iniState2d << odom[0], refined_singuals[0] * cos(odom[2]), 0.0,
                                odom[1], refined_singuals[0] * sin(odom[2]), 0.0;

                    finState2d << targetPose[0], refined_singuals[segnum-1] * cos(targetPose[2]), 0.0,
                                targetPose[1], refined_singuals[segnum-1] * sin(targetPose[2]), 0.0;
                    // ROS_INFO("begin to refine");
                    PolyTrajOpt::UgvTrajectory optTraj;
                    PolyTrajOpt::TrajOpt refinedOpt;
                    double t3 = ros::Time::now().toSec();
                    int flagSs = refinedOpt.OptimizeSe2Trajectory(
                    iniState2d, finState2d, refined_rt,
                    refined_inPs_container, refined_gearPos,
                    refined_angles, &gridMap,  config_, refined_singuals, vis_tool,"kinoastar");
                    double t4 = ros::Time::now().toSec();
                    printf("\033[32mKinoastar path finder time!,time(ms)=%5.3f \n\033[0m", (t2-t1) * 1000.0);
                    printf("\033[32mBackend trajectory optimizer time!,time(ms)=%5.3f \n\033[0m", (t4-t3) * 1000.0);
                  
                }
                else{
                   ROS_ERROR("kinoastar failed!!!");
                }

            }
            //wait
            ros::Duration(10.0).sleep();
        }
    }
}
