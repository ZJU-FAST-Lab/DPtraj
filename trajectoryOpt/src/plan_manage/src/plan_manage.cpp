#include <plan_manage/plan_manage.h>
#include <tf/tf.h>
#include <tools/tic_toc.hpp>
#include <arcPlan/Trajopt_alm.hpp>
#include <difPlan/trajDif_opt.hpp>
// #include <arcPlan/Trajopt_penalty.hpp>
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

    targetSub = nh.subscribe("/move_base_simple/goal", 1, &PlanManager::targetCallback, this);
    odomSub = nh.subscribe("/ugv/odometry", 1, &PlanManager::odomCallback, this);
    vis_tool->registe<nav_msgs::Path>("/visualization/kinoPath");
    vis_tool->registe<nav_msgs::Path>("/visualization/AstarPath");
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
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_dptraj");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_dftpav");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/fullshapeTraj_kinoastar");


    pieceTime = config_->pieceTime;

    kino_path_finder_.reset(new path_searching::KinoAstar);
    kino_path_finder_->init(config_, nh);  
    kino_path_finder_->intialMap(&gridMap);

    // if(config_->useDP){
    //     trajCmdPub = nh.advertise<mpc_controller::DPtrajContainer>("/planner/dptrajectory", 1);  
    // }
    // else{
    //     trajCmdPub = nh.advertise<mpc_controller::PolyTraj>("/planner/trajectory", 1);  
    // }

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
    // targetPose << (6.0+5.3) / 2, 5.5, -M_PI / 2.0;
    // targetPose << 11.701,  4.11058, -1.84895;
    std::cout<<"targetPose: "<<targetPose.transpose()<<std::endl;
    hasTarget = true;
    return;
}
void PlanManager::process(const ros::TimerEvent &){
    if(!gridMap.has_map_()) return;
    // neural_path_finder_->intialMap(&gridMap);
    gridMap.publishPCL();
    // std::cout << "pcl number: "<<gridMap.getCloud().size()<<std::endl;
    if(!hasTarget || !hasOdom) return;
    ROS_WARN("Triggering------------------------------------ we begin to plan a trajectory!");
    hasTarget = false;

    {
    //-9.5,0,0       0,9.5,pi/2        9.5,0,pi      0,-9.5,-pi/2
    //-9.5 5 -2.5 9.5    4.5 5   -2.5 0.5
        vector<Eigen::Vector3d> sampleTraj;
        path_searching::KinoTrajData kino_trajs_;
        std::vector<Eigen::Vector3d> visKinoPath;

        {
            kino_path_finder_->reset();
            Eigen::Vector4d iniFs, finFs;
            iniFs << -9.5,5,   0, 0.0;
            finFs << -2.5,9.5,M_PI_2, 0.0;
            // iniFs <<  0,9.5,-M_PI_2, 0.0;
            // finFs << 9.5,0,0, 0.0;
            TicToc time_profile_tool_;
            time_profile_tool_.tic();
            double model_time;
            double t1 = ros::Time::now().toSec();
            int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
            double t2 = ros::Time::now().toSec();
            std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
            sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
        }
        {
            kino_path_finder_->reset();
            Eigen::Vector4d iniFs, finFs;
            iniFs <<  -2.5,9.5,M_PI_2, 0.0;
            finFs << 4.5,5,M_PI, 0.0;
            TicToc time_profile_tool_;
            time_profile_tool_.tic();
            double model_time;
            double t1 = ros::Time::now().toSec();
            int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
            double t2 = ros::Time::now().toSec();
            std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
            sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
        }
        double ddd = -2.5, sss = 0.5;
        {
            kino_path_finder_->reset();
            Eigen::Vector4d iniFs, finFs;
            iniFs <<  4.5,5,M_PI, 0.0;
            finFs << ddd ,sss,M_PI_2, 0.0;
            TicToc time_profile_tool_;
            time_profile_tool_.tic();
            double model_time;
            double t1 = ros::Time::now().toSec();
            int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
            double t2 = ros::Time::now().toSec();
            std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
            sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
        }

        // {
        //     kino_path_finder_->reset();
        //     Eigen::Vector4d iniFs, finFs;
        //     iniFs <<  0,-9.5,-M_PI_2, 0.0;
        //     finFs << -9.5,0,0, 0.0;
        //     TicToc time_profile_tool_;
        //     time_profile_tool_.tic();
        //     double model_time;
        //     double t1 = ros::Time::now().toSec();
        //     int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
        //     double t2 = ros::Time::now().toSec();
        //     std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
        //     sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
        // }


        kino_path_finder_->getKinoNode(kino_trajs_, sampleTraj);
        // kino_path_finder_->getKinoNode(kino_trajs_);
        for(int i = 0; i < sampleTraj.size(); i++){
            Eigen::Vector3d pos;
            pos.head(2) = sampleTraj[i].head(2);
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
        std::vector<int> pnums;
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
            pnums.push_back(piece_nums);

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
        Eigen::Vector3d ini;ini << -9.5,5.0, 0.0;
        Eigen::Vector3d fin;fin << ddd,sss,M_PI_2;
        // std::cout <<"11111111111111111\n";
        iniState2d << ini[0], refined_singuals[0] * cos(ini[2]), 0.0,
                    ini[1], refined_singuals[0] * sin(ini[2]), 0.0;
        finState2d << fin[0], refined_singuals[segnum-1] * cos(fin[2]), 0.0,
                    fin[1], refined_singuals[segnum-1] * sin(fin[2]), 0.0;
        // ROS_INFO("begin to refine");
        // PolyTrajOpt::TrajOpt refinedOpt;

    // refined_rt[0] = 6.4 / pnums[0];
    // refined_rt[1] = 6.59023 / pnums[1];
    // refined_rt[2] = 6.97846 / pnums[2];
    // refined_rt[3] = 2.83716 / pnums[3];
    refined_rt[0] = 6.6 / pnums[0];
    refined_rt[1] = 7.0 / pnums[1];
    refined_rt[2] = 7.3 / pnums[2];
    refined_rt[3] = 3.6 / pnums[3];
    if(config_->useDP){
        PolyTrajOpt::TrajOpt refinedOpt;
        // 25ms
        double t3 = ros::Time::now().toSec();
        int flagSs = refinedOpt.OptimizeSe2Trajectory(
        iniState2d, finState2d, refined_rt,
        refined_inPs_container, refined_gearPos,
        refined_angles, &gridMap,  config_, refined_singuals, vis_tool,"dptraj");
        PolyTrajOpt::UgvTrajectory optTraj = refinedOpt.getOptTraj();
        // ROS_WARN_STREAM("dptraj total arc: "<<optTraj.getTotalArc());
        // ROS_WARN_STREAM("dptraj traj time: "<<optTraj.getTotalDuration());
        // std::cout << "durations: "<<optTraj.getDurations().transpose()<<std::endl;
        // for(int i = 0; i < optTraj.etas.size(); i++){
        //     std::cout <<"i: "<<i<<" "<<optTraj.Traj_container[i].getTotalDuration()<<std::endl;
        // }
        // ofstream f_dp("/home/han/2023codes/NeuralTraj/dp_traj.txt",ios::out);
        // for(double t = 0.0; t <= optTraj.getTotalDuration(); t+= 0.01){
        //     Eigen::Vector2d pos = optTraj.getPos(t);
        //     double yaw = optTraj.getYaw(t);
        //     f_dp << t <<" "<<pos[0]<<" "<<pos[1]<<" "<<yaw<<std::endl;
        // }


        /*
        ofstream f_omega("/home/han/2023codes/NeuralTraj/data/omega.txt",ios::out);
        ofstream f_phi("/home/han/2023codes/NeuralTraj/data/phi.txt",ios::out);
        ofstream f_t("/home/han/2023codes/NeuralTraj/data/t.txt",ios::out); 
        ofstream f_v("/home/han/2023codes/NeuralTraj/data/v.txt",ios::out); 
        ofstream f_a("/home/han/2023codes/NeuralTraj/data/a.txt",ios::out); 
        double maxomega = 0.0;
        for(double t = 0.0; t <= optTraj.getTotalDuration(); t+=0.01){
            double phi = optTraj.getPhi(t);
            double omega = optTraj.getOmega(t);
            double lonv = optTraj.getVelItem(t);
            f_omega << omega << " ";
            f_phi << phi << " ";
            f_v << lonv << " ";
            f_t << t <<" ";
            if(fabs(omega) > fabs(maxomega)){
                maxomega = omega;
            }
        }
        f_omega.close();
        f_phi.close();
        f_v.close();
        f_a.close();
        f_t.close();*/
        // {
        //     mpc_controller::DPtrajContainer trajmsg;
        //     for(int i = 0; i < optTraj.getSegNum(); i++){
        //         int singual = optTraj.etas[i];
        //         mpc_controller::PolyTrajAC trajSegment;
        //         for(int j = 0; j < optTraj.Traj_container[i].getPieceNum(); j++){
        //             mpc_controller::SinglePolyAC piece;
        //             piece.dt = optTraj.Traj_container[i].tTraj[j].getDuration();
        //             piece.ds = optTraj.Traj_container[i].posTraj[j].getDuration();

        //             Eigen::Matrix<double, 2, 6> c = optTraj.Traj_container[i].posTraj[j].getCoeffMat();
        //             Eigen::Matrix<double, 1, 6> c1 = optTraj.Traj_container[i].tTraj[j].getCoeffMat(); 
        //             for (int k=0; k<6; k++)
        //             {
        //                 piece.coef_x.push_back(c(0, k));
        //                 piece.coef_y.push_back(c(1, k));
        //                 piece.coef_s.push_back(c1(0, k));
        //             }
        //             trajSegment.trajs.push_back(piece);
        //         }
        //         trajmsg.traj_container.push_back(trajSegment);
        //         // trajmsg.reverse
        //         if(singual > 0) trajmsg.reverse.push_back(false);
        //         else trajmsg.reverse.push_back(true);
        //     }
        //     trajmsg.start_time = ros::Time::now();
        //     trajCmdPub.publish(trajmsg);

        // }


    }
    else{
        dftpav::PolyTrajOptimizer refinedOpt;
        // 25ms
        double t3 = ros::Time::now().toSec();
        int flagSs = refinedOpt.OptimizeSe2Trajectory(
        iniState2d, finState2d, refined_rt,
        refined_inPs_container, refined_gearPos,
        refined_angles, &gridMap,  config_, refined_singuals, vis_tool,"dftpav");
        dftpav::DifTrajectory optTraj = refinedOpt.getOptTraj();
        // ROS_WARN_STREAM("dftpav total arc: "<<optTraj.getTotalArc());
        // ROS_WARN_STREAM("dftpav traj time: "<<optTraj.getTotalDuration());
        // ofstream f_dfp("/home/han/2023codes/NeuralTraj/dfp_traj.txt",ios::out);
        // for(double t = 0.0; t <= optTraj.getTotalDuration(); t+= 0.01){
        //     Eigen::Vector2d pos = optTraj.getPos(t);
        //     double yaw = optTraj.getYaw(t);
        //     f_dfp << t <<" "<<pos[0]<<" "<<pos[1]<<" "<<yaw<<std::endl;
        // }

        /*
        ofstream f_omega("/home/han/2023codes/NeuralTraj/omega.txt",ios::out);
        ofstream f_phi("/home/han/2023codes/NeuralTraj/phi.txt",ios::out);
        ofstream f_t("/home/han/2023codes/NeuralTraj/t.txt",ios::out); 
        ofstream f_v("/home/han/2023codes/NeuralTraj/v.txt",ios::out); 
        ofstream f_a("/home/han/2023codes/NeuralTraj/a.txt",ios::out); 
        double maxomega = 0.0;
        for(double t = 0.0; t <= optTraj.getTotalDuration(); t+=0.01){
            double phi = optTraj.getPhi(t);
            double omega = optTraj.getOmega(t);
            double lonv = optTraj.getVelItem(t);
            f_omega << omega << " ";
            f_phi << phi << " ";
            f_v << lonv << " ";
            f_t << t <<" ";
            if(fabs(omega) > fabs(maxomega)){
                maxomega = omega;
            }
        }
        f_omega.close();
        f_phi.close();
        f_v.close();
        f_a.close();
        f_t.close();
        */
        // {
        //     mpc_controller::PolyTraj trajmsg;
        //     for(int i = 0; i < optTraj.getSegNum(); i++){
        //         int singual = optTraj.etas[i];

        //         for(int j = 0; j < optTraj[i].getPieceNum(); j++){
        //             mpc_controller::SinglePoly piece;
        //             if(singual > 0) piece.reverse = false;
        //             else piece.reverse = true;
        //             piece.duration = optTraj[i][j].getDuration();
        //             Eigen::Matrix<double, 2, 6> c = optTraj[i][j].getCoeffMat();
        //             for (int k=0; k<6; k++)
        //             {
        //                 piece.coef_x.push_back(c(0, k));
        //                 piece.coef_y.push_back(c(1, k));
        //             }
        //             trajmsg.trajs.push_back(piece);
        //         }
        //     }
        //     trajmsg.start_time = ros::Time::now();
        //     trajCmdPub.publish(trajmsg);
        // }
    }
    }
}

  