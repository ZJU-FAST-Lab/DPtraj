#include <path_searching/kino_astar.h>
#include <sstream>


#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tools/tic_toc.hpp>

using namespace std;
using namespace Eigen;

namespace path_searching
{
  KinoAstar::~KinoAstar()
  {
    for (int i = 0; i < allocate_num_; i++)
    {
      delete path_node_pool_[i];
    }
  }
   void KinoAstar::stateTransit(Eigen::Vector3d &state0,  Eigen::Vector3d &state1,
              Eigen::Vector2d &ctrl_input){
    //helpful var
    double psi = ctrl_input[0]; double s = ctrl_input[1]; 
    if(fabs(psi)>=0.01){
      double k = wheel_base/tan(psi);
      state1[0] = state0[0] + k*(sin(state0[2]+s/k)-sin(state0[2]));
      state1[1] = state0[1] - k*(cos(state0[2]+s/k)-cos(state0[2]));
      state1[2] = state0[2] + s/k;
    } 
    else{
      state1[0] = state0[0] + s * cos(state0[2]);
      state1[1] = state0[1] + s * sin(state0[2]);
      state1[2] = state0[2]; 
    }
    // std::cout << "state0: "<<state0.transpose()<<std::endl;
    // std::cout << "state1: "<<state1.transpose()<<std::endl;
    // std::cout << "ctrl_input: "<<ctrl_input.transpose()<<std::endl;
  }

  void KinoAstar::nnVis(torch::Tensor map){
    // networkVis
    if(use_network){
      sensor_msgs::PointCloud2 globalMap_pcd;
      pcl::PointCloud<pcl::PointXYZ> cloudMap;
      for(int i = 0; i < frontend_map_itf_->getDim()[0]; i++){
        for(int j = 0; j < frontend_map_itf_->getDim()[1]; j++){
          if(map[i][j].item<double>() > neuralBudget){
            pcl::PointXYZ pt;
            Eigen::Vector2d pos = frontend_map_itf_->intToFloat(Eigen::Vector2i(i,j));
            pt.x = pos[0];
            pt.y = pos[1];
            pt.z = 0.2;
            cloudMap.points.push_back(pt); 
          }
        }
      }
        cloudMap.width = cloudMap.points.size();
        cloudMap.height = 1;
        cloudMap.is_dense = true;
      pcl::toROSMsg(cloudMap, globalMap_pcd);
      globalMap_pcd.header.stamp = ros::Time::now();
      globalMap_pcd.header.frame_id = "world";
      networkVis.publish(globalMap_pcd);
    }
    return;
  }


  int KinoAstar::search(Eigen::Vector4d start_state, Eigen::Vector2d init_ctrl,
                               Eigen::Vector4d end_state,double& model_time,bool use3d)
  {
    model_time = 0.0;
    if(use_network){
      double debugt1 =ros::Time::now().toSec();
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
      torch::Tensor input = torch::cat({env_tensor,context_map,context_cos_map,context_sin_map },1);
      std::vector<torch::IValue> all_input;
      if(use_half){
        all_input.push_back(input.to(at::kCUDA).to(torch::kHalf));
      }
      else{
        all_input.push_back(input.to(at::kCUDA));   
      }
      auto outputs = mpnet_.forward(all_input).toTuple();
      positive = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();//200*200
      // predClass = outputs->elements()[1].toTensor().to(at::kCPU).squeeze();//20*20
      // nnVis(positive);
      // std::cout <<"debugTime: "<<1000.0*(debugt2-debugt1)<<" ms "<<1000.0*(debugt3-debugt2)<<" ms "<<1000.0*(debugt4-debugt3)<<
      // " ms "<<1000.0*(debugt5-debugt4)<<" ms\n";
      double xxxx = ros::Time::now().toSec();
      // nm.reserve(positive.numel());
      // std::copy(positive.data_ptr<float>(), positive.data_ptr<float>() + positive.numel(), std::back_inserter(nm));
      positive = positive.to(torch::kFloat);
      nm.assign(positive.data_ptr<float>(), positive.data_ptr<float>() + positive.numel());
      double yyyy = ros::Time::now().toSec();
      // std::vector<float> nm(positive.data_ptr<float>(),positive.data_ptr<float>()+positive.numel());//将tensor转换为vector
      // for(int x = 0; x<200; x++){
      //   for(int y = 0; y<200; y++){
      //     if(positive[x][y].item<double>()>0.1){
      //       std::cout <<"x: "<<x << " y: "<<y << std::endl;
      //       std::cout <<"positive: "<<positive[x][y].item<double>()<<"\n";
      //       std::cout <<"nmx: "<<nm[frontend_map_itf_->getIndex(Eigen::Vector2i(x,y))]<<"\n";
      //       std::cout <<"nmy: "<<nm[frontend_map_itf_->getIndex(Eigen::Vector2i(y,x))]<<"\n";
      //     }
      //   }
      // }
      // nnVis(positive);
      double debugt2 = ros::Time::now().toSec();

      // ROS_WARN_STREAM("prt time: "<<1000.0*(yyyy-xxxx)<<" ms");
      // ROS_WARN_STREAM("neural map time: "<<1000.0*(debugt2-debugt1)<<" ms ");
      model_time = 1000.0*(debugt2-debugt1);
    }


    double startTime = ros::Time::now().toSec();
    int expanded_num = 0;
    bool isocc = false;  bool initsearch = false;
    TicToc time_profile_tool_;
    time_profile_tool_.tic();
    frontend_map_itf_->CheckIfCollisionUsingPosAndYaw(start_state.head(3),&isocc);
    if(isocc){
      ROS_ERROR("KinoAstar: head is not free!");
      return 0;
    }
    frontend_map_itf_->CheckIfCollisionUsingPosAndYaw(end_state.head(3),&isocc);
    if(isocc){
      ROS_WARN("KinoAstar: end is not free!");
      return 0;
    }
    start_state_ = start_state;
    start_ctrl = init_ctrl;
    end_state_ = end_state;
    Eigen::Vector2i end_index;
    end_index = posToIndex(end_state.head(2));
    /* ---------- initialize ---------- */
    PathNodePtr cur_node = path_node_pool_[0];
    cur_node->parent = NULL;
    cur_node->state = start_state.head(3);
    cur_node->index = posToIndex(start_state.head(2));
    cur_node->yaw_idx = yawToIndex(start_state[2]);
    cur_node->g_score = 0.0;
    cur_node->input = Eigen::Vector2d(0.0,0.0);
    cur_node->singul = getSingularity(start_state(3));
    cur_node->f_score = lambda_heu_ * getHeu(cur_node->state, end_state);
    cur_node->node_state = IN_OPEN_SET;
    open_set_.push(cur_node);
    use_node_num_ += 1;
    if(!use3d)
      expanded_nodes_.insert(cur_node->index, cur_node);
    else
      expanded_nodes_.insert(cur_node->index, yawToIndex(start_state[2]),cur_node);
    PathNodePtr terminate_node = NULL;
    if(cur_node->singul == 0){ initsearch = true;}
    /* ---------- search loop ---------- */
    while (!open_set_.empty())
    {
        // ros::Duration(0.1).sleep();

      // std::cout<<"cur_state: "<<cur_node->state.transpose()<<"\n";
      /* ---------- get lowest f_score node ---------- */
      cur_node = open_set_.top();

      /* ---------- determine termination ---------- */
      // to decide the near end
      bool reach_horizon = (cur_node->state.head(2) - start_state_.head(2)).norm() >= horizon_;
      double t1 = ros::Time::now().toSec();
      //hzc hzc
      if((cur_node->state.head(2) - end_state_.head(2)).norm()<1.5){
    
        is_shot_sucess(cur_node->state,end_state_.head(3));
      }
      double t2 = ros::Time::now().toSec();
      // std::cout<<"one-shot time: "<<(t2-t1)*1000<<" ms"<<std::endl;
      
      if (is_shot_succ_)
      {
        terminate_node = cur_node;
        retrievePath(terminate_node);
        has_path_ = true;
        if (is_shot_succ_)
        {
          /* one shot trajectory */

          // ROS_WARN("one shot! iter num: %d",iter_num_);
          double endTime = ros::Time::now().toSec();
          // ROS_WARN_STREAM("search Time: "<<(endTime-startTime)*1000.0<<" ms");
          return 1;
        }
  
      }
      double runTime = time_profile_tool_.toc() / 1000.0;
      if(runTime > max_seach_time){
        terminate_node = cur_node;
        retrievePath(terminate_node);
        has_path_ = true;
        if (terminate_node->parent == NULL)
        {

          cout << "[34mKino Astar]: terminate_node->parent == NULL" << endl;
          printf("\033[Kino Astar]: NO_PATH \n\033[0m");
          return 0;
        }
        else
        {
          ROS_WARN("KinoSearch: Reach the max seach time");
          return 0;
        }
      }
      /* ---------- pop node and add to close set ---------- */
      open_set_.pop();
      cur_node->node_state = IN_CLOSE_SET;
      // iter_num_ += 1;
      /* ---------- init state propagation ---------- */
      Eigen::Vector3d cur_state = cur_node->state;
      Eigen::Vector3d pro_state;
      Eigen::Vector2d ctrl_input;
      vector<Eigen::Vector2d> inputs;
      double res = steer_res;//hzchzchzc
      // if(!initsearch ){
      //   if(start_state_[3]>0){
      //     for(double arc = resolution_; arc <= 2*resolution_ + 1e-3; arc += resolution_){
      //       for(double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_* 1.0){
      //         ctrl_input<< steer, arc;
      //         inputs.push_back(ctrl_input);
      //       }
      //     }
      //   }
      //   else{
      //     for(double arc = -resolution_; arc >= -2*resolution_ - 1e-3; arc-= resolution_){
      //       for(double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_* 1.0){
      //         ctrl_input<< steer, arc;
      //         inputs.push_back(ctrl_input);
      //       }
      //     } 
      //   }
      //   initsearch = true;
      // }
      // else{
      for (double arc = -step_arc; arc <= step_arc + 1e-3; arc += 0.5*step_arc){
        if(fabs(arc)<0.05) continue;
        
        for (double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_*1.0)
        {
          ctrl_input << steer, arc;
          inputs.push_back(ctrl_input);
        }
      }
      // }
      /* ---------- state propagation loop ---------- */
      for (auto& input:inputs){
        int singul = input[1]>0?1:-1;
        stateTransit(cur_state, pro_state, input);
        // NodeVis(pro_state);
        iter_num_ += 1;


        // std::cout<<"cur_state: "<<cur_state.transpose()<<" pro: "<<pro_state.transpose()<<" input: "<<input.transpose()<<"\n";
        //check if in the positive region
        bool penalty = false;
        if(use_network){
          Eigen::Vector2i pro_idx = frontend_map_itf_->floatToInt(pro_state.head(2));
          double tt1=  ros::Time::now().toSec();
          // double pbl = positive[pro_idx[0]][pro_idx[1]].item<double>();
          double pbl = nm[frontend_map_itf_->getIndex(Eigen::Vector2i(pro_idx[1],pro_idx[0]))];
          double tt2=  ros::Time::now().toSec();
          if(pbl<neuralBudget){
            penalty = true;
          }
        }
        /* inside map range */
        if (pro_state(0) <= origin_(0) || pro_state(0) >= map_size_3d_(0)*0.5 ||
            pro_state(1) <= origin_(1) || pro_state(1) >= map_size_3d_(1)*0.5)
        {
          // std::cout << "[Kino Astar]: out of map range" << endl;
          //hzc  data generation
          // return 0;
          continue;
        }
        /* not in close set */
        Eigen::Vector2i pro_id = posToIndex(pro_state.head(2));
        double pro_yaw_id = yawToIndex(pro_state[2]);
        PathNodePtr pro_node;
        if(use3d)
          pro_node = expanded_nodes_.find(pro_id, pro_yaw_id);
        else
          pro_node = expanded_nodes_.find(pro_id);
        if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET)
        {
          // std::cout<<"in close set!\n";
          continue;
        }

        // /* not in the same voxel */
        Eigen::Vector2i diff = pro_id - cur_node->index;
        int diff_yaw = pro_yaw_id - cur_node->yaw_idx;
        if (diff.norm() == 0 && ((!use3d) || diff_yaw == 0))
        {
          std::cout<<"diff.norm() == 0 && ((!use3d) || diff_yaw == 0)!\n";
          continue;
        }
        /* collision free */
        Eigen::Vector3d xt;
        bool is_occ = false;
        for (int k = 1; k <= check_num_; ++k)
        {
          double tmparc = input[1] * double(k) / double(check_num_);
          Eigen::Vector2d tmpctrl; tmpctrl << input[0],tmparc;
          stateTransit(cur_state, xt, tmpctrl);
          frontend_map_itf_->CheckIfCollisionUsingPosAndYaw(xt, &is_occ);
          if (is_occ)
          {
            // std::cout<<"occ!\n";
            break;
          }

        }




        if (is_occ)  continue;

        /* ---------- compute cost ---------- */
        double tmp_g_score = 0.0;
        double tmp_f_score = 0.0;
        int lastDir = cur_node->singul;
        if(singul>0){
          tmp_g_score +=  std::fabs(input[1]) * traj_forward_penalty;
        }
        else{
          tmp_g_score += std::fabs(input[1]) * traj_back_penalty;
        }
        // std::cout<<"1111111111111111gscore: "<<tmp_g_score<<"\n";
        if(singul * lastDir < 0){
          tmp_g_score += traj_gear_switch_penalty;
        }
        tmp_g_score += traj_steer_penalty * std::fabs(input[0]) * std::fabs(input[1]);
        tmp_g_score += traj_steer_change_penalty * std::fabs(input[0]-cur_node->input[0]);
        tmp_g_score += cur_node->g_score;
        if(penalty){
          tmp_g_score += 100.0;
        }
        // std::cout<<"2222222222222222222222gscrore: "<<tmp_g_score<<"\n";
        tmp_f_score = tmp_g_score + lambda_heu_ * getHeu(pro_state, end_state);
        // tmp_f_score = tmp_g_score + lambda_heu_ * getObsHeu(pro_state.head(2));
        // std::cout<<"tmpfscore: "<<tmp_f_score<<"\n";
        // std::cout<<"g1: "<<( ctrl_input(1)*ctrl_input(1) + 10*ctrl_input(0)*ctrl_input(0) + w_time_) * tau<<" g0: "<<backward_penalty<<
        // "h: "<<lambda_heu_ * getHeu(pro_state, end_state)<<"\n";
        /* ---------- compare expanded node in this loop ---------- */
        if (pro_node == NULL)
        {
          pro_node = path_node_pool_[use_node_num_];
          pro_node->index = pro_id;
          pro_node->state = pro_state;
          pro_node->yaw_idx = pro_yaw_id;
          pro_node->f_score = tmp_f_score;
          pro_node->g_score = tmp_g_score;
          pro_node->input = input;
          pro_node->parent = cur_node;
          pro_node->node_state = IN_OPEN_SET;
          pro_node->singul = singul;
          open_set_.push(pro_node);
          if(use3d)
            expanded_nodes_.insert(pro_id, pro_yaw_id, pro_node);
          else
            expanded_nodes_.insert(pro_id,pro_node);
          use_node_num_ += 1;
          if (use_node_num_ == allocate_num_)
          {
            cout << "run out of memory." << endl;
            return 0;
          }
        }
        else if (pro_node->node_state == IN_OPEN_SET)
        {

          if (tmp_g_score < pro_node->g_score)
          {
            pro_node->index = pro_id;
            pro_node->state = pro_state;
            pro_node->yaw_idx = pro_yaw_id;
            pro_node->f_score = tmp_f_score;
            pro_node->g_score = tmp_g_score;
            pro_node->input = input;
            pro_node->parent = cur_node;
            pro_node->singul = singul;
          }
        }
        else
        {
          cout << "error type in searching: " << pro_node->node_state << endl;
        }
      }
    }

    /* ---------- open set empty, no path ---------- */
    cout << "open set empty, no path." << endl;
    return 0;
  }


  bool KinoAstar::is_shot_sucess(Eigen::Vector3d state1,Eigen::Vector3d state2){
    
    std::vector<Eigen::Vector3d> path_list;
    double len;
    double ct1 = ros::Time::now().toSec();
    computeShotTraj(state1,state2,path_list,len);
    double ct2 = ros::Time::now().toSec();
    // std::cout<<"compute shot traj time: "<<(ct2-ct1)*1000.0<<" ms"<<std::endl;
    bool is_occ = false;
    double t1 = ros::Time::now().toSec();
    for(unsigned int i = 0; i < path_list.size(); ++i){
        frontend_map_itf_->CheckIfCollisionUsingPosAndYaw(path_list[i], &is_occ);
        if(is_occ) return false;
    }
    double t2 = ros::Time::now().toSec();


    is_shot_succ_ = true;
    return true;
  }



  void KinoAstar::computeShotTraj(Eigen::Vector3d &state1, Eigen::Vector3d &state2,
                                    std::vector<Eigen::Vector3d> &path_list,
                                    double& len){
    namespace ob = ompl::base;
    namespace og = ompl::geometric;
    ob::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
    from[0] = state1[0]; from[1] = state1[1]; from[2] = state1[2];
    to[0] = state2[0]; to[1] = state2[1]; to[2] = state2[2];
    std::vector<double> reals;
    len = shotptr->distance(from(), to());

    for (double l = 0.0; l <=len; l += checkl)
    {
      shotptr->interpolate(from(), to(), l/len, s());
      reals = s.reals();
      path_list.push_back(Eigen::Vector3d(reals[0], reals[1], reals[2]));        
    }
  
    return;
  }



  // to retrieve the path to the correct order
  void KinoAstar::retrievePath(PathNodePtr end_node)
  {
    PathNodePtr cur_node = end_node;
    path_nodes_.push_back(cur_node);

    while (cur_node->parent != NULL)
    {
      cur_node = cur_node->parent;
      path_nodes_.push_back(cur_node);
    }

    reverse(path_nodes_.begin(), path_nodes_.end());
  }


  void KinoAstar::intialMap(map_util::OccMapUtil *map_itf)
  {
    frontend_map_itf_ = map_itf;
    // Dp_map.intialMap(map_itf);
    if(use_network){
      Eigen::Vector2i dims = frontend_map_itf_->getDim();
      for(int i = 0; i < dims[0]; i++){
      for(int j = 0; j < dims[1]; j++){
          env_tensor[0][0][i][j] = frontend_map_itf_->getDistance(Eigen::Vector2i(i,j));
        }
      }
    }
  }

  void KinoAstar::init(ConfigPtr cfg_, ros::NodeHandle nh,bool un)
  {
    std::cout << "PyTorch version: "
      << TORCH_VERSION_MAJOR << "."
      << TORCH_VERSION_MINOR << "."
      << TORCH_VERSION_PATCH << std::endl;
    // Dp_map.init(cfg_);
    horizon_ = cfg_->horizon_;
    yaw_resolution_ = cfg_->yaw_resolution_;
    lambda_heu_ = cfg_->lambda_heu_;
    allocate_num_ = cfg_->allocate_num_;
    check_num_ = cfg_->check_num_;
    max_seach_time  = cfg_->max_seach_time;
    step_arc = cfg_->step_arc;
    checkl = cfg_->checkl;
    non_siguav = 0.0;
    wheel_base = cfg_->wheel_base;
    
    



    /* ---------- pre-allocated node ---------- */
    path_node_pool_.resize(allocate_num_);
    for (int i = 0; i < allocate_num_; i++)
    {
      path_node_pool_[i] = new PathNode;
    }

    use_node_num_ = 0;
    iter_num_ = 0;

    map_size_3d_(0) = cfg_->mapX;
    map_size_3d_(1) = cfg_->mapY;

    resolution_ = cfg_->mapRes;
    origin_(0) = -0.5*map_size_3d_(0);
    origin_(1) = -0.5*map_size_3d_(1);
    yaw_origin_ = -M_PI;
         
    max_forward_vel = cfg_->vmax;
    max_forward_vel = cfg_->vmax-0.2;
    max_forward_acc = cfg_->lonAccmax;
    max_backward_vel = max_forward_vel;
    max_backward_acc = max_forward_acc;

    
    
    max_cur_ = cfg_->kmax;//damn it
    std::cout << "max_cur_ = " << max_cur_ << std::endl;
    max_steer_ = std::atan(wheel_base*max_cur_);
    inv_resolution_ = 1.0 / resolution_;
    inv_yaw_resolution_ = 1.0 / yaw_resolution_;
    steer_res = cfg_->steer_res;

    shotptr =std::make_shared<ompl::base::ReedsSheppStateSpace>(1.0/max_cur_);

    expandNodesVis = nh.advertise<sensor_msgs::PointCloud2>("/vis/expanded_nodes", 1);
    networkVis = nh.advertise<sensor_msgs::PointCloud2>("/vis/neuralAstarMap", 1);
    use_network = un;



    if(use_network){
      std::string package_path = ros::package::getPath("plan_manage");
      mpnet_ = torch::jit::load(package_path + "/config/transformer.pt",torch::kCUDA);
      if(use_half){
        mpnet_.to(torch::kHalf);
      }
      mpnet_.eval();
      torch::NoGradGuard no_grad_;
      //warm start
      for(int i = 0; i < 20; i++){
        warm();
      }
    }
  }

  void KinoAstar::warm(){
        std::vector<torch::IValue> ws_inputs;
        torch::Tensor ws0 = torch::zeros({1,4,200,200});
        if(use_half){
            ws_inputs.push_back(ws0.to(at::kCUDA).to(torch::kHalf));
        }
        else{
            ws_inputs.push_back(ws0.to(at::kCUDA));
        }
        auto outputs = mpnet_.forward(ws_inputs).toTuple();
        return;
  }
    

  void KinoAstar::reset()
  {
    expanded_nodes_.clear();
    path_nodes_.clear();

    std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
    open_set_.swap(empty_queue);

    for (int i = 0; i < use_node_num_; i++)
    {
      PathNodePtr node = path_node_pool_[i];
      node->parent = NULL;
      node->node_state = NOT_EXPAND;
    }

    use_node_num_ = 0;
    iter_num_ = 0;
    is_shot_succ_ = false;

    context_map = torch::zeros({1,1,200,200});
    context_cos_map = torch::zeros({1,1,200,200});
    context_sin_map = torch::zeros({1,1,200,200});//reset
  }
  std::vector<Eigen::Vector3d> KinoAstar::getSampleTraj(){
    return SampleTraj;
  }
  /*hzchzc px py yaw*/
  Eigen::Vector3d KinoAstar::evaluatePos(double t){
    t = std::min<double>(std::max<double>(0,t),totalTrajTime);
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
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

    double initv = non_siguav,finv = non_siguav;
    if(index==0) {initv  = startvel;}
    if(index==shot_lengthList.size() - 1) finv = endvel;

    double localtime = shot_timeList[index];
    double locallength = shot_lengthList[index];
    int front = shotindex[index]; int back =  shotindex[index+1];
    std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+front,SampleTraj.begin()+back+1);
    //find the nearest point
    double arclength;
    if(shot_SList[index] > 0)
      arclength= evaluateLength(CutTime,locallength,localtime,max_forward_vel,max_forward_acc, initv,finv);
    else
      arclength= evaluateLength(CutTime,locallength,localtime,max_backward_vel, max_backward_acc, initv,finv);
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
  std::vector<Eigen::Vector4d> KinoAstar::SamplePosList(int N){
      double dt = totalTrajTime/N;
      std::vector<Eigen::Vector4d>  path;
      for(int i=0;i<=N;i++){
        double time = i*dt;
        Eigen::Vector4d posAndt;
        posAndt<<evaluatePos(time),time;
        path.push_back(posAndt);
      }
      return path;
  }
  
  void KinoAstar::NodeVis(Eigen::Vector3d state){
    sensor_msgs::PointCloud2 globalMap_pcd;
    pcl::PointCloud<pcl::PointXYZ> cloudMap;
    pcl::PointXYZ pt;
    pt.x = state[0];
    pt.y = state[1];
    pt.z = 0.2;
    cloudMap.points.push_back(pt); 
    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.stamp = ros::Time::now();
    globalMap_pcd.header.frame_id = "world";
    expandNodesVis.publish(globalMap_pcd);

    return;
  }
std::vector<Eigen::Vector3d> KinoAstar::getRoughSamples(){
 if(use_network){
      nnVis(positive);
    }
    double truncate_len = 25.0;
    bool exceed_len = false;


    std::vector<Eigen::Vector3d> roughSampleList;
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
    PathNodePtr node = path_nodes_.back();
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;
    while(node->parent != NULL){
      for (int k = check_num_; k >0; k--)
      {
        Eigen::Vector3d state;
        double tmparc = node->input[1] * double(k) / double(check_num_);
        Eigen::Vector2d tmpctrl; tmpctrl << node->input[0],tmparc;
        stateTransit(node->parent->state, state, tmpctrl);
        state[2] = normalize_angle(state[2]);
        roughSampleList.push_back(state);

      }
      node = node->parent;
    } 
    start_state_[2] = normalize_angle(start_state_[2]);
    roughSampleList.push_back(start_state_.head(3));
    reverse(roughSampleList.begin(),roughSampleList.end());

    if(is_shot_succ_){
      ompl::base::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
      Eigen::Vector3d state1,state2;
      state1 = roughSampleList.back();
      state2 = end_state_.head(3);
      from[0] = state1[0]; from[1] = state1[1]; from[2] = state1[2];
      to[0] = state2[0]; to[1] = state2[1]; to[2] = state2[2];
      double shotLength = shotptr->distance(from(), to());
      std::vector<double> reals;
      for(double l = checkl; l < shotLength; l += checkl){
        shotptr->interpolate(from(), to(), l/shotLength, s());
        reals = s.reals();
        roughSampleList.push_back(Eigen::Vector3d(reals[0], reals[1], normalize_angle(reals[2])));        
      }
      end_state_[2] = normalize_angle(end_state_[2]);
      roughSampleList.push_back(end_state_.head(3));
    }
    //truncate the init trajectory
    double tmp_len = 0;
    int truncate_idx = 0;
    for(truncate_idx = 0;truncate_idx <roughSampleList.size()-1;truncate_idx++){  
      tmp_len += (roughSampleList[truncate_idx+1]-roughSampleList[truncate_idx]).norm();
      // if(tmp_len>truncate_len){
      //   break;
      // }
    }
    roughSampleList.assign(roughSampleList.begin(),roughSampleList.begin()+truncate_idx+1);

    
    SampleTraj = roughSampleList;
    return SampleTraj;
  }

  void KinoAstar::getKinoNode(KinoTrajData &flat_trajs)
  {
    if(use_network){
      nnVis(positive);
    }
    double truncate_len = 25.0;
    bool exceed_len = false;

    flat_trajs.clear();
    std::vector<Eigen::Vector3d> roughSampleList;
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
    PathNodePtr node = path_nodes_.back();
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;
    while(node->parent != NULL){
      for (int k = check_num_; k >0; k--)
      {
        Eigen::Vector3d state;
        double tmparc = node->input[1] * double(k) / double(check_num_);
        Eigen::Vector2d tmpctrl; tmpctrl << node->input[0],tmparc;
        stateTransit(node->parent->state, state, tmpctrl);
        state[2] = normalize_angle(state[2]);
        roughSampleList.push_back(state);

      }
      node = node->parent;
    } 
    start_state_[2] = normalize_angle(start_state_[2]);
    roughSampleList.push_back(start_state_.head(3));
    reverse(roughSampleList.begin(),roughSampleList.end());

    if(is_shot_succ_){
      ompl::base::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
      Eigen::Vector3d state1,state2;
      state1 = roughSampleList.back();
      state2 = end_state_.head(3);
      from[0] = state1[0]; from[1] = state1[1]; from[2] = state1[2];
      to[0] = state2[0]; to[1] = state2[1]; to[2] = state2[2];
      double shotLength = shotptr->distance(from(), to());
      std::vector<double> reals;
      for(double l = checkl; l < shotLength; l += checkl){
        shotptr->interpolate(from(), to(), l/shotLength, s());
        reals = s.reals();
        roughSampleList.push_back(Eigen::Vector3d(reals[0], reals[1], normalize_angle(reals[2])));        
      }
      end_state_[2] = normalize_angle(end_state_[2]);
      roughSampleList.push_back(end_state_.head(3));
    }
    //truncate the init trajectory
    double tmp_len = 0;
    int truncate_idx = 0;
    for(truncate_idx = 0;truncate_idx <roughSampleList.size()-1;truncate_idx++){  
      tmp_len += (roughSampleList[truncate_idx+1]-roughSampleList[truncate_idx]).norm();
      // if(tmp_len>truncate_len){
      //   break;
      // }
    }
    roughSampleList.assign(roughSampleList.begin(),roughSampleList.begin()+truncate_idx+1);

    
    SampleTraj = roughSampleList;
    /*divide the whole shot traj into different segments*/   
    shot_lengthList.clear();
    shot_timeList.clear();
    shotindex.clear();
    shot_SList.clear(); 
    double tmpl = 0;
    bool ifnewtraj = false;
    int lastS = (SampleTraj[1]-SampleTraj[0]).head(2).dot(Eigen::Vector2d(cos(SampleTraj[0][2]),sin(SampleTraj[0][2])))>=0?1:-1;
    //hzchzc attention please!!! max_v must = min_v  max_acc must = min_acc
    shotindex.push_back(0);
    for(int i = 0; i<SampleTraj.size()-1; i++){
      Eigen::Vector3d state1 = SampleTraj[i];
      Eigen::Vector3d state2 = SampleTraj[i+1];
      int curS = (state2-state1).head(2).dot(Eigen::Vector2d(cos(state1[2]),sin(state1[2]))) >=0 ? 1:-1;
      if(curS*lastS >= 0){
        tmpl += (state2-state1).head(2).norm();
      }
      else{  
        shotindex.push_back(i);
        shot_SList.push_back(lastS);
        shot_lengthList.push_back(tmpl);
        if(lastS>0)
          shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
        else
          shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc, non_siguav,non_siguav));

        tmpl = (state2-state1).head(2).norm();
      }       
      lastS = curS;
    }
    shot_SList.push_back(lastS);
    shot_lengthList.push_back(tmpl);
    if(lastS>0)
      shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
    else
      shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc, non_siguav,non_siguav));
    shotindex.push_back(SampleTraj.size()-1);
    if(shot_timeList.size()>=2){
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,non_siguav);
      else
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel, max_backward_acc, startvel,non_siguav);
      if(shot_SList[shot_timeList.size()-1]>0)
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_forward_vel,max_forward_acc,non_siguav,endvel);
      else
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_backward_vel,max_backward_acc,non_siguav,endvel);
    }
    else{
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,endvel);
      else  
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel,max_backward_acc, startvel,endvel);
    }
    /*extract flat traj  
    the flat traj include the end point but not the first point*/
    for(int i=0;i<shot_lengthList.size();i++){
      double initv = non_siguav,finv = non_siguav;
      Eigen::Vector2d Initctrlinput,Finctrlinput;
      Initctrlinput<<0,0;Finctrlinput<<0,0;
      if(i==0) {initv  = startvel; Initctrlinput = start_ctrl;}
      if(i==shot_lengthList.size() - 1) finv = endvel;

      double locallength = shot_lengthList[i];
      int sig = shot_SList[i];
      std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+shotindex[i],SampleTraj.begin()+shotindex[i+1]+1);
      traj_pts.clear();
      thetas.clear();        
      double samplet;
      double tmparc = 0;
      int index = 0;
      double sampletime = 0.1;
      if(shot_timeList[i]<=sampletime){
        sampletime = shot_timeList[i] / 2.0;
      }
      for(samplet = sampletime; samplet<shot_timeList[i]; samplet+=sampletime){
        double arc;
        if(sig > 0){
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_forward_vel, max_forward_acc, initv, finv);
        }
        else{
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_backward_vel, max_backward_acc, initv, finv);
        }
        
        //find the nearest point
        for(int k = index; k<localTraj.size()-1 ;k++)
        {
          tmparc+= (localTraj[k+1]-localTraj[k]).head(2).norm();
          if(tmparc>=arc){
            index = k;
            double l1 = tmparc-arc;
            double l = (localTraj[k+1]-localTraj[k]).head(2).norm();
            double l2 = l-l1;//l2
            double px = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[0];
            double py = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[1];
            double yaw= (l1/l*localTraj[k]+l2/l*localTraj[k+1])[2];
            if(fabs(localTraj[k+1][2]-localTraj[k][2])>=M_PI){   
              double normalize_yaw;
              if(localTraj[k+1][2]<=0){
                normalize_yaw = l1/l*localTraj[k][2]+l2/l*(localTraj[k+1][2]+2*M_PI);
              }
              else if(localTraj[k][2]<=0){
                normalize_yaw = l1/l*(localTraj[k][2]+2*M_PI)+l2/l*localTraj[k+1][2];
              }
              yaw = normalize_yaw;
            }
            traj_pts.push_back(Eigen::Vector3d(px,py,sampletime));
            thetas.push_back(yaw);
            tmparc -=(localTraj[k+1]-localTraj[k]).head(2).norm();
            break;
          }
        }      
      }
      traj_pts.push_back(Eigen::Vector3d(localTraj.back()[0],localTraj.back()[1],shot_timeList[i]-(samplet-sampletime)));
      thetas.push_back(localTraj.back()[2]);
      FlatTrajData flat_traj;
      Eigen::MatrixXd startS;
      Eigen::MatrixXd endS;
      getFlatState(Eigen::Vector4d(localTraj.front()[0],localTraj.front()[1],localTraj.front()[2],initv),Initctrlinput,startS,sig);
      getFlatState(Eigen::Vector4d(localTraj.back()[0],localTraj.back()[1],localTraj.back()[2],finv),Finctrlinput,endS,sig);
      flat_traj.traj_pts = traj_pts;
      flat_traj.thetas = thetas;
      flat_traj.start_state = startS;
      flat_traj.final_state = endS;
      flat_traj.singul = sig;
      flat_traj.duration = shot_timeList[i];
      flat_trajs.push_back(flat_traj);
    }
    totalTrajTime = 0.0;
    for(const auto dt : shot_timeList){
       totalTrajTime += dt; 
    }
  }
  
  void KinoAstar::getKinoNode(KinoTrajData &flat_trajs, std::vector<Eigen::Vector3d> inputSamples){
    double truncate_len = 25.0;
    bool exceed_len = false;

    flat_trajs.clear();
    double startvel = 0.0;
    double endvel = 0.0;
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;


    
    SampleTraj = inputSamples;
    /*divide the whole shot traj into different segments*/   
    shot_lengthList.clear();
    shot_timeList.clear();
    shotindex.clear();
    shot_SList.clear(); 
    double tmpl = 0;
    bool ifnewtraj = false;
    int lastS = (SampleTraj[1]-SampleTraj[0]).head(2).dot(Eigen::Vector2d(cos(SampleTraj[0][2]),sin(SampleTraj[0][2])))>=0?1:-1;
    //hzchzc attention please!!! max_v must = min_v  max_acc must = min_acc
    shotindex.push_back(0);
    for(int i = 0; i<SampleTraj.size()-1; i++){
      Eigen::Vector3d state1 = SampleTraj[i];
      Eigen::Vector3d state2 = SampleTraj[i+1];
      int curS = (state2-state1).head(2).dot(Eigen::Vector2d(cos(state1[2]),sin(state1[2]))) >=0 ? 1:-1;
      if(curS*lastS >= 0){
        tmpl += (state2-state1).head(2).norm();
      }
      else{  
        shotindex.push_back(i);
        shot_SList.push_back(lastS);
        shot_lengthList.push_back(tmpl);
        if(lastS>0)
          shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
        else
          shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc, non_siguav,non_siguav));

        tmpl = (state2-state1).head(2).norm();
      }       
      lastS = curS;
    }
    shot_SList.push_back(lastS);
    shot_lengthList.push_back(tmpl);
    if(lastS>0)
      shot_timeList.push_back(evaluateDuration(tmpl,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
    else
      shot_timeList.push_back(evaluateDuration(tmpl,max_backward_vel, max_backward_acc, non_siguav,non_siguav));
    shotindex.push_back(SampleTraj.size()-1);
    if(shot_timeList.size()>=2){
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,non_siguav);
      else
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel, max_backward_acc, startvel,non_siguav);
      if(shot_SList[shot_timeList.size()-1]>0)
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_forward_vel,max_forward_acc,non_siguav,endvel);
      else
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_backward_vel,max_backward_acc,non_siguav,endvel);
    }
    else{
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,endvel);
      else  
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel,max_backward_acc, startvel,endvel);
    }
    /*extract flat traj  
    the flat traj include the end point but not the first point*/
    for(int i=0;i<shot_lengthList.size();i++){
      double initv = non_siguav,finv = non_siguav;
      Eigen::Vector2d Initctrlinput,Finctrlinput;
      Initctrlinput<<0,0;Finctrlinput<<0,0;
      if(i==0) {initv  = startvel; Initctrlinput = start_ctrl;}
      if(i==shot_lengthList.size() - 1) finv = endvel;

      double locallength = shot_lengthList[i];
      int sig = shot_SList[i];
      std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+shotindex[i],SampleTraj.begin()+shotindex[i+1]+1);
      traj_pts.clear();
      thetas.clear();        
      double samplet;
      double tmparc = 0;
      int index = 0;
      double sampletime = 0.1;
      if(shot_timeList[i]<=sampletime){
        sampletime = shot_timeList[i] / 2.0;
      }
      for(samplet = sampletime; samplet<shot_timeList[i]; samplet+=sampletime){
        double arc;
        if(sig > 0){
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_forward_vel, max_forward_acc, initv, finv);
        }
        else{
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_backward_vel, max_backward_acc, initv, finv);
        }
        
        //find the nearest point
        for(int k = index; k<localTraj.size()-1 ;k++)
        {
          tmparc+= (localTraj[k+1]-localTraj[k]).head(2).norm();
          if(tmparc>=arc){
            index = k;
            double l1 = tmparc-arc;
            double l = (localTraj[k+1]-localTraj[k]).head(2).norm();
            double l2 = l-l1;//l2
            double px = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[0];
            double py = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[1];
            double yaw= (l1/l*localTraj[k]+l2/l*localTraj[k+1])[2];
            if(fabs(localTraj[k+1][2]-localTraj[k][2])>=M_PI){   
              double normalize_yaw;
              if(localTraj[k+1][2]<=0){
                normalize_yaw = l1/l*localTraj[k][2]+l2/l*(localTraj[k+1][2]+2*M_PI);
              }
              else if(localTraj[k][2]<=0){
                normalize_yaw = l1/l*(localTraj[k][2]+2*M_PI)+l2/l*localTraj[k+1][2];
              }
              yaw = normalize_yaw;
            }
            traj_pts.push_back(Eigen::Vector3d(px,py,sampletime));
            thetas.push_back(yaw);
            tmparc -=(localTraj[k+1]-localTraj[k]).head(2).norm();
            break;
          }
        }      
      }
      traj_pts.push_back(Eigen::Vector3d(localTraj.back()[0],localTraj.back()[1],shot_timeList[i]-(samplet-sampletime)));
      thetas.push_back(localTraj.back()[2]);
      FlatTrajData flat_traj;
      Eigen::MatrixXd startS;
      Eigen::MatrixXd endS;
      getFlatState(Eigen::Vector4d(localTraj.front()[0],localTraj.front()[1],localTraj.front()[2],initv),Initctrlinput,startS,sig);
      getFlatState(Eigen::Vector4d(localTraj.back()[0],localTraj.back()[1],localTraj.back()[2],finv),Finctrlinput,endS,sig);
      flat_traj.traj_pts = traj_pts;
      flat_traj.thetas = thetas;
      flat_traj.start_state = startS;
      flat_traj.final_state = endS;
      flat_traj.singul = sig;
      flat_traj.duration = shot_timeList[i];
      flat_trajs.push_back(flat_traj);
    }
    totalTrajTime = 0.0;
    for(const auto dt : shot_timeList){
       totalTrajTime += dt; 
    }
  }
  double KinoAstar::evaluateDuration(double length, double max_vel, double max_acc, double startV, double endV){
   double critical_len; //the critical length of two-order optimal control, acc is the input
    if(startV>max_vel||endV>max_vel){
      ROS_ERROR("kinoAstar:evaluateDuration:start or end vel is larger that the limit!");
    }
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
  double KinoAstar::evaluateLength(double curt,double locallength,double localtime,double max_vel, double max_acc, double startV, double endV){
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

  std::vector<PathNodePtr> KinoAstar::getVisitedNodes()
  {
    vector<PathNodePtr> visited;
    visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
    return visited;
  }

  Eigen::Vector2i KinoAstar::posToIndex(Eigen::Vector2d pt)
  {
    int idx = std::round((pt[0]-origin_[0])/resolution_);
    int idy = std::round((pt[1]-origin_[1])/resolution_);
    return Eigen::Vector2i(idx,idy);
  }

  int KinoAstar::yawToIndex(double yaw)
  {
    yaw = normalize_angle(yaw);
    // std::cout << "yaw: " << yaw << std::endl;
    if(yaw>3.15||yaw<-3.15){
      ROS_ERROR("yaw>3.15||yaw<-3.15");
    }
    int idx = floor((yaw - yaw_origin_) * inv_yaw_resolution_);
    return idx;
  }




  void KinoAstar::getFlatState(Eigen::Vector4d state, Eigen::Vector2d control_input,
                                  Eigen::MatrixXd &flat_state, int singul)
  {

    flat_state.resize(2, 3);

    double angle = state(2);
    double vel   = state(3); // vel > 0 

    Eigen::Matrix2d init_R;
    init_R << cos(angle),  -sin(angle),
              sin(angle),   cos(angle);

    if (abs(vel) <= non_siguav){
      vel = singul * non_siguav;
    }
    else{
      vel = singul * vel;
    }
    
    flat_state << state.head(2), init_R*Eigen::Vector2d(vel, 0.0), 
                  init_R*Eigen::Vector2d(control_input(1), std::tan(control_input(0)) / wheel_base * std::pow(vel, 2));

  }

} // namespace resilient_planner
