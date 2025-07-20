#ifndef _TRAJ_OPTIMIZER_H_
#define _TRAJ_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <ros/ros.h>

#include "poly_traj_utils.hpp"
#include "lbfgs.hpp"
#include "geoutils2d.hpp"
#include <tools/gridmap.hpp>
#include <tools/config.hpp>
//1.0 0.1 0.01
namespace dftpav
{

  using namespace std;


  class PolyTrajOptimizer
  {

  private:
    Eigen::VectorXi piecenums;
    int trajnum;
    double gslar_ = 0.05;
    double wei_time_ = 500.0;
    double esdfWei = 1000.0;
    double penaWei = 10000.0;
    double kmax = 0.45;
    double kdotmax = 1.0;
    double vmax = 3.0;
    double latAccmax = 3.0;
    double lonAccmax = 3.0;
    double accRatemax = 3.0;


    double phimax = 0.785;
    double omegamax = 0.3;
    double wheelbase = 0.6;

    int traj_res = 16;
    std::vector<Eigen::MatrixXd> cfgHs_;
    std::vector<Eigen::Vector2d> conpts;
    /*lots of alm param*/
    double scaling_wf_ = 1.0;
    Eigen::VectorXd scaling_eqwc_, scaling_uneqwc_;
    double scaling_wf_min_ = 0.01, scaling_wc_min_ = 0.01;
    int eqc_num = 0, neqc_num = 0, totalc_num = 0;
    Eigen::VectorXd lambda, mu, hx, gx;
    double rho = 1.0, rho_max =1000.0;
    double gamma = 1;
    double cons_eps_ = 0.20;
    double safeMargin = 0.15;
    Eigen::VectorXd gradx;
    double almTotalcost;
    Eigen::MatrixXd headState2d_, tailState2d_;
    std::vector<int> singual_;
    map_util::OccMapUtil* map_itf_;

    int eqc_idx = 0;
    int uneqc_idx = 0;
    std::shared_ptr<visualization::Visualization> vis_tool_;
    bool isdebug_;
    bool isfix_;
    double backW_;
    double miniS = 0.1;
    bool isstart;
    double totalCost;
    std::string name_;



    std::vector<dftpav::MinJerkOpt> jerkOpt_container;

    double startT;
    double non_sinv = 0.005;
    //0.005 0.05 0.5
    double debugEnergy;
  public:
    
    PolyTrajOptimizer() {}
    ~PolyTrajOptimizer() {}

    /* main planning API */
    inline int OptimizeSe2Trajectory(
            const Eigen::MatrixXd &headState2d, const Eigen::MatrixXd &tailState2d, Eigen::VectorXd rdTs_container,
            const std::vector<Eigen::MatrixXd> &inPs_container, const std::vector<Eigen::Vector2d> &gearPos,
             const std::vector<double> & angles, map_util::OccMapUtil* map_itf, std::shared_ptr<Config> config_, const std::vector<int> singual,
            std::shared_ptr<visualization::Visualization> vis_tool,std::string name = "dftpav", std::vector<Eigen::VectorXd> ds_container = {}){
      vis_tool_ = vis_tool;
      wei_time_ = config_->wei_time_;
      kdotmax = config_->kdotmax;
      kmax = config_->kmax;
      vmax = config_->vmax;
      latAccmax = config_->latAccmax;
      lonAccmax = config_->lonAccmax;
      accRatemax = config_->accRatemax;
      traj_res = config_->traj_res;
      rho = config_->rho;
      rho_max = config_->rho_max;
      gamma = config_->gamma;
      cons_eps_ = config_->cons_eps_;
      conpts = config_->conpts;
      singual_ = singual;
      isdebug_ = config_->isdebug;
      isfix_ = config_->isfixGear;
      backW_ = config_->backW;
      miniS = config_->miniS;
      safeMargin = config_->safeMargin;
      esdfWei = config_->esdfWei;
      penaWei = config_->penaWei;
      map_itf_ =  map_itf;

      phimax = config_->phimax;
      omegamax = config_->omegamax;
      wheelbase = config_->wheel_base;
      kmax = tan(phimax) / wheelbase; 



      isstart = true;
      non_sinv = config_->non_siguav;
      name_ = name;
      for(int i = 0; i < rdTs_container.size(); i++){
          if(rdTs_container[i] <= gslar_){
              ROS_ERROR("piece time <= gslar_");
              rdTs_container[i] = gslar_+0.05;
          }
      }
      headState2d_ = headState2d;
      tailState2d_ = tailState2d;
      headState2d_.col(1) = headState2d_.col(1) * non_sinv;
      tailState2d_.col(1) = tailState2d_.col(1) * non_sinv;

      trajnum = singual_.size();
      int variable_num_ = 0;
      jerkOpt_container.clear();
      jerkOpt_container.resize(trajnum);
      piecenums.resize(trajnum);
      for(int i = 0; i < trajnum; i++){
          int pn = inPs_container[i].cols() + 1;
          piecenums[i] = pn;
          jerkOpt_container[i].reset(pn);
          variable_num_ += 2 * (pn - 1);
      }
      variable_num_ += trajnum;
      variable_num_ += 2 * (trajnum-1);
      variable_num_ += 1 * (trajnum-1);
    
    //optimization variables dt
    Eigen::VectorXd x;
    x.resize(variable_num_);
    int offset = 0;
    for(int i = 0; i<trajnum; i++){
      memcpy(x.data()+offset,inPs_container[i].data(), inPs_container[i].size() * sizeof(x[0]));
      offset += inPs_container[i].size();
    }
    Eigen::Map<Eigen::VectorXd> Vt(x.data()+offset, rdTs_container.size());
    RealT2VirtualT(rdTs_container, Vt);
    offset += rdTs_container.size();

    for(int i = 0; i < trajnum-1; i++){
      memcpy(x.data()+offset,gearPos[i].data(), 2 * sizeof(x[0]));
      offset += 2;
    }
    Eigen::Map<Eigen::VectorXd> angles_(x.data()+offset, trajnum-1);
    for(int i = 0; i < trajnum - 1; i++){
      angles_[i] = angles[i];
    }


    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs_params.mem_size = config_->mem_size;//128
    lbfgs_params.past = config_->past; //3 
    lbfgs_params.g_epsilon = config_->g_epsilon;
    lbfgs_params.min_step = config_->min_step;
    lbfgs_params.delta = config_->delta;
    lbfgs_params.max_iterations = config_->max_iterations;
    startT = ros::Time::now().toSec();
    int result;
    double final_cost;
    double t1 = ros::Time::now().toSec();
    result = lbfgs::lbfgs_optimize(
      x,
      final_cost,
      PolyTrajOptimizer::costFunctionCallback,
      NULL,
      monitor,
      this,
      lbfgs_params);
    double t2 = ros::Time::now().toSec();
    // ROS_WARN_STREAM("dftpav planning time: "<<1000.0*(t2-t1)<<" ms");


    /* ---------- get result and check collision ---------- */
    if (result == lbfgs::LBFGS_CONVERGENCE ||
        result == lbfgs::LBFGS_CANCELED ||
        result == lbfgs::LBFGS_STOP||result == lbfgs::LBFGSERR_MAXIMUMITERATION)
    {
      // ROS_WARN_STREAM("dif planner worked cost:"<<final_cost);
      
    } 
    else if (result == lbfgs::LBFGSERR_MAXIMUMLINESEARCH){
      ROS_WARN("Lbfgs: The line-search routine reaches the maximum number of evaluations.");
    }
    else
    {
      ROS_WARN("Solver error. Return = %d, %s. Skip this planning.", result, lbfgs::lbfgs_strerror(result));
    }

    
    debugVis();
    // ROS_WARN_STREAM("dftpav energy: " << debugEnergy);
    return true;






  }


  std::vector<Eigen::Vector3d> getTraj(double dt){
      //return {px,py,yaw}
      std::vector<Eigen::Vector3d> poseTraj;
      for(int i = 0; i < trajnum; i++){
          dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
          for(double t = 0.0; t < polytraj.getTotalDuration()-1.0e-3; t+=dt){
              Eigen::Vector2d pos= polytraj.getSigma(t);
              poseTraj.push_back(Eigen::Vector3d(pos[0],pos[1],0.0));
          }   
      }
      return poseTraj;
  }

  void debugVis(){
    vis_tool_ ->visualize_path(getTraj(0.01), "/visualization/refinedTraj_"+name_);
    DifTrajectory  optTraj = getOptTraj();
    vis_tool_->visualize_se2traj(optTraj, "/visualization/fullshapeTraj_kinoastar");
    // std::cout <<"path topic: "<<"/visualization/refinedTraj_"+name_ << "\n";
    ROS_WARN_STREAM("differential flatness max cur: "<<getMaxCur()); 
    // ROS_WARN_STREAM("dftpav max kdot: "<<getMaxCurDot()); 
    ROS_WARN_STREAM("differential flatness max omega: "<<getMaxOmega()); 
  }
  dftpav::DifTrajectory getOptTraj(){
    dftpav::DifTrajectory optTraj;
    for(int i = 0; i < trajnum; i++){
      dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
      optTraj.Traj_container.push_back(polytraj);
      optTraj.etas.push_back(singual_[i]);
    }
    // ROS_WARN_STREAM("max cur: "<<getMaxCur());
    // ROS_WARN_STREAM("max cur dot: "<<getMaxCurDot());
    return optTraj;
  }
  double getMaxCur(){
    dftpav::DifTrajectory optTraj;
    for(int i = 0; i < trajnum; i++){
      dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
      optTraj.Traj_container.push_back(polytraj);
      optTraj.etas.push_back(singual_[i]);
    }
    double maxcur = 0.0;
    for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
      double cur = optTraj.getCur(t);
      if(fabs(cur) > fabs(maxcur)){
        maxcur = cur;
      }
    }

    return maxcur;
  }
  double getMaxCurDot(){
    // dftpav::DifTrajectory optTraj;
    // for(int i = 0; i < trajnum; i++){
    //   dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
    //   optTraj.Traj_container.push_back(polytraj);
    //   optTraj.etas.push_back(singual_[i]);
    // }
    // double maxcurdot = 0.0;
    // double baset = 0.0;
    // for(int id = 0; id < optTraj.etas.size(); id++){
    //     for(int i = 0; i < optTraj.Traj_container[id].getPieceNum(); i++){
    //         double step = optTraj.Traj_container[id].getDurations()[i]/traj_res;
    //         for(int j = 0; j <= traj_res; j++){
    //             double curt = baset + j*step;
    //             double curdot = optTraj.getCurDot(curt);     
    //             if(fabs(curdot) > fabs(maxcurdot)){
    //                 maxcurdot = curdot;
    //             }
    //         }
    //         baset += optTraj.Traj_container[id].getDurations()[i];
    //     }
    // }

    // return maxcurdot;
    dftpav::DifTrajectory optTraj;
    for(int i = 0; i < trajnum; i++){
      dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
      optTraj.Traj_container.push_back(polytraj);
      optTraj.etas.push_back(singual_[i]);
    }
    double maxcurdot = 0.0;
    for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
      double curdot = optTraj.getCurDot(t);
      if(fabs(curdot) > fabs(maxcurdot)){
        maxcurdot = curdot;
      }
    }

    return maxcurdot;
  }

  double getMaxOmega(){
    dftpav::DifTrajectory optTraj;
    for(int i = 0; i < trajnum; i++){
      dftpav::Trajectory polytraj = jerkOpt_container[i].getTraj(singual_[i]);
      optTraj.Traj_container.push_back(polytraj);
      optTraj.etas.push_back(singual_[i]);
    }
    double maxomega = 0.0;
    for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
      double omega = optTraj.getOmega(t);
      if(fabs(omega) > fabs(maxomega)){
        maxomega = omega;
      }
    }

    return maxomega;
  }






  private:
    /* callbacks by the L-BFGS optimizer */
    static int monitor(void *func_data,const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &g,
                                    const double fx,
                                    const double step,
                                    const int k,
                                    const int ls){
    double nowtime = ros::Time::now().toSec();
    PolyTrajOptimizer *opt = reinterpret_cast<PolyTrajOptimizer *>(func_data);
    double budget = 0.1;


    if(nowtime-(opt->startT) > budget){
      // ROS_WARN("reach budget time");
      return 1;
    }
    else{
      return 0;
    }
  }

    static double costFunctionCallback(void *func_data, const Eigen::VectorXd &x, Eigen::VectorXd &grad){
      double total_smcost = 0.0, total_timecost = 0.0, penalty_cost = 0.0;
      PolyTrajOptimizer *opt = reinterpret_cast<PolyTrajOptimizer *>(func_data);
      int offset = 0;
   
      std::vector<Eigen::Map<const Eigen::MatrixXd>> P_container;
      std::vector<Eigen::Map<Eigen::MatrixXd>> gradP_container;;
      for(int trajid = 0; trajid < opt->trajnum; trajid++){
        Eigen::Map<const Eigen::MatrixXd> P(x.data()+offset, 2, opt->piecenums[trajid] - 1);
        Eigen::Map<Eigen::MatrixXd>gradP(grad.data()+offset, 2, opt->piecenums[trajid] - 1);
        offset += 2 * (opt->piecenums[trajid] - 1);
        gradP.setZero();
        P_container.push_back(P);
        gradP_container.push_back(gradP);
      }
      Eigen::Map<const Eigen::VectorXd> Vdts(x.data()+offset, opt->trajnum);
      Eigen::Map<Eigen::VectorXd>gradVdts(grad.data()+offset, opt->trajnum);
      offset += opt -> trajnum;
      Eigen::VectorXd dts(opt->trajnum);
      Eigen::VectorXd gradDts(opt->trajnum); gradDts.setZero();
      opt->VirtualT2RealT(Vdts, dts);

      // Ini/Fin Gear Pos
      std::vector<Eigen::Map<const Eigen::MatrixXd>> Gear_container;
      std::vector<Eigen::Map<Eigen::MatrixXd>> gradGear_container;
      for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
        Eigen::Map<const Eigen::MatrixXd> Gear(x.data()+offset, 2, 1);
        Eigen::Map<Eigen::MatrixXd>gradGear(grad.data()+offset, 2, 1);
        offset += 2;
        gradGear.setZero();
        Gear_container.push_back(Gear);
        gradGear_container.push_back(gradGear);
      }
      //
      Eigen::Map<const Eigen::VectorXd> Angles(x.data()+offset, opt->trajnum-1);
      Eigen::Map<Eigen::VectorXd>gradAngles(grad.data()+offset, opt->trajnum-1);
      gradAngles.setZero();
      
  



      // Eigen::VectorXd gradt;
      for(int trajid = 0; trajid < opt->trajnum; trajid++){
        double smoo_cost = 0.0;
        double pena_cost = 0.0;
        Eigen::MatrixXd IniS,FinS;
        IniS = opt->headState2d_;
        FinS = opt->tailState2d_;

        if(trajid > 0){
          double theta = Angles[trajid-1];
          IniS.col(0) = Gear_container[trajid-1];
          IniS.col(1) = Eigen::Vector2d(opt->singual_[trajid]*opt->non_sinv*cos(theta), opt->singual_[trajid] * opt->non_sinv*sin(theta)); 
        }
        if(trajid < opt->trajnum-1){
          double theta = Angles[trajid];
          FinS.col(0) = Gear_container[trajid];
          FinS.col(1) = Eigen::Vector2d(opt->singual_[trajid] * opt->non_sinv*cos(theta), opt->singual_[trajid]*opt->non_sinv*sin(theta));
        }
        //relax end v?
        // if(trajid == opt->trajnum - 1){
        //   FinS.col(1) = endV;
        // }


        opt->jerkOpt_container[trajid].generate(P_container[trajid],dts[trajid],IniS,FinS);

        
        opt->jerkOpt_container[trajid].initSmGradCost(); // Smoothness cost   
        smoo_cost = opt->jerkOpt_container[trajid].getTrajJerkCost();
        opt->addPVAGradCost2CT(pena_cost, trajid); // Time int cost
        //Get gradT gradC
        total_smcost += smoo_cost;
        penalty_cost +=  pena_cost;
        // std::cout<<"Trajid: "<<trajid<<" penalty: "<<obs_surround_feas_qvar_costs.transpose()<<std::endl;
      }


      for(int trajid = 0; trajid < opt->trajnum; trajid++){
        double time_cost = 0.0;
        Eigen::Matrix<double,2,3> gradIni, gradFin;
        opt->jerkOpt_container[trajid].calGrads_PT(); // gdt gdp gdhead gdtail
        //waypoint
        gradP_container[trajid] = opt->jerkOpt_container[trajid].get_gdP();
        //init Fin
        gradIni = opt->jerkOpt_container[trajid].get_gdHead();
        gradFin = opt->jerkOpt_container[trajid].get_gdTail();
        if(trajid > 0){
          double theta = Angles[trajid-1];
          gradGear_container[trajid-1] += gradIni.col(0);
          // IniS.col(1) = Eigen::Vector2d(opt->singual_[trajid]*opt->non_sinv*cos(theta), opt->singual_[trajid] * opt->non_sinv*sin(theta)); 
          gradAngles[trajid-1] += gradIni.col(1).transpose() * Eigen::Vector2d(-opt->singual_[trajid]*opt->non_sinv * sin(theta), opt->singual_[trajid] *opt->non_sinv*cos(theta));
        }
        if(trajid < opt->trajnum-1){
          double theta = Angles[trajid];
          gradGear_container[trajid] += gradFin.col(0);
          // Eigen::Vector2d(opt->singual_[trajid] * opt->non_sinv*cos(theta), opt->singual_[trajid]*opt->non_sinv*sin(theta));
          gradAngles[trajid] += gradFin.col(1).transpose() * Eigen::Vector2d(-opt->non_sinv *opt->singual_[trajid] * sin(theta), opt->singual_[trajid]*opt->non_sinv*cos(theta));
        }

        time_cost = opt->wei_time_ * dts[trajid]* opt->piecenums[trajid];
        gradDts[trajid] = opt->jerkOpt_container[trajid].get_gdT();
        gradDts[trajid] += opt->wei_time_ * opt->piecenums[trajid];
        opt->Virtual2Grad(Vdts,  gradDts, gradVdts);
        total_timecost += time_cost;


        // opt->VirtualTGradCost(T[trajid],t[trajid],opt->jerkOpt_container[trajid].get_gdT() / opt->piece_num_container[trajid],gradt[trajid],time_cost);

        
        // gradt[trajid] = 0.0;
        
        // std::cout<<"gradp: \n"<<gradP_container[trajid] <<std::endl;
        // std::cout<<"gradeigen p: \n"<<opt->jerkOpt_container[trajid].get_gdP()<<std::endl;
      }

      gradGear_container[0].setZero();
      gradGear_container[1].setZero();
      gradAngles[0]  = 0.0;
      gradAngles[1]  = 0.0;
      gradVdts.setZero();

      opt->debugEnergy = total_smcost;
      return total_smcost + total_timecost + penalty_cost;

    }
    void addPVAGradCost2CT(double &costs, const int trajid){
    int N = piecenums[trajid];
    Eigen::Vector2d sigma, dsigma, ddsigma, dddsigma, ddddsigma;
    Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
    double s1, s2, s3, s4, s5;
    double step, alpha;
    
    int singul_ = singual_[trajid];


    costs = 0.0;
    Eigen::Matrix2d B_h;
    B_h << 0, -1,
           1, 0;
    double z_h0, z_h1, z_h2, z_h3, z_h4;
    double z1, z2, z3;
    double n1, n2, n3, n4, n5, n6;
    Eigen::Matrix2d ego_R, help_R;

    std::vector<int> cosindex;
    // int innerLoop;
    double t = 0;

    Eigen::Vector2d gradPos, gradVel, gradAcc, gradJerk;

    for (int i = 0; i < N; ++i)
    {
      const Eigen::Matrix<double, 6, 2> &c = jerkOpt_container[trajid].getCoeffs().block<6, 2>(i * 6, 0);
      step = jerkOpt_container[trajid].getDt() / traj_res; // T_i /k
      s1 = 0.0;
      for (int j = 0; j <= traj_res; ++j)
      {
        s2 = s1 * s1;
        s3 = s2 * s1;
        s4 = s2 * s2;
        s5 = s4 * s1;
        beta0 << 1.0, s1, s2, s3, s4, s5;
        beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
        beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
        beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
        beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * s1;
        alpha = 1.0 / traj_res * j;
        
        //update s1 for the next iteration
        s1 += step;

        sigma = c.transpose() * beta0;
        dsigma = c.transpose() * beta1;
        ddsigma = c.transpose() * beta2;
        dddsigma = c.transpose() * beta3;
        ddddsigma = c.transpose() * beta4;
        gradPos.setZero();
        gradVel.setZero();
        gradAcc.setZero();
        gradJerk.setZero();
        Eigen::Vector2d pos,vel,acc,jerk;
        pos = sigma;
        vel = dsigma;
        acc = ddsigma;
        jerk = dddsigma;
        Eigen::Matrix2d rotR;
        rotR << vel[0], -vel[1],
                vel[1],  vel[0];
        rotR = (singul_ * rotR) / vel.norm();
        // if(non_sinv < 0.05){
        // if(vel.norm() < 0.05)
        //   continue;
        // }
        
        {
          double vioVel = dsigma.squaredNorm() - vmax * vmax;
          if(vioVel > 0){
              double pena, penaD;
              positiveSmoothedL1(vioVel, pena, penaD);
              costs += penaWei * pena;
              gradVel += penaWei * penaD *  2.0 * dsigma;
          }
        }

        // {
        //   double vioAcc = ddsigma.squaredNorm() - accRatemax * accRatemax;
        //   if(vioAcc > 0){
        //       double pena, penaD;
        //       positiveSmoothedL1(vioAcc, pena, penaD);
        //       costs += penaWei * pena;
        //       gradAcc += penaWei * penaD * 2.0 * ddsigma;
        //   }
        // }
        {
          double lonAcc = ddsigma.dot(dsigma)/  dsigma.norm();
          double vioLonAcc = lonAcc*lonAcc - lonAccmax * lonAccmax;
          if(vioLonAcc > 0){
              double pena, penaD;
              positiveSmoothedL1(vioLonAcc, pena, penaD);
              costs += penaWei * pena;
              gradAcc += penaWei * penaD * 2.0 * lonAcc * dsigma / dsigma.norm();
              gradVel += penaWei * penaD * 2.0 * lonAcc * (ddsigma/dsigma.norm()-dsigma*ddsigma.dot(dsigma)/ (pow(dsigma.norm(),3)));
          }
        }





      {
        for(const Eigen::Vector2d resPt : conpts){
            Eigen::Vector2d gradViolaSdpos;
          
            Eigen::Vector2d absPt = pos + rotR * resPt;
            /*
                rotR << vel[0], -vel[1],
                        vel[1],  vel[0];
                rotR = (sg * rotR) / vel.norm();
            */
            Eigen::Matrix2d temp_l_Bl;
            temp_l_Bl << resPt(0), resPt(1),
                          -resPt(1), resPt(0);          
            double dis = map_itf_->getDistGrad(absPt, gradViolaSdpos);
            double min_dis_ = std::min(dis, min_dis_);
            double vioSdist = (-dis + safeMargin);
            if(vioSdist > 0){
                double pena, penaD;
                positiveSmoothedL1(vioSdist, pena, penaD);
                costs += esdfWei * pena;
                gradPos += esdfWei * penaD * (-1.0) * gradViolaSdpos;
                gradVel += esdfWei * penaD * (-1.0) * 
                (singul_ * temp_l_Bl / vel.norm() - vel * (rotR * resPt).transpose() / (vel.squaredNorm()) )
                * gradViolaSdpos;
            }
            
        }
    }


      {
          double cur = (dsigma[0]*ddsigma[1] - ddsigma[0]*dsigma[1]) / (dsigma.norm() * dsigma.norm() * dsigma.norm());
          double cross = (dsigma[0]*ddsigma[1] - ddsigma[0]*dsigma[1]);
          double vioCur = (cur*cur - kmax*kmax);
          if(vioCur > 0.0){
              double pena, penaD;
              positiveSmoothedL1(vioCur, pena, penaD);
              costs += penaWei * pena;
              Eigen::Vector2d gradViolaCv;
              Eigen::Vector2d gradViolaCa;
              gradViolaCv = (
                            2*cross*Eigen::Vector2d(acc[1], -acc[0])/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()) -
                            6*Eigen::Vector2d(vel[0], vel[1])*cross*cross/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm())
                              ); 
              gradViolaCa = 2.0 * Eigen::Vector2d(-vel[1], vel[0]) * cross / (vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm());
              
              gradVel += penaWei * penaD * gradViolaCv;
              gradAcc += penaWei * penaD * gradViolaCa;
          }
      }

      // {
      //     // double cur = (dsigma[0]*ddsigma[1] - ddsigma[0]*dsigma[1]) / (dsigma.norm() * dsigma.norm() * dsigma.norm());
      //     double cross = (dsigma[0]*ddsigma[1] - ddsigma[0]*dsigma[1]);
      //     double v3 = pow(vel.norm(),3);
      //     double vioCur = cross * cross - kmax * kmax * v3 * v3;
      //     if(vioCur > 0.0){
      //         double pena, penaD;
      //         positiveSmoothedL1(vioCur, pena, penaD);
      //         costs += penaWei * pena;
      //         Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
      //         Eigen::Vector2d gradViolaCv;
      //         Eigen::Vector2d gradViolaCa;
      //         gradViolaCv = 2*cross*Eigen::Vector2d(acc[1], -acc[0]) - kmax * kmax * gradv6_v;
      //         gradViolaCa = 2*cross*Eigen::Vector2d(-vel[1], vel[0]);
              
              
      //         gradVel += penaWei * penaD * gradViolaCv;
      //         gradAcc += penaWei * penaD * gradViolaCa;
      //     }
      // }
    /*
    {     //curdot
          Eigen::Matrix2d B;
          B << 0,-1,
                1, 0;
          double tp1 = jerk.transpose()*B*vel;
          double v3 = pow(vel.norm(),3);
          double tp2 = acc.transpose()*B*vel;
          double tp3 = acc.dot(vel);
          double v1 = vel.norm();
          double v6 = pow(vel.norm(),6);
          double denor = tp1*v3-3.0*tp2*tp3*v1; 
          double kdot = denor / v6;
          double vioKdot = kdot*kdot-kdotmax*kdotmax;
          if(vioKdot > 0){
            double pena, penaD;
            positiveSmoothedL1(vioKdot, pena, penaD);
            costs += penaWei * pena; 


              Eigen::Vector2d gradViolaKdv;
              Eigen::Vector2d gradtp1_v = -B*jerk;
              Eigen::Vector2d gradv3_v = 3.0*vel.norm()*vel;
              Eigen::Vector2d gradtp2_v = -B*acc;
              Eigen::Vector2d gradtp3_v = acc;
              Eigen::Vector2d gradv1_v = vel/vel.norm();
              Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
              Eigen::Vector2d graddenor_v = gradtp1_v*v3+tp1*gradv3_v-3.0*(gradtp2_v*tp3*v1+tp2*gradtp3_v*v1+tp2*tp3*gradv1_v);
              gradViolaKdv = 2.0 * kdot * (graddenor_v*v6-gradv6_v*denor) / (v6*v6) ;

              Eigen::Vector2d gradViolaKda;
              Eigen::Vector2d gradtp2_a = B*vel;
              Eigen::Vector2d gradtp3_a = vel;
              Eigen::Vector2d graddenor_a = -3.0*(gradtp2_a*tp3*v1+tp2*gradtp3_a*v1);
              gradViolaKda = 2.0*kdot*(graddenor_a/v6);

              Eigen::Vector2d gradViolaKdj;
              Eigen::Vector2d gradtp1_j = B*vel;
              Eigen::Vector2d graddenor_j = gradtp1_j*v3;
              gradViolaKdj = 2.0*kdot*(graddenor_j/v6);

              

              gradVel += penaWei * penaD * gradViolaKdv;
              gradAcc += penaWei * penaD * gradViolaKda;
              gradJerk += penaWei * penaD * gradViolaKdj;
          }


      }*/


      {   //omega
              Eigen::Matrix2d B;
              B << 0,-1,
                    1, 0;
              double tp1 = jerk.transpose()*B*vel;
              double v3 = pow(vel.norm(),3);
              double tp2 = acc.transpose()*B*vel;
              double tp3 = acc.dot(vel);
              double v1 = vel.norm();
              double v6 = pow(vel.norm(),6);
              double denor = tp1*v3-3.0*tp2*tp3*v1;
              double nor =  v6 + pow(tp2*wheelbase,2);
              double omega = wheelbase * denor / nor;
              double vioOmega = omega*omega-omegamax*omegamax;

              if(vioOmega > 0){
                double pena, penaD;
                positiveSmoothedL1(vioOmega, pena, penaD);
                costs += penaWei * pena; 

                Eigen::Vector2d gradViolaOmegav;
                Eigen::Vector2d gradtp1_v = -B*jerk;
                Eigen::Vector2d gradv3_v = 3.0*vel.norm()*vel;
                Eigen::Vector2d gradtp2_v = -B*acc;
                Eigen::Vector2d gradtp3_v = acc;
                Eigen::Vector2d gradv1_v = vel/vel.norm();
                Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
                Eigen::Vector2d graddenor_v = gradtp1_v*v3+tp1*gradv3_v-3.0*(gradtp2_v*tp3*v1+tp2*gradtp3_v*v1+tp2*tp3*gradv1_v);
                Eigen::Vector2d gradnor_v = gradv6_v + 2.0*tp2*wheelbase*wheelbase*gradtp2_v;
                gradViolaOmegav = 2.0 * omega * (graddenor_v*nor-gradnor_v*denor) / (nor*nor) *  wheelbase;

                Eigen::Vector2d gradViolaOmegaa;
                Eigen::Vector2d gradtp2_a = B*vel;
                Eigen::Vector2d gradtp3_a = vel;
                Eigen::Vector2d graddenor_a = -3.0*(gradtp2_a*tp3*v1+tp2*gradtp3_a*v1);
                Eigen::Vector2d gradnor_a = 2.0*tp2*wheelbase*wheelbase*gradtp2_a;
                gradViolaOmegaa = 2.0 * omega * (graddenor_a*nor-gradnor_a*denor) / (nor*nor) *  wheelbase;

                Eigen::Vector2d gradViolaOmegaj;
                Eigen::Vector2d gradtp1_j = B*vel;
                Eigen::Vector2d graddenor_j = gradtp1_j*v3;
                gradViolaOmegaj = 2.0*omega*(graddenor_j/nor*wheelbase);


                gradVel += penaWei * penaD * gradViolaOmegav;
                gradAcc += penaWei * penaD * gradViolaOmegaa;
                gradJerk += penaWei * penaD * gradViolaOmegaj;
              }
                        

        }



    //  {
    //       Eigen::Matrix2d B;
    //       B << 0,-1,
    //             1, 0;
    //       double tp1 = jerk.transpose()*B*vel;
    //       double v3 = pow(vel.norm(),3);
    //       double tp2 = acc.transpose()*B*vel;
    //       double tp3 = acc.dot(vel);
    //       double v1 = vel.norm();
    //       double v6 = pow(vel.norm(),6);
    //       double denor = tp1*v3-3.0*tp2*tp3*v1; 
    //       // double kdot = denor / v6;
    //       double vioKdot = denor * denor - v6 * v6 * kdotmax * kdotmax;
    //       if(vioKdot > 0){
    //         double pena, penaD;
    //         positiveSmoothedL1(vioKdot, pena, penaD);
    //         costs += penaWei * pena; 

    //         Eigen::Vector2d gradViolaKdv;
    //         Eigen::Vector2d gradtp1_v = -B*jerk;
    //         Eigen::Vector2d gradv3_v = 3.0*vel.norm()*vel;
    //         Eigen::Vector2d gradtp2_v = -B*acc;
    //         Eigen::Vector2d gradtp3_v = acc;
    //         Eigen::Vector2d gradv1_v = vel/vel.norm();
    //         Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
    //         Eigen::Vector2d graddenor_v = gradtp1_v*v3+tp1*gradv3_v-3.0*(gradtp2_v*tp3*v1+tp2*gradtp3_v*v1+tp2*tp3*gradv1_v);
    //         gradViolaKdv = 2.0*denor*graddenor_v-2*v6*gradv6_v* kdotmax * kdotmax;

    //         Eigen::Vector2d gradViolaKda;
    //         Eigen::Vector2d gradtp2_a = B*vel;
    //         Eigen::Vector2d gradtp3_a = vel;
    //         Eigen::Vector2d graddenor_a = -3.0*(gradtp2_a*tp3*v1+tp2*gradtp3_a*v1);
    //         gradViolaKda = 2.0*denor*graddenor_a;

    //         Eigen::Vector2d gradViolaKdj;
    //         Eigen::Vector2d gradtp1_j = B*vel;
    //         Eigen::Vector2d graddenor_j = gradtp1_j*v3;
    //         gradViolaKdj = 2.0*denor*graddenor_j;

              

    //         gradVel += penaWei * penaD * gradViolaKdv;
    //         gradAcc += penaWei * penaD * gradViolaKda;
    //         gradJerk += penaWei * penaD * gradViolaKdj;
    //       }


    //   }



    jerkOpt_container[trajid].get_gdC().block<6, 2>(i * 6, 0) += beta0 * gradPos.transpose() +
                                      beta1 * gradVel.transpose() +
                                      beta2 * gradAcc.transpose() +
                                      beta3 * gradJerk.transpose();
    jerkOpt_container[trajid].get_gdT()  += (gradPos.dot(vel) +
                    gradVel.dot(acc) +
                    gradAcc.dot(jerk) +
                    gradJerk.dot(ddddsigma)) *
                      alpha;
        
        
        
      }
    }
    }





    void positiveLog(const double &x, double &f, double &df){
      double g,dg;
      positiveSmoothedL1(x, g, dg);
      f = log(1.0+g);
      df = 1.0/(1.0+g)*dg;
      return;
    }


    void positiveSmoothedL1(const double &x, double &f, double &df)
      {
                const double pe = 1.0e-4;
                const double half = 0.5 * pe;
                const double f3c = 1.0 / (pe * pe);
                const double f4c = -0.5 * f3c / pe;
                const double d2c = 3.0 * f3c;
                const double d3c = 4.0 * f4c;

                if (x < pe)
                {
                    f = (f4c * x + f3c) * x * x * x;
                    df = (d3c * x + d2c) * x * x;
                }
                else
                {
                    f = x - half;
                    df = 1.0;
                }
                return;
        }
    void positiveSmoothedL3(const double &x, double &f, double &df){
          df = x * x;
          f = df *x;
          df *= 3.0;
      

          return ;
      }


    template <typename EIGENVEC>
    void VirtualT2RealT(const EIGENVEC &VT, Eigen::VectorXd &RT)
    {
      for (int i = 0; i < VT.size(); ++i)
        {
        RT(i) = VT(i) > 0.0 ? ((0.5 * VT(i) + 1.0) * VT(i) + 1.0) + gslar_
                            : 1.0 / ((0.5 * VT(i) - 1.0) * VT(i) + 1.0) + gslar_;
        }
    }
    void VirtualT2RealT(const  double & VT, double &RT){
        
        
        RT = VT > 0.0 ? ((0.5 * VT + 1.0) * VT + 1.0) + gslar_
                            : 1.0 / ((0.5 * VT - 1.0) * VT + 1.0) + gslar_;
    }
    template <typename EIGENVEC>
    inline void RealT2VirtualT(const Eigen::VectorXd &RT, EIGENVEC &VT)
    {
      for (int i = 0; i < RT.size(); ++i)
        {
            VT(i) = RT(i) > 1.0 + gslar_ 
          ? (sqrt(2.0 * RT(i) - 1.0 - 2 * gslar_) - 1.0)
          : (1.0 - sqrt(2.0 / (RT(i)-gslar_) - 1.0));
        }
    }
    inline void RealT2VirtualT(const double &RT, double &VT)
    {
      VT = RT > 1.0 + gslar_ 
        ? (sqrt(2.0 * RT - 1.0 - 2 * gslar_) - 1.0)
        : (1.0 - sqrt(2.0 / (RT-gslar_) - 1.0));
    }
    template <typename EIGENVEC, typename EIGENVECGD>
    void VirtualTGradCost(const Eigen::VectorXd &RT, const EIGENVEC &VT,const Eigen::VectorXd &gdRT, EIGENVECGD &gdVT,double &costT)
    {
        for (int i = 0; i < VT.size(); ++i)
        {
        double gdVT2Rt;
        if (VT(i) > 0)
        {
            gdVT2Rt = VT(i) + 1.0;
        }
        else
        {
            double denSqrt = (0.5 * VT(i) - 1.0) * VT(i) + 1.0;
            gdVT2Rt = (1.0 - VT(i)) / (denSqrt * denSqrt);
        }

        gdVT(i) = (gdRT(i) + wei_time_) * gdVT2Rt;
        }

        costT = RT.sum() * wei_time_;
    }
    void VirtualTGradCost(const double &RT, const double &VT, const double &gdRT, double &gdVT, double& costT){
        double gdVT2Rt;
        if (VT > 0)
        {
        gdVT2Rt = VT + 1.0;
        }
        else
        {
        double denSqrt = (0.5 * VT - 1.0) * VT + 1.0;
        gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt);
        }

        gdVT = (gdRT + wei_time_) * gdVT2Rt;
        costT = RT * wei_time_;
    }        
    void VirtualTGrad2t(const double &VT, const double &gdRT, double &gdVT){
        double gdVT2Rt;
        if (VT > 0)
        {
        gdVT2Rt = VT + 1.0;
        }
        else
        {
        double denSqrt = (0.5 * VT - 1.0) * VT + 1.0;
        gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt);
        }
        gdVT = gdRT * gdVT2Rt;
    }    
    template <typename EIGENVEC, typename EIGENVECGD>
    void Virtual2Grad(const EIGENVEC &VT, const Eigen::VectorXd &gdRT, EIGENVECGD &gdVT){
        for (int i = 0; i < VT.size(); ++i)
        {
        double gdVT2Rt;
        if (VT(i) > 0)
        {
            gdVT2Rt = VT(i) + 1.0;
        }
        else
        {
            double denSqrt = (0.5 * VT(i) - 1.0) * VT(i) + 1.0;
            gdVT2Rt = (1.0 - VT(i)) / (denSqrt * denSqrt);
        }

        gdVT(i) = (gdRT(i)) * gdVT2Rt;
        }
        
    }        
    double expC2(double t) {
        return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
              : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
    }
    double logC2(double T) {
        return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
    }
    void forwardT(const Eigen::Ref<const Eigen::VectorXd>& t, const double& sT, Eigen::Ref<Eigen::VectorXd> vecT) {
        int M = t.size();
        for (int i = 0; i < M; ++i) {
            vecT(i) = expC2(t(i));
        }
        vecT(M) = 0.0;
        vecT /= 1.0 + vecT.sum();
        vecT(M) = 1.0 - vecT.sum();
        vecT *= sT;
        return;
    }
    void backwardT(const Eigen::Ref<const Eigen::VectorXd>& vecT, Eigen::Ref<Eigen::VectorXd> t) {
        int M = t.size();
        t = vecT.head(M) / vecT(M);
        for (int i = 0; i < M; ++i) {
        t(i) = logC2(vecT(i));
        }
        return;
    }
    




  public:
    typedef unique_ptr<PolyTrajOptimizer> Ptr;

  };

} // namespace plan_manage
#endif