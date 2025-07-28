#ifndef TRAJOPT.HPP
#define TRAJOPT.HPP
#include "Polytraj.hpp"
#include "lbfgs.hpp"
#include "geoutils2d.hpp"
#include <tools/gridmap.hpp>
#include <tools/config.hpp>
// using namespace std;
/*to do list
  vel constraint 
  lon acc constraint
  lat acc constraint
  yaw dot constraint
*/
namespace PolyTrajOpt
{
class TrajOpt{
    private:
        
        Eigen::VectorXi piecenums;
        int trajnum;
        double gslar_ = 0.05;
        double wei_time_ = 500.0;
        double kmax = 0.45;
        double vmax = 3.0;
        double latAccmax = 3.0;
        double lonAccmax = 3.0;
        double accRatemax = 8.0;
        double kdotmax = 5.0;
        double yawdotmax = 3.14;
        double penaWei = 1000.0;
        double esdfWei = 1000.0;
        int traj_res = 16;
        std::vector<Eigen::MatrixXd> cfgHs_;
        std::vector<Eigen::Vector2d> conpts;
        // = {Eigen::Vector2d(3.455, 0.95), 
        //                                        Eigen::Vector2d(3.455, -0.95),
        //                                        Eigen::Vector2d(-1.425, 0.95),
        //                                        Eigen::Vector2d(-1.425, -0.95)};
        // std::vector<Eigen::Vector2d> conpts = {Eigen::Vector2d(3.0, 0.7), 
        // Eigen::Vector2d(3.0, -0.7),
        // Eigen::Vector2d(-1.4, 0.7),
        // Eigen::Vector2d(-1.4, -0.7)};
        double safeMargin = 0.15;
        Eigen::MatrixXd headState2d_, tailState2d_;
        std::vector<int> singual_;
        map_util::OccMapUtil* map_itf_;
        double miniS;

        std::shared_ptr<visualization::Visualization> vis_tool_;
        bool isdebug_;
        bool isfix_;
        double backW_;
        int iter;
        bool isstart;
        std::string name_;
    public:
        // se2Plan::MinJerkOpt<3> jerkOpt;
        std::vector<PolyTrajOpt::MINCO_S3NU<2>> posOpts;
        std::vector<PolyTrajOpt::MINCO_S3NU<1>> tOpts;
        double min_dis_;
        double debugEnergy;
        double debugRhoT;
        double debugEsdf, debugCur, debugDs, debugminVel;


        inline int OptimizeSe2Trajectory(
            const Eigen::MatrixXd &headState2d, const Eigen::MatrixXd &tailState2d, Eigen::VectorXd rdTs_container,
            const std::vector<Eigen::MatrixXd> &inPs_container, const std::vector<Eigen::Vector2d> &gearPos,
             const std::vector<double> & angles, map_util::OccMapUtil* map_itf, std::shared_ptr<Config> config_, const std::vector<int> singual,
            std::shared_ptr<visualization::Visualization> vis_tool,std::string name = "nn", std::vector<Eigen::VectorXd> ds_container = {})
        {
            //rdT_container is the real dt for each piece of the trajectory
            //angles is the angle at the gear shifting state
            //inPs_container[i] is the wayPts for each mincotraj

            name_ = name;
            vis_tool_ = vis_tool;
            /*setparam*/
            wei_time_ = config_->wei_time_;
            kmax = config_->kmax;
            vmax = config_->vmax;
            accRatemax = config_->accRatemax;
            kdotmax = config_->kdotmax;
            yawdotmax = config_->yawdotmax;
            traj_res = config_->traj_res;
            singual_ = singual;
            isdebug_ = config_->isdebug;
            isfix_ = config_->isfixGear;
            backW_ = config_->backW;
            map_itf_ =  map_itf;
            conpts = config_->conpts;
            penaWei = config_->penaWei;
            miniS = config_->miniS;
            safeMargin = config_->safeMargin;
            esdfWei = config_->esdfWei;
            isstart = true;
            iter = 0;
            for(int i = 0; i < rdTs_container.size(); i++){
                if(rdTs_container[i] <= gslar_){
                    ROS_ERROR("piece time <= gslar_");
                    rdTs_container[i] = gslar_+0.01;
                    // return 0; //hzchzc
                }
            }
            

            //totalt is total time
            headState2d_ = headState2d;
            tailState2d_ = tailState2d;
            trajnum = singual_.size();
            posOpts.resize(trajnum);piecenums.resize(trajnum);
            tOpts.resize(trajnum);
            int variable_num_ = 0;
            for(int i = 0; i < trajnum; i++) {
                piecenums[i] = inPs_container[i].cols() + 1;
                posOpts[i].reset(piecenums[i]);
                tOpts[i].reset(piecenums[i]);
                variable_num_ += 2 * (piecenums[i] - 1);
                variable_num_ += (piecenums[i]); // scale (ds
            }
            ROS_WARN("begin to optimize Arc Trajectory!");
            std::cout << "trajnum: " << trajnum << std::endl;
            std::cout << "piecenums: "<< piecenums.transpose() <<std::endl;
            variable_num_ += trajnum;         // dts
            variable_num_ += 2 * (trajnum-1); //gear pos
            variable_num_ += 1 * (trajnum-1); //gear angle

            Eigen::VectorXd x;
            x.resize(variable_num_);

            int offset = 0;
            for(int i = 0; i<trajnum; i++){
                memcpy(x.data()+offset,inPs_container[i].data(), inPs_container[i].size() * sizeof(x[0]));
                offset += inPs_container[i].size();
            }
            Eigen::Map<Eigen::VectorXd> Vdts(x.data()+offset, rdTs_container.size());
            for(int i = 0; i < rdTs_container.size(); i++){
                double va;
                RealT2VirtualT(rdTs_container[i], va);
                Vdts[i] = va;
            }
            offset += rdTs_container.size();
            for(int i = 0; i < trajnum-1; i++){
                memcpy(x.data()+offset,gearPos[i].data(), 2 * sizeof(x[0]));
                offset += 2;
            }
            Eigen::Map<Eigen::VectorXd> angles_(x.data()+offset, trajnum-1);
            for(int i = 0; i < trajnum - 1; i++){
                angles_[i] = angles[i];
            }
            offset += trajnum-1;
            std::vector<Eigen::VectorXd> Rds_container;
            for(int i = 0; i < trajnum; i++){
                Eigen::Map<Eigen::VectorXd> Vds(x.data()+offset, piecenums[i]);
                Eigen::VectorXd Rds(piecenums[i]);
                if(ds_container.size() == 0){
                    {
                        Eigen::Vector2d tmpHead, tmpTail;
                        if(i == 0){
                            tmpHead << headState2d.col(0);
                        }
                        else{
                            tmpHead << gearPos[i-1];
                        }
                        if(i == trajnum - 1){
                            tmpTail << tailState2d.col(0);
                        }
                        else{
                            tmpTail << gearPos[i];
                        }
                        Eigen::MatrixXd tmpPts; tmpPts.resize(2, piecenums[i] + 1);
                        tmpPts.col(0) << tmpHead; tmpPts.col(piecenums[i]) << tmpTail;
                        for(int j = 0; j < piecenums[i] - 1; j++){
                            tmpPts.col(j+1) = inPs_container[i].col(j);
                        }

                        for(int j = 0; j < piecenums[i]; j++){
                            Rds[j] = (tmpPts.col(j) - tmpPts.col(j+1)).norm();//bug hzc hzc
                        }
                    }
                    // Rds.setConstant(rdTs_container[i]);       
                }
                else{
                    Rds = ds_container[i];
                }
                // std::cout << "trajid: " << i << " Rds: " << Rds.transpose() << std::endl;              
                           
                for(int j = 0; j < Rds.size(); j++){
                    if(Rds[j] <= gslar_){
                        ROS_ERROR("RT[i] <= gslar_");
                        Rds[j] = gslar_+0.1;
                        // return 0;//hzchzc
                    }
                }
                Rds_container.push_back(Rds);
                RealT2VirtualT(Rds, Vds);
                offset += piecenums[i];
            }
            //the variables beform optimization
            ROS_INFO("The variables beform optimization: ");
            for(int i = 0; i < trajnum; i++){
                std::cout << "trajid: "<<i<<" scales: "<<Rds_container[i].transpose()<<std::endl;
                std::cout << "trajid: "<<i<<" times: "<<rdTs_container[i]<<std::endl;

                std::cout << "trajid: "<<i<<" waypoints:\n"<<inPs_container[i]<<std::endl;
            }


            



            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.mem_size = config_->mem_size;//128
            lbfgs_params.past = config_->past; //3 
            lbfgs_params.g_epsilon = config_->g_epsilon;
            lbfgs_params.min_step = config_->min_step;
            lbfgs_params.delta = config_->delta;
            lbfgs_params.max_iterations = config_->max_iterations;
            double final_cost;
            bool flag_success = true;

            double t1 = ros::Time::now().toSec();
            
            int result = lbfgs::lbfgs_optimize(
            x,
            final_cost,
            TrajOpt::PenaltyCostFunctionCallback,
            NULL,
            NULL,
            this,
            lbfgs_params);
            double t2 = ros::Time::now().toSec();

            if (result == lbfgs::LBFGS_CONVERGENCE ||
            result == lbfgs::LBFGS_CANCELED ||
            result == lbfgs::LBFGS_STOP||result == lbfgs::LBFGSERR_MAXIMUMITERATION)
            {
            // if(final_cost >= 50000.0){
            //     return 0;
            // }
            // ROS_INFO_STREAM("se2 trajectory generation success! final cost: " << final_cost << " iter: " << iter );
            printf("\033[32mse2 trajectory generation success!,time(ms)=%5.3f, final cost=%5.3f, iter=%d \n\033[0m", (t2-t1) * 1000.0, final_cost, iter);
            } 
            else if (result == lbfgs::LBFGSERR_MAXIMUMLINESEARCH){
            ROS_WARN("Lbfgs: The line-search routine reaches the maximum number of evaluations.");
                return 0;
            }
            else
            {
                return 0;
                ROS_WARN("Solver error. Return = %d, %s. Skip this planning.", result, lbfgs::lbfgs_strerror(result));
            }

            // ROS_INFO_STREAM("solve time: "<<(t2-t1) * 1000.0 << " ms" );
            {
                ROS_INFO("Optimized scale s, times, waypoints: ");
                for(int i = 0; i < posOpts.size(); i++){
                    std::cout << "trajid: " << i << " scales: " << posOpts[i].T1.transpose() << std::endl;
                    std::cout << "trajid: " << i << " timess: " << tOpts[i].T1.transpose() << std::endl;
                    PolyTrajOpt::PolyTrajectory<2> traj = posOpts[i].getTraj();
                    std::cout << "waypoints: \n" << traj.getWpts() << std::endl;
                }
            }
            debugVis();
            
            return flag_success;
        }
        static inline double PenaltyCostFunctionCallback(void *func_data, const Eigen::VectorXd &x, Eigen::VectorXd &grad){
            double smcost = 0.0, timecost = 0.0, penalty = 0.0;
            TrajOpt *opt = reinterpret_cast<TrajOpt *>(func_data);
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

            std::vector<Eigen::Map<const Eigen::Vector2d>> Gear_container;
            std::vector<Eigen::Map<Eigen::Vector2d>> gradGear_container;
            for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
                Eigen::Map<const Eigen::Vector2d> Gear(x.data()+offset, 2);
                Eigen::Map<Eigen::Vector2d>gradGear(grad.data()+offset, 2);
                offset += 2;
                gradGear.setZero();
                Gear_container.push_back(Gear);
                gradGear_container.push_back(gradGear);
            }
            Eigen::Map<const Eigen::VectorXd> Angles(x.data()+offset, opt->trajnum-1);
            Eigen::Map<Eigen::VectorXd>gradAngles(grad.data()+offset, opt->trajnum-1);
            gradAngles.setZero();
            offset += opt->trajnum-1;

            std::vector<Eigen::Map<const Eigen::VectorXd>> Vs_container;
            std::vector<Eigen::Map<Eigen::VectorXd>> gradVs_container;;
            std::vector<Eigen::VectorXd> Rs_container;
            std::vector<Eigen::VectorXd> gradRs_container;;
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                Eigen::Map<const Eigen::VectorXd> Vs(x.data()+offset, opt->piecenums[trajid]);
                Eigen::Map<Eigen::VectorXd>gradVs(grad.data()+offset, opt->piecenums[trajid]);
                Vs_container.push_back(Vs);
                gradVs.setZero();
                gradVs_container.push_back(gradVs); 
                Eigen::VectorXd Rs(opt->piecenums[trajid]);
                Eigen::VectorXd gradRs(opt->piecenums[trajid]); gradRs.setZero();
                opt->VirtualT2RealT(Vs, Rs);
                Rs_container.push_back(Rs);
                gradRs_container.push_back(gradRs);
                offset += opt->piecenums[trajid];
            }



            std::vector<Eigen::MatrixXd> gdS2c_2d(opt->trajnum), gdS2c_1d(opt->trajnum), gdC2c_2d(opt->trajnum), gdC2c_1d(opt->trajnum), gdTotalc_2d(opt->trajnum), gdTotalc_1d(opt->trajnum);
            std::vector<Eigen::VectorXd> gdS2t_2d(opt->trajnum), gdC2t_2d(opt->trajnum), gdTotalt_2d(opt->trajnum),gdS2t_1d(opt->trajnum), gdC2t_1d(opt->trajnum), gdTotalt_1d(opt->trajnum);
            
            opt->debugEnergy = 0.0;
            opt->debugEsdf = 0.0;
            opt->debugCur = -INFINITY;
            opt->debugDs = 0.0;
            opt->debugminVel  = 0.0;
            opt->debugRhoT = 0.0;
            opt->min_dis_ = INFINITY;
        
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                double s = 1.0 * opt->singual_[trajid];
                double k;
                if(s > 0){
                    k= 1.0;
                }
                else{
                    k = opt->backW_;
                }
                Eigen::Matrix<double, 2, 3> IniState, FinState;
                if(trajid == 0){
                    IniState = opt->headState2d_;
                }
                else{
                    IniState.col(0) << Gear_container[trajid - 1];
                    IniState.col(1) << s * cos(Angles[trajid -1]), s * sin(Angles[trajid -1]);
                    IniState.col(2) << 0.0, 0.0;
                }
                if(trajid == opt->trajnum-1){
                    FinState = opt->tailState2d_;
                }
                else{
                    FinState.col(0) << Gear_container[trajid];
                    FinState.col(1) << s * cos(Angles[trajid]), s * sin(Angles[trajid]);
                    FinState.col(2) << 0.0, 0.0;
                }
                // std::cout <<"trajid: " <<trajid << "\n";
                // std::cout << "IniState: \n"<<IniState<<"\n";
                // std::cout << "FinState: \n"<<FinState<<"\n";

                opt->posOpts[trajid].generate(P_container[trajid],Rs_container[trajid],IniState,FinState);
                opt->posOpts[trajid].initSmGradCost(gdS2c_2d[trajid], gdS2t_2d[trajid]);

                
                // smcost +=  k*opt->posOpts[trajid].getTrajJerkCost();
                // gdS2c_2d[trajid] = gdS2c_2d[trajid] * k;
                // gdS2t_2d[trajid]=  gdS2t_2d[trajid] * k;//hzchzc
                gdS2c_2d[trajid].setZero();
                gdS2t_2d[trajid].setZero();

                {
                    Eigen::MatrixXd wpTs, IniSt, FinSt;
                    wpTs.resize(1, opt->piecenums[trajid]-1); IniSt.resize(1,3); FinSt.resize(1,3);
                    for(int idx = 0; idx <  opt->piecenums[trajid]-1; idx++){
                        wpTs.col(idx) << Rs_container[trajid].segment(0, idx + 1).sum();
                    }
                    IniSt << 0.0, 0.0, 0.0;
                    FinSt << Rs_container[trajid].sum(), 0.0,0.0;
                    Eigen::VectorXd cdts(opt->piecenums[trajid]);
                    cdts.setConstant(dts[trajid]);
                    opt->tOpts[trajid].generate(wpTs, cdts,IniSt, FinSt);
                    opt->tOpts[trajid].initSmGradCost(gdS2c_1d[trajid], gdS2t_1d[trajid]);
                    // smcost +=  k*opt->tOpts[trajid].getTrajJerkCost();
                    // gdS2c_1d[trajid] = gdS2c_1d[trajid] * k;
                    // gdS2t_1d[trajid] = gdS2t_1d[trajid] * k;
                    gdS2c_1d[trajid].setZero();
                    gdS2t_1d[trajid].setZero();
                }

                opt->penaltyGradCost2CT(penalty, opt->posOpts[trajid], opt->tOpts[trajid], gdC2c_2d[trajid], gdC2t_2d[trajid], gdC2c_1d[trajid], gdC2t_1d[trajid], trajid); // Time int cost
                gdTotalc_2d[trajid] = gdC2c_2d[trajid] + gdS2c_2d[trajid];
                gdTotalt_2d[trajid] = gdC2t_2d[trajid] + gdS2t_2d[trajid];
                gdTotalc_1d[trajid] = gdC2c_1d[trajid] + gdS2c_1d[trajid];
                gdTotalt_1d[trajid] = gdC2t_1d[trajid] + gdS2t_1d[trajid];
            }
            //gradient propagation 
            for(int trajid = 0; trajid < opt->trajnum; trajid++) {
                Eigen::MatrixXd propogate_gradp_2d, propogate_gradIni_2d, propogate_gradFin_2d;
                opt->posOpts[trajid].calGrads_PT(gdTotalc_2d[trajid], gdTotalt_2d[trajid], propogate_gradp_2d, propogate_gradIni_2d, propogate_gradFin_2d);
                double sg = 1.0 * opt->singual_[trajid];
                if(trajid > 0){
                    double theta = Angles[trajid-1];
                    gradGear_container[trajid-1] += propogate_gradIni_2d.col(0);
                    gradAngles[trajid-1] += propogate_gradIni_2d.col(1).transpose() * Eigen::Vector2d(-sg * sin(theta), sg*cos(theta));
                }
                if(trajid < opt->trajnum-1){
                    double theta = Angles[trajid];
                    gradGear_container[trajid] += propogate_gradFin_2d.col(0);
                    gradAngles[trajid] += propogate_gradFin_2d.col(1).transpose() * Eigen::Vector2d(-sg * sin(theta), sg*cos(theta));
                }
                gradP_container[trajid] = propogate_gradp_2d;
                double k;
                if(sg > 0){
                    k = 1.0;
                }
                else{
                    k = opt->backW_;
                }
                // gradArcs[trajid] = gdTotalt_2d[trajid] / opt->piecenums[trajid] + opt->wei_time_ * k;
                // timecost += k * opt->wei_time_ * arcs[trajid];
                gradRs_container[trajid] = gdTotalt_2d[trajid];
                
                Eigen::MatrixXd propogate_gradp_1d, propogate_gradIni_1d, propogate_gradFin_1d;
                opt->tOpts[trajid].calGrads_PT(gdTotalc_1d[trajid], gdTotalt_1d[trajid], propogate_gradp_1d, propogate_gradIni_1d, propogate_gradFin_1d);
                for(int p1 = 0; p1 < opt->piecenums[trajid] - 1; p1++){
                    gradRs_container[trajid].segment(0, p1 + 1).array() += propogate_gradp_1d.col(p1)[0];
                }
                gradRs_container[trajid].array() += propogate_gradFin_1d.col(0)[0];
                
                gradDts[trajid] = gdTotalt_1d[trajid].sum();
                gradDts[trajid] += k *  opt->wei_time_ * opt->piecenums[trajid];
                timecost += k * opt->wei_time_ * dts[trajid]* opt->piecenums[trajid];
                opt->debugRhoT += k * opt->wei_time_ * dts[trajid]* opt->piecenums[trajid];
            }
            opt->Virtual2Grad(Vdts,  gradDts, gradVdts);
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                // gradRs_container[trajid].setConstant(gradRs_container[trajid].sum() / opt->piecenums[trajid]);
                opt->Virtual2Grad(Vs_container[trajid],  gradRs_container[trajid], gradVs_container[trajid]);    
                // gradVT_container[trajid].setZero();
                // gradP_container[trajid].setZero();
            }
            // gradVarcs.setZero();
            // opt->Virtual2Grad(VT,  gradRT, gradVT);
            if(opt->isfix_){
                for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
                    gradGear_container[trajid].setZero();
                    gradAngles[trajid]  = 0.0;
                }
            }
            if(opt->isdebug_){
                // std::cout << "------------------------------------------------------------------\n";
                std::cout << "Energycost: "<<opt->debugEnergy << std::endl;
                std::cout << "EsdfCost: " << opt->debugEsdf << std::endl;
                
                std::cout << "MinVelCost: " << opt->debugminVel << std::endl;
                // std::cout << "dsCur: " << opt->debugDs << std::endl;
                // std::cout << "penalty: " << penalty <<std::endl;
                // std::cout << "smcost: " << smcost << std::endl;
                // std::cout << "timecost: " << timecost << std::endl;
                // std::cout << "totalcost: " << smcost + timecost + penalty << std::endl;
                // std::cout << "times: "<< RT.transpose() << std::endl;
                // std::cout << "gradRT: "<<gradRT.transpose() << std::endl;
                // std::cout << "gradVT: " << gradVT.transpose() << std::endl;
                // std::cout << "angles: "<<Angles.transpose() << std::endl;
                // std::cout << "gradAngles: "<<gradAngles.transpose() << std::endl;
                if(opt->debugEnergy > 1.0e10){
                    std::cout << "cost is too high!"<<"\n";
                }   
                else{
                    opt->debugVis();
                }
            }
            opt->iter ++;
            return smcost + timecost + penalty;
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
        void penaltyGradCost2CT(double &cost,const PolyTrajOpt::MINCO_S3NU<2>& posOpt, const PolyTrajOpt::MINCO_S3NU<1>& tOpt, Eigen::MatrixXd& gdC2c_2d, Eigen::VectorXd& gdC2t_2d, Eigen::MatrixXd& gdC2c_1d, Eigen::VectorXd& gdC2t_1d, int trajid){
            // output gradT gradC 
            int N = piecenums[trajid];
            double sg = 1.0 * singual_[trajid];
            double s, ds, dds, ddds, dddds, latAcc, lonAcc, vnorm, cur;
            Eigen::Vector2d pos, vel, acc, jerk, snap;
            Eigen::Vector2d realVel, realAcc, realJerk;
            double energy;
            double gradS, gradDs, gradDds, gradDdds;
            Eigen::Vector2d gradPos, gradVel, gradAcc, gradJerk;
            Eigen::Matrix<double, 6, 1> beta0_2d, beta1_2d, beta2_2d, beta3_2d, beta4_2d;
            Eigen::Matrix<double, 6, 1> beta0_1d, beta1_1d, beta2_1d, beta3_1d, beta4_1d;
            Eigen::Matrix2d rotR, drotR;
            Eigen::Vector2d outerNormal;
            Eigen::Matrix<double, 6, 1> t_2d, t_1d;
            double step, alpha, eqf;
            double vioCur, vioVel, vioLonAcc, vioLatAcc, vioAcc, vioYawdot, vioCor, vioDs;
            int corId = -1;
            double baset = 0.0;
            t_2d(0) = 1.0;
            t_1d(0) = 1.0;

            gdC2c_2d.resize(6 * N, 2);
            gdC2c_2d.setZero();
            gdC2t_2d.resize(N);
            gdC2t_2d.setZero();
            gdC2c_1d.resize(6 * N, 1);
            gdC2c_1d.setZero();
            gdC2t_1d.resize(N);
            gdC2t_1d.setZero();
            double baseS = 0.0;



            for (int i = 0; i < N; ++i)
            {
                const Eigen::Matrix<double, 6, 1> &c_1d = tOpt.getCoeffs().block<6, 1>(i * 6, 0);
                const Eigen::Matrix<double, 6, 2> &c_2d = posOpt.getCoeffs().block<6, 2>(i * 6, 0);
                
                step = tOpt.T1(i) / traj_res; // T_i /k
                // innerLoop = traj_res;
                for (int j = 0; j <= traj_res; ++j)
                {
                    
                    /*analyse t*/
                    t_1d(1) = step * j;
                    t_1d(2) = t_1d(1) * t_1d(1);
                    t_1d(3) = t_1d(2) * t_1d(1);
                    t_1d(4) = t_1d(2) * t_1d(2);
                    t_1d(5) = t_1d(4) * t_1d(1);
                    beta0_1d  = t_1d;
                    beta1_1d << 0.0, 1.0, 2.0 * t_1d(1), 3.0 * t_1d(2), 4.0 * t_1d(3), 5.0 * t_1d(4);
                    beta2_1d << 0.0, 0.0, 2.0, 6.0 * t_1d(1), 12.0 * t_1d(2), 20.0 * t_1d(3);
                    beta3_1d << 0.0, 0.0, 0.0, 6.0, 24.0 * t_1d(1), 60.0 * t_1d(2);
                    beta4_1d << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * t_1d(1);
                    s = c_1d.transpose() * beta0_1d;
                    ds = c_1d.transpose() * beta1_1d;
                    dds = c_1d.transpose() * beta2_1d;
                    ddds = c_1d.transpose() * beta3_1d;
                    dddds = c_1d.transpose() * beta4_1d;
                    gradS = 0.0;
                    gradDs = 0.0;
                    gradDds = 0.0;
                    gradDdds = 0.0;
                    /*analyse pos*/
                    t_2d(1) = s - baseS;
                    t_2d(2) = t_2d(1) * t_2d(1);
                    t_2d(3) = t_2d(2) * t_2d(1);
                    t_2d(4) = t_2d(2) * t_2d(2);
                    t_2d(5) = t_2d(4) * t_2d(1);
                    beta0_2d  = t_2d;
                    beta1_2d << 0.0, 1.0, 2.0 * t_2d(1), 3.0 * t_2d(2), 4.0 * t_2d(3), 5.0 * t_2d(4);
                    beta2_2d << 0.0, 0.0, 2.0, 6.0 * t_2d(1), 12.0 * t_2d(2), 20.0 * t_2d(3);
                    beta3_2d << 0.0, 0.0, 0.0, 6.0, 24.0 * t_2d(1), 60.0 * t_2d(2);
                    beta4_2d << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * t_2d(1);
                    pos = c_2d.transpose() * beta0_2d;
                    vel = c_2d.transpose() * beta1_2d;
                    acc = c_2d.transpose() * beta2_2d;
                    jerk = c_2d.transpose() * beta3_2d;
                    snap = c_2d.transpose() * beta4_2d;
                    gradPos.setZero();
                    gradVel.setZero();
                    gradAcc.setZero();
                    gradJerk.setZero();
                    alpha = 1.0 / traj_res * j;
                    cur = (vel[0]*acc[1] - acc[0]*vel[1]);
                    rotR << vel[0], -vel[1],
                            vel[1],  vel[0];
                    rotR = (sg * rotR) / vel.norm();
                    realVel = vel * ds;
                    realAcc = dds * vel + ds*acc*ds;
                    realJerk = ddds * vel + acc * dds * ds +2.0 * ds * acc * dds + ds * ds * ds * jerk;

                    
                    /*v > 0.9 penalty*/
                    {
                        double vionegV = miniS * miniS - vel.dot(vel);
                        if(vionegV > 0){
                            double pena, penaD;
                            positiveSmoothedL1(vionegV, pena, penaD);
                            cost += penaWei * pena;
                            gradVel += penaWei * penaD * (-2.0) * vel;
                            debugminVel += penaWei * pena;
                        }
                    }
                    /*v > 1.1 penalty hzc*/
                    // {
                    //     double vioposV = vel.dot(vel)-1.1*1.1;
                    //     if(vioposV > 0){
                    //         double pena, penaD;
                    //         positiveSmoothedL1(vioposV, pena, penaD);
                    //         cost += penaWei * pena;
                    //         gradVel += penaWei * penaD * (2.0) * vel;
                    //         // debugminVel += penaWei * pena;
                    //     }
                    // }


                    /*curvature constraint penalty*/ 
                    {
                        cur = (vel[0]*acc[1] - acc[0]*vel[1]) / (vel.norm() * vel.norm() * vel.norm());
                        double cross = (vel[0]*acc[1] - acc[0]*vel[1]);
                        vioCur = (cur*cur - kmax*kmax);
                        debugCur = std::max(debugCur, fabs(cur));
                        if(vioCur > 0.0){
                            double pena, penaD;
                            positiveSmoothedL1(vioCur, pena, penaD);
                            cost += penaWei * pena;
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
                    //     double cross = (vel[0]*acc[1] - acc[0]*vel[1]);
                    //     vioCur = cross * cross - kmax * kmax * vel.squaredNorm() * vel.squaredNorm() * vel.squaredNorm();
                    //     if(vioCur > 0.0){
                    //         double pena, penaD;
                    //         positiveSmoothedL1(vioCur, pena, penaD);
                    //         cost += penaWei * pena;
                    //         debugCur += penaWei * pena;
                    //         Eigen::Vector2d gradViolaCv;
                    //         Eigen::Vector2d gradViolaCa;
                    //         gradViolaCv = 2.0*cross*Eigen::Vector2d(acc[1], -acc[0]) - 6.0*kmax*kmax*vel.squaredNorm()*vel.squaredNorm()*vel;
                    //         gradViolaCa = 2.0 * Eigen::Vector2d(-vel[1], vel[0]) * cross;
                    //         gradVel += penaWei * penaD * gradViolaCv;
                    //         gradAcc += penaWei * penaD * gradViolaCa;
                    //     }

                    // }

                    /*s' >=0 penalty*/
                    {
                        
                        vioDs = -ds;
                        if(vioDs > 0){
                            double pena, penaD;
                            positiveSmoothedL1(vioDs, pena, penaD);
                            cost += penaWei * pena;
                            debugDs += penaWei * pena;
                            gradDs += penaWei * penaD * -1.0;
                        }
                    }
                    {
                        //realVel (vel * ds)
                        vioVel = realVel.squaredNorm() - vmax * vmax;
                        if(vioVel > 0){
                            double pena, penaD;
                            positiveSmoothedL1(vioVel, pena, penaD);
                            cost += penaWei * pena;
                            gradVel += penaWei * penaD *  ds * ds * 2.0 * vel;
                            gradDs += penaWei * penaD *  2.0 * ds * vel.squaredNorm();
                        }
                    }
                    {
                        //realAcc = dds * vel + ds*acc*ds;
                        vioAcc = realAcc.squaredNorm() - accRatemax * accRatemax;
                        if(vioAcc > 0){
                            double pena, penaD;
                            positiveSmoothedL1(vioAcc, pena, penaD);
                            cost += penaWei * pena;
                            Eigen::Vector2d gradViolaAv;
                            Eigen::Vector2d gradViolaAa;
                            double gradViolaAds;
                            double gradViolaAdds;
                            gradViolaAv = dds * dds * 2.0 * vel + 2.0 * dds * ds * ds * acc;
                            gradViolaAa = 2.0 * dds * ds * ds * vel + ds * ds * ds * ds * 2.0 * acc;
                            gradViolaAds = 2.0 * ds * 2.0 * dds * vel.transpose() * acc + 4.0 * ds * ds * ds * acc.squaredNorm();
                            gradViolaAdds = 2.0 * dds * vel.squaredNorm() + 2.0 * ds * ds * vel.transpose() * acc;
                            gradVel += penaWei * penaD * gradViolaAv;
                            gradAcc += penaWei * penaD * gradViolaAa;
                            gradDs += penaWei * penaD * gradViolaAds;
                            gradDds += penaWei * penaD * gradViolaAdds;

                        }
                       
                    }
                    /*static collsion avoidance*/
                    //full shape model
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
                            min_dis_ = std::min(dis, min_dis_);
                            double vioSdist = (-dis + safeMargin);
                            if(vioSdist > 0){
                                double pena, penaD;
                                positiveSmoothedL1(vioSdist, pena, penaD);
                                // positiveSmoothedL3(vioSdist, pena, penaD);
                                cost += esdfWei * pena;
                                debugEsdf += esdfWei * pena;
                                gradPos += esdfWei * penaD * (-1.0) * gradViolaSdpos;
                                gradVel += esdfWei * penaD * (-1.0) * 
                                (sg * temp_l_Bl / vel.norm() - vel * (rotR * resPt).transpose() / (vel.squaredNorm()) )
                                * gradViolaSdpos;
                            }
                            
                        }
                    }
                   
                    //realAcc = dds * vel + ds*acc*ds;
                    //realJerk = ddds * vel + acc * dds * ds +2.0 * ds * acc * dds + ds * ds * ds * jerk;
                    //step = tOpt.getDt() / traj_res;
                    {
                        double wpw = 1.0;
                        if(sg > 0){
                            wpw = 1.0;
                        }
                        else{
                            wpw = backW_;
                        }
                        
                        cost += wpw * step * realJerk.squaredNorm();
                        gradVel += wpw * step * 2.0 * ddds * realJerk;
                        gradAcc += wpw * step * 2.0 * 3.0 * ds * dds * realJerk;
                        gradJerk += wpw * step * 2.0 * ds * ds * ds * realJerk;

                        gradDs += wpw * step * 2.0 * realJerk.transpose() * (3*acc*dds + 3*ds*ds*jerk);
                        gradDds += wpw * step * 2.0 * realJerk.transpose() * (3*acc*ds);
                        gradDdds += wpw * step * 2.0 * realJerk.transpose() * (vel);
                        gdC2t_1d[i] += wpw * realJerk.squaredNorm() / traj_res;
                        
                        debugEnergy += wpw * step * realJerk.squaredNorm();

                    }
                    
                    double gradResS = (gradPos.dot(vel) +
                                 gradVel.dot(acc) +
                                 gradAcc.dot(jerk) +
                                 gradJerk.dot(snap));
                    gradS += gradResS;
                    gdC2c_1d.block<6, 1>(i * 6, 0) += beta0_1d * gradS +
                                                    beta1_1d * gradDs +
                                                    beta2_1d * gradDds +
                                                    beta3_1d * gradDdds;
                    gdC2t_1d[i] += (gradS * (ds) +
                                 gradDs * (dds) +
                                 gradDds * (ddds) +
                                 gradDdds * (dddds)) *
                                    alpha;


                    gdC2c_2d.block<6, 2>(i * 6, 0) += beta0_2d * gradPos.transpose() +
                                                    beta1_2d * gradVel.transpose() +
                                                    beta2_2d * gradAcc.transpose() +
                                                    beta3_2d * gradJerk.transpose();
                    if(i > 0){
                        gdC2t_2d.segment(0, i).array() +=  -gradResS;
                    }
                                
                }
                baseS += posOpt.T1(i) ;
            }
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
       
        
        // std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> getSe2traj(double dt){
        //     std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> arrowTraj;
        //     Se2Trajectory traj;
        //     traj.posTraj = posJerkOpt.getTraj();
        //     traj.angleTraj = yawJerkOpt.getTraj();
        //     for(double t = 0.0; t < traj.getTotalDuration()-1.0e-3; t+=dt){
        //         Eigen::Vector2d pos= traj.getPos(t);
        //         double angle = traj.getAngle(t);
        //         Eigen::Vector3d p1,p2;
        //         p1 << pos[0], pos[1], 0.0;
        //         p2 << pos[0] + 1.0 * cos(angle), pos[1] + 1.0 * sin(angle), 0.0;
        //         arrowTraj.push_back(std::pair<Eigen::Vector3d,Eigen::Vector3d>(p1,p2));
        //     }
        //     Eigen::Vector2d pos= traj.getPos(traj.getTotalDuration());
        //     double angle = traj.getAngle(traj.getTotalDuration());
        //     Eigen::Vector3d p1,p2;
        //     p1 << pos[0], pos[1], 0.0;
        //     p2 << pos[0] + 1.0 * cos(angle), pos[1] + 1.0 * sin(angle), 0.0;
        //     arrowTraj.push_back(std::pair<Eigen::Vector3d,Eigen::Vector3d>(p1,p2));
        //     return arrowTraj;
        // }
        std::vector<Eigen::Vector3d> getTraj(double dt){
            //return {px,py,yaw}
            std::vector<Eigen::Vector3d> poseTraj;
            // std::cout << "durations: ";
            // for(int i = 0; i < trajnum; i++){
            //     PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
            //     PolyTrajOpt::PolyTrajectory<1> ttraj = tOpts[i].getTraj();
            //     // std::cout << polytraj.getTotalDuration() << " ";
            //     for(double t = 0.0; t < polytraj.getTotalDuration()-1.0e-3; t+=dt){
            //         Eigen::Vector2d pos = polytraj.getSigma(t);
            //         poseTraj.push_back(Eigen::Vector3d(pos[0],pos[1],0));
            //         Eigen::Vector2d vel = polytraj.getdSigma(t);
            //         Eigen::Vector2d acc = polytraj.getddSigma(t);
            //         double curvatue = (vel[0]*acc[1] - acc[0]*vel[1]);
            //         // std::cout << curvatue << std::endl;
            //     }
            //     for(double t = 0.0; t < ttraj.getTotalDuration()-1.0e-3; t+=dt){
            //         Eigen::VectorXd ds = ttraj.getdSigma(t);
            //         // std::cout << ds[0] << std::endl;
            //     }
            // }
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
                for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=dt){
                    double s = Ttraj.getSigma(t)[0];
                    double ds = Ttraj.getdSigma(t)[0];
                    Eigen::Vector2d pos= polytraj.getSigma(s);
                    Eigen::Vector2d vel= polytraj.getdSigma(s);
                    poseTraj.push_back(Eigen::Vector3d(pos[0],pos[1],ds*vel.norm()));
                }   
            }
            
            // ROS_WARN_STREAM("MAX CUR : " << getMaxCur());
            return poseTraj;
        }

        double getMaxCur(){
            double maxcur = 0.0;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
                for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=0.01){
                    double s = Ttraj.getSigma(t)[0];
                    Eigen::Vector2d vel= polytraj.getdSigma(s);
                    Eigen::Vector2d acc= polytraj.getddSigma(s);
                    double cur = (vel[0]*acc[1] - acc[0]*vel[1]) / (vel.norm() * vel.norm() * vel.norm());
                    if(fabs(cur) > fabs(maxcur)){
                        maxcur = cur;
                    }
                }   
            }
            return maxcur;
        }
        double getLength(int i){
            double length = 0;
            PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
            PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
            for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=0.01){
                double s = Ttraj.getSigma(t)[0];
                double ds = Ttraj.getdSigma(t)[0];
                Eigen::Vector2d vel= polytraj.getdSigma(s);
                Eigen::Vector2d realV = vel * ds;
                length += 0.01 * realV.norm();
            }   
            return length;
        }
        double getTotalDuration(int i){
            PolyTrajOpt::PolyTrajectory<1> tTraj = tOpts[i].getTraj();
            return tTraj.getTotalDuration();
        }
        // Se2Trajectory getTraj(){
        //     Se2Trajectory traj;
        //     traj.posTraj = posJerkOpt.getTraj();
        //     traj.angleTraj = yawJerkOpt.getTraj();
        //     return traj;
        // }
        void debugVis(){
            if(isstart&&isdebug_){
                ROS_WARN("initial Guess");
            }
            vis_tool_ ->visualize_path(getTraj(0.01), "/visualization/refinedTraj_"+name_);
            std::vector<Eigen::Vector2d> wpts;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> traj = posOpts[i].getTraj();
                wpts.push_back(traj.getSigma(0.0));
                Eigen::MatrixXd interWpts = traj.getWpts();
                for(int j = 0; j < interWpts.cols(); j++){
                    wpts.push_back(interWpts.col(j));
                }
                wpts.push_back(traj.getSigma(traj.getTotalDuration()));
            }
            vis_tool_ ->visualize_pointcloud(wpts, "/visualization/wapoints_"+name_);

            UgvTrajectory  optTraj = getOptTraj();
            vis_tool_->visualize_se2traj(optTraj, "/visualization/fullshapeTraj_"+name_);
        

            if(isstart&&isdebug_){
                ros::Duration(7.5).sleep();   
                isstart = false; 
            }
        }
        UgvTrajectory getOptTraj(){
            UgvTrajectory optTraj;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();   
                DpTrajectory dpTraj;
                dpTraj.posTraj = polytraj;
                dpTraj.tTraj = Ttraj;
                optTraj.Traj_container.push_back(dpTraj);
                optTraj.etas.push_back(singual_[i]);
            }
            return optTraj;
        }
};
} 
#endif 