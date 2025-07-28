#ifndef TRAJOPT.HPP
#define TRAJOPT.HPP
#include "Polytraj.hpp"
#include "lbfgs.hpp"
#include "geoutils2d.hpp"
#include <tools/gridmap.hpp>
#include <tools/config.hpp>
#include <fstream>
// using namespace std;

namespace PolyTrajOpt
{
class TrajOpt{
    private:
        
        Eigen::VectorXi piecenums;
        int trajnum;
        double gslar_ = 0.05;
        double wei_time_ = 500.0;
        double esdfWei = 1000.0;
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



    public:
        // se2Plan::MinJerkOpt<3> jerkOpt;
        std::vector<PolyTrajOpt::MINCO_S3NU<2>> posOpts;
        std::vector<PolyTrajOpt::MINCO_S3NU<1>> tOpts;
        double min_dis_;
        double debugEnergy, debugRhoT,debugEsdf, debugCur, debugMinis;

        

        inline int OptimizeSe2Trajectory(
            const Eigen::MatrixXd &headState2d, const Eigen::MatrixXd &tailState2d, Eigen::VectorXd rdTs_container,
            const std::vector<Eigen::MatrixXd> &inPs_container, const std::vector<Eigen::Vector2d> &gearPos,
             const std::vector<double> & angles, map_util::OccMapUtil* map_itf, std::shared_ptr<Config> config_, const std::vector<int> singual,
            std::shared_ptr<visualization::Visualization> vis_tool,std::string name = "nn", std::vector<Eigen::VectorXd> ds_container = {})
        {
            //arcs is total length for each mincotraj
            //angles is the angle at the gear shifting state
            //inPs_container[i] is the wayPts for each mincotraj

            name_ = name;
            vis_tool_ = vis_tool;




            /*setparam*/
            wei_time_ = config_->wei_time_;
            kdotmax = config_->kdotmax;
            kmax = config_->kmax;
            // kmax = 10000.0;
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
            map_itf_ =  map_itf;

            phimax = config_->phimax;
            omegamax = config_->omegamax;
            wheelbase = config_->wheel_base;
            kmax = tan(phimax) / wheelbase; 


            isstart = true;

            for(int i = 0; i < rdTs_container.size(); i++){
                if(rdTs_container[i] <= gslar_){
                    ROS_ERROR("piece time <= gslar_");
                    rdTs_container[i] = gslar_ + 0.1;
                }
            }
            
            //totalt is total time
            headState2d_ = headState2d;
            tailState2d_ = tailState2d;
            trajnum = singual_.size();
            // ROS_WARN_STREAM("trajnum: "<<trajnum);
            posOpts.resize(trajnum);piecenums.resize(trajnum);
            tOpts.resize(trajnum);
            int variable_num_ = 0;
            for(int i = 0; i < trajnum; i++) {
                piecenums[i] = inPs_container[i].cols() + 1;
                posOpts[i].reset(piecenums[i]);
                tOpts[i].reset(piecenums[i]);
                variable_num_ += 2 * (piecenums[i] - 1);
                variable_num_ += (piecenums[i]); // scale (ds
                eqc_num += piecenums[i] * (traj_res+1) * 0; // vnorm = 1
                neqc_num += piecenums[i] * (traj_res+1) * 5; //  vel>0.1 cur  s' >= 0   vel acc  omega
                // neqc_num += piecenums[i] * (traj_res+1) * conpts.size(); //   esdf
            }
            ROS_INFO("begin to optimize Arc Trajectory!");
            // std::cout << "trajnum: " << trajnum << std::endl;
            // std::cout << "piecenums: "<< piecenums.transpose() <<std::endl;
            variable_num_ += trajnum;         // dts
            variable_num_ += 2 * (trajnum-1); //gear pos
            variable_num_ += 1 * (trajnum-1); //gear angle

            totalc_num = eqc_num + neqc_num;
            scaling_wf_ = 1.0;
            scaling_eqwc_.resize(eqc_num); scaling_eqwc_.setConstant(1.0);
            scaling_uneqwc_.resize(totalc_num); scaling_uneqwc_.setConstant(1.0);
            hx.resize(eqc_num); lambda.resize(eqc_num);
            hx.setZero();       lambda.setZero();
            gx.resize(neqc_num); mu.resize(neqc_num);
            gx.setZero();        mu.setZero(); 
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
                            Rds[j] = (tmpPts.col(j) - tmpPts.col(j+1)).norm();
                        }
                    }
                    // std::cout << "trajid: " << i << " Rds: " << Rds.transpose() << std::endl;              
                    // Rds.setConstant(rdTs_container[i]);      
                }
                else{
                    Rds = ds_container[i];
                }
                           
                for(int j = 0; j < Rds.size(); j++){
                    if(Rds[j] <= gslar_){
                        ROS_ERROR("RT[i] <= gslar_");
                        Rds[j] = gslar_+ 0.1;
                    }
                }
                Rds_container.push_back(Rds);
                RealT2VirtualT(Rds, Vds);
                offset += piecenums[i];
            }

            //the variables beform optimization
            // ROS_INFO("The variables beform optimization: ");
            // for(int i = 0; i < trajnum; i++){
            //     std::cout << "trajid: "<<i<<" scales: "<<Rds_container[i].transpose()<<std::endl;
            //     std::cout << "trajid: "<<i<<" times: "<<rdTs_container[i]<<std::endl;
            //     std::cout << "trajid: "<<i<<" waypoints:\n"<<inPs_container[i]<<std::endl;
            //     std::cout << "trajid: "<<i<<" singuals: "<<singual_[i]<<std::endl;
            //     std::cout << "headState2d: \n"<<headState2d<<std::endl;
            //     std::cout << "tailState2d: \n"<<tailState2d<<std::endl;
            //     if(i < trajnum-1){
            //     std::cout << "gearPos: "<<gearPos[i].transpose()<<std::endl;
            //     std::cout << "angles: "<<angles[i]<<std::endl;}
            // }




            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.mem_size = config_->mem_size;//128
            lbfgs_params.past = config_->past; //3 
            lbfgs_params.g_epsilon = config_->g_epsilon;
            lbfgs_params.min_step = config_->min_step;
            lbfgs_params.delta = config_->delta;
            lbfgs_params.max_iterations = config_->max_iterations;
            double final_cost;
            bool flag_success = true;

            /*ALM*/
            double t1 = ros::Time::now().toSec();
            int iter = 0;
            while(true){
                int result = lbfgs::lbfgs_optimize(
                x,
                final_cost,
                TrajOpt::almCostFunctionCallback,
                NULL,
                NULL,
                this,
                lbfgs_params);
                if (result == lbfgs::LBFGS_CONVERGENCE ||
                result == lbfgs::LBFGS_CANCELED ||
                result == lbfgs::LBFGS_STOP||result == lbfgs::LBFGSERR_MAXIMUMITERATION)
                {
                // ROS_INFO_STREAM("se2 trajectory generation success! cost: " << final_cost );            
                 } 
                else if (result == lbfgs::LBFGSERR_MAXIMUMLINESEARCH){
                    ROS_WARN("Lbfgs: The line-search routine reaches the maximum number of evaluations.");
                    return 0;
                }
                else
                {
                    ROS_WARN("Solver error. Return = %d, %s. Skip this planning.", result, lbfgs::lbfgs_strerror(result));
                    return 0;
                }
                updateDualVar();
                if(ifConvergence()){
                    // ROS_WARN_STREAM("Convergence! iter: "<<iter);
                    break;
                }
                iter++;
                if(iter > config_->amlMaxIter){
                    ROS_WARN("Reach Max iteration");
                    return 0;
                    break;

                }
            }
            double t2 = ros::Time::now().toSec();
            debugVis();
            // printf("\033[32mse2 trajectory generation success!time(ms)=%5.3f, totalCost=%5.3f \n\033[0m", (t2-t1) * 1000.0,totalCost);
            // {
            //     ROS_INFO("Optimized scale s, times, waypoints: ");
            //     for(int i = 0; i < posOpts.size(); i++){
            //         std::cout << "trajid: " << i << " scales: " << posOpts[i].T1.transpose() << std::endl;
            //         std::cout << "trajid: " << i << " timess: " << tOpts[i].T1.transpose() << std::endl;
            //         PolyTrajOpt::PolyTrajectory<2> traj = posOpts[i].getTraj();
            //         std::cout << "waypoints: \n" << traj.getWpts() << std::endl;
            //     }
            //     UgvTrajectory optTraj = getOptTraj();
            //     for(int i = 0; i < optTraj.etas.size()-1; i++){
            //         std::cout << "i: "<<i<<" "<<optTraj.getGearPos(i).transpose() << " " << optTraj.getGearYaw(i) << "\n";
            //     }
            // }
            // ROS_WARN_STREAM("dptraj energy: " << debugEnergy);
            return flag_success;
        }


        static inline double almCostFunctionCallback(void *func_data, const Eigen::VectorXd &x, Eigen::VectorXd &grad){
            double smcost = 0.0, timecost = 0.0, almcost = 0.0;
            TrajOpt *opt = reinterpret_cast<TrajOpt *>(func_data);
            opt->totalCost = 0.0;

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
            std::vector<Eigen::VectorXd> gdS2t_2d(opt->trajnum), gdC2t_2d(opt->trajnum), gdTotalt_2d(opt->trajnum), gdS2t_1d(opt->trajnum), gdC2t_1d(opt->trajnum), gdTotalt_1d(opt->trajnum);

            opt->eqc_idx = 0;
            opt->uneqc_idx = 0;
            opt->debugEnergy = 0.0;
            opt->debugEsdf = 0.0;
            opt->min_dis_ = INFINITY;
            opt->debugRhoT = 0.0;
            opt->debugCur = -INFINITY;
            opt->debugMinis = INFINITY;
            // std::cout << "-------------------------------------\n";

            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                double s = 1.0 * opt->singual_[trajid];
                // double k;
                // if(s > 0){
                //     k= 1.0;
                // }
                // else{
                //     k = 1.0;
                // }
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
                // gdS2t_2d[trajid]=  gdS2t_2d[trajid] * k;
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
                    opt->tOpts[trajid].generate(wpTs, cdts, IniSt, FinSt);
                    opt->tOpts[trajid].initSmGradCost(gdS2c_1d[trajid], gdS2t_1d[trajid]);
                    // smcost += opt->tOpts[trajid].getTrajJerkCost() * k;
                    // gdS2c_1d[trajid] =  gdS2c_1d[trajid] * k;
                    // gdS2t_1d[trajid] = gdS2t_1d[trajid] * k;
                    gdS2c_1d[trajid].setZero();
                    gdS2t_1d[trajid].setZero();
                }

                // gdS2c_2d[trajid].setZero(); gdS2t_2d.setZero();
                opt->almGradCost2CT(almcost, opt->posOpts[trajid], opt->tOpts[trajid], gdC2c_2d[trajid], gdC2t_2d[trajid], gdC2c_1d[trajid], gdC2t_1d[trajid], trajid); // Time int cost
                // gdC2c_2d[trajid].setZero();
                // gdC2t_2d[trajid]=0.0;
                // gdC2c_1d[trajid].setZero();
                // gdC2t_1d[trajid].setZero();
                // almcost =  0.0;
                gdTotalc_2d[trajid] = gdC2c_2d[trajid] + gdS2c_2d[trajid];
                gdTotalt_2d[trajid] = gdC2t_2d[trajid] + gdS2t_2d[trajid];
                gdTotalc_1d[trajid] = gdC2c_1d[trajid] + gdS2c_1d[trajid];
                gdTotalt_1d[trajid] = gdC2t_1d[trajid] + gdS2t_1d[trajid];
                // gdTotalc_2d[trajid] = gdC2c_2d[trajid];
                // gdTotalt_2d[trajid] = gdC2t_2d[trajid];
            }
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
                //特殊
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
            }
            if(opt->isfix_){
                for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
                    gradGear_container[trajid].setZero();
                    gradAngles[trajid]  = 0.0;
                }
            }
            //hzchzc
            // gradGear_container[0].setZero();
            // gradGear_container[1].setZero();
            // gradAngles[0]  = 0.0;
            // gradAngles[1]  = 0.0;
            // gradVdts.setZero();
            if(opt->isdebug_){
                // std::cout << "------------------------------------------------------------------\n";
                std::cout << "min_dis_: "<<opt->min_dis_ << std::endl;
                std::cout << "debugCur: "<<opt->debugCur << std::endl;
                std::cout << "debugMinis: "<<opt->debugMinis << std::endl;
                // // std::cout << "gradVT: " << gradVT.transpose() << std::endl;
                // std::cout << "angles: "<<Angles.transpose() << std::endl;
                // std::cout << "gradAngles: "<<gradAngles.transpose() << std::endl;
                if(opt->debugEnergy > 1.0e10){
                    std::cout << "cost is too high!"<<"\n";
                }   
                else{
                    opt->debugVis();
                    ros::Duration(0.01).sleep();
                }
            }
            opt->totalCost += timecost;
            return smcost + timecost + almcost;
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
        void almGradCost2CT(double &cost,const PolyTrajOpt::MINCO_S3NU<2>& posOpt, const PolyTrajOpt::MINCO_S3NU<1>& tOpt, Eigen::MatrixXd& gdC2c_2d, Eigen::VectorXd& gdC2t_2d, Eigen::MatrixXd& gdC2c_1d, Eigen::VectorXd& gdC2t_1d, int trajid){
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

                    /*test vnorm = 1*/
                    // {
                    //     double arcscale = 1.0;
                    //     double eqf2 = arcscale * (vel.dot(vel) - 1.0);
                    //     double swc = scaling_eqwc_[eqc_idx];
                    //     double lbd = lambda[eqc_idx];
                    //     cost +=  swc * eqf2 * (lbd + 0.5 * rho * swc * eqf2);
                    //     hx[eqc_idx] = eqf2;
                    //     Eigen::Vector2d gradViolaM2v;
                    //     gradViolaM2v = 2.0 * arcscale * vel; 
                    //     gradVel += addAlmGradPro(swc, lbd, eqf2, gradViolaM2v);
                    //     eqc_idx ++;
                        
                    // }
                    // {
                    //     double arcscale = 10.0;
                    //     double eqf2 = arcscale * (vel.dot(vel) - 1.0) * step;
                    //     double swc = scaling_eqwc_[eqc_idx];
                    //     double lbd = lambda[eqc_idx];
                    //     cost +=  swc * eqf2 * (lbd + 0.5 * rho * swc * eqf2);
                    //     hx[eqc_idx] = eqf2;
                    //     Eigen::Vector2d gradViolaM2v;
                    //     gradViolaM2v = 2.0 * arcscale * vel * step;
                    //     double gradViolaM2t;
                    //     gradViolaM2t = arcscale * (vel.dot(vel) - 1.0) / traj_res;

                    //     gradVel += addAlmGradPro(swc, lbd, eqf2, gradViolaM2v);
                    //     gdC2t_1d += addAlmGradPro(swc, lbd, eqf2, gradViolaM2t);
                    //     eqc_idx ++;
                        
                    // }
                    /*v > 0.9*/
                    {
                        debugMinis = std::min(debugMinis, vel.norm());
                        double vionegV = miniS * miniS - vel.dot(vel);
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vionegV;
                        if(tmu + rho * swc * vionegV > 0){
                            cost += swc * vionegV * (tmu + 0.5 * rho * swc * vionegV);
                            Eigen::Vector2d gradViolanegv;
                            gradViolanegv = -2.0*vel;
                            gradVel += addAlmGradPro(swc, tmu, vionegV, gradViolanegv);
                        }
                        
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;

                    }
                    // {
                    //     double scalenegv = 10000.0;
                    //     double vionegV = 0.9 - vel.squaredNorm();
                    //     if(vionegV > 0){
                    //         double pena, penaD;
                    //         positiveSmoothedL1(vionegV, pena, penaD);
                    //         cost += scalenegv * pena;
                    //         gradVel += scalenegv * penaD * (-2.0) * vel;
                    //     }

                    // }
                    /*curvature constraint*/ 
                    {
                        double curs = 1.0;
                        cur = (vel[0]*acc[1] - acc[0]*vel[1]) / (vel.norm() * vel.norm() * vel.norm());
                        // if(fabs(cur)>100.0){
                        //     std::cout << "i: "<<i << " j: "<<j << std::endl;
                        //     std::cout <<"cur: "<< cur << std::endl;
                        //     std::cout <<"vel: "<< vel.transpose() << std::endl;
                        //     std::cout <<"acc: "<< acc << std::endl;
                        // }
                        double cross = (vel[0]*acc[1] - acc[0]*vel[1]);
                        vioCur = curs*(cur*cur - kmax*kmax);
                        debugCur = std::max(debugCur, fabs(cur));

                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioCur;
                        if(tmu + rho * swc * vioCur > 0){
                            cost += swc * vioCur * (tmu + 0.5 * rho * swc * vioCur);
                            Eigen::Vector2d gradViolaCv;
                            Eigen::Vector2d gradViolaCa;
                            
                            gradViolaCv = curs *(
                                          2*cross*Eigen::Vector2d(acc[1], -acc[0])/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()) -
                                          6*Eigen::Vector2d(vel[0], vel[1])*cross*cross/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm())
                                            ); 

                            gradViolaCa = curs * 2 * Eigen::Vector2d(-vel[1], vel[0]) * cross / (vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm());
                            gradVel += addAlmGradPro(swc, tmu, vioCur, gradViolaCv);
                            gradAcc += addAlmGradPro(swc, tmu, vioCur, gradViolaCa);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    // {
                    //     double curs = 1.0;
                    //     vioCur = curs*(cur*cur - kmax*kmax);
                    //     double swc = scaling_uneqwc_[uneqc_idx];
                    //     double tmu = mu[uneqc_idx];
                    //     gx[uneqc_idx] = vioCur;
                    //     if(tmu + rho * swc * vioCur > 0){
                    //         cost += swc * vioCur * (tmu + 0.5 * rho * swc * vioCur);
                    //         Eigen::Vector2d gradViolaCv;
                    //         Eigen::Vector2d gradViolaCa;
                    //         gradViolaCv = curs * 2 * cur * Eigen::Vector2d(acc[1], -acc[0]);
                    //         gradViolaCa = curs * 2 * cur * Eigen::Vector2d(-vel[1], vel[0]);
                    //         gradVel += addAlmGradPro(swc, tmu, vioCur, gradViolaCv);
                    //         gradAcc += addAlmGradPro(swc, tmu, vioCur, gradViolaCa);
                    //     }
                    //     else{
                    //         cost += -0.5 * tmu * tmu / rho;
                    //     }
                    //     uneqc_idx++;
                    // }
                    /*
                    {   //kdot
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
                        double kdot = denor / v6 * ds;
                        double vioKdot = kdot*kdot-kdotmax*kdotmax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioKdot;
                        if(tmu + rho * swc * vioKdot > 0){
                            cost += swc * vioKdot * (tmu + 0.5 * rho * swc * vioKdot);

                            Eigen::Vector2d gradViolaKdv;
                            Eigen::Vector2d gradtp1_v = -B*jerk;
                            Eigen::Vector2d gradv3_v = 3.0*vel.norm()*vel;
                            Eigen::Vector2d gradtp2_v = -B*acc;
                            Eigen::Vector2d gradtp3_v = acc;
                            Eigen::Vector2d gradv1_v = vel/vel.norm();
                            Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
                            Eigen::Vector2d graddenor_v = gradtp1_v*v3+tp1*gradv3_v-3.0*(gradtp2_v*tp3*v1+tp2*gradtp3_v*v1+tp2*tp3*gradv1_v);
                            gradViolaKdv = 2.0 * kdot * (graddenor_v*v6-gradv6_v*denor) / (v6*v6) * ds;

                            Eigen::Vector2d gradViolaKda;
                            Eigen::Vector2d gradtp2_a = B*vel;
                            Eigen::Vector2d gradtp3_a = vel;
                            Eigen::Vector2d graddenor_a = -3.0*(gradtp2_a*tp3*v1+tp2*gradtp3_a*v1);
                            gradViolaKda = 2.0*kdot*(graddenor_a/v6*ds);

                            Eigen::Vector2d gradViolaKdj;
                            Eigen::Vector2d gradtp1_j = B*vel;
                            Eigen::Vector2d graddenor_j = gradtp1_j*v3;
                            gradViolaKdj = 2.0*kdot*(graddenor_j/v6*ds);

                            double gradViolaKdds =  2.0*kdot*denor/v6;

                            gradVel += addAlmGradPro(swc, tmu, vioKdot, gradViolaKdv);
                            gradAcc += addAlmGradPro(swc, tmu, vioKdot, gradViolaKda);
                            gradJerk += addAlmGradPro(swc, tmu, vioKdot, gradViolaKdj);
                            gradDs +=addAlmGradPro(swc, tmu, vioKdot, gradViolaKdds);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }*/
                    /*
                    {   //omega
                        double womega = 5.0;
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
                        double omega = wheelbase * denor / nor * ds;
                        double vioOmega = womega * (omega*omega-omegamax*omegamax);

                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioOmega;
                        if(tmu + rho * swc * vioOmega > 0){
                            cost += swc * vioOmega * (tmu + 0.5 * rho * swc * vioOmega);

                            Eigen::Vector2d gradViolaOmegav;
                            Eigen::Vector2d gradtp1_v = -B*jerk;
                            Eigen::Vector2d gradv3_v = 3.0*vel.norm()*vel;
                            Eigen::Vector2d gradtp2_v = -B*acc;
                            Eigen::Vector2d gradtp3_v = acc;
                            Eigen::Vector2d gradv1_v = vel/vel.norm();
                            Eigen::Vector2d gradv6_v = 6.0*pow(vel.norm(),4)*vel;
                            Eigen::Vector2d graddenor_v = gradtp1_v*v3+tp1*gradv3_v-3.0*(gradtp2_v*tp3*v1+tp2*gradtp3_v*v1+tp2*tp3*gradv1_v);
                            Eigen::Vector2d gradnor_v = gradv6_v + 2.0*tp2*wheelbase*wheelbase*gradtp2_v;
                            gradViolaOmegav = womega * 2.0 * omega * (graddenor_v*nor-gradnor_v*denor) / (nor*nor) * ds * wheelbase;

                            Eigen::Vector2d gradViolaOmegaa;
                            Eigen::Vector2d gradtp2_a = B*vel;
                            Eigen::Vector2d gradtp3_a = vel;
                            Eigen::Vector2d graddenor_a = -3.0*(gradtp2_a*tp3*v1+tp2*gradtp3_a*v1);
                            Eigen::Vector2d gradnor_a = 2.0*tp2*wheelbase*wheelbase*gradtp2_a;
                            gradViolaOmegaa = womega *2.0 * omega * (graddenor_a*nor-gradnor_a*denor) / (nor*nor) * ds * wheelbase;

                            Eigen::Vector2d gradViolaOmegaj;
                            Eigen::Vector2d gradtp1_j = B*vel;
                            Eigen::Vector2d graddenor_j = gradtp1_j*v3;
                            gradViolaOmegaj = womega *2.0*omega*(graddenor_j/nor*ds*wheelbase);

                            double gradViolaOmegads =  womega *2.0*omega*denor/nor*wheelbase;

                            gradVel += addAlmGradPro(swc, tmu, vioOmega, gradViolaOmegav);
                            gradAcc += addAlmGradPro(swc, tmu, vioOmega, gradViolaOmegaa);
                            gradJerk += addAlmGradPro(swc, tmu, vioOmega, gradViolaOmegaj);
                            gradDs +=addAlmGradPro(swc, tmu, vioOmega, gradViolaOmegads);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }

                    */

                    /*s' >=0*/
                    {
                        
                        vioDs = -ds;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioDs;
                        if(tmu + rho * swc * vioDs > 0){
                            cost += swc * vioDs * (tmu + 0.5 * rho * swc * vioDs);
                            gradDs += addAlmGradPro(swc, tmu, vioDs, -1.0);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    {
                        //realVel (vel * ds)
                        vioVel = realVel.squaredNorm() - vmax * vmax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioVel;
                        if(tmu + rho * swc * vioVel > 0){
                            cost += swc * vioVel * (tmu + 0.5 * rho * swc * vioVel);
                            Eigen::Vector2d gradViolaVv;
                            double gradViolaVds;
                            gradViolaVv = ds * ds * 2.0 * vel;
                            gradViolaVds = 2.0 * ds * vel.squaredNorm();
                            gradVel += addAlmGradPro(swc, tmu, vioVel, gradViolaVv);
                            gradDs += addAlmGradPro(swc, tmu, vioVel, gradViolaVds);

                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    
                    
                    /*
                    {
                        //realAcc = dds * vel + ds*acc*ds;
                        vioAcc = realAcc.squaredNorm() - accRatemax * accRatemax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioAcc;
                        if(tmu + rho * swc * vioAcc > 0){
                            cost += swc * vioAcc * (tmu + 0.5 * rho * swc * vioAcc);
                            Eigen::Vector2d gradViolaAv;
                            Eigen::Vector2d gradViolaAa;
                            double gradViolaAds;
                            double gradViolaAdds;
                            gradViolaAv = dds * dds * 2.0 * vel + 2.0 * dds * ds * ds * acc;
                            gradViolaAa = 2.0 * dds * ds * ds * vel + ds * ds * ds * ds * 2.0 * acc;
                            gradViolaAds = 2.0 * ds * 2.0 * dds * vel.transpose() * acc + 4.0 * ds * ds * ds * acc.squaredNorm();
                            gradViolaAdds = 2.0 * dds * vel.squaredNorm() + 2.0 * ds * ds * vel.transpose() * acc;
                            gradVel += addAlmGradPro(swc, tmu, vioAcc, gradViolaAv);
                            gradAcc += addAlmGradPro(swc, tmu, vioAcc, gradViolaAa);
                            gradDs += addAlmGradPro(swc, tmu, vioAcc, gradViolaAds);
                            gradDds += addAlmGradPro(swc, tmu, vioAcc, gradViolaAdds);

                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }*/



                    {
                        //realAcc = dds * vel + ds*acc*ds;
                        lonAcc = dds * vel.norm() + ds * ds * acc.dot(vel) / vel.norm(); 
                        vioLonAcc = lonAcc * lonAcc - lonAccmax * lonAccmax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioLonAcc;

                        if(tmu + rho * swc * vioLonAcc > 0){
                            cost += swc * vioLonAcc * (tmu + 0.5 * rho * swc * vioLonAcc);
                            Eigen::Vector2d gradViolaLonAv;
                            Eigen::Vector2d gradViolaLonAa;
                            double gradViolaLonAds;
                            double gradViolaLonAdds;
                            gradViolaLonAv = 2.0*lonAcc*(dds*vel/vel.norm()+ds*ds*acc/vel.norm() - ds*ds*vel*acc.dot(vel)/(pow(vel.norm(),3)));
                            gradViolaLonAa = 2.0*lonAcc*(ds*ds*vel/vel.norm());
                            gradViolaLonAds = 2.0*lonAcc*(2*ds*acc.dot(vel) / vel.norm());
                            gradViolaLonAdds = 2.0*lonAcc*vel.norm(); 
                          
                            gradVel += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAv);
                            gradAcc += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAa);
                            gradDs += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAds);
                            gradDds += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAdds);

                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
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

                            double vioSdist = (-dis + safeMargin);//hzchzchzc
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
                    //point model
                    // {
                    //     Eigen::Vector2d gradViolaSdpos;
                    //     double dis = map_itf_->getDistGrad(pos, gradViolaSdpos);
                    //     double vioSdist = (-dis + 0.0);
                    //     if(vioSdist > 0){
                    //         double pena, penaD;
                    //         positiveSmoothedL1(vioSdist, pena, penaD);
                    //         // positiveSmoothedL3(vioSdist, pena, penaD);
                    //         cost += esdfWei * pena;
                    //         debugEsdf += esdfWei * pena;
                    //         gradPos += esdfWei * penaD * (-1.0) * gradViolaSdpos;
                    //     }
                    // }

                    // {
                    //     for(const Eigen::Vector2d resPt : conpts){
                    //         Eigen::Vector2d gradViolaSdpos;
                    //         Eigen::Vector2d absPt = pos + rotR * resPt;
                    //         Eigen::Matrix2d temp_l_Bl;
                    //         temp_l_Bl << resPt(0), resPt(1),
                    //                      -resPt(1), resPt(0);          
                    //         double dis = map_itf_->getDistGrad(absPt, gradViolaSdpos);
                    //         double vioSdist = (-dis + safeMargin);
                    //         double swc = scaling_uneqwc_[uneqc_idx];
                    //         double tmu = mu[uneqc_idx];
                    //         gx[uneqc_idx] = vioSdist;
                    //         // std::cout << "gx[uneqc_idx]" << gx[uneqc_idx] <<std::endl;
                    //         if(tmu + rho * swc * vioSdist > 0){
                    //             cost += swc * vioSdist * (tmu + 0.5 * rho * swc * vioSdist);
                    //             Eigen::Vector2d gradViolaSpos = (-1.0) * gradViolaSdpos;
                    //             Eigen::Vector2d gradViolaSvel = (-1.0) * 
                    //             (sg * temp_l_Bl / vel.norm() - vel * (rotR * resPt).transpose() / (vel.squaredNorm()) )
                    //             * gradViolaSdpos;
                    //             gradPos += addAlmGradPro(swc, tmu, vioSdist, gradViolaSpos);
                    //             gradVel += addAlmGradPro(swc, tmu, vioSdist, gradViolaSvel);
                    //         }
                    //         else{
                    //             cost += -0.5 * tmu * tmu / rho;
                    //         }
                    //         uneqc_idx++;
                    //     }
                    // }

                  
                    //realAcc = dds * vel + ds*acc*ds;
                    //realJerk = ddds * vel + 3.0 * ds * acc * dds + ds * ds * ds * jerk;
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
                        totalCost += wpw * step * realJerk.squaredNorm();
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
                baseS += posOpt.T1(i);
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
       
        void updateDualVar(){
            /*update lambda*/
            {
                for(int i = 0; i < eqc_num; i++){
                    lambda[i] += rho * scaling_eqwc_[i] * hx[i];
                }
            }
            /*update mu*/
            {
                for(int i = 0; i < neqc_num; i++){
                    if(mu[i] + rho * scaling_uneqwc_[i] * gx[i] >= 0){
                        mu[i] += rho * scaling_uneqwc_[i] * gx[i];
                    }
                    else{
                        mu[i] = 0;
                    }
                }
            }
            /*update rho*/
            {
                rho = std::min((1 + gamma) * rho, rho_max);
            }
        }
        bool ifConvergence(){
            //to do
            Eigen::VectorXd maxgx(neqc_num), maxhx(eqc_num);
            for(int i =0 ;i < eqc_num; i++){
                // maxhx[i] = hx[i] * lambda[i];
                maxhx[i] = hx[i];
            }
            for(int i = 0; i < neqc_num; i++){
                maxgx[i] = std::max(0.0, gx[i]);
            }
            double reshx;
            if(maxhx.size()==0){
                reshx = 0.0;
            }
            else{
                reshx = maxhx.cwiseAbs().maxCoeff();
            }
            int gxidx;
            double resgx = maxgx.cwiseAbs().maxCoeff(&gxidx);
            /*hzc may ???*/
            //&& (resgrad) < prec_eps_
            // std::cout << "reshx: "<<reshx << " resgx: "<<resgx << " gxidx: "<<gxidx%5<<std::endl;
            if (std::max(reshx, resgx) < cons_eps_ )
            {
                return true;
            }
            return false;
        }
        template <typename EIGENVEC>
        EIGENVEC addAlmGradPro(const double &ws, const double &lbd, const double &cons, const EIGENVEC &grad_){
            EIGENVEC output = ws * lbd * grad_ + rho * ws * ws * cons * grad_;
            return output;
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
                // std::cout <<" trajid: "<< i << " durations: "<< Ttraj.getDurations().transpose() << std::endl;
            }
            return poseTraj;
        }
        void writeTraj(){
            std::fstream f;
            f.open("/home/han/2023codes/NeuralTraj/GTtraj.txt",std::ios::out|std::ios::app);
            UgvTrajectory optTraj = getOptTraj();
            for(double t = 0.0; t <= optTraj.getTotalDuration(); t+=0.01){
                Eigen::Vector2d pos = optTraj.getPos(t);
                f << t <<" "<<pos[0] <<" "<<pos[1] <<" "<<optTraj.getYaw(t)<<std::endl;
            }
            f.close();
            return;
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
        // Se2Trajectory getTraj(){
        //     Se2Trajectory traj;
        //     traj.posTraj = posJerkOpt.getTraj();
        //     traj.angleTraj = yawJerkOpt.getTraj();
        //     return traj;
        // }
        double getTotalDuration(int i){
            PolyTrajOpt::PolyTrajectory<1> tTraj = tOpts[i].getTraj();
            return tTraj.getTotalDuration();
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
        double getMaxCurDot(){
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
            double maxcurdot = 0.0;

            // double baset = 0.0;
            // for(int id = 0; id < optTraj.etas.size(); id++){
            //     for(int i = 0; i < optTraj.Traj_container[id].tTraj.getPieceNum(); i++){
            //         double step = optTraj.Traj_container[id].tTraj.getDurations()[i]/traj_res;
            //         for(int j = 0; j <= traj_res; j++){
            //             double curt = baset + j*step;
            //             double curdot = optTraj.getCurDot(curt);     
            //             if(fabs(curdot) > fabs(maxcurdot)){
            //                 maxcurdot = curdot;
            //             }
            //         }
            //         baset += optTraj.Traj_container[id].tTraj.getDurations()[i];
            //     }

            // }

            for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
                double curdot = optTraj.getCurDot(t);        
                if(fabs(curdot) > fabs(maxcurdot)){
                    maxcurdot = curdot;
                }
            }
            return maxcurdot;
        }
        
        double getMaxOmega(){
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
            double maxomega = 0.0;

            // double baset = 0.0;
            // for(int id = 0; id < optTraj.etas.size(); id++){
            //     for(int i = 0; i < optTraj.Traj_container[id].tTraj.getPieceNum(); i++){
            //         double step = optTraj.Traj_container[id].tTraj.getDurations()[i]/traj_res;
            //         for(int j = 0; j <= traj_res; j++){
            //             double curt = baset + j*step;
            //             double curdot = optTraj.getCurDot(curt);     
            //             if(fabs(curdot) > fabs(maxcurdot)){
            //                 maxcurdot = curdot;
            //             }
            //         }
            //         baset += optTraj.Traj_container[id].tTraj.getDurations()[i];
            //     }

            // }

            for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
                double omega = optTraj.getOmega(t);        
                if(fabs(omega) > fabs(maxomega)){
                    maxomega = omega;
                }
            }
            return maxomega;
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
};
} 
#endif 