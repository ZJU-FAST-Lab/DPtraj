#ifndef POLYTRAJ_HPP
#define POLYTRAJ_HPP

#include <ros/ros.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>
#include <Eigen/Eigen>
#include "root_finder.hpp"
namespace PolyTrajOpt
{
    constexpr double PI = 3.1415926;
    // typedef Eigen::Matrix<double, 3, 6> CoefficientMat;
    // typedef Eigen::Matrix<double, 3, 5> dsigmaCoefficientMat;
    // typedef Eigen::Matrix<double, 3, 4> ddsigmaCoefficientMat;

    template<int D>
    using CoefficientMat = Eigen::Matrix<double, D, 6>;
    template<int D>
    using dsigmaCoefficientMat = Eigen::Matrix<double, D, 5>;
    template<int D>
    using ddsigmaCoefficientMat = Eigen::Matrix<double, D, 4>;

    
    template <int Dim> 
    class Piece // component from poly
    {
    private:  // duration + coeffMat
        double duration;
        CoefficientMat<Dim> coeffMat;
        int order = 5;

    public:
        Piece() = default;

        Piece(double dur, const CoefficientMat<Dim> &cMat)
            : duration(dur), coeffMat(cMat) {}
        
        inline int getDim() const
        {
            return Dim;
        }
        
        inline int getOrder() const
        {
            return order;
        }

        inline double getDuration() const
        {
            return duration;
        }

        inline const CoefficientMat<Dim> &getCoeffMat() const
        {
            return coeffMat;
        }
        inline CoefficientMat<Dim> normalizeSigmaCoeffMat() const
        {
            CoefficientMat<Dim> nSigmaCoeffsMat;
            double t = 1.0;
            for (int i = order; i >= 0; i--)
            {
                nSigmaCoeffsMat.col(i) = coeffMat.col(i) * t;
                t *= duration;
            }
            return nSigmaCoeffsMat;
        }
        inline dsigmaCoefficientMat<Dim> normalizeDsigmaCoeffMat() const
        {
            dsigmaCoefficientMat<Dim> nDsigmalCoeffMat;
            int n = 1;
            double t = duration;
            for (int i = order - 1; i >= 0; i--)
            {
                nDsigmalCoeffMat.col(i) = n * coeffMat.col(i) * t;
                t *= duration;
                n++;
            }
            return nDsigmalCoeffMat;
        }

        inline ddsigmaCoefficientMat<Dim> normalizeDdsigmaCoeffMat() const
        {
            ddsigmaCoefficientMat<Dim> nDdsigmaCoeffMat;
            int n = 2;
            int m = 1;
            double t = duration * duration;
            for (int i = order - 2; i >= 0; i--)
            {
                nDdsigmaCoeffMat.col(i) = n * m * coeffMat.col(i) * t;
                n++;
                m++;
                t *= duration;
            }
            return nDdsigmaCoeffMat;
        }
        inline Eigen::Matrix<double, Dim, 1> getSigma(const double &t) const{
            Eigen::Matrix<double, Dim, 1>  Sigma;
            Sigma.setZero();
            double tn = 1.0;
            for (int i = order; i >= 0; i--)
            {
                Sigma += tn * coeffMat.col(i);
                tn *= t;
            }
            return Sigma;
        }
        inline Eigen::Matrix<double, Dim, 1> getdSigma(const double &t) const
        {
            Eigen::Matrix<double, Dim, 1> dSigma;
            dSigma.setZero();
            double tn = 1.0;
            int n = 1;

            for (int i = order-1; i >= 0; i--)
            {
                dSigma += n * tn * coeffMat.col(i);
                tn *= t;
                n++;
            }
            return dSigma;
        }
        inline Eigen::Matrix<double, Dim, 1> getddSigma(const double &t) const
        {
            Eigen::Matrix<double, Dim, 1> ddSigma;
            ddSigma.setZero();
            double tn = 1.0;
            int m = 1;
            int n = 2;
            for (int i = order-2; i >= 0; i--)
            {
                ddSigma += m * n * tn * coeffMat.col(i);
                tn *= t;
                m++;
                n++;
            }
            return ddSigma;
        }

        inline Eigen::Matrix<double, Dim, 1> getdddSigma(const double &t) const
        {
            Eigen::Matrix<double, Dim, 1> dddSigma;
            dddSigma.setZero();
            double tn = 1.0;
            int m = 1;
            int n = 2;
            int l = 3;
            for (int i = order-3; i >= 0; i--)
            {
                dddSigma += m * n * l * tn * coeffMat.col(i);
                tn *= t;
                m++;
                n++;
                l++;
            }

            return dddSigma;
        }


        
        inline double getMaxDsigmaNorm() const
        {
            dsigmaCoefficientMat<Dim> nDsigmaCoeffMat = normalizeDsigmaCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nDsigmaCoeffMat.row(0));
            for(int i = 1; i < Dim; i++){
                coeff = coeff + RootFinder::polySqr(nDsigmaCoeffMat.row(i));
            }
            int N = coeff.size();
            int n = N - 1;
            for (int i = 0; i < N; i++)
            {
                coeff(i) *= n;
                n--;
            }
            if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON)
            {
                return getdSigma(0.0).norm();
            }
            else
            {
                double l = -0.0625;
                double r = 1.0625;
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON)
                {
                    l = 0.5 * l;
                }
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON)
                {
                    r = 0.5 * (r + 1.0);
                }
                std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r,
                                                                        FLT_EPSILON / duration);
                candidates.insert(0.0);
                candidates.insert(1.0);
                double maxVelRateSqr = -INFINITY;
                double tempNormSqr;
                for (std::set<double>::const_iterator it = candidates.begin();
                    it != candidates.end();
                    it++)
                {
                    if (0.0 <= *it && 1.0 >= *it)
                    {
                        tempNormSqr = getdSigma((*it) * duration).norm() * getdSigma((*it) * duration).norm();
                        maxVelRateSqr = maxVelRateSqr < tempNormSqr ? tempNormSqr : maxVelRateSqr;
                    }
                }
                return sqrt(maxVelRateSqr);
            }
        }
        inline double getMaxDdsigmaNorm() const
        {
            ddsigmaCoefficientMat<Dim> nDdsigmaCoeffMat = normalizeDdsigmaCoeffMat();
            Eigen::VectorXd coeff = RootFinder::polySqr(nDdsigmaCoeffMat.row(0));
            for(int i = 1; i < Dim; i++){
                coeff = coeff + RootFinder::polySqr(nDdsigmaCoeffMat.row(i));
            }
            int N = coeff.size();
            int n = N - 1;
            for (int i = 0; i < N; i++)
            {
                coeff(i) *= n;
                n--;
            }
            if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON)
            {
                return getddSigma(0.0).norm();
            }
            else
            {
                double l = -0.0625;
                double r = 1.0625;
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON)
                {
                    l = 0.5 * l;
                }
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON)
                {
                    r = 0.5 * (r + 1.0);
                }
                std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r,
                                                                        FLT_EPSILON / duration);
                candidates.insert(0.0);
                candidates.insert(1.0);
                double maxAccRateSqr = -INFINITY;
                double tempNormSqr;
                for (std::set<double>::const_iterator it = candidates.begin();
                    it != candidates.end();
                    it++)
                {
                    if (0.0 <= *it && 1.0 >= *it)
                    {
                        tempNormSqr = getddSigma((*it) * duration).norm() * getddSigma((*it) * duration).norm();
                        maxAccRateSqr = maxAccRateSqr < tempNormSqr ? tempNormSqr : maxAccRateSqr;
                    }
                }
                return sqrt(maxAccRateSqr);
            }
        }
    };


    template <int Dim>
    class PolyTrajectory
    {
    private:
        using Pieces = std::vector<Piece<Dim>>;
        Pieces pieces;

    public:
        PolyTrajectory() = default;

        PolyTrajectory(const std::vector<double> &durs,
                const std::vector<CoefficientMat<Dim>> &cMats)
        {
            int N = std::min(durs.size(), cMats.size());
            pieces.reserve(N);
            for (int i = 0; i < N; i++)
            {
                pieces.emplace_back(durs[i], cMats[i]);
            }
        }

        inline int getPieceNum() const
        {
            return pieces.size();
        }
        inline Eigen::MatrixXd getWpts()const{
            int N = getPieceNum();
            Eigen::MatrixXd wpts;
            wpts.resize(Dim, N-1);
            for(int i = 0; i < N - 1; i++){
                wpts.col(i) << pieces[i].getSigma(pieces[i].getDuration());
            }
            return wpts;
        }
        inline Eigen::MatrixXd getKypoints()const{
            int N = getPieceNum();
            Eigen::MatrixXd keyPoints;
            keyPoints.resize(Dim, N+1);
            keyPoints.col(0) << pieces[0].getSigma(0); 
            for(int i = 0; i < N - 1; i++){
                keyPoints.col(i+1) << pieces[i].getSigma(pieces[i].getDuration());
            }
            keyPoints.col(N) << pieces[N-1].getSigma(pieces[N-1].getDuration());
            return keyPoints;
        }


        inline Eigen::MatrixXd getInPts()const{
            int N = getPieceNum();
            Eigen::MatrixXd InPts;
            InPts.resize(Dim, N-1);
            for(int i = 0; i < N - 1; i++){
                InPts.col(i) << pieces[i].getSigma(pieces[i].getDuration());
            }
            return InPts;
        }


        inline Eigen::VectorXd getDurations() const
        {
            int N = getPieceNum();
            Eigen::VectorXd durations(N);
            for (int i = 0; i < N; i++)
            {
                durations(i) = pieces[i].getDuration();
            }
            return durations;
        }

        inline double getTotalDuration() const
        {
            int N = getPieceNum();
            double totalDuration = 0.0;
            for (int i = 0; i < N; i++)
            {
                totalDuration += pieces[i].getDuration();
            }
            return totalDuration;
        }
        inline const Piece<Dim> &operator[](int i) const
        {
            return pieces[i];
        }

        inline Piece<Dim> &operator[](int i)
        {
            return pieces[i];
        }

        inline void clear(void)
        {
            pieces.clear();
            return;
        }

        inline void reserve(const int &n)
        {
            pieces.reserve(n);
            return;
        }

        inline void emplace_back(const Piece<Dim> &piece)
        {
            pieces.emplace_back(piece);
            return;
        }

        inline void emplace_back(const double &dur,
                                const CoefficientMat<Dim> &cMat)
        {
            pieces.emplace_back(dur, cMat);
            return;
        }

        inline void append(const PolyTrajectory<Dim> &traj)
        {
            pieces.insert(pieces.end(), traj.begin(), traj.end());
            return;
        }

        inline int locatePieceIdx(double &t) const
        {
            int N = getPieceNum();
            int idx;
            double dur;
            for (idx = 0;
                idx < N &&
                t > (dur = pieces[idx].getDuration());
                idx++)
            {
                t -= dur;
            }
            if (idx == N)
            {
                idx--;
                t += pieces[idx].getDuration();
            }
            return idx;
        }
        inline Eigen::Matrix<double, Dim, 1> getSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getSigma(t);
        }
        inline Eigen::Matrix<double, Dim, 1> getdSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getdSigma(t);
        }
        inline Eigen::Matrix<double, Dim, 1> getddSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getddSigma(t);
        }
        inline Eigen::Matrix<double, Dim, 1> getdddSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getdddSigma(t);
        }
        inline double getMaxDsigmaNorm() const{
            int N = getPieceNum();
            double maxDsigmaNorm = -INFINITY;
            double tempNorm;
            for (int i = 0; i < N; i++)
            {
                tempNorm = pieces[i].getMaxDsigmaNorm();
                maxDsigmaNorm = maxDsigmaNorm < tempNorm ? tempNorm : maxDsigmaNorm;
            }
            return maxDsigmaNorm;
        }
         inline double getMaxDdsigmaNorm() const{
            int N = getPieceNum();
            double maxDdsigmaNorm = -INFINITY;
            double tempNorm;
            for (int i = 0; i < N; i++)
            {
                tempNorm = pieces[i].getMaxDdsigmaNorm();
                maxDdsigmaNorm = maxDdsigmaNorm < tempNorm ? tempNorm : maxDdsigmaNorm;
            }
            return maxDdsigmaNorm;
        }
    };
    class DpTrajectory
    {
    private:

    public:
        DpTrajectory() = default;
        PolyTrajectory<2> posTraj;
        PolyTrajectory<1> tTraj;
        inline double getTotalDuration() const
        {
            return tTraj.getTotalDuration();
        }
        inline int getPieceNum() const{
            return posTraj.getPieceNum();
        }
        
        inline Eigen::Vector2d getPos(double t) const
        {
            return posTraj.getSigma(tTraj.getSigma(t)[0]);
        }
        inline Eigen::Vector2d getVel(double t) const{
            double s = tTraj.getSigma(t)[0];
            double ds = tTraj.getdSigma(t)[0];
            Eigen::Vector2d vel= posTraj.getdSigma(s);
            Eigen::Vector2d realV = vel * ds;
            return realV;
        }
        inline Eigen::Vector2d getAcc(double t) const{
            double s = tTraj.getSigma(t)[0];
            double ds = tTraj.getdSigma(t)[0];
            double dds = tTraj.getddSigma(t)[0];
            Eigen::Vector2d vel = posTraj.getdSigma(s);
            Eigen::Vector2d acc = posTraj.getddSigma(s);
            Eigen::Vector2d realA = dds * vel + ds*acc*ds;
            return realA;
        }


        inline Eigen::Vector2d getScale(double t) const{
            double s = tTraj.getSigma(t)[0];
            Eigen::Vector2d vel= posTraj.getdSigma(s);
            return vel;
        }
        inline Eigen::VectorXd getDurations() const{
            return tTraj.getDurations();
        }

        inline Eigen::MatrixXd getInPts()const{
            return posTraj.getInPts();
        }
        inline Eigen::VectorXd getDs()const{
            return posTraj.getDurations();
        }


        double getVelNorm(double t) const{
            return getVel(t).norm();
        }
        double getArc(double t) const{
            t = std::min(t, getTotalDuration());
            t = std::max(t, 0.0);
            if(t < 1.0e-3) return 0.0;
            double length = 0.0;
            for(double dt = 0.0; dt <= t-1.0e-3; dt += 0.001){
                double s = tTraj.getSigma(dt)[0];
                double ds = tTraj.getdSigma(dt)[0];
                Eigen::Vector2d vel= posTraj.getdSigma(s);
                Eigen::Vector2d realV = vel * ds;
                length += 0.001 * realV.norm();
            }
            return length;
        }


        inline double getTotalArc() const{
            double length = 0.0;
            for(double t = 0.0; t < tTraj.getTotalDuration()-1.0e-3; t+=0.01){
                double s = tTraj.getSigma(t)[0];
                double ds = tTraj.getdSigma(t)[0];
                Eigen::Vector2d vel= posTraj.getdSigma(s);
                Eigen::Vector2d realV = vel * ds;
                length += 0.01 * realV.norm();
            }   
            return length;
        }
        inline Eigen::Vector2d getPosByArc(double arc) const{
            double len = 0.0;
            if(arc ==0.0){
                return getPos(0.0);
            }
            else {
                for(double t = 0.0; t <= getTotalDuration(); t+=0.01){
                    double vel = getVelNorm(t);
                    len += vel * 0.01;
                    if(len >=arc){
                        return getPos(t);
                    }
                }
                return getPos(getTotalDuration());
            }
        }
        inline double getTByArc(double arc) const{
            double len = 0.0;
            if(arc ==0.0){
                return 0.0;
            }
            else {
                for(double t = 0.0; t <= getTotalDuration(); t+=0.01){
                    double vel = getVelNorm(t);
                    len += vel * 0.01;
                    if(len >=arc){
                        return t;
                    }
                }
                return getTotalDuration();
            }
        }
        inline Eigen::Vector2d getScaleByArc(double arc) const{
            double len = 0.0;
            if(arc ==0.0){
                return getScale(0.0);
            }
            else {
                for(double t = 0.0; t <= getTotalDuration(); t+=0.01){
                    double vel = getVelNorm(t);
                    len += vel * 0.01;
                    if(len >=arc){
                        return getScale(t);
                    }
                }
                return getScale(getTotalDuration());
            }
        }
    };
    class UgvTrajectory{
        private:

    public:
        UgvTrajectory() = default;
        std::vector<DpTrajectory> Traj_container;
        std::vector<int> etas;
        inline int getSegNum() const { return etas.size();}
        inline int locateTrajIdx(double &t) const
        {
            int N = etas.size();
            int idx;
            double dur;
            for (idx = 0;
                idx < N &&
                t > (dur = Traj_container[idx].getTotalDuration()+1.0e-6);
                idx++)
            {
                t -= dur;
            }
            if (idx == N)
            {
                idx--;
                t += Traj_container[idx].getTotalDuration();
            }
            return idx;
        }
        inline int locateTrajIdxByArc(double & arc) const{
            int N = etas.size();
            int idx;
            double dur;
            for (idx = 0;
                idx < N &&
                arc > (dur = Traj_container[idx].getTotalArc());
                idx++)
            {
                arc -= dur;
            }
            if (idx == N)
            {
                idx--;
                arc += Traj_container[idx].getTotalArc();
            }
            return idx;
        }
        inline int getDirection(double t) const{
            int idx = locateTrajIdx(t);
            if(etas[idx] > 0){
                return 1;
            }
            else{
                return -1;
            }
        }
        inline double getTotalDuration() const
        {
            double totalT = 0.0;
            for(const auto traj : Traj_container){
                totalT += traj.getTotalDuration();
            }
            return totalT;
        }

        
        inline double getTotalArc() const
        {
            double totalArc = 0.0;
            for(const auto traj : Traj_container){
                totalArc += traj.getTotalArc();
            }
            return totalArc;
        }
        inline double getSegArc(int i) const
        {
            return Traj_container[i].getTotalArc();
        }


        inline Eigen::Vector2d getPos(double t) const
        {
            int idx = locateTrajIdx(t);
            return Traj_container[idx].getPos(t);
        }
        inline double getArc(double t) const{
            double arc = 0.0;
            int idx = locateTrajIdx(t);
            if(idx > 0){
                for(int i = 0; i < idx; i++){
                    arc += Traj_container[i].getTotalArc();
                }
            }
            return arc + Traj_container[idx].getArc(t);
        }

        inline Eigen::Vector2d getPosByArc(double arc) const
        {
            int idx = locateTrajIdxByArc(arc);
            return Traj_container[idx].getPosByArc(arc);
        }
        double getVelItemByArc(double arc) const{
            double abst = getTByArc(arc);
            return getVelItem(abst);
        }
        double getCurByArc(double arc) const{
            double abst = getTByArc(arc);
            return getCur(abst);
        }


        inline double getTByArc(double arc) const{
            int idx = locateTrajIdxByArc(arc);
            double tt = 0.0;
            if(idx > 0){
                for(int i = 0; i < idx; i++){
                    tt += Traj_container[i].getTotalDuration();
                }
            }
            return tt + Traj_container[idx].getTByArc(arc);
        }


        double getVelItem(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];
            return Traj_container[idx].getVelNorm(t) * eta;
        }

        double getLonAcc(double t) const{
            int idx = locateTrajIdx(t);
            Eigen::Vector2d scale = Traj_container[idx].getScale(t);
            double yaw = atan2(etas[idx] * scale[1], etas[idx] * scale[0]);
            Eigen::Vector2d acc = Traj_container[idx].getAcc(t);
            return acc.dot(Eigen::Vector2d(cos(yaw), sin(yaw)));
        }
        double getCur(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];
            double s = Traj_container[idx].tTraj.getSigma(t)[0];
            Eigen::Vector2d vel = Traj_container[idx].posTraj.getdSigma(s);
            Eigen::Vector2d acc = Traj_container[idx].posTraj.getddSigma(s);
            return eta*(vel[0]*acc[1]-vel[1]*acc[0])/(vel.norm()*vel.norm()*vel.norm());
        }
        double getCurDot(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];

            double s = Traj_container[idx].tTraj.getSigma(t)[0];
            
            double ds = Traj_container[idx].tTraj.getdSigma(t)[0];
            Eigen::Vector2d vel = Traj_container[idx].posTraj.getdSigma(s);
            Eigen::Vector2d acc = Traj_container[idx].posTraj.getddSigma(s);
            Eigen::Vector2d jerk = Traj_container[idx].posTraj.getdddSigma(s);

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
            return kdot * eta;
        }
        double getPhi(double t, double wheelbase = 0.6){
            double c = getCur(t);
            return atan(c * wheelbase);
        }
        double getOmega(double t, double wheelbase = 0.6){
            int idx = locateTrajIdx(t);
            int eta = etas[idx];

            double s = Traj_container[idx].tTraj.getSigma(t)[0];
            double ds = Traj_container[idx].tTraj.getdSigma(t)[0];
            Eigen::Vector2d vel = Traj_container[idx].posTraj.getdSigma(s);
            Eigen::Vector2d acc = Traj_container[idx].posTraj.getddSigma(s);
            Eigen::Vector2d jerk = Traj_container[idx].posTraj.getdddSigma(s);

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
            double nor =  v6+tp2*tp2*wheelbase*wheelbase;
            double omega = wheelbase * denor / nor * ds;


            return omega * eta;
        }



        inline double getYaw(double t) const{
            int idx = locateTrajIdx(t);
            Eigen::Vector2d scale = Traj_container[idx].getScale(t);
            double yaw = atan2(etas[idx] * scale[1], etas[idx] * scale[0]);
            return yaw;
        }
        inline double getGearYaw(int idx) const{
            double T = Traj_container[idx].getTotalDuration();
            Eigen::Vector2d scale = Traj_container[idx].getScale(T);
            double yaw = atan2(etas[idx] * scale[1], etas[idx] * scale[0]);
            return yaw;
        }

        inline Eigen::Vector2d getGearPos(int idx) const
        {
            double T = Traj_container[idx].getTotalDuration();
            return Traj_container[idx].getPos(T);
        }
        


        inline double getYawByArc(double arc) const{
            int idx = locateTrajIdxByArc(arc);
            Eigen::Vector2d scale = Traj_container[idx].getScaleByArc(arc);
            double yaw = atan2(etas[idx] * scale[1], etas[idx] * scale[0]);
            return yaw;
        }




        inline Eigen::VectorXd getDurations() const{
            std::vector<double> stdDurations;
            for(const auto traj : Traj_container){
                Eigen::VectorXd tmpDurations = traj.getDurations();
                for(int i = 0; i < tmpDurations.size(); i++){
                    stdDurations.push_back(tmpDurations[i]);
                }
            }
            Eigen::VectorXd durations;
            durations.resize(stdDurations.size());
            for(int i = 0; i < stdDurations.size(); i++){
                durations[i] = stdDurations[i];
            }
            return durations;
        }
        inline int getPieceNum() const{
            int pieceNum = 0;
            for(const auto traj : Traj_container){
                pieceNum += traj.getPieceNum();
            }
            return pieceNum;
        }
        
       

    };
    

    // The banded system class is used for solving
    // banded linear system Ax=b efficiently.
    // A is an N*N band matrix with lower band width lowerBw
    // and upper band width upperBw.
    // Banded LU factorization has O(N) time complexity.
    class BandedSystem {
        public:
        // The size of A, as well as the lower/upper
        // banded width p/q are needed
        inline void create(const int &n, const int &p, const int &q) {
            // In case of re-creating before destroying
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        inline void destroy() {
            if (ptrData != nullptr) {
            delete[] ptrData;
            ptrData = nullptr;
            }
            return;
        }

        private:
        int N;
        int lowerBw;
        int upperBw;
        // Compulsory nullptr initialization here
        double *ptrData = nullptr;

        public:
        // Reset the matrix to zero
        inline void reset(void) {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        // The band matrix is stored as suggested in "Matrix Computation"
        inline const double &operator()(const int &i, const int &j) const {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j) {
            return ptrData[(i - j + upperBw) * N + j];
        }

        // This function conducts banded LU factorization in place
        // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
        inline void factorizeLU() {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; k++) {
            iM = std::min(k + lowerBw, N - 1);
            cVl = operator()(k, k);
            for (int i = k + 1; i <= iM; i++) {
                if (operator()(i, k) != 0.0) {
                operator()(i, k) /= cVl;
                }
            }
            jM = std::min(k + upperBw, N - 1);
            for (int j = k + 1; j <= jM; j++) {
                cVl = operator()(k, j);
                if (cVl != 0.0) {
                for (int i = k + 1; i <= iM; i++) {
                    if (operator()(i, k) != 0.0) {
                    operator()(i, j) -= operator()(i, k) * cVl;
                    }
                }
                }
            }
            }
            return;
        }

        // This function solves Ax=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const {
            int iM;
            for (int j = 0; j <= N - 1; j++) {
            iM = std::min(j + lowerBw, N - 1);
            for (int i = j + 1; i <= iM; i++) {
                if (operator()(i, j) != 0.0) {
                b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
            }
            for (int j = N - 1; j >= 0; j--) {
            b.row(j) /= operator()(j, j);
            iM = std::max(0, j - upperBw);
            for (int i = iM; i <= j - 1; i++) {
                if (operator()(i, j) != 0.0) {
                b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
            }
            return;
        }

        // This function solves ATx=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const {
            int iM;
            for (int j = 0; j <= N - 1; j++) {
            b.row(j) /= operator()(j, j);
            iM = std::min(j + upperBw, N - 1);
            for (int i = j + 1; i <= iM; i++) {
                if (operator()(j, i) != 0.0) {
                b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
            }
            for (int j = N - 1; j >= 0; j--) {
            iM = std::max(0, j - lowerBw);
            for (int i = iM; i <= j - 1; i++) {
                if (operator()(j, i) != 0.0) {
                b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
            }
            return;
        }
    };




template <int Dim>  class MinJerkOpt
    {
    public:
        MinJerkOpt() = default;
        ~MinJerkOpt() { A.destroy(); }


    private:
        int N; // pieceNum
        Eigen::MatrixXd headPVA, tailPVA;
        Eigen::MatrixXd b, c, adjScaledGrad;  // 6*N, 2
        BandedSystem A;  // 6 * N, 6 * N
        Eigen::Matrix<double, 6, 1> t, tInv;
    public:
        inline void reset(const int &pieceNum)
        {
            
            
            N = pieceNum;
            A.create(6 * N, 6, 6);
            b.resize(6 * N, Dim);
            c.resize(6 * N, Dim);
            adjScaledGrad.resize(6 * N, Dim);
        

            t(0) = 1.0;

            A(0, 0) = 1.0;
            A(1, 1) = 1.0;
            A(2, 2) = 2.0;
            for (int i = 0; i < N - 1; i++) {
            A(6 * i + 3, 6 * i + 3) = 6.0;
            A(6 * i + 3, 6 * i + 4) = 24.0;
            A(6 * i + 3, 6 * i + 5) = 60.0;
            A(6 * i + 3, 6 * i + 9) = -6.0;
            A(6 * i + 4, 6 * i + 4) = 24.0;
            A(6 * i + 4, 6 * i + 5) = 120.0;
            A(6 * i + 4, 6 * i + 10) = -24.0;
            A(6 * i + 5, 6 * i) = 1.0;
            A(6 * i + 5, 6 * i + 1) = 1.0;
            A(6 * i + 5, 6 * i + 2) = 1.0;
            A(6 * i + 5, 6 * i + 3) = 1.0;
            A(6 * i + 5, 6 * i + 4) = 1.0;
            A(6 * i + 5, 6 * i + 5) = 1.0;
            A(6 * i + 6, 6 * i) = 1.0;
            A(6 * i + 6, 6 * i + 1) = 1.0;
            A(6 * i + 6, 6 * i + 2) = 1.0;
            A(6 * i + 6, 6 * i + 3) = 1.0;
            A(6 * i + 6, 6 * i + 4) = 1.0;
            A(6 * i + 6, 6 * i + 5) = 1.0;
            A(6 * i + 6, 6 * i + 6) = -1.0;
            A(6 * i + 7, 6 * i + 1) = 1.0;
            A(6 * i + 7, 6 * i + 2) = 2.0;
            A(6 * i + 7, 6 * i + 3) = 3.0;
            A(6 * i + 7, 6 * i + 4) = 4.0;
            A(6 * i + 7, 6 * i + 5) = 5.0;
            A(6 * i + 7, 6 * i + 7) = -1.0;
            A(6 * i + 8, 6 * i + 2) = 2.0;
            A(6 * i + 8, 6 * i + 3) = 6.0;
            A(6 * i + 8, 6 * i + 4) = 12.0;
            A(6 * i + 8, 6 * i + 5) = 20.0;
            A(6 * i + 8, 6 * i + 8) = -2.0;
            }
            A(6 * N - 3, 6 * N - 6) = 1.0;
            A(6 * N - 3, 6 * N - 5) = 1.0;
            A(6 * N - 3, 6 * N - 4) = 1.0;
            A(6 * N - 3, 6 * N - 3) = 1.0;
            A(6 * N - 3, 6 * N - 2) = 1.0;
            A(6 * N - 3, 6 * N - 1) = 1.0;
            A(6 * N - 2, 6 * N - 5) = 1.0;
            A(6 * N - 2, 6 * N - 4) = 2.0;
            A(6 * N - 2, 6 * N - 3) = 3.0;
            A(6 * N - 2, 6 * N - 2) = 4.0;
            A(6 * N - 2, 6 * N - 1) = 5.0;
            A(6 * N - 1, 6 * N - 4) = 2.0;
            A(6 * N - 1, 6 * N - 3) = 6.0;
            A(6 * N - 1, 6 * N - 2) = 12.0;
            A(6 * N - 1, 6 * N - 1) = 20.0;
            A.factorizeLU();

            return;
        }

        inline void generate(const Eigen::MatrixXd &inPs,
                            const double &dT,
                            const Eigen::MatrixXd &headState,
                            const Eigen::MatrixXd &tailState)
        {
            headPVA = headState;
            tailPVA = tailState;

            t(1) = dT;
            t(2) = t(1) * t(1);
            t(3) = t(2) * t(1);
            t(4) = t(2) * t(2);
            t(5) = t(4) * t(1);
            tInv = t.cwiseInverse();

            b.setZero();
            b.row(0) = headPVA.col(0).transpose();
            b.row(1) = headPVA.col(1).transpose() * t(1);
            b.row(2) = headPVA.col(2).transpose() * t(2);
            for (int i = 0; i < N - 1; i++) {
                b.row(6 * i + 5) = inPs.col(i).transpose();
            }
            b.row(6 * N - 3) = tailPVA.col(0).transpose();
            b.row(6 * N - 2) = tailPVA.col(1).transpose() * t(1);
            b.row(6 * N - 1) = tailPVA.col(2).transpose() * t(2);

            A.solve(b);
            for (int i = 0; i < N; i++) {
            c.block<6, Dim>(6 * i, 0) =
                b.block<6, Dim>(6 * i, 0).array().colwise() * tInv.array();
            }
            return;
        }
        inline PolyTrajectory<Dim> getTraj() const
        {
            PolyTrajectory<Dim> polytraj;
            polytraj.reserve(N);
            for (int i = 0; i < N; i++)
            {
                polytraj.emplace_back(t(1), c.block<6, Dim>(6 * i, 0).transpose().rowwise().reverse());
            }

            return polytraj;
        }
        inline double getTrajJerkCost() const {
            double energy = 0.0;
            for (int i = 0; i < N; i++) {
            energy += 36.0 * c.row(6 * i + 3).squaredNorm() * t(1) +
                144.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * t(2) +
                192.0 * c.row(6 * i + 4).squaredNorm() * t(3) +
                240.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * t(3) +
                720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * t(4) +
                720.0 * c.row(6 * i + 5).squaredNorm() * t(5);
            }
            return energy;
        }
        

        inline void initSmGradCost(Eigen::MatrixXd& gdC, double& gdT) {
            gdC.resize(6 * N, Dim); 
            for (int i = 0; i < N; i++) {
                gdC.row(6 * i + 5) = 240.0 * c.row(6 * i + 3) * t(3) +
                                    720.0 * c.row(6 * i + 4) * t(4) +
                                    1440.0 * c.row(6 * i + 5) * t(5);
                gdC.row(6 * i + 4) = 144.0 * c.row(6 * i + 3) * t(2) +
                                    384.0 * c.row(6 * i + 4) * t(3) +
                                    720.0 * c.row(6 * i + 5) * t(4);
                gdC.row(6 * i + 3) = 72.0 * c.row(6 * i + 3) * t(1) +
                                    144.0 * c.row(6 * i + 4) * t(2) +
                                    240.0 * c.row(6 * i + 5) * t(3);
                gdC.block<3, Dim>(6 * i, 0).setZero();
            }
            gdT = 0.0;
            for (int i = 0; i < N; i++) {
                gdT += 36.0 * c.row(6 * i + 3).squaredNorm() +
                    288.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * t(1) +
                    576.0 * c.row(6 * i + 4).squaredNorm() * t(2) +
                    720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * t(2) +
                    2880.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * t(3) +
                    3600.0 * c.row(6 * i + 5).squaredNorm() * t(4);
            }
            return;
        }
        
        
        inline void calGrads_PT(const Eigen::MatrixXd& gdC, double& gdT, Eigen::MatrixXd& gdHead, Eigen::MatrixXd& gdTail,
        Eigen::MatrixXd& gdP) {
            gdP.resize(Dim, N - 1);
            gdHead.resize(Dim, 3);
            gdTail.resize(Dim, 3);
            for (int i = 0; i < N; i++) {
            adjScaledGrad.block<6, Dim>(6 * i, 0) =
                gdC.block<6, Dim>(6 * i, 0).array().colwise() * tInv.array();
            }
            A.solveAdj(adjScaledGrad);

            for (int i = 0; i < N - 1; i++) {
                gdP.col(i) = adjScaledGrad.row(6 * i + 5).transpose();
            }
            gdHead = adjScaledGrad.topRows(3).transpose() * t.head<3>().asDiagonal();
            gdTail = adjScaledGrad.bottomRows(3).transpose() * t.head<3>().asDiagonal();

            gdT += headPVA.col(1).dot(adjScaledGrad.row(1));
            gdT += headPVA.col(2).dot(adjScaledGrad.row(2)) * 2.0 * t(1);
            gdT += tailPVA.col(1).dot(adjScaledGrad.row(6 * N - 2));
            gdT += tailPVA.col(2).dot(adjScaledGrad.row(6 * N - 1)) * 2.0 * t(1);
            Eigen::Matrix<double, 6, 1> gdtInv;
            gdtInv(0) = 0.0;
            gdtInv(1) = -1.0 * tInv(2);
            gdtInv(2) = -2.0 * tInv(3);
            gdtInv(3) = -3.0 * tInv(4);
            gdtInv(4) = -4.0 * tInv(5);
            gdtInv(5) = -5.0 * tInv(5) * tInv(1);
            const Eigen::VectorXd gdcol = gdC.cwiseProduct(b).rowwise().sum();
            for (int i = 0; i < N; i++) {
                gdT += gdtInv.dot(gdcol.segment<6>(6 * i));
            }
            return;
        }
        inline void calGrads_PT(const Eigen::MatrixXd& gdC, double& gdT, Eigen::MatrixXd& gdP) {
            gdP.resize(Dim, N - 1);
            for (int i = 0; i < N; i++) {
            adjScaledGrad.block<6, Dim>(6 * i, 0) =
                gdC.block<6, Dim>(6 * i, 0).array().colwise() * tInv.array();
            }
            A.solveAdj(adjScaledGrad);

            for (int i = 0; i < N - 1; i++) {
                gdP.col(i) = adjScaledGrad.row(6 * i + 5).transpose();
            }

            gdT += headPVA.col(1).dot(adjScaledGrad.row(1));
            gdT += headPVA.col(2).dot(adjScaledGrad.row(2)) * 2.0 * t(1);
            gdT += tailPVA.col(1).dot(adjScaledGrad.row(6 * N - 2));
            gdT += tailPVA.col(2).dot(adjScaledGrad.row(6 * N - 1)) * 2.0 * t(1);
            Eigen::Matrix<double, 6, 1> gdtInv;
            gdtInv(0) = 0.0;
            gdtInv(1) = -1.0 * tInv(2);
            gdtInv(2) = -2.0 * tInv(3);
            gdtInv(3) = -3.0 * tInv(4);
            gdtInv(4) = -4.0 * tInv(5);
            gdtInv(5) = -5.0 * tInv(5) * tInv(1);
            const Eigen::VectorXd gdcol = gdC.cwiseProduct(b).rowwise().sum();
            for (int i = 0; i < N; i++) {
                gdT += gdtInv.dot(gdcol.segment<6>(6 * i));
            }
            return;
        }

        inline const int &getPieceNum (void) const{
            return N;
        }
        inline const Eigen::MatrixXd &getCoeffs(void) const {
            return c;
        }
        inline const double &getDt(void) const {
            return  t(1);
        }
    };








    



   


    typedef MinJerkOpt<1> MinJerkOpt1d;
    
    typedef MinJerkOpt<2> MinJerkOpt2d;

    typedef MinJerkOpt<3> MinJerkOpt3d;

 template <int Dim> class MINCO_S3NU
        {
        public:
            MINCO_S3NU() = default;
            ~MINCO_S3NU() { A.destroy(); }
            Eigen::VectorXd T1;


        private:
            int N;
            Eigen::MatrixXd headPVA;
            Eigen::MatrixXd tailPVA;
            BandedSystem A;
            Eigen::MatrixXd c;
            Eigen::VectorXd T2;
            Eigen::VectorXd T3;
            Eigen::VectorXd T4;
            Eigen::VectorXd T5;

        public:
            inline void reset(const int &pieceNum)
            {
                N = pieceNum;
                A.create(6 * N, 6, 6);
                c.resize(6 * N, Dim);
                T1.resize(N);
                T2.resize(N);
                T3.resize(N);
                T4.resize(N);
                T5.resize(N);
                return;
            }

            inline void generate(const Eigen::MatrixXd &inPs,
                                    const Eigen::VectorXd &ts,
                                     const Eigen::MatrixXd &headState,
                            const Eigen::MatrixXd &tailState)
            {
                headPVA = headState;
                tailPVA = tailState;
                T1 = ts;
                T2 = T1.cwiseProduct(T1);
                T3 = T2.cwiseProduct(T1);
                T4 = T2.cwiseProduct(T2);
                T5 = T4.cwiseProduct(T1);

                A.reset();
                c.setZero();

                A(0, 0) = 1.0;
                A(1, 1) = 1.0;
                A(2, 2) = 2.0;
                c.row(0) = headPVA.col(0).transpose();
                c.row(1) = headPVA.col(1).transpose();
                c.row(2) = headPVA.col(2).transpose();

                for (int i = 0; i < N - 1; i++)
                {
                    A(6 * i + 3, 6 * i + 3) = 6.0;
                    A(6 * i + 3, 6 * i + 4) = 24.0 * T1(i);
                    A(6 * i + 3, 6 * i + 5) = 60.0 * T2(i);
                    A(6 * i + 3, 6 * i + 9) = -6.0;
                    A(6 * i + 4, 6 * i + 4) = 24.0;
                    A(6 * i + 4, 6 * i + 5) = 120.0 * T1(i);
                    A(6 * i + 4, 6 * i + 10) = -24.0;
                    A(6 * i + 5, 6 * i) = 1.0;
                    A(6 * i + 5, 6 * i + 1) = T1(i);
                    A(6 * i + 5, 6 * i + 2) = T2(i);
                    A(6 * i + 5, 6 * i + 3) = T3(i);
                    A(6 * i + 5, 6 * i + 4) = T4(i);
                    A(6 * i + 5, 6 * i + 5) = T5(i);
                    A(6 * i + 6, 6 * i) = 1.0;
                    A(6 * i + 6, 6 * i + 1) = T1(i);
                    A(6 * i + 6, 6 * i + 2) = T2(i);
                    A(6 * i + 6, 6 * i + 3) = T3(i);
                    A(6 * i + 6, 6 * i + 4) = T4(i);
                    A(6 * i + 6, 6 * i + 5) = T5(i);
                    A(6 * i + 6, 6 * i + 6) = -1.0;
                    A(6 * i + 7, 6 * i + 1) = 1.0;
                    A(6 * i + 7, 6 * i + 2) = 2 * T1(i);
                    A(6 * i + 7, 6 * i + 3) = 3 * T2(i);
                    A(6 * i + 7, 6 * i + 4) = 4 * T3(i);
                    A(6 * i + 7, 6 * i + 5) = 5 * T4(i);
                    A(6 * i + 7, 6 * i + 7) = -1.0;
                    A(6 * i + 8, 6 * i + 2) = 2.0;
                    A(6 * i + 8, 6 * i + 3) = 6 * T1(i);
                    A(6 * i + 8, 6 * i + 4) = 12 * T2(i);
                    A(6 * i + 8, 6 * i + 5) = 20 * T3(i);
                    A(6 * i + 8, 6 * i + 8) = -2.0;

                    c.row(6 * i + 5) = inPs.col(i).transpose();
                }

                A(6 * N - 3, 6 * N - 6) = 1.0;
                A(6 * N - 3, 6 * N - 5) = T1(N - 1);
                A(6 * N - 3, 6 * N - 4) = T2(N - 1);
                A(6 * N - 3, 6 * N - 3) = T3(N - 1);
                A(6 * N - 3, 6 * N - 2) = T4(N - 1);
                A(6 * N - 3, 6 * N - 1) = T5(N - 1);
                A(6 * N - 2, 6 * N - 5) = 1.0;
                A(6 * N - 2, 6 * N - 4) = 2 * T1(N - 1);
                A(6 * N - 2, 6 * N - 3) = 3 * T2(N - 1);
                A(6 * N - 2, 6 * N - 2) = 4 * T3(N - 1);
                A(6 * N - 2, 6 * N - 1) = 5 * T4(N - 1);
                A(6 * N - 1, 6 * N - 4) = 2;
                A(6 * N - 1, 6 * N - 3) = 6 * T1(N - 1);
                A(6 * N - 1, 6 * N - 2) = 12 * T2(N - 1);
                A(6 * N - 1, 6 * N - 1) = 20 * T3(N - 1);

                c.row(6 * N - 3) = tailPVA.col(0).transpose();
                c.row(6 * N - 2) = tailPVA.col(1).transpose();
                c.row(6 * N - 1) = tailPVA.col(2).transpose();

                A.factorizeLU();
                A.solve(c);

                return;
            }

            inline PolyTrajectory<Dim> getTraj() const
            {
                PolyTrajectory<Dim> polytraj;
                polytraj.reserve(N);
                for (int i = 0; i < N; i++)
                {
                
                    polytraj.emplace_back(T1(i),
                                    c.block<6, Dim>(6 * i, 0)
                                        .transpose()
                                        .rowwise()
                                        .reverse());
                }
                return polytraj;
            }

            inline double getTrajJerkCost() const
            {
                double energy = 0.0;
                for (int i = 0; i < N; i++)
                {
                    energy += 36.0 * c.row(6 * i + 3).squaredNorm() * T1(i) +
                            144.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * T2(i) +
                            192.0 * c.row(6 * i + 4).squaredNorm() * T3(i) +
                            240.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * T3(i) +
                            720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * T4(i) +
                            720.0 * c.row(6 * i + 5).squaredNorm() * T5(i);
                }
                return energy;
            }

            inline const Eigen::MatrixXd &getCoeffs(void) const
            {
                return c;
            }

            inline void initSmGradCost(Eigen::MatrixXd& gdC, Eigen::VectorXd &gdT) const
            {
                gdC.resize(6 * N, Dim);
                for (int i = 0; i < N; i++)
                {
                    gdC.row(6 * i + 5) = 240.0 * c.row(6 * i + 3) * T3(i) +
                                        720.0 *  c.row(6 * i + 4) * T4(i) +
                                        1440.0 * c.row(6 * i + 5) * T5(i);
                    gdC.row(6 * i + 4) = 144.0 * c.row(6 * i + 3) * T2(i) +
                                        384.0 * c.row(6 * i + 4) * T3(i) +
                                        720.0 * c.row(6 * i + 5) * T4(i);
                    gdC.row(6 * i + 3) = 72.0 * c.row(6 * i + 3) * T1(i) +
                                        144.0 * c.row(6 * i + 4) * T2(i) +
                                        240.0 * c.row(6 * i + 5) * T3(i);
                    gdC.block<3, Dim>(6 * i, 0).setZero();
                }
                gdT.resize(N);
                for (int i = 0; i < N; i++)
                {
                    gdT(i) = 36.0 * c.row(6 * i + 3).squaredNorm() +
                            288.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * T1(i) +
                            576.0 * c.row(6 * i + 4).squaredNorm() * T2(i) +
                            720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * T2(i) +
                            2880.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * T3(i) +
                            3600.0 * c.row(6 * i + 5).squaredNorm() * T4(i);
                }

                return;
            }



            inline void calGrads_PT(const Eigen::MatrixXd& gdC, Eigen::VectorXd& gdT, Eigen::MatrixXd& gdP,
            Eigen::MatrixXd& gdHead, Eigen::MatrixXd& gdTail)

            {
                gdP.resize(Dim, N - 1);
                gdHead.resize(Dim, 3);
                gdTail.resize(Dim, 3);
                Eigen::MatrixXd adjGrad = gdC;
                A.solveAdj(adjGrad);

                for (int i = 0; i < N - 1; i++)
                {
                    gdP.col(i) = adjGrad.row(6 * i + 5).transpose();
                }
                gdHead = adjGrad.topRows(3).transpose();
                gdTail = adjGrad.bottomRows(3).transpose();

                Eigen::Matrix<double, 6, Dim> B1;
                Eigen::Matrix<double, 3, Dim> B2;
                for (int i = 0; i < N - 1; i++)
                {
                    // negative velocity
                    B1.row(2) = -(c.row(i * 6 + 1) +
                                2.0 * T1(i) * c.row(i * 6 + 2) +
                                3.0 * T2(i) * c.row(i * 6 + 3) +
                                4.0 * T3(i) * c.row(i * 6 + 4) +
                                5.0 * T4(i) * c.row(i * 6 + 5));
                    B1.row(3) = B1.row(2);

                    // negative acceleration
                    B1.row(4) = -(2.0 * c.row(i * 6 + 2) +
                                6.0 * T1(i) * c.row(i * 6 + 3) +
                                12.0 * T2(i) * c.row(i * 6 + 4) +
                                20.0 * T3(i) * c.row(i * 6 + 5));

                    // negative jerk
                    B1.row(5) = -(6.0 * c.row(i * 6 + 3) +
                                24.0 * T1(i) * c.row(i * 6 + 4) +
                                60.0 * T2(i) * c.row(i * 6 + 5));

                    // negative snap
                    B1.row(0) = -(24.0 * c.row(i * 6 + 4) +
                                120.0 * T1(i) * c.row(i * 6 + 5));

                    // negative crackle
                    B1.row(1) = -120.0 * c.row(i * 6 + 5);

                    gdT(i) += B1.cwiseProduct(adjGrad.block<6, Dim>(6 * i + 3, 0)).sum();
                }

                // negative velocity
                B2.row(0) = -(c.row(6 * N - 5) +
                            2.0 * T1(N - 1) * c.row(6 * N - 4) +
                            3.0 * T2(N - 1) * c.row(6 * N - 3) +
                            4.0 * T3(N - 1) * c.row(6 * N - 2) +
                            5.0 * T4(N - 1) * c.row(6 * N - 1));

                // negative acceleration
                B2.row(1) = -(2.0 * c.row(6 * N - 4) +
                            6.0 * T1(N - 1) *  c.row(6 * N - 3) +
                            12.0 * T2(N - 1) * c.row(6 * N - 2) +
                            20.0 * T3(N - 1) * c.row(6 * N - 1));

                // negative jerk
                B2.row(2) = -(6.0 * c.row(6 * N - 3) +
                            24.0 * T1(N - 1) * c.row(6 * N - 2) +
                            60.0 * T2(N - 1) * c.row(6 * N - 1));
                gdT(N - 1) += B2.cwiseProduct(adjGrad.block<3, Dim>(6 * N - 3, 0)).sum();

            }
        };
} //namespace plan_utils
#endif