#pragma once

#include <ros/ros.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>

#include <Eigen/Eigen>


#include "root_finder.hpp"

namespace dftpav
{
    constexpr double PI = 3.1415926;
    typedef Eigen::Matrix<double, 2, 6> CoefficientMat;
    typedef Eigen::Matrix<double, 2, 5> VelCoefficientMat;
    typedef Eigen::Matrix<double, 2, 4> AccCoefficientMat;
    

    class Piece // component from poly
    {
    private:  // duration + coeffMat
        double duration;
        CoefficientMat coeffMat;

        int dim = 2;
        int order = 5;
        int singul;

    public:
        Piece() = default;

        Piece(double dur, const CoefficientMat &cMat, int s)
            : duration(dur), coeffMat(cMat), singul(s) {}
        //@yuwei
        inline int getDim() const
        {
            return dim;
        }
        //@yuwei
        inline int getOrder() const
        {
            return order;
        }

        inline double getDuration() const
        {
            return duration;
        }

        inline const CoefficientMat &getCoeffMat() const
        {
            return coeffMat;
        }

        inline VelCoefficientMat getVelCoeffMat() const
        {
            VelCoefficientMat velCoeffMat;
            int n = 1;
            for (int i = 4; i >= 0; i--)
            {
                velCoeffMat.col(i) = n * coeffMat.col(i);
                n++;
            }
            return velCoeffMat;
        }


        // the point in the rear axle center
        inline Eigen::Vector2d getSigma(const double &t) const
        {
            Eigen::Vector2d pos(0.0, 0.0);
            double tn = 1.0;
            for (int i = order; i >= 0; i--)
            {
                pos += tn * coeffMat.col(i);
                tn *= t;
            }
            return pos;
        }

        inline Eigen::Matrix2d getR(const double &t) const
        {
            Eigen::Vector2d current_v = getdSigma(t);
            Eigen::Matrix2d rotation_matrix;
            rotation_matrix << current_v(0), -current_v(1),
                               current_v(1), current_v(0);
            rotation_matrix = singul * rotation_matrix / current_v.norm();

            return rotation_matrix;
        }

        inline Eigen::Matrix2d getRdot(const double &t) const
        {
            Eigen::Vector2d current_v = getdSigma(t);
            Eigen::Vector2d current_a = getddSigma(t);
            Eigen::Matrix2d temp_a_ba, temp_v_bv;
            temp_a_ba << current_a(0), -current_a(1),
                         current_a(1), current_a(0);
            temp_v_bv << current_v(0), -current_v(1),
                         current_v(1), current_v(0);
            Eigen::Matrix2d R_dot = singul * (temp_a_ba / current_v.norm() - temp_v_bv / pow(current_v.norm(), 3) * (current_v.transpose() * current_a)); 
        
            return R_dot;
        }

      


        inline Eigen::Vector2d getdSigma(const double &t) const
        {


            Eigen::Vector2d dsigma(0.0, 0.0);
            double tn = 1.0;
            int n = 1;

            for (int i = order-1; i >= 0; i--)
            {
                dsigma += n * tn * coeffMat.col(i);
                tn *= t;
                n++;
            }
            return dsigma;
        }

        inline Eigen::Vector2d getddSigma(const double &t) const
        {
            Eigen::Vector2d ddsigma(0.0, 0.0);
            double tn = 1.0;
            int m = 1;
            int n = 2;

            for (int i = order-2; i >= 0; i--)
            {
                ddsigma += m * n * tn * coeffMat.col(i);
                tn *= t;
                m++;
                n++;
            }
            return ddsigma;
        }

        inline Eigen::Vector2d getdddSigma(const double &t) const
        {
            Eigen::Vector2d dddsigma(0.0, 0.0);
            double tn = 1.0;
            int l = 1;
            int m = 2;
            int n = 3;
  
            for (int i = order-3; i >= 0; i--)
            {
                dddsigma += l * m * n * tn * coeffMat.col(i);

                tn *= t;
                l++;
                m++;
                n++;
            }
            return dddsigma;
        }


        inline CoefficientMat normalizePosCoeffMat() const
        {
            CoefficientMat nPosCoeffsMat;
            double t = 1.0;
            for (int i = order; i >= 0; i--)
            {
                nPosCoeffsMat.col(i) = coeffMat.col(i) * t;
                t *= duration;
            }
            return nPosCoeffsMat;
        }
     
    };


    class Trajectory
    {
    private:
        typedef std::vector<Piece> Pieces;
        Pieces pieces;

    public:
        Trajectory() = default;

        Trajectory(const std::vector<double> &durs,
                const std::vector<CoefficientMat> &cMats, int s)
        {
            int N = std::min(durs.size(), cMats.size());
            pieces.reserve(N);
            for (int i = 0; i < N; i++)
            {
                pieces.emplace_back(durs[i], cMats[i], s);
            }
        }

      


        inline int getPieceNum() const
        {
            return pieces.size();
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

 

        inline const Piece &operator[](int i) const
        {
            return pieces[i];
        }

        inline Piece &operator[](int i)
        {
            return pieces[i];
        }

        inline void clear(void)
        {
            pieces.clear();
            return;
        }

        inline Pieces::const_iterator begin() const
        {
            return pieces.begin();
        }

        inline Pieces::const_iterator end() const
        {
            return pieces.end();
        }

        inline Pieces::iterator begin()
        {
            return pieces.begin();
        }

        inline Pieces::iterator end()
        {
            return pieces.end();
        }

        inline void reserve(const int &n)
        {
            pieces.reserve(n);
            return;
        }

        inline void emplace_back(const Piece &piece)
        {
            pieces.emplace_back(piece);
            return;
        }

        inline void emplace_back(const double &dur,
                                const CoefficientMat &cMat, int s)
        {
            pieces.emplace_back(dur, cMat, s);
            return;
        }

        inline void append(const Trajectory &traj)
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


        inline int locatePieceIdx2(double t) const
        {
            int N = getPieceNum();
            std::cout << "N = getPieceNum() is"<< N << std::endl;
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


        inline Eigen::Vector2d getSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getSigma(t);
        }


        
        inline Eigen::Vector2d getdSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getdSigma(t);
        }

        double getVelNorm(double t) const{
            return getdSigma(t).norm();
        }
        inline Eigen::Vector2d getddSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            // std::cout << "pieceIdx is"<< pieceIdx << std::endl;
            // std::cout << "t is"<< t << std::endl;                        
            return pieces[pieceIdx].getddSigma(t);
        }

        inline Eigen::Vector2d getdddSigma(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getdddSigma(t);
        }

        // hzc
        inline std::pair<int, double> locatePieceIdxWithRatio(double &t) const
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
            std::pair<int, double> idx_ratio;
            idx_ratio.first = idx;
            idx_ratio.second = t / dur;
            return idx_ratio;
        }
    };

    class DifTrajectory{
        private:

    public:
        DifTrajectory() = default;
        std::vector<Trajectory> Traj_container;
        std::vector<int> etas;
        inline int getSegNum() const { return etas.size(); }
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
        double getVelItem(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];
            return Traj_container[idx].getVelNorm(t) * eta;
        }
        inline double getTotalArc() const{
            double length = 0.0;
            for(double t = 0.0; t < getTotalDuration()-1.0e-3; t+=0.01){
                length += 0.01 * fabs(getVelItem(t));
            }   
            return length;
        }

        
     


        inline Eigen::Vector2d getPos(double t) const
        {
            int idx = locateTrajIdx(t);
            return Traj_container[idx].getSigma(t);
        }
        double getCur(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];
            Eigen::Vector2d vel = Traj_container[idx].getdSigma(t);
            Eigen::Vector2d acc = Traj_container[idx].getddSigma(t);
            return eta*(vel[0]*acc[1]-vel[1]*acc[0])/(vel.norm()*vel.norm()*vel.norm());
        }
        double getCurDot(double t) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];

            Eigen::Vector2d vel = Traj_container[idx].getdSigma(t);
            Eigen::Vector2d acc = Traj_container[idx].getddSigma(t);
            Eigen::Vector2d jerk = Traj_container[idx].getdddSigma(t);

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
            return kdot * eta;

        }
        double getPhi(double t, double wheelbase = 0.6){
            double c = getCur(t);
            return atan(c * wheelbase);
        }
        double getOmega(double t, double wheelbase = 0.6) const{
            int idx = locateTrajIdx(t);
            int eta = etas[idx];




            Eigen::Vector2d vel = Traj_container[idx].getdSigma(t);
            Eigen::Vector2d acc = Traj_container[idx].getddSigma(t);
            Eigen::Vector2d jerk = Traj_container[idx].getdddSigma(t);

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

            return omega * eta;

        }

        inline double getYaw(double t) const{
            int idx = locateTrajIdx(t);
            Eigen::Vector2d vel = Traj_container[idx].getdSigma(t);
            double yaw = atan2(etas[idx] * vel[1], etas[idx] * vel[0]);
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
        inline int getTotalPieceNum() const{
            int pieceNum = 0;
            for(const auto traj : Traj_container){
                pieceNum += traj.getPieceNum();
            }
            return pieceNum;
        }
        inline const Trajectory &operator[](int i) const
        {
            return Traj_container[i];
        }

        inline Trajectory &operator[](int i)
        {
            return Traj_container[i];
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

    class MinJerkOpt
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

        // for outside use
        /*polynomial descrips*/
        Eigen::MatrixXd gdC;
        double gdT;
        /*MINCO descrips*/
        Eigen::MatrixXd gdHead;
        Eigen::MatrixXd gdTail;
        Eigen::MatrixXd gdP;
        

    public:
        inline void reset(const int &pieceNum)
        {
            
            
            N = pieceNum;
            A.create(6 * N, 6, 6);
            b.resize(6 * N, 2);
            c.resize(6 * N, 2);
            adjScaledGrad.resize(6 * N, 2);
            gdC.resize(6 * N, 2); 
            gdP.resize(2, N - 1);

            gdHead.resize(2, 3);
            gdTail.resize(2, 3);

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
            c.block<6, 2>(6 * i, 0) =
                b.block<6, 2>(6 * i, 0).array().colwise() * tInv.array();
            }
            return;
        }
        inline Trajectory getTraj(int s) const
        {
            Trajectory traj;
            traj.reserve(N);
            for (int i = 0; i < N; i++)
            {
                traj.emplace_back(t(1), c.block<6, 2>(6 * i, 0).transpose().rowwise().reverse(), s);
            }

            return traj;
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
        

        inline void initSmGradCost() {
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
                gdC.block<3, 2>(6 * i, 0).setZero();
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
        
        inline void calGrads_PT() {
            for (int i = 0; i < N; i++) {
            adjScaledGrad.block<6, 2>(6 * i, 0) =
                gdC.block<6, 2>(6 * i, 0).array().colwise() * tInv.array();
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


        inline const Eigen::MatrixXd &getCoeffs(void) const {
            return c;
        }
        inline const double &getDt(void) const {
            return  t(1);
        }

        
        inline Eigen::MatrixXd &get_gdC()
        {
            return gdC;
        }
        inline double &get_gdT()
        {
            return gdT;
        }

        inline  Eigen::MatrixXd  get_gdHead(void)  {
            return  gdHead;
        }
        inline  Eigen::MatrixXd  get_gdTail(void)  {
            return  gdTail;
        }
        inline  Eigen::MatrixXd  get_gdP(void)  {
            return  gdP;
        }
    };

} //namespace plan_utils
