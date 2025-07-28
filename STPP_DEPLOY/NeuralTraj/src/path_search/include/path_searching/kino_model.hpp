#ifndef _KINOMODEL_H
#define _KINOMODEL_H
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <iostream>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <iostream>


#include <Eigen/Geometry>
#include <Eigen/StdVector>


namespace  path_searching{
  
    struct FlatTrajData
    {
      int singul;
      std::vector<Eigen::Vector3d> traj_pts;      // 3, N  x,y dur
      std::vector<double> thetas;
      Eigen::MatrixXd start_state;   // start flat state (2, 3)
      Eigen::MatrixXd final_state;   // end flat state (2, 3)
      double duration;

    };
    typedef std::vector<FlatTrajData> KinoTrajData;


}
#endif