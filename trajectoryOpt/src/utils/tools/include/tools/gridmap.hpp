#ifndef GRIDMAP_H
#define GRIDMAP_H


#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <map>
#include <tools/config.hpp>
#include <decomp_basis/data_type.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
namespace map_util{
    using Tmap = std::vector<signed char>;
    template <int Dim> class MapUtil {
    public:
      ///Simple constructor
      MapUtil() {}
      ///Get map data
      Tmap getMap() { return map_; }
      //
      bool has_map_(){return has_map;}
      ///Get resolution
      double getRes() { return res_; }
      ///Get dimensions
      Veci<Dim> getDim() { return dim_; }
      ///Get origin
      Vecf<Dim> getOrigin() { return origin_d_; }
      Vecf<Dim> getMapSize() { return map_size; }
      void setParam(ConfigPtr config_, ros::NodeHandle& nh){
        res_ = config_->mapRes;
        int buffer_size;
        expand_size = config_->expandSize;
        priv_config_ = config_;
        if(Dim == 3){
            map_size(0) = config_->mapX;
            map_size(1) = config_->mapY;
            map_size(2) = config_->mapZ;
            origin_d_[0] = -map_size(0)/2;
            origin_d_[1] = -map_size(1)/2;
            origin_d_[2] = 0;
            dim_(0) = map_size(0)/res_;
            dim_(1) = map_size(1)/res_;
            dim_(2) = map_size(2)/res_;
            buffer_size = dim_(0)*dim_(1)*dim_(2);
        }
        else if(Dim==2){
            map_size(0) = config_->mapX + 1.0e-4;
            map_size(1) = config_->mapY + 1.0e-4;
            origin_d_[0] = -map_size(0)/2;
            origin_d_[1] = -map_size(1)/2;
            dim_(0) = map_size(0)/res_;
            dim_(1) = map_size(1)/res_;
            buffer_size = dim_(0)*dim_(1);

            /*ESDF*/
            distance_buffer_ = std::vector<double>(buffer_size, 10000);
            distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);
            distance_buffer_all_ = std::vector<double>(buffer_size, 10000);
            tmp_buffer1_ = std::vector<double>(buffer_size, 0);
            occupancy_buffer_neg = std::vector<char>(buffer_size, 0);
            esdf_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/global_esdf", 10);
            pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/global_pcl", 10);

        }
        else{
            ROS_ERROR("grid map dimensions must be 2 or 3!");
        }
        map_.resize(buffer_size,val_free);
        point_cloud_sub_ = nh.subscribe("/global_map", 10, &MapUtil::MapBuild, this);
      }
      void setEnv(const sensor_msgs::PointCloud2 & pointcloud_map){
        //only used in 2D environment
        int buffer_size;
        buffer_size = dim_(0)*dim_(1);
        distance_buffer_ = std::vector<double>(buffer_size, 10000);
        distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);
        distance_buffer_all_ = std::vector<double>(buffer_size, 10000);
        tmp_buffer1_ = std::vector<double>(buffer_size, 0);
        occupancy_buffer_neg = std::vector<char>(buffer_size, 0);
        std::fill(map_.begin(), map_.end(), val_free);
    


        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(pointcloud_map, cloud);
        if( (int)cloud.points.size() == 0 ) return;
        pcl::PointXYZ pt;
        for (int idx = 0; idx < (int)cloud.points.size(); idx++)
        {
          pt = cloud.points[idx];
          setObs(pt);
        }
        has_map = true;
        // std::cout << "!map build\n";
        updateESDF2d();
        publishESDF();
        publishPCL();
      }
      void MapBuild(const sensor_msgs::PointCloud2 & pointcloud_map){
        if(has_map) {
          publishESDF();
          return;}
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(pointcloud_map, cloud);
        if( (int)cloud.points.size() == 0 ) return;
        pcl::PointXYZ pt;
        // cloud.points.clear();//data generaion

        for (int idx = 0; idx < (int)cloud.points.size(); idx++)
        {
          pt = cloud.points[idx];
          setObs(pt);
        }
        has_map = true;
        updateESDF2d();
        publishESDF();
      }
      void setObs(pcl::PointXYZ pt){
        if(Dim==3){
            double coord_x = pt.x;
            double coord_y = pt.y;
            double coord_z = pt.z;
            Vec3f pt = Vec3f(coord_x,coord_y,coord_z);
            Veci<Dim> index3i;
            for(int i = 0; i < Dim; i++)
              index3i(i) = std::round((pt(i) - origin_d_(i)) / res_ - 0.5);
            if(isOutside(index3i))
                return;
            for (int i=-expand_size;i<=expand_size;i++)
            for (int j=-expand_size;j<=expand_size;j++)
            for (int k=-expand_size;k<=expand_size;k++)
                {
                    Veci<Dim> temp_index;
                    temp_index(0) = index3i(0)+i;
                    temp_index(1) = index3i(1)+j;
                    temp_index(2) = index3i(2)+k;
                    if(isOutside(temp_index)) continue;
                    map_[getIndex(temp_index)] = val_occ;
                }
        }
        else{
            double coord_x = pt.x;
            double coord_y = pt.y;
            Vec2f pt = Vec2f(coord_x,coord_y);
            Veci<Dim> index2i;
            for(int i = 0; i < Dim; i++)
              index2i(i) = std::round((pt(i) - origin_d_(i)) / res_ - 0.5);
            if(isOutside(index2i))
                return;
            for (int i=-expand_size;i<=expand_size;i++)
            for (int j=-expand_size;j<=expand_size;j++)
                {
                    Veci<Dim> temp_index;
                    temp_index(0) = index2i(0)+i;
                    temp_index(1) = index2i(1)+j;
                    if(isOutside(temp_index)) continue;
                    map_[getIndex(temp_index)] = val_occ;
                }

        }
        return;

      }
      ///Get index of a cell
      int getIndex(const Veci<Dim>& pn) {
          return Dim == 2 ? pn(0) + dim_(0) * pn(1) :
                            pn(0) + dim_(0) * pn(1) + dim_(0) * dim_(1) * pn(2);
      }
      ///Check if the given cell is outside of the map in i-the dimension
      bool isOutsideXYZ(const Veci<Dim> &n, int i) { return n(i) < 0 || n(i) >= dim_(i); }
      ///Check if the cell is free by index
      bool isFree(int idx) { return map_[idx] == val_free; }
      ///Check if the cell is unknown by index
      bool isUnknown(int idx) { return map_[idx] == val_unknown; }
      ///Check if the cell is occupied by index
      bool isOccupied(int idx) { return map_[idx] > val_free; }
      //free 0 occ 100 unknow -1
      ///Check if the cell is outside by coordinate
      bool isOutside(const Veci<Dim> &pn) {
        for(int i = 0; i < Dim; i++)
          if (pn(i) < 0 || pn(i) >= dim_(i))
            return true;
        return false;
      }
      ///Check if the given cell is free by coordinate
      bool isFree(const Veci<Dim> &pn) {
        if (isOutside(pn))
          return false;
        else
          return isFree(getIndex(pn));
      }
      ///Check if the given cell is occupied by coordinate
      bool isOccupied(const Veci<Dim> &pn) {
        if (isOutside(pn))
          return true;
        else
          return isOccupied(getIndex(pn));
      }
      bool isOccupied(const Vecf<Dim> &pt) {
        Veci<Dim> pn = floatToInt(pt);
        if (isOutside(pn))
          return true;
        else
          return isOccupied(getIndex(pn));
      }
      
      
      ///Check if the given cell is unknown by coordinate
      bool isUnknown(const Veci<Dim> &pn) {
        if (isOutside(pn))
          return false;
        return map_[getIndex(pn)] == val_unknown;
      }

      /**
       * @brief Set map
       *
       * @param ori origin position
       * @param dim number of cells in each dimension
       * @param map array of cell values
       * @param res map resolution
       */
      void setMap(const Vecf<Dim>& ori, const Veci<Dim>& dim, const Tmap &map, double res) {
        map_ = map;
        dim_ = dim;
        origin_d_ = ori;
        res_ = res;
      }

      ///Print basic information about the util
      void info() {
        Vecf<Dim> range = dim_.template cast<double>() * res_;
        std::cout << "MapUtil Info ========================== " << std::endl;
        std::cout << "   res: [" << res_ << "]" << std::endl;
        std::cout << "   origin: [" << origin_d_.transpose() << "]" << std::endl;
        std::cout << "   range: [" << range.transpose() << "]" << std::endl;
        std::cout << "   dim: [" << dim_.transpose() << "]" << std::endl;
      };

      ///Float position to discrete cell coordinate
      Veci<Dim> floatToInt(const Vecf<Dim> &pt) {
        Veci<Dim> pn;
        for(int i = 0; i < Dim; i++)
          pn(i) = floor((pt(i) - origin_d_(i)) / res_);
        return pn;
      }
      ///Discrete cell coordinate to float position
      Vecf<Dim> intToFloat(const Veci<Dim> &pn) {
        //return pn.template cast<double>() * res_ + origin_d_;
        return (pn.template cast<double>() + Vecf<Dim>::Constant(0.5)) * res_ + origin_d_;
      }

      ///Raytrace from float point pt1 to pt2
      vec_Veci<Dim> rayTrace(const Vecf<Dim> &pt1, const Vecf<Dim> &pt2) {
        Vecf<Dim> diff = pt2 - pt1;
        double k = 0.8;
        int max_diff = (diff / res_).template lpNorm<Eigen::Infinity>() / k;
        double s = 1.0 / max_diff;
        Vecf<Dim> step = diff * s;

        vec_Veci<Dim> pns;
        Veci<Dim> prev_pn = Veci<Dim>::Constant(-1);
        for (int n = 1; n < max_diff; n++) {
          Vecf<Dim> pt = pt1 + step * n;
          Veci<Dim> new_pn = floatToInt(pt);
          if (isOutside(new_pn))
            break;
          if (new_pn != prev_pn)
            pns.push_back(new_pn);
          prev_pn = new_pn;
        }
        return pns;
      }

      ///Check if the ray from p1 to p2 is occluded
      bool isBlocked(const Vecf<Dim>& p1, const Vecf<Dim>& p2, int8_t val = 100) {
        vec_Veci<Dim> pns = rayTrace(p1, p2);
        for (const auto &pn : pns) {
          if (map_[getIndex(pn)] >= val)
            return true;
        }
        return false;
      }

      ///Get occupied voxels
      vec_Vecf<Dim> getCloud() {
        vec_Vecf<Dim> cloud;
        Veci<Dim> n;
        if(Dim == 3) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              for (n(2) = 0; n(2) < dim_(2); n(2)++) {
                if (isOccupied(getIndex(n)))
                  cloud.push_back(intToFloat(n));
              }
            }
          }
        }
        else if (Dim == 2) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              if (isOccupied(getIndex(n)))
                cloud.push_back(intToFloat(n));
            }
          }
        }

        return cloud;
      }
      void updateESDF2d() {
        if(Dim != 2){
          ROS_ERROR("Dimension mismatch!");
        }
        Eigen::Vector2i min_esdf = Eigen::Vector2i(0,0);
        Eigen::Vector2i max_esdf = Eigen::Vector2i(dim_(0) - 1, dim_(1) - 1);

        /* ========== compute positive DT ========== */

        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
          fillESDF(
              [&](int y) {
                return isOccupied(Vec2i(x, y))?
                    0 :
                    std::numeric_limits<double>::max();
              },
              [&](int y, double val) {tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],
              max_esdf[1], 1);
        }

        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },
                    [&](int x, double val) {
                      distance_buffer_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);
                      //  min(mp_.resolution_ * std::sqrt(val),
                      //      md_.distance_buffer_[toAddress(x, y, z)]);
                    },
                    min_esdf[0], max_esdf[0], 0);
        }

        /* ========== compute negative distance ========== */
        for (int x = min_esdf(0); x <= max_esdf(0); ++x)
          for (int y = min_esdf(1); y <= max_esdf(1); ++y)
          {
            int idx = getIndex(Vec2i(x, y));
            if (isFree(idx)) {
              occupancy_buffer_neg[idx] = 1;

            } else if (isOccupied(idx)) {
              occupancy_buffer_neg[idx] = 0;
            } else {
              ROS_ERROR("what?");
            }
          }

        ros::Time t1, t2;

        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
          fillESDF(
              [&](int y) {
                return occupancy_buffer_neg[getIndex(Vec2i(x, y))] == 1 ?
                    0 :
                    std::numeric_limits<double>::max();
              },
              [&](int y, double val) { tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],
              max_esdf[1], 1);
        }

        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },
                    [&](int x, double val) {
                      distance_buffer_neg_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);
                    },
                    min_esdf[0], max_esdf[0], 0);
        }

        /* ========== combine pos and neg DT ========== */
        for (int x = min_esdf(0); x <= max_esdf(0); ++x)
          for (int y = min_esdf(1); y <= max_esdf(1); ++y)
          {
            int idx = getIndex(Vec2i(x, y));
            distance_buffer_all_[idx] = distance_buffer_[idx];

            if (distance_buffer_neg_[idx] > 0.0){
              distance_buffer_all_[idx] += (-distance_buffer_neg_[idx] + res_);
              }
          }

      }
      template <typename F_get_val, typename F_set_val>
      void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int d) {
        int v[dim_(d)];
        double z[dim_(d) + 1];

        int k = start;
        v[start] = start;
        z[start] = -std::numeric_limits<double>::max();
        z[start + 1] = std::numeric_limits<double>::max();

        for (int q = start + 1; q <= end; q++) {
          k++;
          double s;

          do {
            k--;
            s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
          } while (s <= z[k]);

          k++;

          v[k] = q;
          z[k] = s;
          z[k + 1] = std::numeric_limits<double>::max();
        }

        k = start;

        for (int q = start; q <= end; q++) {
          while (z[k + 1] < q) k++;
          double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
          f_set_val(q, val);
        }
      }
      inline double getDistance(const Eigen::Vector2d& pos) {
        Eigen::Vector2i id;
        id = floatToInt(pos);
        boundIndex(id);
        return distance_buffer_all_[getIndex(id)];
      }
      inline double getDistance(const Eigen::Vector2i& id) {
      Eigen::Vector2i id1 = id;
      boundIndex(id1);
      return distance_buffer_all_[getIndex(id1)];
      }
      inline void boundIndex(Eigen::Vector2i& id) {
        Eigen::Vector2i id1;
        id1(0) = std::max(std::min(id(0), dim_(0) - 1), 0);
        id1(1) = std::max(std::min(id(1), dim_(1) - 1), 0);
        id = id1;
    }
    void publishPCL(){
      vec_Vecf<Dim> pcs = getCloud();
      pcl::PointCloud<pcl::PointXYZI> cloud;
      pcl::PointXYZI pt;
      for(const auto& pc : pcs){
        pt.x = pc[0];
        pt.y = pc[1];
        pt.z = 0.0;
        cloud.push_back(pt);
      }
      cloud.width = cloud.points.size();
      cloud.height = 1;
      cloud.is_dense = true;
      cloud.header.frame_id = "world";
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(cloud, cloud_msg);
      pcl_pub_.publish(cloud_msg);

    }
    void publishESDF() {
      double dist;
      pcl::PointCloud<pcl::PointXYZI> cloud;
      pcl::PointXYZI pt;
      const double min_dist = -4.0;
      const double max_dist = 4.0;
      Eigen::Vector2i min_cut = Eigen::Vector2i(0,0);
      Eigen::Vector2i max_cut = Eigen::Vector2i(dim_(0)-1,dim_(1)-1);

      for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y) {
          Vec2f pos;
          pos = intToFloat(Vec2i(x, y));
          dist = getDistance(pos);
          // if(dist<min_dist) dist = min_dist;
          // if(dist>max_dist) dist = max_dist;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = dist;
          pt.intensity = (dist - min_dist) / (max_dist - min_dist);
          cloud.push_back(pt);
          // }
        }
      cloud.width = cloud.points.size();
      cloud.height = 1;
      cloud.is_dense = true;
      cloud.header.frame_id = "world";
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(cloud, cloud_msg);
      esdf_pub_.publish(cloud_msg);
    }
      inline double getDistGrad(Eigen::Vector2d po, Eigen::Vector2d& grad) {
        Eigen::Vector2d pos(po(0),po(1));
        if (isOutside(floatToInt(pos))) {
          grad.setZero();
          return 0;
        }

        /* use bilinear interpolation */
        Eigen::Vector2d pos_m = pos - 0.5 * res_ * Eigen::Vector2d::Ones();

        Eigen::Vector2i idx;
        idx = floatToInt(pos_m);
        Eigen::Vector2d idx_pos, diff;
        idx_pos = intToFloat(idx);

        diff = (pos - idx_pos) / res_;

        double values[2][2];
        for (int x = 0; x < 2; x++) {
          for (int y = 0; y < 2; y++) {
              Eigen::Vector2i current_idx = idx + Eigen::Vector2i(x, y);
              values[x][y] = getDistance(current_idx);
          }   
        }
        double v00 = (1-diff(0)) * values[0][0] + diff(0) * values[1][0];
        double v10 = (1-diff(0)) * values[0][1] + diff(0) * values[1][1];
        double v0  = (1-diff(1)) * v00 + diff(1) * v10;
        // double dist = v0;
        grad[1] = (v10 - v00) / res_;
        grad[0] = ((1-diff(1)) * (values[1][0]-values[0][0]) + diff(1) * (values[1][1]-values[0][1])) / res_;

        return v0;
      }
      
      ///Get free voxels
      vec_Vecf<Dim> getFreeCloud() {
        vec_Vecf<Dim> cloud;
        Veci<Dim> n;
        if(Dim == 3) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              for (n(2) = 0; n(2) < dim_(2); n(2)++) {
                if (isFree(getIndex(n)))
                  cloud.push_back(intToFloat(n));
              }
            }
          }
        }
        else if (Dim == 2) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              if (isFree(getIndex(n)))
                cloud.push_back(intToFloat(n));
            }
          }
        }

        return cloud;
      }

      ///Get unknown voxels
      vec_Vecf<Dim> getUnknownCloud() {
        vec_Vecf<Dim> cloud;
        Veci<Dim> n;
        if(Dim == 3) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              for (n(2) = 0; n(2) < dim_(2); n(2)++) {
                if (isUnknown(getIndex(n)))
                  cloud.push_back(intToFloat(n));
              }
            }
          }
        }
        else if (Dim == 2) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              if (isUnknown(getIndex(n)))
                cloud.push_back(intToFloat(n));
            }
          }
        }

        return cloud;
      }

      ///Dilate occupied cells
      void dilate(const vec_Veci<Dim>& dilate_neighbor) {
        Tmap map = map_;
        Veci<Dim> n = Veci<Dim>::Zero();
        if(Dim == 3) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              for (n(2) = 0; n(2) < dim_(2); n(2)++) {
                if (isOccupied(getIndex(n))) {
                  for (const auto &it : dilate_neighbor) {
                    if (!isOutside(n + it))
                      map[getIndex(n + it)] = val_occ;
                  }
                }
              }
            }
          }
        }
        else if(Dim == 2) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              if (isOccupied(getIndex(n))) {
                for (const auto &it : dilate_neighbor) {
                  if (!isOutside(n + it))
                    map[getIndex(n + it)] = val_occ;
                }
              }
            }
          }
        }

        map_ = map;
      }

      ///Free unknown voxels
      void freeUnknown() {
        Veci<Dim> n;
        if(Dim == 3) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              for (n(2) = 0; n(2) < dim_(2); n(2)++) {
                if (isUnknown(getIndex(n)))
                  map_[getIndex(n)] = val_free;
              }
            }
          }
        }
        else if (Dim == 2) {
          for (n(0) = 0; n(0) < dim_(0); n(0)++) {
            for (n(1) = 0; n(1) < dim_(1); n(1)++) {
              if (isUnknown(getIndex(n)))
                map_[getIndex(n)] = val_free;
            }
          }
        }
      }
      void CheckIfCollisionUsingPosAndYaw(const Eigen::Vector3d &state, bool *res, double flat = 0.0){
        if(Dim==3){
          ROS_ERROR("CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");
          return;
        }
        else{
          Eigen::Matrix2d rotM;
          double yaw = state[2];
          rotM << cos(yaw), -sin(yaw),
                  sin(yaw),  cos(yaw);
          Eigen::Vector2d p = state.head(2);
          for(int i = 0; i < priv_config_->conpts.size(); i++){
            // Eigen::Vector2d rep1 = priv_config_->conpts[i];
            // Eigen::Vector2d rep2 = priv_config_->conpts[(i+1)%priv_config_->conpts.size()];
            // if(rep1[0] > 0){
            //   rep1[0] += res_;
            // }
            // else{
            //   rep1[0] -= res_;
            // }
            // if(rep1[1] > 0){
            //   rep1[1] += res_;
            // }
            // else{
            //   rep1[1] -= res_;
            // }
            // if(rep2[0] > 0){
            //   rep2[0] += res_;
            // }
            // else{
            //   rep2[0] -= res_;
            // }
            // if(rep2[1] > 0){
            //   rep2[1] += res_;
            // }
            // else{
            //   rep2[1] -= res_;
            // }
            // Eigen::Vector2d pt1 = p + rotM * rep1;
            // Eigen::Vector2d pt2 = p + rotM * rep2;
            // if(isBlocked(pt1, pt2)){
            //   *res = true;
            //   return;
            // }
            
            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];
            Eigen::Vector2d grad;
            // if(flat > 0){
            // std::cout << "dis: "<<getDistGrad(pt, grad)<<"\n";
            // }
            if(getDistGrad(pt, grad)<priv_config_->safeMargin + flat){
              *res = true;
              return;
            }

          }
        }
        *res = false;
        return;
      }


      void CheckIfCollisionForBackEnd(const Eigen::Vector3d &state, bool *res){
        if(Dim==3){
          ROS_ERROR("CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");
          return;
        }
        else{
          Eigen::Matrix2d rotM;
          double yaw = state[2];
          rotM << cos(yaw), -sin(yaw),
                  sin(yaw),  cos(yaw);
          Eigen::Vector2d p = state.head(2);
          for(int i = 0; i < priv_config_->conpts.size(); i++){
            // Eigen::Vector2d rep1 = priv_config_->conpts[i];
            // Eigen::Vector2d rep2 = priv_config_->conpts[(i+1)%priv_config_->conpts.size()];
            // if(rep1[0] > 0){
            //   rep1[0] += res_;
            // }
            // else{
            //   rep1[0] -= res_;
            // }
            // if(rep1[1] > 0){
            //   rep1[1] += res_;
            // }
            // else{
            //   rep1[1] -= res_;
            // }
            // if(rep2[0] > 0){
            //   rep2[0] += res_;
            // }
            // else{
            //   rep2[0] -= res_;
            // }
            // if(rep2[1] > 0){
            //   rep2[1] += res_;
            // }
            // else{
            //   rep2[1] -= res_;
            // }
            // Eigen::Vector2d pt1 = p + rotM * rep1;
            // Eigen::Vector2d pt2 = p + rotM * rep2;
            // if(isBlocked(pt1, pt2)){
            //   *res = true;
            //   return;
            // }
            
            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];
            Eigen::Vector2d grad;
            if(getDistGrad(pt, grad)<priv_config_->safeMargin){
              *res = true;
              return;
            }

          }
        }
        *res = false;
        return;
      }
      ///Map entity
      Tmap map_;
      std::vector<double> distance_buffer_;
      std::vector<double> distance_buffer_neg_;
      std::vector<char> occupancy_buffer_neg;
      std::vector<double> distance_buffer_all_;
      std::vector<double> tmp_buffer1_;
      std::string world_frame_id;

    protected:
      ///Resolution
      double res_;
      ///Origin, float type
      Vecf<Dim> origin_d_;
      ///Dimension, int type
      Veci<Dim> dim_;
      Vecf<Dim> map_size;
      ///Assume occupied cell has value 100
      int8_t val_occ = 100;
      ///Assume free cell has value 0
      int8_t val_free = 0;
      ///Assume unknown cell has value -1
      int8_t val_unknown = -1;
      ///inflate size
      int  expand_size = 1;
      bool has_map = false;
      ros::Subscriber point_cloud_sub_;
      ConfigPtr priv_config_;
      ros::Publisher esdf_pub_;
      ros::Publisher pcl_pub_;
  };
  typedef MapUtil<2> OccMapUtil;
  typedef MapUtil<3> VoxelMapUtil;


}


#endif
