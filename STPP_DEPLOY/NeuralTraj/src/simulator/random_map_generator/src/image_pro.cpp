#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/filters/random_sample.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tools/gridmap.hpp>
#include <tools/config.hpp>
#include <ros/package.h>

using namespace std;

// pcl
pcl::PointCloud<pcl::PointXYZ> cloud_map;
pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
pcl::PointXYZ sensor_pose;
vector<int> pointIdxRadiusSearch;
vector<float> pointRadiusSquaredDistance;
vector<Eigen::Vector2d> cors;
// random
random_device rd;
default_random_engine eng(rd());
uniform_real_distribution<double> rand_x;
uniform_real_distribution<double> rand_y;
uniform_real_distribution<double> rand_w;
uniform_real_distribution<double> rand_theta;
uniform_real_distribution<double> rand_radius;

// ros
ros::Publisher local_map_pub;
ros::Publisher global_map_pub;
ros::Publisher mesh_map_pub;
ros::Subscriber odom_sub;
ros::Timer sensor_timer;
sensor_msgs::PointCloud2 global_msg;
visualization_msgs::Marker mesh_msg;


// params
bool has_odom = false;
bool has_map = false;
vector<int> obs_num = {1, 1, 1};
double resolution = 0.1;
double size_x = 30.0;
double size_y = 30.0;
double min_width = 0.3;
double max_width = 0.8;
double sensor_rate = 10.0;
double sensor_range = 5.0;
double min_dis = 0.3;
// laser
constexpr int LINE_NUM = 256;
double laser_res = 2.0 * M_PI / LINE_NUM;
Eigen::VectorXi idx_map = Eigen::VectorXi::Constant(LINE_NUM, -1);
Eigen::VectorXd dis_map = Eigen::VectorXd::Constant(LINE_NUM, 9999.0);
bool checkDis(double x, double y){
    bool result = false;
    Eigen::Vector2d tmp(x, y);
    for(const auto pt : cors){
        if((tmp-pt).norm() < min_dis){
            result = true;
            return result;
        }
    }
    return result;
}
bool crossBoolean2(Eigen::Vector2d a, Eigen::Vector2d b)
{
    return (a(0)*b(1)-b(0)*a(1) > 0);
}

pcl::PointCloud<pcl::PointXYZ> fillConvexPolygon(vector<Eigen::Vector2d> poly_vs)
{
    pcl::PointCloud<pcl::PointXYZ> cloud_polygon;

    if (poly_vs.size() < 3)
        return cloud_polygon;
    
    double down = 9999.0;
    double up = -9999.0;
    double left = 9999.0;
    double right = -9999.0;
    
    // AABB box
    for (size_t i=0; i<poly_vs.size(); i++)
    {
        if (poly_vs[i][0] > right)
            right = poly_vs[i][0];
        if (poly_vs[i][0] < left)
            left = poly_vs[i][0];
        if (poly_vs[i][1] > up)
            up = poly_vs[i][1];
        if (poly_vs[i][1] < down)
            down = poly_vs[i][1];
    }

    for (double x=left; x<right+resolution; x+=resolution)
    {
        for (double y=down; y<up+resolution; y+=resolution)
        {
            bool in_poly = false;
            Eigen::Vector2d O(x, y);

            for (size_t i=0; i<poly_vs.size() - 2; i++)
            {
                // if a point is in triangle
                Eigen::Vector2d A = poly_vs[0];
                Eigen::Vector2d B = poly_vs[i+1];
                Eigen::Vector2d C = poly_vs[i+2];
                if (crossBoolean2(B-A, O-A) && \
                    crossBoolean2(C-B, O-B) && \
                    crossBoolean2(A-C, O-C) )
                {
                    in_poly = true;
                    break;
                }                
            }

            if (in_poly)
            {
                pcl::PointXYZ pt;
                pt.x = x;
                pt.y = y;
                pt.z = 0.0;
                cloud_polygon.push_back(pt);
            }
        }
    }
    
    return cloud_polygon;
}

pair<vector<Eigen::Vector2d>, pcl::PointCloud<pcl::PointXYZ>> generatePolygon(int K)
{
    pcl::PointCloud<pcl::PointXYZ> cloud_polygon;

    rand_w = uniform_real_distribution<double>(min_width, max_width);
    rand_theta = uniform_real_distribution<double>(-M_PI, M_PI);

    double radius = rand_w(eng);
    double theta = rand_theta(eng);
    double angle_res = 2.0 * M_PI / K;
    double small_r = radius * sin(angle_res/2.0);

    rand_radius = uniform_real_distribution<double>(-small_r, small_r);

    vector<Eigen::Vector2d> vs;
    for (int i=0; i<K; i++)
    {
        double a = angle_res * i + theta;
        double delta_theta = rand_theta(eng);
        double delta_radius = rand_radius(eng);
        Eigen::Vector2d p(cos(a)*radius + cos(a+delta_theta)*delta_radius, \
                          sin(a)*radius + sin(a+delta_theta)*delta_radius);
        vs.push_back(p);
    }
    cloud_polygon = fillConvexPolygon(vs);

    return std::make_pair(vs, cloud_polygon);
} 

void generateMap()
{
    pcl::PointXYZ pt_random;
    cors.clear();
    cloud_map.points.clear();
    mesh_msg.points.clear();
    rand_x = uniform_real_distribution<double>(-size_x / 2.0, size_x / 2.0);
    rand_y = uniform_real_distribution<double>(-size_y / 2.0, size_y / 2.0);

    // generate polygon obs
    for (int k = 0; k<obs_num.size(); k++)
    {
        for (int i = 0; i < obs_num[k]; i++) 
        {
            double x, y;
            x = rand_x(eng);
            y = rand_y(eng);

            if (sqrt(pow(x, 2) + pow(y, 2)) < 2.0) 
            {
                i--;
                continue;
            }

            x = floor(x / resolution) * resolution + resolution / 2.0;
            y = floor(y / resolution) * resolution + resolution / 2.0;
            if(checkDis(x, y)){
                i--;
                continue;
            }
            cors.push_back(Eigen::Vector2d(x, y));
            pair<vector<Eigen::Vector2d>, pcl::PointCloud<pcl::PointXYZ>> cloud_polygon = generatePolygon(k+3);
            for (size_t i=0; i<cloud_polygon.second.points.size(); i++)
            {
                pt_random.x = cloud_polygon.second.points[i].x + x;
                pt_random.y = cloud_polygon.second.points[i].y + y;
                pt_random.z = 0.0;
                cloud_map.points.push_back(pt_random);
            }

            vector<Eigen::Vector2d> vector_polygon = cloud_polygon.first;
            geometry_msgs::Point init_p;
            init_p.x = vector_polygon[0].x() + x;
            init_p.y = vector_polygon[0].y() + y;
            init_p.z = 1.0;
            for (int i=1; i<k+2; i++)
            {
                mesh_msg.points.push_back(init_p);
                geometry_msgs::Point p;
                p.x = vector_polygon[i].x() + x;
                p.y = vector_polygon[i].y() + y;
                p.z = 1.0;
                mesh_msg.points.push_back(p);
                p.x = vector_polygon[i+1].x() + x;
                p.y = vector_polygon[i+1].y() + y;
                p.z = 1.0;
                mesh_msg.points.push_back(p);
            }
        }
    }
    
    cloud_map.width = cloud_map.points.size();
    cloud_map.height = 1;
    cloud_map.is_dense = true;
    kd_tree.setInputCloud(cloud_map.makeShared());
    has_map = true;

    pcl::toROSMsg(cloud_map, global_msg);
    global_msg.header.frame_id = "world";

 	mesh_msg.id = 0;
 	mesh_msg.type = visualization_msgs::Marker::TRIANGLE_LIST;
 	mesh_msg.action = visualization_msgs::Marker::ADD;
 	mesh_msg.scale.x = 1.0;
 	mesh_msg.scale.y = 1.0;
 	mesh_msg.scale.z = 1.0;
 	mesh_msg.color.r = 0.2;
 	mesh_msg.color.g = 0.2;
 	mesh_msg.color.b = 0.2;
 	mesh_msg.color.a = 1.0;
    mesh_msg.header.frame_id = "world";
}

void rcvOdomCallBack(const nav_msgs::OdometryConstPtr msg)
{
    sensor_pose.x = msg->pose.pose.position.x;
    sensor_pose.y = msg->pose.pose.position.y;
    sensor_pose.z = 0.0;
    has_odom = true;
}

void sensorCallback(const ros::TimerEvent &e)
{
    if (!has_map || !has_odom)
        return;
    static int count = 0;
    count++;
    
    global_map_pub.publish(global_msg);
    mesh_map_pub.publish(mesh_msg);

    pcl::PointCloud<pcl::PointXYZ> local_map;

    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();
    idx_map.setConstant(-1);
    dis_map.setConstant(9999.0);

    pcl::PointXYZ pt;
    if (kd_tree.radiusSearch(sensor_pose, sensor_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) 
    {
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++) 
        {
            pt = cloud_map.points[pointIdxRadiusSearch[i]];
            int idx = floor((atan2(pt.y - sensor_pose.y, pt.x - sensor_pose.x) + M_PI + laser_res / 2.0) / laser_res);
            if (idx >= 0 && idx < LINE_NUM && dis_map[idx] > pointRadiusSquaredDistance[i])
            {
                idx_map[idx] = idx;
                dis_map[idx] = pointRadiusSquaredDistance[i];
            }
        }

        for (int i=0; i<LINE_NUM; i++)
        {
            if (idx_map[i] != -1)
            {
                double angle = idx_map[i] * laser_res - M_PI;
                double dist = sqrt(dis_map[i]);
                pt.x = dist*cos(angle) + sensor_pose.x;
                pt.y = dist*sin(angle) + sensor_pose.y;
                pt.z = 0.0;
                local_map.push_back(pt);
            }
        }
    }    

    local_map.width = local_map.points.size();
    local_map.height = 1;
    local_map.is_dense = true;

    sensor_msgs::PointCloud2 local_msg;
    pcl::toROSMsg(local_map, local_msg);
    local_msg.header.frame_id = "world";
    local_map_pub.publish(local_msg);
}
// void main_loop(){
//    for(int i = 40; i <= 70; i++){
//         for(int j = 40; j <= 70; j++){
//             for(int k = 0; k <= 30; k++){
//                 obs_num[0] = i;
//                 obs_num[1] = j;
//                 generateMap();
//                 gridmap.setEnv(global_msg);
//                 // global_map_pub.publish(global_msg);
//                 mesh_map_pub.publish(mesh_msg);
//                 ros::Duration(1.0).sleep();
//             }
//         }
//    } 
// }
int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "random_map_node");
    ros::NodeHandle nh("~");

    nh.param<std::vector<int>>("map/obs_num", obs_num, std::vector<int>());
	nh.getParam("map/resolution", resolution);
	nh.getParam("map/size_x", size_x);
	nh.getParam("map/size_y", size_y);
	nh.getParam("map/min_width", min_width);
	nh.getParam("map/max_width", max_width);
	nh.getParam("map/sensor_rate", sensor_rate);
	nh.getParam("map/sensor_range", sensor_range);
    nh.getParam("map/min_dis", min_dis);
	
    ConfigPtr config_;
    config_.reset(new Config(nh));


    odom_sub  = nh.subscribe("odom", 1000, rcvOdomCallBack);
    local_map_pub = nh.advertise<sensor_msgs::PointCloud2>("local_cloud", 1);
    global_map_pub = nh.advertise<sensor_msgs::PointCloud2>("global_cloud", 1);
    mesh_map_pub = nh.advertise<visualization_msgs::Marker>("mesh_obstacles", 1);
    
    map_util::OccMapUtil gridmap;
    gridmap.setParam(config_, nh);
    int count = 0;
    std::string packPath = ros::package::getPath("random_map_generator");
    std::cout << packPath << std::endl;
    /*read obcs from files*/
    {   
        for(int i = 882; i <= 30000; i++){
            double flatObcs[7200] = {0};
            std::ifstream fin(std::string("/home/han/2023codes/data2/obcs/obc")+std::to_string(i)+ std::string(".dat"), std::ios::binary);
            fin.read((char*)flatObcs, sizeof(double) * 7200);
            fin.close();
            cloud_map.points.clear();
            for(int j = 0; j < 7200; j+=2){
                pcl::PointXYZ pt;
                pt.x = flatObcs[j];
                pt.y = flatObcs[j+1];
                pt.z = 0.0;
                cloud_map.points.push_back(pt);
            }
            cloud_map.width = cloud_map.points.size();
            cloud_map.height = 1;
            cloud_map.is_dense = true;
            pcl::toROSMsg(cloud_map, global_msg);
            global_msg.header.frame_id = "world";
            gridmap.setEnv(global_msg);
            global_map_pub.publish(global_msg);
            int dimx = gridmap.getDim()[0];
            int dimy = gridmap.getDim()[1];
            std::cout << "dimx: " << dimx << "dimy: " << dimy <<std::endl;
            {
                cv::Mat image(dimx, dimy, CV_8UC1);
                for (int ix = 0; ix < dimx; ix++) {
                    for (int jy = 0; jy < dimy; jy++) {
                        if (gridmap.isOccupied(Vec2i(ix,jy))){
                            image.at<uchar>(ix,jy) = 1;
                        }
                        else{
                            image.at<uchar>(ix,jy) = 0;
                        }
                    }
                }
                cv::imwrite(std::string("/home/han/2023codes/data2/gridmaps/")+std::to_string(i)+string(".png"), image);
            }
            //save esdf image
            {   
                double max_dis = 10.0;
                cv::Mat esdfImage(dimx, dimy, CV_64FC3);
                for (int ix = 0; ix < dimx; ix++) {
                    for (int jy = 0; jy < dimy; jy++) {
                        double dis = gridmap.getDistance(Eigen::Vector2i(ix,jy));
                        Eigen::Vector2d gd;
                        gridmap.getDistGrad(gridmap.intToFloat( Eigen::Vector2i(ix,jy)), gd);
                        // cv::Vec<double, 100> a;
                        esdfImage.at<cv::Vec3d>(ix,jy)=cv::Vec3d(dis, gd[0], gd[1]);
                        // if(i==1 && ix ==63 && jy==130){
                        //     std::cout << "dis: "<<dis << " " << gd.transpose() << std::endl;
                        // }
                    }
                }
                cv::FileStorage fs(std::string("/home/han/2023codes/data2/esdfmaps/")+std::to_string(i)+string(".xml"), cv::FileStorage::WRITE);
                fs << "instance" << esdfImage;
                fs.release();
            }
        }
    }
    
	ros::spin();

    return 0;
}