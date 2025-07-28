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
    // ROS_WARN("begin to generate map!");
    cors.clear();
    cloud_map.points.clear();
    mesh_msg.points.clear();



    pcl::PointXYZ pt_random;


    //add rectangle
    uniform_real_distribution<double> rand_ColorRow = uniform_real_distribution<double>(-1, 1);//hzchzc
    double iscol = rand_ColorRow(eng);
    // ROS_ERROR("1111111111111111111111111111111");
    // std::cout << "iscol = " << iscol << std::endl;
    
    uniform_int_distribution<int> gap = uniform_int_distribution<int>(1, 3);
    uniform_real_distribution<double> half = uniform_real_distribution<double>(0.6, 0.6);

    if(iscol>0){
        int last_gapnum = -1;
        for(double rx = -6.0; rx<=6.1; ){
            int gap_num = gap(eng);
            if(gap_num==last_gapnum||(last_gapnum+gap_num)%2==0){
                continue;
            }
            else{
                last_gapnum = gap_num;
            }
            double half_l = half(eng);
            double last_y = 10.0;
            for(int idx = gap_num; idx >= 1; idx--){
                double center = 1.0*idx*20.0 / (1.0*(gap_num+1))-10.0;
                cors.push_back(Eigen::Vector2d(rx, center));
                pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
                vector<Eigen::Vector2d> vs;
                Eigen::Vector2d p1(rx, last_y);
                vs.push_back(p1);
                Eigen::Vector2d p2(rx, center+half_l);
                vs.push_back(p2);
                Eigen::Vector2d p3(rx+0.2, center+half_l);//?
                vs.push_back(p3);
                Eigen::Vector2d p4(rx+0.2, last_y);
                vs.push_back(p4);
                cloud_polygon = fillConvexPolygon(vs);
                for (size_t i=0; i<cloud_polygon.points.size(); i++)
                {
                    pt_random.x = cloud_polygon.points[i].x;
                    pt_random.y = cloud_polygon.points[i].y;
                    pt_random.z = 0.0;
                    cloud_map.points.push_back(pt_random);
                }
                last_y = center-half_l;
            }
            {
                pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
                vector<Eigen::Vector2d> vs;
                Eigen::Vector2d p1(rx, last_y);
                vs.push_back(p1);
                Eigen::Vector2d p2(rx, -10.0);
                vs.push_back(p2);
                Eigen::Vector2d p3(rx+0.2, -10.0);//?
                vs.push_back(p3);
                Eigen::Vector2d p4(rx+0.2, last_y);
                vs.push_back(p4);
                cloud_polygon = fillConvexPolygon(vs);
                for (size_t i=0; i<cloud_polygon.points.size(); i++)
                {
                    pt_random.x = cloud_polygon.points[i].x;
                    pt_random.y = cloud_polygon.points[i].y;
                    pt_random.z = 0.0;
                    cloud_map.points.push_back(pt_random);
                }
            }
            rx+=3.0; 
        }
    }
    else
    {   
        int last_gapnum = -1;
        for(double ry = -6.0; ry<=6.1;){
            int gap_num = gap(eng);
            if(gap_num==last_gapnum||(last_gapnum+gap_num)%2==0){
                continue;
            }
            else{
                last_gapnum = gap_num;
            }
            double half_l = half(eng);
            double last_x = -10.0;
            for(int idx = 1; idx <= gap_num; idx++){
                double center = 1.0*idx*20.0 / (1.0*(gap_num+1))-10.0;
                cors.push_back(Eigen::Vector2d(center, ry));

                pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
                vector<Eigen::Vector2d> vs;
                Eigen::Vector2d p1(last_x, ry+0.2);
                vs.push_back(p1);
                Eigen::Vector2d p2(last_x, ry);
                vs.push_back(p2);
                Eigen::Vector2d p3(center-half_l, ry);
                vs.push_back(p3);
                Eigen::Vector2d p4(center-half_l, ry+0.2);
                vs.push_back(p4);
                cloud_polygon = fillConvexPolygon(vs);
                for (size_t i=0; i<cloud_polygon.points.size(); i++)
                {
                    pt_random.x = cloud_polygon.points[i].x;
                    pt_random.y = cloud_polygon.points[i].y;
                    pt_random.z = 0.0;
                    cloud_map.points.push_back(pt_random);
                }
                last_x = center+half_l;
            }   
                {
                    pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
                    vector<Eigen::Vector2d> vs;
                    Eigen::Vector2d p1(last_x, ry+0.2);
                    vs.push_back(p1);
                    Eigen::Vector2d p2(last_x, ry);
                    vs.push_back(p2);
                    Eigen::Vector2d p3(10.0, ry);
                    vs.push_back(p3);
                    Eigen::Vector2d p4(10.0, ry+0.2);
                    vs.push_back(p4);
                    cloud_polygon = fillConvexPolygon(vs);
                    for (size_t i=0; i<cloud_polygon.points.size(); i++)
                    {
                        pt_random.x = cloud_polygon.points[i].x;
                        pt_random.y = cloud_polygon.points[i].y;
                        pt_random.z = 0.0;
                        cloud_map.points.push_back(pt_random);
                        // cors.push_back(Eigen::Vector2d(pt_random.x, pt_random.y));
                    }
                }
                 ry+=3.0; 
        }
    }

    // rand_x = uniform_real_distribution<double>(-size_x / 2.0, size_x / 2.0);
    // rand_y = uniform_real_distribution<double>(-size_y / 2.0, size_y / 2.0);//hzchzc

    rand_x = uniform_real_distribution<double>(-size_x / 2.0, 7.5);
    rand_y = uniform_real_distribution<double>(-size_y / 2.0, 7.5);//hzchzc
    // generate polygon obs
    for (int k = 0; k<obs_num.size(); k++)
    {
        for (int i = 0; i < obs_num[k]; i++) 
        {
            double x, y;
            x = rand_x(eng);
            y = rand_y(eng);
            //-9.5 -9.5
            if (sqrt(pow(x+9.5, 2) + pow(y+9.5, 2)) < 4.0) 
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
            for (size_t p=0; p<cloud_polygon.second.points.size(); p++)
            {
                pt_random.x = std::max(std::min(cloud_polygon.second.points[p].x + x, 7.5),-size_x / 2.0 );
                pt_random.y = std::max(std::min(cloud_polygon.second.points[p].y + y, 7.5),-size_y / 2.0 );
                pt_random.z = 0.0;
                cloud_map.points.push_back(pt_random);
            }

            vector<Eigen::Vector2d> vector_polygon = cloud_polygon.first;
            geometry_msgs::Point init_p;
            init_p.x = vector_polygon[0].x() + x;
            init_p.y = vector_polygon[0].y() + y;
            init_p.z = 0.0;
            for (int m=1; m<k+2; m++)
            {
                mesh_msg.points.push_back(init_p);
                geometry_msgs::Point p;
                p.x = vector_polygon[m].x() + x;
                p.y = vector_polygon[m].y() + y;
                p.z = 0.0;
                mesh_msg.points.push_back(p);
                p.x = vector_polygon[m+1].x() + x;
                p.y = vector_polygon[m+1].y() + y;
                p.z = 0.0;
                mesh_msg.points.push_back(p);
            }
        }
    }


    {
        pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
        vector<Eigen::Vector2d> vs;
        Eigen::Vector2d p1(-10, 10);
        vs.push_back(p1);
        Eigen::Vector2d p2(-10, -10);
        vs.push_back(p2);
        Eigen::Vector2d p3(-9.9, -10);
        vs.push_back(p3);
        Eigen::Vector2d p4(-9.9, 10);
        vs.push_back(p4);
        cloud_polygon = fillConvexPolygon(vs);
        for (size_t i=0; i<cloud_polygon.points.size(); i++)
        {
            pt_random.x = cloud_polygon.points[i].x;
            pt_random.y = cloud_polygon.points[i].y;
            pt_random.z = 0.0;
            cloud_map.points.push_back(pt_random);
        }
    }
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
        vector<Eigen::Vector2d> vs;
        Eigen::Vector2d p1(-10, -9.9);
        vs.push_back(p1);
        Eigen::Vector2d p2(-10, -10);
        vs.push_back(p2);
        Eigen::Vector2d p3(10.0, -10);
        vs.push_back(p3);
        Eigen::Vector2d p4(10, -9.9);
        vs.push_back(p4);
        cloud_polygon = fillConvexPolygon(vs);
        for (size_t i=0; i<cloud_polygon.points.size(); i++)
        {
            pt_random.x = cloud_polygon.points[i].x;
            pt_random.y = cloud_polygon.points[i].y;
            pt_random.z = 0.0;
            cloud_map.points.push_back(pt_random);
        }
    }

    {
        pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
        vector<Eigen::Vector2d> vs;
        Eigen::Vector2d p1(9.9, 10);
        vs.push_back(p1);
        Eigen::Vector2d p2(9.9, -10);
        vs.push_back(p2);
        Eigen::Vector2d p3(10, -10);
        vs.push_back(p3);
        Eigen::Vector2d p4(10, 10);
        vs.push_back(p4);
        cloud_polygon = fillConvexPolygon(vs);
        for (size_t i=0; i<cloud_polygon.points.size(); i++)
        {
            pt_random.x = cloud_polygon.points[i].x;
            pt_random.y = cloud_polygon.points[i].y;
            pt_random.z = 0.0;
            cloud_map.points.push_back(pt_random);
        }
    }
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
        vector<Eigen::Vector2d> vs;
        Eigen::Vector2d p1(-10, 10);
        vs.push_back(p1);
        Eigen::Vector2d p2(-10, 9.9);
        vs.push_back(p2);
        Eigen::Vector2d p3(10, 9.9);
        vs.push_back(p3);
        Eigen::Vector2d p4(10, 10);
        vs.push_back(p4);
        cloud_polygon = fillConvexPolygon(vs);
        for (size_t i=0; i<cloud_polygon.points.size(); i++)
        {
            pt_random.x = cloud_polygon.points[i].x;
            pt_random.y = cloud_polygon.points[i].y;
            pt_random.z = 0.0;
            cloud_map.points.push_back(pt_random);
        }
    }

    // ROS_WARN("Finish  generating map!");

    //     vector<Eigen::Vector2d> vector_polygon = vs;
    //     geometry_msgs::Point init_p;
    //     init_p.x = vector_polygon[0].x();
    //     init_p.y = vector_polygon[0].y();
    //     init_p.z = 0.0;
    //     for (int i=1; i<3; i++)
    //     {
    //         mesh_msg.points.push_back(init_p);
    //         geometry_msgs::Point p;
    //         p.x = vector_polygon[i].x();
    //         p.y = vector_polygon[i].y();
    //         p.z = 0.0;
    //         mesh_msg.points.push_back(p);
    //         p.x = vector_polygon[i+1].x();
    //         p.y = vector_polygon[i+1].y();
    //         p.z = 0.0;
    //         mesh_msg.points.push_back(p);
    //     }

    // }
    // {
    //     pcl::PointCloud<pcl::PointXYZ> cloud_polygon;
    //     vector<Eigen::Vector2d> vs;
    //     Eigen::Vector2d p1(7, 20);
    //     vs.push_back(p1);
    //     Eigen::Vector2d p2(6, 20);
    //     vs.push_back(p2);
    //     Eigen::Vector2d p3(6, 4);
    //     vs.push_back(p3);
    //     Eigen::Vector2d p4(7, 4);
    //     vs.push_back(p4);
    //     cloud_polygon = fillConvexPolygon(vs);
    //     for (size_t i=0; i<cloud_polygon.points.size(); i++)
    //     {
    //         pt_random.x = cloud_polygon.points[i].x;
    //         pt_random.y = cloud_polygon.points[i].y;
    //         pt_random.z = 0.0;
    //         cloud_map.points.push_back(pt_random);
    //     }

    //     vector<Eigen::Vector2d> vector_polygon = vs;
    //     geometry_msgs::Point init_p;
    //     init_p.x = vector_polygon[0].x();
    //     init_p.y = vector_polygon[0].y();
    //     init_p.z = 0.0;
    //     for (int i=1; i<3; i++)
    //     {
    //         mesh_msg.points.push_back(init_p);
    //         geometry_msgs::Point p;
    //         p.x = vector_polygon[i].x();
    //         p.y = vector_polygon[i].y();
    //         p.z = 0.0;
    //         mesh_msg.points.push_back(p);
    //         p.x = vector_polygon[i+1].x();
    //         p.y = vector_polygon[i+1].y();
    //         p.z = 0.0;
    //         mesh_msg.points.push_back(p);
    //     }



    cloud_map.width = cloud_map.points.size();
    cloud_map.height = 1;
    cloud_map.is_dense = true;
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
    // {   
    //     for(int i = 0; i <= 30000; i++){
    //         double flatObcs[2800] = {0};
    //         std::ifstream fin(std::string("/home/hzc/2022Codes/MPNet/S2D/dataset/obs_cloud/obc")+std::to_string(i)+ std::string(".dat"), std::ios::binary);
    //         fin.read((char*)flatObcs, sizeof(double) * 2800);
    //         fin.close();
    //         cloud_map.points.clear();
    //         for(int j = 0; j < 2800; j+=2){
    //             pcl::PointXYZ pt;
    //             pt.x = flatObcs[j];
    //             pt.y = flatObcs[j+1];
    //             pt.z = 0.0;
    //             cloud_map.points.push_back(pt);
    //         }
    //         cloud_map.width = cloud_map.points.size();
    //         cloud_map.height = 1;
    //         cloud_map.is_dense = true;
    //         pcl::toROSMsg(cloud_map, global_msg);
    //         global_msg.header.frame_id = "world";
    //         gridmap.setEnv(global_msg);
    //         global_map_pub.publish(global_msg);
    //         int dimx = gridmap.getDim()[0];
    //         int dimy = gridmap.getDim()[1];
    //         std::cout << "dimx: " << dimx << "dimy: " << dimy <<std::endl;
    //         {
    //             cv::Mat image(dimx, dimy, CV_8UC1);
    //             for (int ix = 0; ix < dimx; ix++) {
    //                 for (int jy = 0; jy < dimy; jy++) {
    //                     if (gridmap.isOccupied(Vec2i(ix,jy))){
    //                         image.at<uchar>(ix,jy) = 1;
    //                     }
    //                     else{
    //                         image.at<uchar>(ix,jy) = 0;
    //                     }
    //                 }
    //             }
    //             cv::imwrite(std::string("/home/hzc/2022Codes/MPNet/S2D/dataset/obs_map/obc")+std::to_string(i)+string(".png"), image);
    //         }
    //         //save esdf image
    //         // {   
    //         //     double max_dis = 10.0;
    //         //     cv::Mat esdfImage(dimx, dimy, CV_8UC1);
    //         //     for (int ix = 0; ix < dimx; ix++) {
    //         //         for (int jy = 0; jy < dimy; jy++) {
    //         //             double dis = gridmap.getDistance(Eigen::Vector2i(ix,jy));
    //         //             dis = std::min(dis, max_dis);
    //         //             dis = std::max(0.0, dis);
    //         //             //map 0~max- 0~255
    //         //             esdfImage.at<uchar>(ix,jy) = std::round((dis/max_dis)*255);
    //         //         }
    //         //     }
    //         //     cv::imwrite(std::string("/home/hzc/2022Codes/MPNet/S2D/dataset/obs_esdf/obc")+std::to_string(i)+string(".png"), esdfImage);
    //         //     double dis = gridmap.getDistance(Eigen::Vector2i(0,0));
    //         //     dis = std::min(dis, max_dis);
    //         //     dis = std::max(0.0, dis);
    //         // }
    //     }
    // }
    /*image data
    for(int i = 40; i <= 70; i++){
        for(int j = 40; j <= 70; j++){
            for(int k = 0; k <= 30; k++){
                obs_num[0] = i;
                obs_num[1] = j;
                generateMap();
                
                gridmap.setEnv(global_msg);
                // global_map_pub.publish(global_msg);
                mesh_map_pub.publish(mesh_msg);
                vec_Vec2f vec_obs = gridmap.getCloud();
                std::cout << "points: " << vec_obs.size() << std::endl;
                // ros::Duration(1.0).sleep();
                //save 0-1 image
                int dimx = gridmap.getDim()[0];
                int dimy = gridmap.getDim()[1];
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
                    cv::imwrite(packPath + string("/data/occMap/occMap")+std::to_string(count)+string(".png"), image);

                }
                //save esdf image
                {
                    double offset = 20.0;
                    cv::Mat esdfImage(dimx, dimy, CV_64FC1);
                    for (int ix = 0; ix < dimx; ix++) {
                        for (int jy = 0; jy < dimy; jy++) {
                            esdfImage.at<double>(ix,jy) = gridmap.getDistance(Eigen::Vector2i(ix,jy)) + offset;
                        }
                    }
                    cv::imwrite(packPath + string("/data/esdfMap/esdfMap")+std::to_string(count)+string(".png"), esdfImage);
                    
                }
                count ++;
             }
        }
   } 
   */
    // sensor_timer = nh.createTimer(ros::Duration(1.0/sensor_rate), sensorCallback);
    // 10000
    int map_idx = 1;
    nh.getParam("map/map_id", map_idx);
    for(int k = 1; k <= 30000; k++){
    // for(int k = map_idx; k < map_idx+9999; k++){
        // std::cout <<"k: "<<k << std::endl;
        std::string prefix = "/home/hzc/learning_code/harddata/e"+std::to_string(k);
        std::ifstream fpath(prefix+std::string("/path50.dat"));
        if (fpath.good()){
            continue;
        }

        // std::cout << "map idx: "<<k<<std::endl;
        // uniform_int_distribution<unsigned> u1(8,23);
        // uniform_int_distribution<unsigned> u2(8,23);
        // uniform_int_distribution<unsigned> u3(8,23);
        uniform_int_distribution<unsigned> u1(0,0);
        uniform_int_distribution<unsigned> u2(10,25);
        uniform_int_distribution<unsigned> u3(10,25);
        obs_num[0] = u1(eng);
        obs_num[1] = u2(eng);
        obs_num[2] = u3(eng);
        generateMap();
        gridmap.setEnv(global_msg);
        mesh_map_pub.publish(mesh_msg);
        vec_Vec2f vec_obs = gridmap.getCloud();
        // std::cout << "points: " << vec_obs.size() << std::endl;
        // std::cout << "obsnum: "<<obs_num[0]<<" "<<obs_num[1]<<" "<<obs_num[2]<<"\n";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sample(new pcl::PointCloud<pcl::PointXYZ>);
        for(const auto pt:vec_obs){
            pcl::PointXYZ pxyz;
            pxyz.x = pt[0];
            pxyz.y = pt[1];
            pxyz.z = 0.0;
            cloud->push_back(pxyz);
        }
        // std::cout << "points: " << cloud->size() << std::endl;
        if(cloud->size()>9000){
            ROS_ERROR("11111111111111111");
        }
        if(cloud->size()==10000){
            cloud_sample = cloud;
        }
        else if(cloud->size()>10000) {
            pcl::RandomSample<pcl::PointXYZ> rs;
            rs.setInputCloud(cloud);
            rs.setSample(10000);
            rs.filter(*cloud_sample);
        }
        else{
            // < 10000 repeat
            int resL = 10000 - cloud->size();
            for(int resi = 0; resi< resL; resi++){
                cloud->push_back(cloud->at(0));
            }
            cloud_sample = cloud;
        }
        double writeObc[20000];
        for(int idx = 0; idx < 10000; idx++){
            pcl::PointXYZ pxyz = cloud_sample->at(idx);
            double px = pxyz.x;
            double py = pxyz.y;
            writeObc[2*idx] = px;
            writeObc[2*idx + 1] = py;
        }

        // std::ofstream fout(std::string("/home/hzc/learning_code/harddata/obcs/obc")+std::to_string(k)+std::string(".dat"), std::ios::binary);
        // fout.write((char*)writeObc, sizeof(double)*20000);
        // fout.close();
        
        int dimx = gridmap.getDim()[0];
        int dimy = gridmap.getDim()[1];

        {   
            cv::Mat esdfImage(dimx, dimy, CV_32FC1);
            for (int ix = 0; ix < dimx; ix++) {
                for (int jy = 0; jy < dimy; jy++) {
                    double dis = gridmap.getDistance(Eigen::Vector2i(ix,jy));
                    // Eigen::Vector2d gd;
                    // gridmap.getDistGrad(gridmap.intToFloat( Eigen::Vector2i(ix,jy)), gd);
                    // cv::Vec<double, 100> a;
                    esdfImage.at<float>(ix,jy)=dis;
                    // if(i==1 && ix ==63 && jy==130){
                    //     std::cout << "dis: "<<dis << " " << gd.transpose() << std::endl;
                    // }
                }
            }

            // cv::FileStorage fs(std::string("/home/hzc/learning_code/harddata/esdfmaps/")+std::to_string(k)+string(".xml"), cv::FileStorage::WRITE);
            // fs << "instance" << esdfImage;
            // fs.release();
        }
    }
    std::cout <<"finished" << std::endl;
	ros::spin();

    return 0;
}