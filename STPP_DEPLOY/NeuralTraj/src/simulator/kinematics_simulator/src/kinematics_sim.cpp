#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include "kinematics_simulator/ControlCmd.h"

#define ACKERMANN 	   2

using namespace std;

// ros interface
ros::Subscriber command_sub;
ros::Publisher  odom_pub;
ros::Publisher  mesh_pub;
ros::Timer simulate_timer;
ros::Time get_cmdtime;
visualization_msgs::Marker marker;

// simulator variables
std::default_random_engine generator;
std::normal_distribution<double> distribution{0.0, 1.0};
// std::uniform_real_distribution<double> real_uniform{-1.0, 1.0};
std::uniform_real_distribution<double> real_uniform{-0.5, 1.5};
kinematics_simulator::ControlCmd immediate_cmd;
vector<kinematics_simulator::ControlCmd> cmd_buff;
double x = 0.0;
double y = 0.0;
double v = 0.0;
double steer = 0.0;
double yaw = 0.0;
bool rcv_cmd = false;
double vx = 0.0;
double vy = 0.0;
double w = 0.0;

// simulator parameters
double init_x = 0.0;
double init_y = 0.0;
double init_v = 0.0;
double init_steer = 0.0;
double init_yaw = 0.0;
double time_resolution = 0.01;
double max_longitude_acc = 15.0;
double max_longitude_vel = 1.5;
double max_steer = 0.7;
double max_steer_vel = 7.0;
double time_delay = 0.0;
double wheel_base = 0.6;
double noise_percent = 0.0;
// double noise_percent = 0.1;
double noise_std_longitude_speed = 0.1;
double noise_std_steer = 0.1;
Eigen::Quaterniond q_mesh;
Eigen::Vector3d pos_mesh;

// utils
void normYaw(double& th)
{
	while (th > M_PI)
		th -= M_PI * 2;
	while (th < -M_PI)
		th += M_PI * 2;
}

Eigen::Vector2d guassRandom2d(double std)
{
	return std * Eigen::Vector2d(distribution(generator), distribution(generator));
}

Eigen::Vector3d guassRandom3d(double std)
{
	return std * Eigen::Vector3d(distribution(generator), distribution(generator), distribution(generator));
}

double guassRandom(double std)
{	
	return std * distribution(generator);
}

double uniformRandom(double limit)
{	
	return limit * real_uniform(generator);
	return limit;
}

// callBackes
void rcvCmdCallBack(const kinematics_simulator::ControlCmdConstPtr cmd)
{	
	if (rcv_cmd==false)
	{
		rcv_cmd = true;
		cmd_buff.push_back(*cmd);
		get_cmdtime = ros::Time::now();
	}
	else
	{
		cmd_buff.push_back(*cmd);
		if ((ros::Time::now() - get_cmdtime).toSec() > time_delay)
		{
			immediate_cmd = cmd_buff[0];
			cmd_buff.erase(cmd_buff.begin());
		}
	}
}

void simCallback(const ros::TimerEvent &e)
{
	nav_msgs::Odometry new_odom;

	new_odom.header.stamp    = ros::Time::now();
	new_odom.header.frame_id = "world";
	
	vx = v * cos(yaw);
	vy = v * sin(yaw);
	w = v * tan(steer) / wheel_base;

	vx += uniformRandom(noise_percent*fabs(vx));
	vy += uniformRandom(noise_percent*fabs(vy));
	w += uniformRandom(noise_percent*fabs(w));

	x = x + vx * time_resolution;
	y = y + vy * time_resolution;
	yaw = yaw + w * time_resolution;
	
	immediate_cmd.steer_vel += uniformRandom(noise_percent*fabs(immediate_cmd.steer_vel));
	immediate_cmd.longitude_acc += uniformRandom(noise_percent*fabs(immediate_cmd.longitude_acc));

	steer = steer + max(min(immediate_cmd.steer_vel, max_steer_vel), -max_steer_vel)  * time_resolution;
	steer = max(min(steer, max_steer), -max_steer);
	v = v + max(min(immediate_cmd.longitude_acc, max_longitude_acc), -max_longitude_acc) * time_resolution;
	v = max(min(v, max_longitude_vel), -max_longitude_vel);

	normYaw(yaw);

	new_odom.pose.pose.position.x  = x;
	new_odom.pose.pose.position.y  = y;
	new_odom.pose.pose.position.z  = 0;
	new_odom.pose.pose.orientation.w  = cos(yaw/2.0);
	new_odom.pose.pose.orientation.x  = 0;
	new_odom.pose.pose.orientation.y  = 0;
	new_odom.pose.pose.orientation.z  = sin(yaw/2.0);
	new_odom.twist.twist.linear.x  = vx;
	new_odom.twist.twist.linear.y  = vy;
	new_odom.twist.twist.linear.z  = 0;
	new_odom.twist.twist.angular.x = v;
	new_odom.twist.twist.angular.y = steer;
	new_odom.twist.twist.angular.z = w;

	Eigen::Quaterniond qyaw(cos(yaw/2.0), 0.0, 0.0, sin(yaw/2.0));
	Eigen::Quaterniond q = (qyaw * q_mesh).normalized();
	Eigen::Matrix3d R(qyaw);
	Eigen::Vector3d dp = R*pos_mesh;
	marker.pose.position.x = x - dp.x();
	marker.pose.position.y = y - dp.y();
	marker.pose.position.z = 0.0;
	marker.pose.orientation.w = q.w();
	marker.pose.orientation.x = q.x();
	marker.pose.orientation.y = q.y();
	marker.pose.orientation.z = q.z();

	odom_pub.publish(new_odom);
	mesh_pub.publish(marker);
}

// main loop
int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "simulator_node");
    ros::NodeHandle nh("~");

	nh.getParam("simulator/init_x", init_x);
	nh.getParam("simulator/init_y", init_y);
	nh.getParam("simulator/init_yaw", init_yaw);
	nh.getParam("simulator/time_resolution", time_resolution);
	nh.getParam("simulator/max_longitude_vel", max_longitude_vel);
	nh.getParam("simulator/max_longitude_acc", max_longitude_acc);
	nh.getParam("simulator/max_steer", max_steer);
	nh.getParam("simulator/max_steer_vel", max_steer_vel);
	nh.getParam("simulator/time_delay", time_delay);
	nh.getParam("simulator/wheel_base", wheel_base);
	nh.getParam("simulator/noise_percent", noise_percent);
	nh.getParam("simulator/noise_std_longitude_speed", noise_std_longitude_speed);
	nh.getParam("simulator/noise_std_steer", noise_std_steer);
	
	noise_percent /= 100.0;
    command_sub  = nh.subscribe("cmd", 1000, rcvCmdCallBack);
    odom_pub  = nh.advertise<nav_msgs::Odometry>("odom", 10);
	mesh_pub = nh.advertise<visualization_msgs::Marker>("mesh", 10);

	immediate_cmd.longitude_acc = 0.0;
	immediate_cmd.steer_vel = 0.0;
	
	marker.header.frame_id = "world";
	marker.id = 0;
	marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = x = init_x;
	marker.pose.position.y = y = init_y;
	marker.pose.position.z = 0.0;
	marker.pose.orientation.w = 0.5;
	marker.pose.orientation.x = 0.5;
	marker.pose.orientation.y = 0.5;
	marker.pose.orientation.z = 0.5;
	marker.color.a = 1.0;
	marker.color.r = 0.5;
	marker.color.g = 0.5;
	marker.color.b = 0.5;
	
	marker.scale.x = 0.00025;
	marker.scale.y = 0.00025;
	marker.scale.z = 0.00025;
	q_mesh = Eigen::Quaterniond(1.0/sqrt(2), 0.0, 0.0, 1.0/sqrt(2));
	pos_mesh = Eigen::Vector3d(-0.75, 0.35, 0.0);
	marker.mesh_resource = "package://kinematics_simulator/meshes/ackermann_model.STL";

    simulate_timer = nh.createTimer(ros::Duration(time_resolution), simCallback);

	ros::spin();

    return 0;
}