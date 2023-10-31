

//c++ lib
#include <cmath>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>

//ros lib
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

//pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//local lib
#include "lidar.h"
#include "odomEstimationClass.h"

Odom_ES_EstimationClass odom_ES_Estimation;
std::mutex mutex_lock;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudEdgeBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudSurfBuf;
lidar::Lidar lidar_param;
std::string sensorFrameId;
double weightType_para;

ros::Publisher pubLaserOdometry;
ros::Publisher pubmapSurfPoints;
ros::Publisher pubmapEdgePoints;
void velodyneSurfHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    mutex_lock.lock();
    pointCloudSurfBuf.push(laserCloudMsg);
    mutex_lock.unlock();
}
void velodyneEdgeHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    mutex_lock.lock();
    pointCloudEdgeBuf.push(laserCloudMsg);
    mutex_lock.unlock();
}

bool is_odom_inited = false;
double total_time =0;
int total_frame=0;
void odom_edge_surf_estimation(){
    while(1){
        if(!pointCloudEdgeBuf.empty() && !pointCloudSurfBuf.empty()){

            //read data
            mutex_lock.lock();
            if(!pointCloudSurfBuf.empty() && (pointCloudSurfBuf.front()->header.stamp.toSec()<pointCloudEdgeBuf.front()->header.stamp.toSec()-0.5*lidar_param.scan_period)){
                pointCloudSurfBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;  
            }

            if(!pointCloudEdgeBuf.empty() && (pointCloudEdgeBuf.front()->header.stamp.toSec()<pointCloudSurfBuf.front()->header.stamp.toSec()-0.5*lidar_param.scan_period)){
                pointCloudEdgeBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;  
            }
            //if time aligned 
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_tmp(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_edge_in(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_surf_in(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::fromROSMsg(*pointCloudEdgeBuf.front(), *pointcloud_tmp);
            pcl::copyPointCloud(*pointcloud_tmp, *pointcloud_edge_in);
            pcl::fromROSMsg(*pointCloudSurfBuf.front(), *pointcloud_tmp);
            pcl::copyPointCloud(*pointcloud_tmp, *pointcloud_surf_in);

            ros::Time pointcloud_time = (pointCloudSurfBuf.front())->header.stamp;
            pointCloudEdgeBuf.pop();
            pointCloudSurfBuf.pop();
            mutex_lock.unlock();

            if(is_odom_inited == false){
                odom_ES_Estimation.initMapWithPoints(pointcloud_edge_in, pointcloud_surf_in);
                is_odom_inited = true;
                ROS_INFO("odom inited");
            }else{
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
                odom_ES_Estimation.updatePointsToMap(pointcloud_edge_in, pointcloud_surf_in);
                end = std::chrono::system_clock::now();
                std::chrono::duration<float> elapsed_seconds = end - start;
                total_frame++;
                float time_temp = elapsed_seconds.count() * 1000;
                total_time+=time_temp;
                ROS_INFO("average odom estimation time %f ms \n \n", time_temp);
            }



            Eigen::Quaterniond q_current(odom_ES_Estimation.odom.rotation());
            //q_current.normalize();
            Eigen::Vector3d t_current = odom_ES_Estimation.odom.translation();

            static tf::TransformBroadcaster br;
            tf::Transform transform;
            transform.setOrigin( tf::Vector3(t_current.x(), t_current.y(), t_current.z()) );
            tf::Quaternion q(q_current.x(),q_current.y(),q_current.z(),q_current.w());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", sensorFrameId));

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "map";
            laserOdometry.child_frame_id = sensorFrameId;
            laserOdometry.header.stamp = pointcloud_time;
            laserOdometry.pose.pose.orientation.x = q_current.x();
            laserOdometry.pose.pose.orientation.y = q_current.y();
            laserOdometry.pose.pose.orientation.z = q_current.z();
            laserOdometry.pose.pose.orientation.w = q_current.w();
            laserOdometry.pose.pose.position.x = t_current.x();
            laserOdometry.pose.pose.position.y = t_current.y();
            laserOdometry.pose.pose.position.z = t_current.z();
            pubLaserOdometry.publish(laserOdometry);
            sensor_msgs::PointCloud2 cloudMsg;
            if(pubmapEdgePoints.getNumSubscribers() > 0){
                pcl::toROSMsg(*odom_ES_Estimation.laserCloudCornerMap, cloudMsg);
                cloudMsg.header.stamp = pointcloud_time;
                cloudMsg.header.frame_id = "map";
                pubmapEdgePoints.publish(cloudMsg);
            }
            
            if(pubmapSurfPoints.getNumSubscribers() > 0){
                pcl::toROSMsg(*odom_ES_Estimation.laserCloudSurfMap, cloudMsg);
                cloudMsg.header.stamp = pointcloud_time;
                cloudMsg.header.frame_id = "map";
                pubmapSurfPoints.publish(cloudMsg);
            }
        }
        //sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    int scan_line = 64;
    double vertical_angle = 2.0;
    double scan_period= 0.1;
    double max_dis = 60.0;
    double min_dis = 2.0;
    double map_resolution = 0.4;
    std::string k_new_in;
    std::string theta_p_in;
    std::string theta_max_in;
    int k_new = 0;
    float theta_p = 0.4; // 判定持久性阈值
    int theta_max = 75; //局部特征点永久保留阈值
    nh.getParam("scan_period", scan_period); 
    nh.getParam("vertical_angle", vertical_angle); 
    nh.getParam("max_dis", max_dis);
    nh.getParam("min_dis", min_dis);
    nh.getParam("scan_line", scan_line);
    nh.getParam("map_resolution", map_resolution);
    nh.getParam("k_new", k_new_in);
    nh.getParam("theta_p", theta_p_in);
    nh.getParam("theta_max", theta_max_in);
    nh.getParam("sensorFrameId", sensorFrameId);
    nh.getParam("weightType", weightType_para);

    

    lidar_param.setScanPeriod(scan_period);
    lidar_param.setVerticalAngle(vertical_angle);
    lidar_param.setLines(scan_line);
    lidar_param.setMaxDistance(max_dis);
    lidar_param.setMinDistance(min_dis);
    k_new = std::stoi(k_new_in.c_str());
    theta_max = std::stoi(theta_max_in.c_str());
    theta_p = std::stof(theta_p_in.c_str());
    odom_ES_Estimation.init(lidar_param, map_resolution, k_new, theta_p, theta_max, weightType_para);
    ros::Subscriber subEdgeLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_edge", 100, velodyneEdgeHandler);
    ros::Subscriber subSurfLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100, velodyneSurfHandler);

    pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    pubmapSurfPoints = nh.advertise<sensor_msgs::PointCloud2>("/surf_local_map", 1000); 
    pubmapEdgePoints = nh.advertise<sensor_msgs::PointCloud2>("/edge_local_map", 1000); 
    ROS_INFO("startODOM");
    std::thread odom_estimation_process{odom_edge_surf_estimation};

    ros::spin();

    return 0;
}

