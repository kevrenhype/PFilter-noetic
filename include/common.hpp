#include <yaml-cpp/yaml.h>

// c++ lib
#include <cmath>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <map>
#include <unordered_map>
// ros lib
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "std_msgs/Header.h"
#include "std_msgs/Float64.h"
// pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <pcl/filters/random_sample.h>
#include <omp.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
typedef pcl::PointXYZI pointType;
typedef pcl::PointXYZRGBL pointTypeRGBL;
typedef pcl::PointCloud<pcl::PointXYZI> pointTypeCloud;
typedef pcl::PointCloud<pcl::PointXYZRGBL> pointTypeRGBLCloud;
typedef pcl::PointXYZINormal pointTypeNormal ;
typedef pcl::PointCloud<pcl::PointXYZINormal> pointTypeCloudNormal ;