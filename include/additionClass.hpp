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

typedef pcl::PointXYZI pointType;
typedef pcl::PointXYZRGBL pointTypeRGBL;
typedef pcl::PointCloud<pcl::PointXYZI> pointTypeCloud;
typedef pcl::PointCloud<pcl::PointXYZRGBL> pointTypeRGBLCloud;

class curvedVoxel
{
public:
    curvedVoxel(ros::NodeHandle nh):nh_(nh){};
    ~curvedVoxel();
    void init(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgPtr);
    bool voxelFilter(std::vector<int> &label_info);
    /**
     * @brief get the index value in the polar radial direction
     * @param radius, polar diameter
     * @return polar diameter index
     */
    int getPolarIndex(double &radius);

    /**
     * @brief converting rectangular coordinate to polar coordinates
     * @param cloud_in_, input point cloud
     * @return void
     */
    void convertToPolar(const pointTypeCloud &cloud_in_);

    /**
     * @brief Create a hash table
     * @param void
     * @return true if success otherwise false
     */
    bool createHashTable(void);

    /**
     * @brief search for neighboring voxels
     * @param polar_index, polar diameter index
     * @param pitch_index, pitch angular index
     * @param azimuth_index, azimuth angular index
     * @param out_neighIndex, output adjacent voxel index set
     * @return void
     */
    void searchKNN(
        int &polar_index,
        int &pitch_index,
        int &azimuth_index,
        std::vector<int> &out_neighIndex) const;

    void labelAnalysis(std::vector<int> &label_info);

    bool colorSegmentation();

    void run(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgPtr);

    void resetParams();

    void publishData();


    template<typename T1,typename T2>
    void copyPointXYZ(T1 src, T2 &tgt){
        tgt.x = src.x;
        tgt.y = src.y;
        tgt.z = src.z;
    }
    template<typename T>
    void setPointRGB(T &tgt,std::vector<int> colorRGB){
        tgt.r = colorRGB[0];
        tgt.g = colorRGB[1];
        tgt.b = colorRGB[2];
    }
    struct segInfo
    {
        int clusterNum{-1};
        std::vector<int> index{};
    };
    pointTypeCloud::Ptr pointCloudPtr;
    pointTypeCloud::Ptr pointCloudSegPtr;
    pointTypeRGBLCloud::Ptr pointCloudSegRGBLPtr;


    std_msgs::Header cloudHeader;

    std::vector<jsk_recognition_msgs::BoundingBox> boxInfo{};

    
    std::string yamlConfigFile;
    std::string velodyne_points;
    std::string velodyne_points_2;
    std::string sensorFrameId;
    

private:
    // >>>>>>>>>> LiDAR params
    int sensorModel{64};
    double scanPeriod{0.0};
    double verticalRes{0.0};
    double initAngle{0.0};
    double sensorHeight{0.0};
    double sensorMinRange{0.0};
    double sensorMaxRange{0.0};
    double near_dis{3.0};
    // <<<<<<<<<<< LiDAR params

    // >>>>>>>>>> Voxels params
    double minPitch{0.0};
    double maxPitch{0.0};
    double minPolar{5.0};
    double maxPolar{5.0};
    double startR{0.0};
    double deltaR{0.0};
    double deltaP{0.0};
    double deltaA{0.0};
    int width{0};
    int height{0};
    int minSeg{0};
    int polarNum{0};
    std::vector<double> polarBounds{};
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> polarCor;
    // std::unordered_map<int, Voxel> voxelMap{};
    std::unordered_map<int, std::vector<int>> voxelMap{};
    std::vector<std::pair<int, segInfo>> labelRecords;
    std::vector<std::vector<int>> colorList;

    // Publisher && Subscriber
    ros::NodeHandle nh_;
    ros::Publisher pubCurvedPointCloud;
    ros::Publisher pubCurvedPointCloudRGBA;
    ros::Publisher pubBoundingBox;

    // <<<<<<<<<<< Voxels params
};

class segmentation
{
public:
    curvedVoxel curvevoxel;

    /**
     * @brief thread function of ground extraction
     * @param q, quadrant index
     * @param cloud_in_, input point cloud
     * @param out_no_ground, output non-ground point cloud
     * @param out_ground, output ground point cloud
     * @return true if success otherwise false
     */
    bool segmentGroundThread(
        int q,
        const pointTypeCloud &cloud_in_,
        pointTypeCloud &out_no_ground,
        pointTypeCloud &out_ground);

    /**
     * @brief ground extraction
     * @param void
     * @return true if success otherwise false
     */
    bool groundRemove(void);
};

template <typename T>
void printout(T value);
void printvalue(std::string valuename, double value);