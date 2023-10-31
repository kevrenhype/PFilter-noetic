#include "additionClass.hpp"
#include "preProcess.hpp"
int groundfilter = 0;
int curvedfilter = 0;
int featurePreExtract = 0;
template <typename T>
void printout(T value)
{
    std::cout << value << std::endl;
}

void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgPtr, curvedVoxel &curvedvoxel, groundSeg &groundseg,nongroundExtract &nongroundextra)
{
    pointTypeCloud::Ptr inputCloudPtr(new pointTypeCloud());

    std_msgs::Header header;

    pcl::fromROSMsg(*laserCloudMsgPtr, *inputCloudPtr);
    header = laserCloudMsgPtr->header;

    if (groundfilter)
    {
        groundseg.groundInit(inputCloudPtr, header);
        groundseg.ground_seg(groundseg.groundSeginputCloudPtr, groundseg.groundCloudPtr, groundseg.nonGroundCloudPtr, groundseg.gf_grid_pt_num_thre, groundseg.gf_grid_resolution, groundseg.gf_max_grid_height_diff, groundseg.gf_neighbor_height_diff, groundseg.gf_max_ground_height, groundseg.gf_min_ground_height);
        groundseg.groundPubCloud();
        inputCloudPtr = groundseg.nonGroundCloudPtr;
    }
    // =======================================================================================//
    if (curvedfilter)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        curvedvoxel.run(inputCloudPtr, header);
        end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        float time_temp = elapsed_seconds.count() * 1000;
        ROS_INFO("CurvedVoxel time %f ms \n \n", time_temp);
        inputCloudPtr = curvedvoxel.pointCloudSegPtr;
    }
    if (featurePreExtract){
        nongroundextra.featureInit(inputCloudPtr,header);
        ROS_INFO("init");
        nongroundextra.pc2pc<pointTypeCloud,pointTypeCloudNormal>(nongroundextra.featureSeginputCloudPtr,nongroundextra.normalCloud);
        nongroundextra.featureExtract<pointTypeNormal>(nongroundextra.normalCloud);
        nongroundextra.pubFeatureCloud();
    }
    // =======================================================================================//
    sensor_msgs::PointCloud2 outputlaserMsg;
    outputlaserMsg.header = header;
    pcl::toROSMsg(*inputCloudPtr,outputlaserMsg);
    curvedvoxel.pubCurvedPointCloud.publish(outputlaserMsg);


}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");

    ros::NodeHandle nh;

    curvedVoxel curvedvoxel(nh); // 实例化
    groundSeg groundseg(nh);
    nongroundExtract nongroundextra(nh);
    nh.getParam("velodyne_points", curvedvoxel.velodyne_points);
    nh.getParam("pfilter_input_cloud", curvedvoxel.pfilter_input_cloud);
    nh.getParam("yamlConfigFile", curvedvoxel.yamlConfigFile);
    nh.getParam("sensorFrameId", curvedvoxel.sensorFrameId);
    nh.getParam("curvedfilter", curvedfilter);
    nh.getParam("groundfilter", groundfilter);
    nh.getParam("featurePreExtract", featurePreExtract);

    ROS_INFO("=============================================================");
    ROS_INFO("curvedfilter %d",curvedfilter);
    ROS_INFO("groundfilter %d",groundfilter);
    ROS_INFO("featurePreExtract %d",featurePreExtract);

    ROS_INFO("=============================================================");

    // Publishers
    curvedvoxel.pubBoundingBox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/boundingBoxLabel", 100);
    curvedvoxel.pubCurvedPointCloud = nh.advertise<sensor_msgs::PointCloud2>(curvedvoxel.pfilter_input_cloud, 100);
    curvedvoxel.pubCurvedPointCloudRGBA = nh.advertise<sensor_msgs::PointCloud2>("/curvedPointCloudRGBA", 100);
    groundseg.pubGround = nh.advertise<sensor_msgs::PointCloud2>("/groundCloud", 100);
    groundseg.pubNonGround = nh.advertise<sensor_msgs::PointCloud2>("/nonGroundCloud", 100);
    nongroundextra.pubBeam = nh.advertise<sensor_msgs::PointCloud2>  ("/beamCloud", 100);
    nongroundextra.pubPillar = nh.advertise<sensor_msgs::PointCloud2>("/pillarCloud", 100);
    nongroundextra.pubFacade = nh.advertise<sensor_msgs::PointCloud2>("/facadeCloud", 100);
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(curvedvoxel.velodyne_points, 100,
                                                                           boost::bind(&velodyneHandler, _1, curvedvoxel, groundseg,nongroundextra));

    ros::spin();

    return 0;
}
