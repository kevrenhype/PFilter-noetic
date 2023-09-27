#include "additionClass.hpp"

template <typename T>
void printout(T value)
{
    std::cout << value << std::endl;
}

void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgPtr, ros::NodeHandle &nh)
{
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    curvedVoxel curvedvoxel(nh); // 实例化
    curvedvoxel.run(laserCloudMsgPtr);
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    float time_temp = elapsed_seconds.count() * 1000;
    ROS_INFO("CurvedVoxel time %f ms \n \n", time_temp);

}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");

    ros::NodeHandle nh;

    std::string velodyne_points;

    nh.getParam("velodyne_points", velodyne_points);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(velodyne_points, 100,
                                                                           boost::bind(&velodyneHandler, _1, nh));

    ros::spin();

    return 0;
}
