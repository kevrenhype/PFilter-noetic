#include "additionClass.hpp"

#include <omp.h>
// curvedVoxel::curvedVoxel()
// {
//     ROS_INFO("startCurve");
// };
curvedVoxel::~curvedVoxel()
{
    ROS_INFO("destroy");
};

void curvedVoxel::init(pointTypeCloud::Ptr &inputCloud, std_msgs::Header header)
{

    // 参数配置
    YAML::Node node = YAML::LoadFile(yamlConfigFile);
    const YAML::Node lidar_config = node["velodyne"];
    const YAML::Node voxel_config = node["curvedVoxel"];
    const YAML::Node color_config = node["colorlist"];

    sensorModel = lidar_config["sensorModel"].as<int>();
    scanPeriod = lidar_config["scanPeriod"].as<double>();
    verticalRes = lidar_config["verticalRes"].as<double>();
    initAngle = lidar_config["initAngle"].as<double>();
    sensorHeight = lidar_config["sensorHeight"].as<double>();
    sensorMinRange = lidar_config["sensorMinRange"].as<double>();
    sensorMaxRange = lidar_config["sensorMaxRange"].as<double>();
    near_dis = lidar_config["near_dis"].as<double>();

    startR = voxel_config["startR"].as<double>();
    deltaR = voxel_config["deltaR"].as<double>();
    deltaP = voxel_config["deltaP"].as<double>();
    deltaA = voxel_config["deltaA"].as<double>();
    minSeg = voxel_config["minSeg"].as<int>();

    // 遍历 "colors" 节点并将颜色值添加到容器中

    for (const auto &colorNode : color_config)
    {
        std::string colorStr = colorNode.as<std::string>();
        std::istringstream colorStream(colorStr);
        int r, g, b;
        char comma;
        colorStream >> r >> comma >> g >> comma >> b;
        // 将RGB值存储在内部向量中
        std::vector<int> rgb;
        rgb.emplace_back(r);
        rgb.emplace_back(g);
        rgb.emplace_back(b);
        colorList.emplace_back(rgb);
    }

    // 读取heade
    cloudHeader = header;
    pointCloudPtr = inputCloud;
    // printvalue("pointCloudPtr", pointCloudPtr->size());
}

/**
 * @brief get the index value in the polar radial direction
 * @param radius, polar diameter
 * @return polar diameter index
 */
int curvedVoxel::getPolarIndex(double &radius)
{
    for (auto r = 0; r < polarNum; ++r)
    {
        if (radius < polarBounds[r])
            return r;
    }

    return polarNum - 1;
}

/**
 * @brief converting rectangular coordinate to polar coordinates
 * @param cloud_in_, input point cloud
 * @return void
 */
void curvedVoxel::convertToPolar(const pointTypeCloud &cloud_in_)
{
    if (cloud_in_.empty())
    {
        ROS_ERROR("object cloud don't have point, please check !");
        return;
    }

    auto azimuthCal = [&](double x, double y) -> double
    {
        auto angle = static_cast<double>(std::atan2(y, x));
        return angle > 0.0 ? angle * 180 / M_PI : (angle + 2 * M_PI) * 180 / M_PI;
    };

    size_t totalSize = cloud_in_.points.size();
    polarCor.resize(totalSize);

    Eigen::Vector3d cur = Eigen::Vector3d::Zero();
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
    #pragma omp parallel for
    for (size_t i = 0; i < totalSize; i++)
    {
        Eigen::Vector3d rpa = Eigen::Vector3d::Zero();
        cur = {cloud_in_.points[i].x, cloud_in_.points[i].y, cloud_in_.points[i].z};
        rpa.x() = cur.norm();                                  // polar
        rpa.y() = std::asin(cur.z() / rpa.x()) * 180.0 / M_PI; // pitch
        rpa.z() = azimuthCal(cur.x(), cur.y());                // azimuth

        if (rpa.x() >= sensorMaxRange || rpa.x() <= sensorMinRange)
            continue;

        minPitch = rpa.y() < minPitch ? rpa.y() : minPitch;
        maxPitch = rpa.y() > maxPitch ? rpa.y() : maxPitch;
        minPolar = rpa.x() < minPolar ? rpa.x() : minPolar;
        maxPolar = rpa.x() > maxPolar ? rpa.x() : maxPolar;

        polarCor[i] = rpa;
    }
    polarCor.shrink_to_fit();
    // printout("polarCor");
    // printout(polarCor.size());
    polarNum = 0;
    polarBounds.clear();
    width = static_cast<int>(std::round(360.0 / deltaA) + 1);
    height = static_cast<int>((maxPitch - minPitch) / deltaP);
    double range = minPolar;
    int step = 1;
    while (range <= maxPolar)
    {
        range += (startR - step * deltaR);
        polarBounds.emplace_back(range);
        polarNum++, step++;
    }
}

/**
 * @brief 创建哈希表
 * @param void
 * @return true if success otherwise false
 */
bool curvedVoxel::createHashTable(void)
{
    size_t totalSize = polarCor.size();
    if (totalSize <= 0)
    {
        return false;
    }

    Eigen::Vector3d cur = Eigen::Vector3d::Zero();
    int polarIndex, pitchIndex, azimuthIndex, voxelIndex;
    voxelMap.reserve(totalSize);
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
    for (size_t item = 0; item < totalSize; ++item)
    {
        cur = polarCor[item];
        polarIndex = getPolarIndex(cur.x());
        pitchIndex = static_cast<int>(std::round((cur.y() - minPitch) / deltaP));
        azimuthIndex = static_cast<int>(std::round(cur.z() / deltaA));

        voxelIndex = (azimuthIndex * (polarNum + 1) + polarIndex) + pitchIndex * (polarNum + 1) * (width + 1);

        auto iter = voxelMap.find(voxelIndex);
        if (iter != voxelMap.end())
        {
            // iter->second.index.emplace_back(item);
            iter->second.emplace_back(item);
        }
        else
        {
            std::vector<int> index{};
            index.emplace_back(item);
            voxelMap.insert(std::make_pair(voxelIndex, index));
        }
    }

    return true;
}

/**
 * @brief search for neighboring voxels
 * @param polar_index, polar diameter index
 * @param pitch_index, pitch angular index
 * @param azimuth_index, azimuth angular index
 * @param out_neighIndex, output adjacent voxel index set
 * @return void
 */
void curvedVoxel::searchKNN(int &polar_index, int &pitch_index, int &azimuth_index,
                            std::vector<int> &out_neighIndex) const
{
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
    for (auto z = pitch_index - 1; z <= pitch_index + 1; ++z)
    {
        if (z < 0 || z > height)
            continue;
        for (int y = polar_index - 1; y <= polar_index + 1; ++y)
        {
            if (y < 0 || y > polarNum)
                continue;

            for (int x = azimuth_index - 1; x <= azimuth_index + 1; ++x)
            {
                int ax = x;
                if (ax < 0)
                    ax = width - 1;
                if (ax > 300)
                    ax = 300;

                out_neighIndex.emplace_back((ax * (polarNum + 1) + y) + z * (polarNum + 1) * (width + 1));
            }
        }
    }
}

/**
 * @brief the Dynamic Curved-Voxle Clustering algoithm for fast and precise point cloud segmentaiton  得到带标签的点云，分类后
 * @param label_info, output the category information of each point
 * @return true if success otherwise false
 */
bool curvedVoxel::voxelFilter(std::vector<int> &label_info)
{
    int labelCount = 0;
    size_t totalSize = polarCor.size();
    if (totalSize <= 0)
    {
        ROS_ERROR("there are not enough point clouds to complete the DCVC algorithm");
        return false;
    }

    label_info.resize(totalSize, -1);
    Eigen::Vector3d cur = Eigen::Vector3d::Zero();
    int polar_index, pitch_index, azimuth_index, voxel_index, currInfo, neighInfo;
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
    for (size_t i = 0; i < totalSize; ++i)
    {
        if (label_info[i] != -1)
            continue;
        cur = polarCor[i];

        polar_index = getPolarIndex(cur.x());
        pitch_index = static_cast<int>(std::round((cur.y() - minPitch) / deltaP));
        azimuth_index = static_cast<int>(std::round(cur.z() / deltaA));
        voxel_index = (azimuth_index * (polarNum + 1) + polar_index) + pitch_index * (polarNum + 1) * (width + 1);

        auto iter_find = voxelMap.find(voxel_index);
        std::vector<int> neighbors;
        if (iter_find != voxelMap.end())
        {

            std::vector<int> KNN{};
            searchKNN(polar_index, pitch_index, azimuth_index, KNN);

            for (auto &k : KNN)
            {
                iter_find = voxelMap.find(k);

                if (iter_find != voxelMap.end())
                {
                    neighbors.reserve(iter_find->second.size());
                    for (auto &id : iter_find->second)
                    {
                        neighbors.emplace_back(id);
                    }
                }
            }
        }

        neighbors.swap(neighbors);

        if (!neighbors.empty())
        {
            for (auto &id : neighbors)
            {
                currInfo = label_info[i];   // current label index
                neighInfo = label_info[id]; // voxel label index
                if (currInfo != -1 && neighInfo != -1 && currInfo != neighInfo)
                {
                    for (auto &seg : label_info)
                    {
                        if (seg == currInfo)
                            seg = neighInfo;
                    }
                }
                else if (neighInfo != -1)
                {
                    label_info[i] = neighInfo;
                }
                else if (currInfo != -1)
                {
                    label_info[id] = currInfo;
                }
                else
                {
                    continue;
                }
            }
        }

        // If there is no category information yet, then create a new label information
        if (label_info[i] == -1)
        {
            labelCount++;
            label_info[i] = labelCount;
            for (auto &id : neighbors)
            {
                label_info[id] = labelCount;
            }
        }
    }

    // free memory
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(polarCor);

    return true;
}

/**
 * @brief obtain the point cloud data of each category
 * @param label_info, input category information
 * @return void
 */
void curvedVoxel::labelAnalysis(std::vector<int> &label_info)
{

    std::unordered_map<int, segInfo> histCounts; // <分类标签 - [标签下点云数量 , 点云序号VECTOR] >键值对
    size_t totalSize = label_info.size();
    for (size_t i = 0; i < totalSize; ++i)
    {
        if (histCounts.find(label_info[i]) == histCounts.end())
        {
            histCounts[label_info[i]].clusterNum = 1;
            histCounts[label_info[i]].index.emplace_back(i);
        }
        else
        {
            histCounts[label_info[i]].clusterNum += 1;
            histCounts[label_info[i]].index.emplace_back(i);
        }
    }

    std::vector<std::pair<int, segInfo>> labelStatic(histCounts.begin(), histCounts.end()); // 根据标签数量保存不同分类号的排序
    std::sort(labelStatic.begin(), labelStatic.end(), [&](std::pair<int, segInfo> &a, std::pair<int, segInfo> &b) -> bool
              { return a.second.clusterNum > b.second.clusterNum; });

    auto labelCount{1};
    auto labelpointcount{0};
    for (auto &info : labelStatic)
    {
        if (info.second.clusterNum > minSeg)
        {
            labelRecords.emplace_back(std::make_pair(labelCount, info.second));
            labelCount++;
            labelpointcount += info.second.clusterNum;
        }
    }
    // printvalue("labelStatic", labelCount);
    // printvalue("labelpointcount", labelpointcount);
}

/**
 * @brief statistics label information and render the colors for better visualization
 * @param void
 * @return true if success otherwise false
 */
bool curvedVoxel::colorSegmentation()
{
    pointCloudSegPtr.reset(new pointTypeCloud());
    pointCloudSegRGBLPtr.reset(new pointTypeRGBLCloud());
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
    for (auto &label : labelRecords)
    {
        // box
        jsk_recognition_msgs::BoundingBox box;
        float min_x = std::numeric_limits<double>::max();
        float max_x = -std::numeric_limits<double>::max();
        float min_y = std::numeric_limits<double>::max();
        float max_y = -std::numeric_limits<double>::max();
        float min_z = std::numeric_limits<double>::max();
        float max_z = -std::numeric_limits<double>::max();

        for (auto &id : label.second.index)
        {
            pointCloudSegPtr->push_back(pointCloudPtr->points[id]);

            min_x = std::min(min_x, pointCloudPtr->points[id].x);
            max_x = std::max(max_x, pointCloudPtr->points[id].x);
            min_y = std::min(min_y, pointCloudPtr->points[id].y);
            max_y = std::max(max_y, pointCloudPtr->points[id].y);
            min_z = std::min(min_z, pointCloudPtr->points[id].z);
            max_z = std::max(max_z, pointCloudPtr->points[id].z);

            pointTypeRGBL pp;
            copyPointXYZ(pointCloudPtr->points[id], pp);
            setPointRGB(pp, colorList[label.first % colorList.size()]);
            pointCloudSegRGBLPtr->points.push_back(pp);
        }

        double lengthBox = max_x - min_x;
        double widthBox = max_y - min_y;
        double heightBox = max_z - min_z;
        box.header = cloudHeader;
        box.label = label.first;
        Eigen::Vector3d box_in_map(min_x + lengthBox / 2.0, min_y + widthBox / 2.0, min_z + heightBox / 2.0);
        box.pose.position.x = box_in_map.x();
        box.pose.position.y = box_in_map.y();
        box.pose.position.z = box_in_map.z();

        box.dimensions.x = ((lengthBox < 0) ? -1 * lengthBox : lengthBox);
        box.dimensions.y = ((widthBox < 0) ? -1 * widthBox : widthBox);
        box.dimensions.z = ((heightBox < 0) ? -1 * heightBox : heightBox);

        boxInfo.emplace_back(box);
    }
    // printvalue("boxInfo", boxInfo.size());

    return true;
}

void curvedVoxel::publishData()
{

    // 发布每个标签的包围盒
    jsk_recognition_msgs::BoundingBoxArray boxArray;
    for (auto &box : boxInfo)
    {
        boxArray.boxes.emplace_back(box);
    }
    boxArray.header = cloudHeader;
    // printvalue("boxArray", boxArray.boxes.size());
    pubBoundingBox.publish(boxArray);

    // 聚类后带颜色的点云
    sensor_msgs::PointCloud2 laserCloudMsgOut2;
    pcl::toROSMsg(*pointCloudSegRGBLPtr, laserCloudMsgOut2);
    laserCloudMsgOut2.header = cloudHeader;
    pubCurvedPointCloudRGBA.publish(laserCloudMsgOut2);
}

void curvedVoxel::resetParams()
{
    width = 0.0;
    height = 0.0;
    minPitch = maxPitch = 0.0;
    minPolar = maxPolar = 0.0;

    boxInfo.clear();
    polarCor.clear();
    voxelMap.clear();
    labelRecords.clear();
    polarBounds.clear();

}

void curvedVoxel::run(pointTypeCloud::Ptr &inputCloud, std_msgs::Header header)
{
    init(inputCloud, header);
    /// step 1.0 convert point cloud to polar coordinate system
    if ((*pointCloudPtr).empty())
    {
        ROS_ERROR("not enough point to convert");
        return;
    }

    if (!(*pointCloudPtr).empty())
    {
        convertToPolar((*pointCloudPtr));
    }
    else
    {
        ROS_ERROR("not enough point to convert");
        return;
    }

    // step 2.0 Create a hash table
    createHashTable();

    /// step 3.0 DCVC segmentation
    std::vector<int> labelInfo{}; //
    if (!voxelFilter(labelInfo))
    {
        ROS_ERROR("DCVC algorithm segmentation failure");
        return;
    }

    /// step 4.0 statistics category record
    labelAnalysis(labelInfo); //

    colorSegmentation();

    publishData();

    resetParams();
    return;
}

template <typename T>
void printout(T value)
{
    std::cout << value << std::endl;
}

void printvalue(std::string valuename, double value)
{
    std::cout << valuename << ": " << value << std::endl;
}
