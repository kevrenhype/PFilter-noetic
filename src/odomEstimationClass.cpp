
#include "odomEstimationClass.h"

/**
 * 根据稀疏地图的g和r值，筛选满足持续性的点加入地图
 */
void OdomBaseClass::extractstablepoint(pcl::PointCloud<PointType>::Ptr lasecloudMap_input, int k_new, float theta_p, int theta_max)
{
    std::vector<int> value_index;
    for (std::size_t i = 0; i < lasecloudMap_input->points.size(); i++)
    {
        if (lasecloudMap_input->points[i].g < lasecloudMap_input->points[i].r * theta_p // 判定是否满足条件
            && lasecloudMap_input->points[i].r > k_new && lasecloudMap_input->points[i].g < theta_max + 1)
            continue;
        value_index.push_back(i);
    }
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(value_index);
    // Create the filtering object
    pcl::ExtractIndices<PointType> extract;
    // Extract the inliers
    extract.setInputCloud(lasecloudMap_input);
    extract.setIndices(index_ptr);
    extract.setNegative(false); // 如果设为true,可以提取指定index之外的点云
    extract.filter(*lasecloudMap_input);
}

/**
 * 1、计算整个点云的边界  点的最大最小值
 * 2、将所有点云按照体素号进行排序，用《点云序号-体素号》键值对存储每个点
 * 3、选择点云数量足够的体素，记录每个体素的开始和结束点号
 * 4、最后计算体素内的质心，并计算这里面最大的r和g值
 * 5、最后output的存储是每个体素内的点云质心点
 */
pcl::PointCloud<PointType>::Ptr OdomBaseClass::rgbds(pcl::PointCloud<PointType>::Ptr input, float dsleaf, unsigned min_points_per_voxel_ = 0)
{
    pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>); // 存储体素内的点云质心及最大的r和g值
    output->height = 1;
    output->is_dense = true;

    Eigen::Vector4f min_p, max_p;      // 点云中 最大最小的xyz值
    Eigen::Vector4i min_b_, max_b_;    // 边界体素序号值
    Eigen::Vector4i div_b_, divb_mul_; // 三个方向上的体素数量；用于计算点云中的体素索引的动态系数
    pcl::getMinMax3D<PointType>(*input, min_p, max_p);

    // 存储六个方向上，边界体素的序号值
    min_b_[0] = static_cast<int>(floor(min_p[0] / dsleaf));
    max_b_[0] = static_cast<int>(floor(max_p[0] / dsleaf));
    min_b_[1] = static_cast<int>(floor(min_p[1] / dsleaf));
    max_b_[1] = static_cast<int>(floor(max_p[1] / dsleaf));
    min_b_[2] = static_cast<int>(floor(min_p[2] / dsleaf));
    max_b_[2] = static_cast<int>(floor(max_p[2] / dsleaf));

    // 取得三个方向上的体素数量
    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
    div_b_[3] = 0;
    divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

    std::vector<cloud_point_index_idx> index_vector; // 存 点云-体素 序号对的容器  大小是输入点云的大小
    index_vector.reserve(input->points.size());

    for (std::size_t i = 0; i < input->points.size(); i++)
    {
        int ijk0 = static_cast<int>(floor(input->points[i].x / dsleaf) - static_cast<float>(min_b_[0]));
        int ijk1 = static_cast<int>(floor(input->points[i].y / dsleaf) - static_cast<float>(min_b_[1]));
        int ijk2 = static_cast<int>(floor(input->points[i].z / dsleaf) - static_cast<float>(min_b_[2]));

        // Compute the centroid leaf index
        int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];        // 体素号
        index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int>(idx), i)); // 体素号-点云号  键值对
    }

    // Second pass: sort the index_vector vector using value representing target cell as index
    // in effect all points belonging to the same output cell will be next to each other
    std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>()); // 按照体素序号排序 <键值对>

    // Third pass: count output cells
    // we need to skip all the same, adjacenent idx values
    unsigned int total = 0;
    unsigned int index = 0;
    // first_and_last_indices_vector[i] represents the index in index_vector of the first point in
    // index_vector belonging to the voxel which corresponds to the i-th output point,
    // and of the first point not belonging to.
    std::vector<std::pair<unsigned int, unsigned int>> first_and_last_indices_vector; // 存储每个体素中的开始和结束点的序号
    // Worst case size
    first_and_last_indices_vector.reserve(index_vector.size());
    while (index < index_vector.size()) // 遍历整个点云，选择点云数足够多的Voxel，保存该体素的头尾点序号
    {
        unsigned int i = index + 1;
        while (i < index_vector.size() && index_vector[i].idx == index_vector[index].idx)
            ++i;
        if (i - index >= min_points_per_voxel_)
        {
            ++total;
            first_and_last_indices_vector.push_back(std::pair<unsigned int, unsigned int>(index, i));
        }
        index = i;
    }

    // Fourth pass: compute centroids, insert them into their final position
    output->points.resize(total); // 将输入点云改为之前筛选出来的点对
    index = 0;
    for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp)
    {
        // calculate centroid - sum values from all input points, that have the same idx value in index_vector array
        unsigned int first_index = first_and_last_indices_vector[cp].first;
        unsigned int last_index = first_and_last_indices_vector[cp].second;

        Eigen::Vector4f centroid(Eigen::Vector4f::Zero());

        int r_max = -1;
        float g_max = -1;
        for (unsigned int li = first_index; li < last_index; ++li)
        {
            centroid += input->points[index_vector[li].cloud_point_index].getVector4fMap();
            if (input->points[index_vector[li].cloud_point_index].r > r_max)
            {
                r_max = input->points[index_vector[li].cloud_point_index].r;
                // centroid = input->points[index_vector[li].cloud_point_index].getVector4fMap ();
            }
            if (input->points[index_vector[li].cloud_point_index].g > g_max)
            {
                g_max = input->points[index_vector[li].cloud_point_index].g;
            }
        }
        centroid /= static_cast<float>(last_index - first_index); // 计算质心
        output->points[index].getVector4fMap() = centroid;
        output->points[index].r = r_max;
        output->points[index].g = g_max;

        ++index;
    }
    output->width = static_cast<uint32_t>(output->points.size());
    return output;
}

void OdomBaseClass::observeMean(std::vector<double> &observe_vec)
{
    double min_element = *std::min_element(observe_vec.begin(), observe_vec.end());
    double max_element = *std::max_element(observe_vec.begin(), observe_vec.end());
    // std::cout << "min: " << double(min_element) << std::endl;
    // std::cout << "max: " << double(max_element) << std::endl;
    auto length = (max_element - min_element);
    if (length == 0)
        return;
    for (auto &ele : observe_vec)
    {
        ele = (ele - min_element) / length;
        ele -= 1.0;
        ele = abs(ele);
        ele *= 2.0;
        ele = std::max(0.1, ele);
    }
    // writefile(observe_vec, "/home/r/catkin_wss/pfilter_ws/src/PFilter-noetic/observe.dat");
    // min_element = *std::min_element(observe_vec.begin(), observe_vec.end());
    //  max_element = *std::max_element(observe_vec.begin(), observe_vec.end());
    //  std::cout << "min: " << double(min_element) << std::endl;
    //  std::cout << "max: " << double(max_element) << std::endl;
    //  double sum = std::accumulate(observe_vec.begin(), observe_vec.end(), 0.0);
    //  double mean = sum / double(observe_vec.size());
}

void OdomBaseClass::pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b;
    // po->intensity = pi->intensity;
    // po->intensity = 1.0;
}

void OdomBaseClass::downSamplingToMap(const pcl::PointCloud<PointType>::Ptr &pc_in, pcl::PointCloud<PointType>::Ptr &pc_out, pcl::VoxelGrid<PointType> &downGrid)
{
    downGrid.setInputCloud(pc_in);
    downGrid.filter(*pc_out);
}

void Odom_ES_EstimationClass::init(lidar::Lidar lidar_param, double map_resolution_in, int k_new_para, float theta_p_para, int theta_max_para, double weightType_para)
{
    // init local map
    laserCloudCornerMap = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    laserCloudSurfMap = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

    // downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution_in, map_resolution_in, map_resolution_in);
    downSizeFilterSurf.setLeafSize(map_resolution_in * 2, map_resolution_in * 2, map_resolution_in * 2);

    // kd-tree
    kdtreeEdgeMap = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfMap = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    optimization_count = 2;
    k_new_surf = k_new_para;
    theta_p_surf = theta_p_para;
    theta_max_surf = theta_max_para;
    k_new_edge = k_new_para;
    theta_p_edge = theta_p_para;
    theta_max_edge = theta_max_para;
    weightType = weightType_para;

    map_resolution = map_resolution_in;
}

void Odom_ES_EstimationClass::getMap(pcl::PointCloud<PointType>::Ptr &laserCloudMap)
{

    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

void Odom_ES_EstimationClass::initMapWithPoints(const pcl::PointCloud<PointType>::Ptr &edge_in, const pcl::PointCloud<PointType>::Ptr &surf_in)
{
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    optimization_count = 12;
}

/**
 * 更新点到地图
 * 1、计算scan2scan的里程计
 * 2、更新持久性点到地图，删除临时点
 */
void Odom_ES_EstimationClass::updatePointsToMap(const pcl::PointCloud<PointType>::Ptr &edge_in, const pcl::PointCloud<PointType>::Ptr &surf_in)
{

    if (optimization_count > 2)
        optimization_count--;

    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    pcl::PointCloud<PointType>::Ptr downsampledEdgeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr downsampledSurfCloud(new pcl::PointCloud<PointType>());
    downSamplingToMap(edge_in, downsampledEdgeCloud, downSizeFilterEdge);
    downSamplingToMap(surf_in, downsampledSurfCloud, downSizeFilterSurf);
    // ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
    if (laserCloudCornerMap->points.size() > 10 && laserCloudSurfMap->points.size() > 50)
    {
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);

        for (int iterCount = 0; iterCount < optimization_count; iterCount++)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);

            problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

            addEdgeCostFactor(downsampledEdgeCloud, laserCloudCornerMap, problem, loss_function);
            addSurfCostFactor(downsampledSurfCloud, laserCloudSurfMap, problem, loss_function);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);
        }
    }
    else
    {
        printf("not enough points in map to associate, map error");
    }
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    addPointsToMap(downsampledEdgeCloud, downsampledSurfCloud);
}

void Odom_ES_EstimationClass::addEdgeCostFactor(const pcl::PointCloud<PointType>::Ptr &edge_cloud, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int corner_num = 0;
    std::vector<edgeInfo> edge_valid_vec;
    std::vector<double> pointSparsity_vec;
    std::vector<double> observe_vec;
    edgeInfo tmp_edge_info;
    for (int i = 0; i < (int)edge_cloud->points.size(); i++)
    {
        bool point_valid = false;
        PointType point_temp;
        pointAssociateToMap(&(edge_cloud->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[4] < 1.0) // 保证邻居点距离足够近
        {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0); // 邻居点质心
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d neigh_p(map_in->points[pointSearchInd[j]].x,
                                        map_in->points[pointSearchInd[j]].y,
                                        map_in->points[pointSearchInd[j]].z);
                center = center + neigh_p;
                nearCorners.push_back(neigh_p);
            }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero(); // 协方差矩阵
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); // 存储了主成分方向 ———— 最大特征值对应的特征向量
            Eigen::Vector3d curr_point(edge_cloud->points[i].x, edge_cloud->points[i].y, edge_cloud->points[i].z);

            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b; // 在穿过质心的，最大特征值方向上的两个点
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                float observe = (map_in->points[pointSearchInd[0]].g +
                                 map_in->points[pointSearchInd[1]].g +
                                 map_in->points[pointSearchInd[2]].g +
                                 map_in->points[pointSearchInd[3]].g +
                                 map_in->points[pointSearchInd[4]].g) /
                                    5.0 +
                                1;
                float round = (map_in->points[pointSearchInd[0]].r +
                               map_in->points[pointSearchInd[1]].r +
                               map_in->points[pointSearchInd[2]].r +
                               map_in->points[pointSearchInd[3]].r +
                               map_in->points[pointSearchInd[4]].r) /
                              5.0;
                for (int j = 0; j < 5; j++)
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);

                if (observe / round > 5)
                    observe = 255;
                if (observe < round * theta_p_edge && round > k_new_edge && observe < theta_max_edge)
                {
                    continue;
                }
                edge_cloud->points[i].r = std::min(255, int(round));
                edge_cloud->points[i].g = std::min(255, int(observe));

                corner_num++;

                point_valid = true;
                tmp_edge_info.curr_point = curr_point;
                tmp_edge_info.point_a = point_a;
                tmp_edge_info.point_b = point_b;
                tmp_edge_info.observe = observe;
                tmp_edge_info.round = round;
                edge_valid_vec.emplace_back(tmp_edge_info);

                float sum = 0;
                Eigen::Vector3d centerNeigh(0, 0, 0);
                std::vector<Eigen::Vector3d> p_neighbors;
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d p_Neighbor(
                        map_in->points[pointSearchInd[j]].x,
                        map_in->points[pointSearchInd[j]].y,
                        map_in->points[pointSearchInd[j]].z);
                    p_neighbors.emplace_back(p_Neighbor);
                    centerNeigh = centerNeigh + p_Neighbor;
                }
                centerNeigh /= 5;
                for (auto p : p_neighbors)
                {
                    sum += (centerNeigh - p).norm();
                }
                sum /= 5.0;
                pointSparsity_vec.emplace_back(sum);
            }
        }
    }
    if (weightType == 1 || weightType == 12)
    {

        for (int i = 0; i < (int)edge_valid_vec.size(); i++)
        {
            observe_vec.emplace_back(edge_valid_vec[i].observe);
        }
        observeMean(observe_vec);
    }
    if (weightType == 2 || weightType == 12)
    {
        pointSparsityMean(pointSparsity_vec);
    }

    for (int i = 0; i < (int)edge_valid_vec.size(); i++)
    {
        Eigen::Vector3d curr_point, point_a, point_b;
        float observe, round;
        curr_point = edge_valid_vec[i].curr_point;
        ;
        point_a = edge_valid_vec[i].point_a;
        point_b = edge_valid_vec[i].point_b;
        observe = edge_valid_vec[i].observe;
        round = edge_valid_vec[i].round;
        ceres::CostFunction *cost_function;
        if (weightType == 0)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
        else if (weightType == 1)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, observe_vec[i]);
        else if (weightType == 2)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, pointSparsity_vec[i]);
        else if (weightType == 12)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, (pointSparsity_vec[i] + observe_vec[i]) / 2);
        else
            ROS_ERROR("STH WRONG!");

        problem.AddResidualBlock(cost_function, loss_function, parameters);
    }
    edge_valid_vec.clear();
    if (corner_num < 20)
    {
        ROS_ERROR("not enough correct points");
    }
}

void Odom_ES_EstimationClass::addSurfCostFactor(const pcl::PointCloud<PointType>::Ptr &surf_cloud, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    std::vector<surfInfo> surf_valid_vec;
    std::vector<double> pointSparsity_vec;
    std::vector<double> observe_vec;
    surfInfo tmp_surf_info;
    int surf_num = 0;
    for (int i = 0; i < (int)surf_cloud->points.size(); i++)
    {
        PointType point_temp;
        pointAssociateToMap(&(surf_cloud->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis); // scan-to-map 的特征匹配

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {

            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm(); // 法向量模长倒数？
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(surf_cloud->points[i].x, surf_cloud->points[i].y, surf_cloud->points[i].z);
            if (planeValid)
            {
                float observe = (map_in->points[pointSearchInd[0]].g +
                                 map_in->points[pointSearchInd[1]].g +
                                 map_in->points[pointSearchInd[2]].g +
                                 map_in->points[pointSearchInd[3]].g +
                                 map_in->points[pointSearchInd[4]].g) /
                                    5.0 +
                                1; // observe 是  p-Index
                float round = (map_in->points[pointSearchInd[0]].r +
                               map_in->points[pointSearchInd[1]].r +
                               map_in->points[pointSearchInd[2]].r +
                               map_in->points[pointSearchInd[3]].r +
                               map_in->points[pointSearchInd[4]].r) /
                              5.0;
                for (int j = 0; j < 5; j++)
                {
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);
                }
                if (observe / round > 5)
                    observe = 255;
                if (observe < round * theta_p_surf && round > k_new_surf && observe < theta_max_surf)
                {
                    continue;
                }
                surf_cloud->points[i].r = std::min(255, int(round));
                surf_cloud->points[i].g = std::min(255, int(observe));

                surf_num++;
                tmp_surf_info.curr_point = curr_point;
                tmp_surf_info.norm = norm;
                tmp_surf_info.negative_OA_dot_norm = negative_OA_dot_norm;
                tmp_surf_info.observe = observe;
                tmp_surf_info.round = round;
                surf_valid_vec.emplace_back(tmp_surf_info);
                float sum = 0;

                Eigen::Vector3d centerNeigh(0, 0, 0);
                std::vector<Eigen::Vector3d> p_neighbors;
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d p_Neighbor(
                        map_in->points[pointSearchInd[j]].x,
                        map_in->points[pointSearchInd[j]].y,
                        map_in->points[pointSearchInd[j]].z);
                    p_neighbors.emplace_back(p_Neighbor);
                    centerNeigh = centerNeigh + p_Neighbor;
                }
                centerNeigh /= 5;
                for (auto p : p_neighbors)
                {
                    sum += (centerNeigh - p).norm();
                }
                sum /= 5.0;
                pointSparsity_vec.emplace_back(sum);
            }
        }
    }
    if (weightType == 1 || weightType == 12)
    {
        for (int i = 0; i < (int)surf_valid_vec.size(); i++)
        {
            observe_vec.emplace_back(surf_valid_vec[i].observe);
        }
        observeMean(observe_vec);
    }
    if (weightType == 2 || weightType == 12)
    {
        pointSparsityMean(pointSparsity_vec);
    }

    for (int i = 0; i < (int)surf_valid_vec.size(); i++)
    {
        Eigen::Vector3d curr_point, norm;
        float observe, round, negative_OA_dot_norm;
        curr_point = surf_valid_vec[i].curr_point;
        norm = surf_valid_vec[i].norm;
        negative_OA_dot_norm = surf_valid_vec[i].negative_OA_dot_norm;
        observe = surf_valid_vec[i].observe;
        round = surf_valid_vec[i].round;
        ceres::CostFunction *cost_function;
        if (weightType == 0)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
        else if (weightType == 1)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, observe_vec[i]);
        else if (weightType == 2)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pointSparsity_vec[i]);
        else if (weightType == 12)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, (observe_vec[i] + pointSparsity_vec[i]) / 2);
        else
            ROS_ERROR("STH WRONG!");

        problem.AddResidualBlock(cost_function, loss_function, parameters);
    }
    surf_valid_vec.clear();

    if (surf_num < 20)
    {
        printf("not enough correct points");
    }
}

/**
 * 融合当前帧与地图
 * 1、将点云按照位子变换加入地图
 * 2、更新地图维护边界
 * 3、裁剪点云，至地图边界内
 * 4、根据体素筛选点云，计算每个体素的质心
 * 4、提取稳定点
 * 5、若点云足够多，变为红色？
 */
void Odom_ES_EstimationClass::addPointsToMap(const pcl::PointCloud<PointType>::Ptr &downsampledEdgeCloud, const pcl::PointCloud<PointType>::Ptr &downsampledSurfCloud)
{

    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
    {
        PointType point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp);
    }

    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        PointType point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
    }

    double x_min = +odom.translation().x() - 100;
    double y_min = +odom.translation().y() - 100;
    double z_min = +odom.translation().z() - 100;
    double x_max = +odom.translation().x() + 100;
    double y_max = +odom.translation().y() + 100;
    double z_max = +odom.translation().z() + 100;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);

    pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);

    laserCloudSurfMap = rgbds(tmpSurf, map_resolution * 2);
    laserCloudCornerMap = rgbds(tmpCorner, map_resolution);
    // downSizeFilterSurf.setInputCloud(tmpSurf);
    // downSizeFilterSurf.filter(*laserCloudSurfMap);
    // downSizeFilterEdge.setInputCloud(tmpCorner);
    // downSizeFilterEdge.filter(*laserCloudCornerMap);
    extractstablepoint(laserCloudSurfMap, k_new_surf, theta_p_surf, theta_max_surf);
    extractstablepoint(laserCloudCornerMap, k_new_edge, theta_p_edge, theta_max_edge);
    // 更新点的持久帧次
    for (int i = 0; i < laserCloudSurfMap->points.size(); i++)
    {
        if (laserCloudSurfMap->points[i].r > 250)
            laserCloudSurfMap->points[i].r = 255;
        else
            laserCloudSurfMap->points[i].r += 2;
    }

    for (int i = 0; i < laserCloudCornerMap->points.size(); i++)
        if (laserCloudCornerMap->points[i].r > 250)
            laserCloudCornerMap->points[i].r = 255;
        else
            laserCloudCornerMap->points[i].r += 2;
}

void Odom_BPF_EstimationClass::init(lidar::Lidar lidar_param, double map_resolution_in, int k_new_para, float theta_p_para, int theta_max_para, double weightType_para)

{
    // init local map
    laserCloudBeamMap = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    laserCloudPillarMap = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    laserCloudFacadeMap = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

    // downsampling size
    downSizeFilterBeam.setLeafSize(map_resolution_in, map_resolution_in, map_resolution_in);
    downSizeFilterPillar.setLeafSize(map_resolution_in, map_resolution_in, map_resolution_in);
    downSizeFilterFacade.setLeafSize(map_resolution_in * 2, map_resolution_in * 2, map_resolution_in * 2);

    // kd-tree
    kdtreeBeamMap = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    kdtreePillarMap = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    kdtreeFacadeMap = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    optimization_count = 2;
    k_new_surf = k_new_para;
    theta_p_surf = theta_p_para;
    theta_max_surf = theta_max_para;
    k_new_edge = k_new_para;
    theta_p_edge = theta_p_para;
    theta_max_edge = theta_max_para;
    weightType = weightType_para;

    map_resolution = map_resolution_in;
}

void Odom_BPF_EstimationClass::getMap(pcl::PointCloud<PointType>::Ptr &laserCloudMap)
{

    *laserCloudMap += *laserCloudBeamMap;
    *laserCloudMap += *laserCloudPillarMap;
    *laserCloudMap += *laserCloudFacadeMap;
}

void Odom_BPF_EstimationClass::initMapWithPoints(const pcl::PointCloud<PointType>::Ptr &beam_in, const pcl::PointCloud<PointType>::Ptr &pillar_in, const pcl::PointCloud<PointType>::Ptr &facade_in)
{
    *laserCloudFacadeMap += *facade_in;
    *laserCloudBeamMap += *beam_in;
    *laserCloudPillarMap += *pillar_in;
    optimization_count = 12;
}

/**
 * 更新点到地图
 * 1、计算scan2scan的里程计
 * 2、更新持久性点到地图，删除临时点
 */
void Odom_BPF_EstimationClass::updatePointsToMap(const pcl::PointCloud<PointType>::Ptr &beam_in, const pcl::PointCloud<PointType>::Ptr &pillar_in, const pcl::PointCloud<PointType>::Ptr &facade_in)
{

    if (optimization_count > 2)
        optimization_count--;

    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    pcl::PointCloud<PointType>::Ptr downsampledBeamCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr downsampledPillarCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr downsampledFacadeCloud(new pcl::PointCloud<PointType>());
    downSamplingToMap(beam_in, downsampledBeamCloud, downSizeFilterBeam);
    downSamplingToMap(pillar_in, downsampledPillarCloud, downSizeFilterPillar);
    downSamplingToMap(facade_in, downsampledFacadeCloud, downSizeFilterFacade);
    // ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
    if (laserCloudBeamMap->points.size() > 10 && laserCloudPillarMap->points.size() > 10 && laserCloudFacadeMap->points.size() > 50)
    {
        kdtreeBeamMap->setInputCloud(laserCloudBeamMap);
        kdtreePillarMap->setInputCloud(laserCloudPillarMap);
        kdtreeFacadeMap->setInputCloud(laserCloudFacadeMap);

        for (int iterCount = 0; iterCount < optimization_count; iterCount++)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);

            problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

            addBeamCostFactor(downsampledBeamCloud, laserCloudBeamMap, problem, loss_function);
            addPillarCostFactor(downsampledPillarCloud, laserCloudPillarMap, problem, loss_function);
            addFacadeCostFactor(downsampledFacadeCloud, laserCloudFacadeMap, problem, loss_function);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);
        }
    }
    else
    {
        printf("not enough points in map to associate, map error");
    }
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    addPointsToMap(downsampledBeamCloud, downsampledPillarCloud, downsampledFacadeCloud);
    mergeFeatures(1);
}

void Odom_BPF_EstimationClass::addBeamCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int corner_num = 0;
    std::vector<edgeInfo> edge_valid_vec;
    std::vector<double> pointSparsity_vec;
    std::vector<double> observe_vec;
    edgeInfo tmp_edge_info;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        bool point_valid = false;
        PointType point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeBeamMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[4] < 1.0) // 保证邻居点距离足够近
        {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0); // 邻居点质心
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d neigh_p(map_in->points[pointSearchInd[j]].x,
                                        map_in->points[pointSearchInd[j]].y,
                                        map_in->points[pointSearchInd[j]].z);
                center = center + neigh_p;
                nearCorners.push_back(neigh_p);
            }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero(); // 协方差矩阵
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); // 存储了主成分方向 ———— 最大特征值对应的特征向量
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);

            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b; // 在穿过质心的，最大特征值方向上的两个点
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                float observe = (map_in->points[pointSearchInd[0]].g +
                                 map_in->points[pointSearchInd[1]].g +
                                 map_in->points[pointSearchInd[2]].g +
                                 map_in->points[pointSearchInd[3]].g +
                                 map_in->points[pointSearchInd[4]].g) /
                                    5.0 +
                                1;
                float round = (map_in->points[pointSearchInd[0]].r +
                               map_in->points[pointSearchInd[1]].r +
                               map_in->points[pointSearchInd[2]].r +
                               map_in->points[pointSearchInd[3]].r +
                               map_in->points[pointSearchInd[4]].r) /
                              5.0;
                for (int j = 0; j < 5; j++)
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);

                if (observe / round > 5)
                    observe = 255;
                if (observe < round * theta_p_edge && round > k_new_edge && observe < theta_max_edge)
                {
                    continue;
                }
                pc_in->points[i].r = std::min(255, int(round));
                pc_in->points[i].g = std::min(255, int(observe));

                corner_num++;

                point_valid = true;
                tmp_edge_info.curr_point = curr_point;
                tmp_edge_info.point_a = point_a;
                tmp_edge_info.point_b = point_b;
                tmp_edge_info.observe = observe;
                tmp_edge_info.round = round;
                edge_valid_vec.emplace_back(tmp_edge_info);

                float sum = 0;
                Eigen::Vector3d centerNeigh(0, 0, 0);
                std::vector<Eigen::Vector3d> p_neighbors;
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d p_Neighbor(
                        map_in->points[pointSearchInd[j]].x,
                        map_in->points[pointSearchInd[j]].y,
                        map_in->points[pointSearchInd[j]].z);
                    p_neighbors.emplace_back(p_Neighbor);
                    centerNeigh = centerNeigh + p_Neighbor;
                }
                centerNeigh /= 5;
                for (auto p : p_neighbors)
                {
                    sum += (centerNeigh - p).norm();
                }
                sum /= 5.0;
                pointSparsity_vec.emplace_back(sum);
            }
        }
    }
    if (weightType == 1 || weightType == 12)
    {

        for (int i = 0; i < (int)edge_valid_vec.size(); i++)
        {
            observe_vec.emplace_back(edge_valid_vec[i].observe);
        }
        observeMean(observe_vec);
    }
    if (weightType == 2 || weightType == 12)
    {
        pointSparsityMean(pointSparsity_vec);
    }

    for (int i = 0; i < (int)edge_valid_vec.size(); i++)
    {
        Eigen::Vector3d curr_point, point_a, point_b;
        float observe, round;
        curr_point = edge_valid_vec[i].curr_point;
        ;
        point_a = edge_valid_vec[i].point_a;
        point_b = edge_valid_vec[i].point_b;
        observe = edge_valid_vec[i].observe;
        round = edge_valid_vec[i].round;
        ceres::CostFunction *cost_function;
        if (weightType == 0)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
        else if (weightType == 1)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, observe_vec[i]);
        else if (weightType == 2)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, pointSparsity_vec[i]);
        else if (weightType == 12)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, (pointSparsity_vec[i] + observe_vec[i]) / 2);
        else
            ROS_ERROR("STH WRONG!");

        problem.AddResidualBlock(cost_function, loss_function, parameters);
    }
    edge_valid_vec.clear();
    if (corner_num < 20)
    {
        ROS_ERROR("not enough Beam points");
    }
}

void Odom_BPF_EstimationClass::addPillarCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int corner_num = 0;
    std::vector<edgeInfo> edge_valid_vec;
    std::vector<double> pointSparsity_vec;
    std::vector<double> observe_vec;
    edgeInfo tmp_edge_info;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        bool point_valid = false;
        PointType point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreePillarMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[4] < 1.0) // 保证邻居点距离足够近
        {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0); // 邻居点质心
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d neigh_p(map_in->points[pointSearchInd[j]].x,
                                        map_in->points[pointSearchInd[j]].y,
                                        map_in->points[pointSearchInd[j]].z);
                center = center + neigh_p;
                nearCorners.push_back(neigh_p);
            }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero(); // 协方差矩阵
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); // 存储了主成分方向 ———— 最大特征值对应的特征向量
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);

            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b; // 在穿过质心的，最大特征值方向上的两个点
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                float observe = (map_in->points[pointSearchInd[0]].g +
                                 map_in->points[pointSearchInd[1]].g +
                                 map_in->points[pointSearchInd[2]].g +
                                 map_in->points[pointSearchInd[3]].g +
                                 map_in->points[pointSearchInd[4]].g) /
                                    5.0 +
                                1;
                float round = (map_in->points[pointSearchInd[0]].r +
                               map_in->points[pointSearchInd[1]].r +
                               map_in->points[pointSearchInd[2]].r +
                               map_in->points[pointSearchInd[3]].r +
                               map_in->points[pointSearchInd[4]].r) /
                              5.0;
                for (int j = 0; j < 5; j++)
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);

                if (observe / round > 5)
                    observe = 255;
                if (observe < round * theta_p_edge && round > k_new_edge && observe < theta_max_edge)
                {
                    continue;
                }
                pc_in->points[i].r = std::min(255, int(round));
                pc_in->points[i].g = std::min(255, int(observe));

                corner_num++;

                point_valid = true;
                tmp_edge_info.curr_point = curr_point;
                tmp_edge_info.point_a = point_a;
                tmp_edge_info.point_b = point_b;
                tmp_edge_info.observe = observe;
                tmp_edge_info.round = round;
                edge_valid_vec.emplace_back(tmp_edge_info);

                float sum = 0;
                Eigen::Vector3d centerNeigh(0, 0, 0);
                std::vector<Eigen::Vector3d> p_neighbors;
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d p_Neighbor(
                        map_in->points[pointSearchInd[j]].x,
                        map_in->points[pointSearchInd[j]].y,
                        map_in->points[pointSearchInd[j]].z);
                    p_neighbors.emplace_back(p_Neighbor);
                    centerNeigh = centerNeigh + p_Neighbor;
                }
                centerNeigh /= 5;
                for (auto p : p_neighbors)
                {
                    sum += (centerNeigh - p).norm();
                }
                sum /= 5.0;
                pointSparsity_vec.emplace_back(sum);
            }
        }
    }
    if (weightType == 1 || weightType == 12)
    {

        for (int i = 0; i < (int)edge_valid_vec.size(); i++)
        {
            observe_vec.emplace_back(edge_valid_vec[i].observe);
        }
        observeMean(observe_vec);
    }
    if (weightType == 2 || weightType == 12)
    {
        pointSparsityMean(pointSparsity_vec);
    }

    for (int i = 0; i < (int)edge_valid_vec.size(); i++)
    {
        Eigen::Vector3d curr_point, point_a, point_b;
        float observe, round;
        curr_point = edge_valid_vec[i].curr_point;
        ;
        point_a = edge_valid_vec[i].point_a;
        point_b = edge_valid_vec[i].point_b;
        observe = edge_valid_vec[i].observe;
        round = edge_valid_vec[i].round;
        ceres::CostFunction *cost_function;
        if (weightType == 0)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
        else if (weightType == 1)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, observe_vec[i]);
        else if (weightType == 2)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, pointSparsity_vec[i]);
        else if (weightType == 12)
            cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, (pointSparsity_vec[i] + observe_vec[i]) / 2);
        else
            ROS_ERROR("STH WRONG!");

        problem.AddResidualBlock(cost_function, loss_function, parameters);
    }
    edge_valid_vec.clear();
    if (corner_num < 20)
    {
        ROS_ERROR("not enough Pillar points");
    }
}

void Odom_BPF_EstimationClass::addFacadeCostFactor(const pcl::PointCloud<PointType>::Ptr &surf_cloud, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    std::vector<surfInfo> surf_valid_vec;
    std::vector<double> pointSparsity_vec;
    std::vector<double> observe_vec;
    surfInfo tmp_surf_info;
    int surf_num = 0;
    for (int i = 0; i < (int)surf_cloud->points.size(); i++)
    {
        PointType point_temp;
        pointAssociateToMap(&(surf_cloud->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeFacadeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis); // scan-to-map 的特征匹配

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {

            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm(); // 法向量模长倒数？
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(surf_cloud->points[i].x, surf_cloud->points[i].y, surf_cloud->points[i].z);
            if (planeValid)
            {
                float observe = (map_in->points[pointSearchInd[0]].g +
                                 map_in->points[pointSearchInd[1]].g +
                                 map_in->points[pointSearchInd[2]].g +
                                 map_in->points[pointSearchInd[3]].g +
                                 map_in->points[pointSearchInd[4]].g) /
                                    5.0 +
                                1; // observe 是  p-Index
                float round = (map_in->points[pointSearchInd[0]].r +
                               map_in->points[pointSearchInd[1]].r +
                               map_in->points[pointSearchInd[2]].r +
                               map_in->points[pointSearchInd[3]].r +
                               map_in->points[pointSearchInd[4]].r) /
                              5.0;
                for (int j = 0; j < 5; j++)
                {
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);
                }
                if (observe / round > 5)
                    observe = 255;
                if (observe < round * theta_p_surf && round > k_new_surf && observe < theta_max_surf)
                {
                    continue;
                }
                surf_cloud->points[i].r = std::min(255, int(round));
                surf_cloud->points[i].g = std::min(255, int(observe));

                surf_num++;
                tmp_surf_info.curr_point = curr_point;
                tmp_surf_info.norm = norm;
                tmp_surf_info.negative_OA_dot_norm = negative_OA_dot_norm;
                tmp_surf_info.observe = observe;
                tmp_surf_info.round = round;
                surf_valid_vec.emplace_back(tmp_surf_info);
                float sum = 0;

                Eigen::Vector3d centerNeigh(0, 0, 0);
                std::vector<Eigen::Vector3d> p_neighbors;
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d p_Neighbor(
                        map_in->points[pointSearchInd[j]].x,
                        map_in->points[pointSearchInd[j]].y,
                        map_in->points[pointSearchInd[j]].z);
                    p_neighbors.emplace_back(p_Neighbor);
                    centerNeigh = centerNeigh + p_Neighbor;
                }
                centerNeigh /= 5;
                for (auto p : p_neighbors)
                {
                    sum += (centerNeigh - p).norm();
                }
                sum /= 5.0;
                pointSparsity_vec.emplace_back(sum);
            }
        }
    }
    if (weightType == 1 || weightType == 12)
    {
        for (int i = 0; i < (int)surf_valid_vec.size(); i++)
        {
            observe_vec.emplace_back(surf_valid_vec[i].observe);
        }
        observeMean(observe_vec);
    }
    if (weightType == 2 || weightType == 12)
    {
        pointSparsityMean(pointSparsity_vec);
    }

    for (int i = 0; i < (int)surf_valid_vec.size(); i++)
    {
        Eigen::Vector3d curr_point, norm;
        float observe, round, negative_OA_dot_norm;
        curr_point = surf_valid_vec[i].curr_point;
        norm = surf_valid_vec[i].norm;
        negative_OA_dot_norm = surf_valid_vec[i].negative_OA_dot_norm;
        observe = surf_valid_vec[i].observe;
        round = surf_valid_vec[i].round;
        ceres::CostFunction *cost_function;
        if (weightType == 0)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
        else if (weightType == 1)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, observe_vec[i]);
        else if (weightType == 2)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pointSparsity_vec[i]);
        else if (weightType == 12)
            cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, (observe_vec[i] + pointSparsity_vec[i]) / 2);
        else
            ROS_ERROR("STH WRONG!");

        problem.AddResidualBlock(cost_function, loss_function, parameters);
    }
    surf_valid_vec.clear();

    if (surf_num < 20)
    {
        printf("not enough correct points");
    }
}

/**
 * 融合当前帧与地图
 * 1、将点云按照位子变换加入地图
 * 2、更新地图维护边界
 * 3、裁剪点云，至地图边界内
 * 4、根据体素筛选点云，计算每个体素的质心
 * 4、提取稳定点
 * 5、若点云足够多，变为红色？
 */
void Odom_BPF_EstimationClass::addPointsToMap(const pcl::PointCloud<PointType>::Ptr &downsampledBeamCloud, const pcl::PointCloud<PointType>::Ptr &downsampledPillarCloud, const pcl::PointCloud<PointType>::Ptr &downsampledFacadeCloud)
{

    for (auto b : *downsampledBeamCloud)
    {
        PointType point_temp;
        pointAssociateToMap(&b, &point_temp);
        laserCloudBeamMap->push_back(point_temp);
    }

    for (auto p : *downsampledPillarCloud)
    {
        PointType point_temp;
        pointAssociateToMap(&p, &point_temp);
        laserCloudPillarMap->push_back(point_temp);
    }
    for (auto f : *downsampledFacadeCloud)
    {
        PointType point_temp;
        pointAssociateToMap(&f, &point_temp);
        laserCloudFacadeMap->push_back(point_temp);
    }

    double x_min = +odom.translation().x() - 100;
    double y_min = +odom.translation().y() - 100;
    double z_min = +odom.translation().z() - 100;
    double x_max = +odom.translation().x() + 100;
    double y_max = +odom.translation().y() + 100;
    double z_max = +odom.translation().z() + 100;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);

    pcl::PointCloud<PointType>::Ptr tmpPillar(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr tmpBeam(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr tmpFacade(new pcl::PointCloud<PointType>());
    cropBoxFilter.setInputCloud(laserCloudBeamMap);
    cropBoxFilter.filter(*tmpBeam);
    cropBoxFilter.setInputCloud(laserCloudPillarMap);
    cropBoxFilter.filter(*tmpPillar);
    cropBoxFilter.setInputCloud(laserCloudFacadeMap);
    cropBoxFilter.filter(*tmpFacade);

    laserCloudFacadeMap = rgbds(tmpFacade, map_resolution * 2);
    laserCloudPillarMap = rgbds(tmpPillar, map_resolution);
    laserCloudBeamMap = rgbds(tmpBeam, map_resolution);
    // downSizeFilterSurf.setInputCloud(tmpSurf);
    // downSizeFilterSurf.filter(*laserCloudSurfMap);
    // downSizeFilterEdge.setInputCloud(tmpCorner);
    // downSizeFilterEdge.filter(*laserCloudCornerMap);
    extractstablepoint(laserCloudFacadeMap, k_new_surf, theta_p_surf, theta_max_surf);
    extractstablepoint(laserCloudPillarMap, k_new_edge, theta_p_edge, theta_max_edge);
    extractstablepoint(laserCloudBeamMap, k_new_edge, theta_p_edge, theta_max_edge);
    // 更新点的持久帧次
    for (int i = 0; i < laserCloudFacadeMap->points.size(); i++)
    {
        if (laserCloudFacadeMap->points[i].r > 250)
            laserCloudFacadeMap->points[i].r = 255;
        else
            laserCloudFacadeMap->points[i].r += 2;
    }

    for (int i = 0; i < laserCloudBeamMap->points.size(); i++)
    {
        if (laserCloudBeamMap->points[i].r > 250)
            laserCloudBeamMap->points[i].r = 255;
        else
            laserCloudBeamMap->points[i].r += 2;
    }
    for (int i = 0; i < laserCloudPillarMap->points.size(); i++)
    {
        if (laserCloudPillarMap->points[i].r > 250)
            laserCloudPillarMap->points[i].r = 255;
        else
            laserCloudPillarMap->points[i].r += 2;
    }
}

void Odom_BPF_EstimationClass::mergeFeatures(int mergeGround = 1)
{
    laserCloudMergeMap.reset(new pcl::PointCloud<PointType>());
    *laserCloudMergeMap += *laserCloudBeamMap;
    *laserCloudMergeMap += *laserCloudFacadeMap;
    *laserCloudMergeMap += *laserCloudPillarMap;


 
}