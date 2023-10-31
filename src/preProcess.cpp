
#include "preProcess.hpp"

bool groundSeg::random_downsample_pcl(typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in_out, int keep_number)
{
    if (cloud_in_out->points.size() <= keep_number)
        return false;
    else
    {
        if (keep_number == 0)
        {
            cloud_in_out.reset(new typename pcl::PointCloud<pointTypeNormal>());
            return false;
        }
        else
        {
            typename pcl::PointCloud<pointTypeNormal>::Ptr cloud_temp(new pcl::PointCloud<pointTypeNormal>);
            pcl::RandomSample<pointTypeNormal> ran_sample(true); // Extract removed indices
            ran_sample.setInputCloud(cloud_in_out);
            ran_sample.setSample(keep_number);
            ran_sample.filter(*cloud_temp);
            cloud_temp->points.swap(cloud_in_out->points);
            return true;
        }
    }
}

bool groundSeg::random_downsample_pcl(typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in,
                                       typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_out, int keep_number)
{
    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

    if (cloud_in->points.size() <= keep_number)
    {
        cloud_out = cloud_in;
        return false;
    }
    else
    {
        if (keep_number == 0)
            return false;
        else
        {
            pcl::RandomSample<pointTypeNormal> ran_sample(true); // Extract removed indices
            ran_sample.setInputCloud(cloud_in);
            ran_sample.setSample(keep_number);
            ran_sample.filter(*cloud_out);
            std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
            // LOG(INFO) << "Random downsampling done in [" << time_used.count() * 1000.0 << "] ms.";
            return true;
        }
    }
}

template <typename PointT>
bool groundSeg::fast_ground_filter(const typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in,
                                    typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_ground,
                                    typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_ground_down,
                                    typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_unground,
                                    typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_curb,
                                    int min_grid_pt_num, float grid_resolution, float max_height_difference,
                                    float neighbor_height_diff, float max_ground_height,
                                    int ground_random_down_rate, int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
                                    int estimate_ground_normal_method, float normal_estimation_radius, // estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
                                    int distance_weight_downsampling_method, float standard_distance,  // standard distance: the distance where the distance_weight is 1
                                    bool fixed_num_downsampling = false, int down_ground_fixed_num = 1000,
                                    bool detect_curb_or_not = false, float intensity_thre = FLT_MAX,
                                    bool apply_grid_wise_outlier_filter = false, float outlier_std_scale = 3.0) // current intensity_thre is for kitti dataset (TODO: disable it)
{
    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

    PrincipleComponentAnalysis<PointT> pca_estimator;

    typename pcl::PointCloud<pointTypeNormal>::Ptr cloud_ground_full(new pcl::PointCloud<pointTypeNormal>());

    int reliable_grid_pts_count_thre = min_grid_pt_num - 1;
    int count_checkpoint = 0;
    float sum_height = 0.001;
    float appro_mean_height;
    float min_ground_height = max_ground_height;
    float underground_noise_thre = -FLT_MAX;
    float non_ground_height_thre;
    float distance_weight;
    // int ground_random_down_rate_temp = ground_random_down_rate;
    // int nonground_random_down_rate_temp = nonground_random_down_rate;

    // For some points,  calculating the approximate mean height
    // 计算近似的点高度 减少计算量
    for (int j = 0; j < cloud_in->points.size(); j++)
    {
        if (j % 100 == 0)
        {
            sum_height += cloud_in->points[j].z;
            count_checkpoint++;
        }
    }
    appro_mean_height = sum_height / count_checkpoint;

    non_ground_height_thre = appro_mean_height + max_ground_height;
    // sometimes, there would be some underground ghost points (noise), however, these points would be removed by scanner filter
    // float underground_noise_thre = appro_mean_height - max_ground_height;  // this is a keyparameter.

    bounds_t bounds;
    centerpoint_t center_pt;
    get_cloud_bbx_cpt(cloud_in, bounds, center_pt); // Inherited from its parent class, use this->

    // Construct Grid
    int row, col, num_grid;
    row = ceil((bounds.max_y - bounds.min_y) / grid_resolution);
    col = ceil((bounds.max_x - bounds.min_x) / grid_resolution);
    num_grid = row * col;

    std::chrono::steady_clock::time_point toc_1_1 = std::chrono::steady_clock::now();

    grid_t *grid = new grid_t[num_grid];

    // Each grid  这个应该可以放在初始化本
    for (int i = 0; i < num_grid; i++)
    {
        grid[i].min_z = FLT_MAX;
        grid[i].neighbor_min_z = FLT_MAX;
    }

    // Each point ---> determine the grid to which the point belongs
    for (int j = 0; j < cloud_in->points.size(); j++)
    {
        int temp_row, temp_col, temp_id;
        temp_col = floor((cloud_in->points[j].x - bounds.min_x) / grid_resolution);
        temp_row = floor((cloud_in->points[j].y - bounds.min_y) / grid_resolution);
        temp_id = temp_row * col + temp_col; // 当前点所属grid 的序号
        if (temp_id >= 0 && temp_id < num_grid)
        {
            if (distance_weight_downsampling_method > 0 && !grid[temp_id].pts_count)
            {
                grid[temp_id].dist2station = std::sqrt(cloud_in->points[j].x * cloud_in->points[j].x + cloud_in->points[j].y * cloud_in->points[j].y + cloud_in->points[j].z * cloud_in->points[j].z);
            }

            if (cloud_in->points[j].z > non_ground_height_thre)
            {
                distance_weight = 1.0 * standard_distance / (grid[temp_id].dist2station + 0.0001); // avoiding Floating point exception
                int nonground_random_down_rate_temp = nonground_random_down_rate;
                if (distance_weight_downsampling_method == 1) // linear weight
                    nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
                else if (distance_weight_downsampling_method == 2) // quadratic weight
                    nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);

                if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[j].intensity > intensity_thre)
                {
                    cloud_in->points[j].data[3] = cloud_in->points[j].z - (appro_mean_height - 3.0); // data[3] stores the approximate point height above ground
                    cloud_unground->points.push_back(cloud_in->points[j]);
                }
            }
            else if (cloud_in->points[j].z > underground_noise_thre)
            {
                grid[temp_id].pts_count++;
                grid[temp_id].point_id.push_back(j);
                if (cloud_in->points[j].z < grid[temp_id].min_z) //
                {
                    grid[temp_id].min_z = cloud_in->points[j].z;
                    grid[temp_id].neighbor_min_z = cloud_in->points[j].z;
                }
            }
        }
    }
    std::chrono::steady_clock::time_point toc_1_3 = std::chrono::steady_clock::now();

    // Each grid
    for (int m = 0; m < num_grid; m++)
    {
        int temp_row, temp_col;
        temp_row = m / col;
        temp_col = m % col;
        if (temp_row >= 1 && temp_row <= row - 2 && temp_col >= 1 && temp_col <= col - 2)
        {
            for (int j = -1; j <= 1; j++) // row
            {
                for (int k = -1; k <= 1; k++) // col
                {
                    grid[m].neighbor_min_z = std::min(grid[m].neighbor_min_z, grid[m + j * col + k].min_z);
                    if (grid[m + j * col + k].pts_count > reliable_grid_pts_count_thre)
                        grid[m].reliable_neighbor_grid_num++;
                }
            }
        }
    }

    double consuming_time_ransac = 0.0;

    std::chrono::steady_clock::time_point toc_1_4 = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<pointTypeNormal>::Ptr> grid_ground_pcs(num_grid);
    std::vector<typename pcl::PointCloud<pointTypeNormal>::Ptr> grid_unground_pcs(num_grid);
    for (int i = 0; i < num_grid; i++)
    {
        typename pcl::PointCloud<pointTypeNormal>::Ptr grid_ground_pc_temp(new pcl::PointCloud<pointTypeNormal>);
        grid_ground_pcs[i] = grid_ground_pc_temp;
        typename pcl::PointCloud<pointTypeNormal>::Ptr grid_unground_pc_temp(new pcl::PointCloud<pointTypeNormal>);
        grid_unground_pcs[i] = grid_unground_pc_temp;
    }

    std::chrono::steady_clock::time_point toc_1 = std::chrono::steady_clock::now();

    // For each grid
    omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
    for (int i = 0; i < num_grid; i++)
    {

        typename pcl::PointCloud<PointT>::Ptr grid_ground(new pcl::PointCloud<PointT>);
        // Filtering some grids with too little points
        if (grid[i].pts_count >= min_grid_pt_num && grid[i].reliable_neighbor_grid_num >= reliable_neighbor_grid_num_thre)
        {
            int ground_random_down_rate_temp = ground_random_down_rate;
            int nonground_random_down_rate_temp = nonground_random_down_rate;
            distance_weight = 1.0 * standard_distance / (grid[i].dist2station + 0.0001);
            if (distance_weight_downsampling_method == 1) // linear weight
            {
                ground_random_down_rate_temp = (int)(distance_weight * ground_random_down_rate + 1);
                nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
            }
            else if (distance_weight_downsampling_method == 2) // quadratic weight
            {
                ground_random_down_rate_temp = (int)(distance_weight * distance_weight * ground_random_down_rate + 1);
                nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);
            }
            // LOG(WARNING) << ground_random_down_rate_temp << "," << nonground_random_down_rate_temp;
            if (grid[i].min_z - grid[i].neighbor_min_z < neighbor_height_diff)
            {
                for (int j = 0; j < grid[i].point_id.size(); j++)
                {
                    if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre)
                    {
                        if (cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z < max_height_difference)
                        {
                            // cloud_ground_full->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                            if (estimate_ground_normal_method == 3)
                                grid_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                            else
                            {
                                if (j % ground_random_down_rate_temp == 0) // for example 10
                                {
                                    if (estimate_ground_normal_method == 0)
                                    {
                                        cloud_in->points[grid[i].point_id[j]].normal_x = 0.0;
                                        cloud_in->points[grid[i].point_id[j]].normal_y = 0.0;
                                        cloud_in->points[grid[i].point_id[j]].normal_z = 1.0;
                                    }
                                    grid_ground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                                    // cloud_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to ground points
                                }
                            }
                        }
                        else // inner grid unground points
                        {
                            if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre) // extract more points on signs and vehicle license plate
                            {
                                cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z; // data[3] stores the point height above ground
                                grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                                // cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to nonground points
                            }
                        }
                    }
                }
            }
            else // unground grid
            {
                for (int j = 0; j < grid[i].point_id.size(); j++)
                {
                    if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre &&
                        (j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre))
                    {
                        cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].neighbor_min_z; // data[3] stores the point height above ground
                        grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                        // cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to nonground points
                    }
                }
            }
            pcl::PointCloud<pointTypeNormal>().swap(*grid_ground);
        }
    }

    // combine the ground and unground points
    for (int i = 0; i < num_grid; i++)
    {
        cloud_ground->points.insert(cloud_ground->points.end(), grid_ground_pcs[i]->points.begin(), grid_ground_pcs[i]->points.end());
        cloud_unground->points.insert(cloud_unground->points.end(), grid_unground_pcs[i]->points.begin(), grid_unground_pcs[i]->points.end());
    }

    // free memory
    delete[] grid;

    std::chrono::steady_clock::time_point toc_2 = std::chrono::steady_clock::now();

    int normal_estimation_neighbor_k = 2 * min_grid_pt_num;
    pcl::PointCloud<pcl::Normal>::Ptr ground_normal(new pcl::PointCloud<pcl::Normal>);
    if (estimate_ground_normal_method == 1)
        pca_estimator.get_normal_pcar(cloud_ground, normal_estimation_radius, ground_normal);
    else if (estimate_ground_normal_method == 2)
        pca_estimator.get_normal_pcak(cloud_ground, normal_estimation_neighbor_k, ground_normal);

    for (int i = 0; i < cloud_ground->points.size(); i++)
    {
        if (estimate_ground_normal_method == 1 || estimate_ground_normal_method == 2)
        {
            cloud_ground->points[i].normal_x = ground_normal->points[i].normal_x;
            cloud_ground->points[i].normal_y = ground_normal->points[i].normal_y;
            cloud_ground->points[i].normal_z = ground_normal->points[i].normal_z;
        }
        if (!fixed_num_downsampling)
        {
            // LOG(INFO)<<cloud_ground->points[i].normal_x << "," << cloud_ground->points[i].normal_y << "," << cloud_ground->points[i].normal_z;
            if (i % ground_random_down_down_rate == 0)
                cloud_ground_down->points.push_back(cloud_ground->points[i]);
        }
    }

    if (fixed_num_downsampling)
        random_downsample_pcl(cloud_ground, cloud_ground_down, down_ground_fixed_num);

    pcl::PointCloud<pcl::Normal>().swap(*ground_normal);

    std::chrono::steady_clock::time_point toc_3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> ground_seg_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc_2 - tic);
    std::chrono::duration<double> ground_seg_prepare_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 - tic);
    std::chrono::duration<double> ground_normal_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc_3 - toc_2);

    std::cout << "Ground: [" << cloud_ground->points.size() << " | " << cloud_ground_down->points.size() << "] Unground: [" << cloud_unground->points.size() << "]." << std::endl;

    if (estimate_ground_normal_method == 3)
    {
        // LOG(INFO) << "Ground segmentation done in [" << ground_seg_time.count() * 1000.0 - consuming_time_ransac << "] ms.";
        // LOG(INFO) << "Ground Normal Estimation done in [" << consuming_time_ransac << "] ms.";
        std::cout << "Ground segmentation and normal estimation in [" << ground_seg_time.count() * 1000.0 << "] ms."
                  << ",in which preparation costs [" << ground_seg_prepare_time.count() * 1000.0 << "] ms." << std::endl;
        // output detailed consuming time
        // LOG(INFO) << prepare_1.count() * 1000.0 << "," << prepare_2.count() * 1000.0 << "," << prepare_3.count() * 1000.0 << "," << prepare_4.count() * 1000.0 << "," << prepare_5.count() * 1000.0;
    }
    else
    {
        std::cout << "Ground segmentation done in [" << ground_seg_time.count() * 1000.0 << "] ms." << std::endl;
        std::cout << "Ground Normal Estimation done in [" << ground_normal_time.count() * 1000.0 << "] ms."
                  << " preparation in [" << ground_seg_prepare_time.count() * 1000.0 << "] ms." << std::endl;
    }
    return 1;
}

