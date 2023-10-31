#include "common.hpp"

class structList
{
public:
    struct grid_t
    {
        std::vector<int> point_id;
        float min_z;
        float max_z;
        float delta_z;
        float min_z_x; // X of Lowest Point in the Voxel;
        float min_z_y; // Y of Lowest Point in the Voxel;
        float min_z_outlier_thre;
        float neighbor_min_z;
        int pts_count;
        int reliable_neighbor_grid_num;
        float mean_z;
        float dist2station;

        grid_t()
        {
            min_z = min_z_x = min_z_y = neighbor_min_z = mean_z = 0.f;
            pts_count = 0;
            reliable_neighbor_grid_num = 0;
            delta_z = 0.0;
            dist2station = 0.001;
            min_z_outlier_thre = -FLT_MAX;
        }
    };
    struct centerpoint_t
    {
        double x;
        double y;
        double z;
        centerpoint_t(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    };

    // regular bounding box whose edges are parallel to x,y,z axises
    struct bounds_t
    {
        double min_x;
        double min_y;
        double min_z;
        double max_x;
        double max_y;
        double max_z;
        int type;

        bounds_t()
        {
            min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
        }
        void inf_x()
        {
            min_x = -DBL_MAX;
            max_x = DBL_MAX;
        }
        void inf_y()
        {
            min_y = -DBL_MAX;
            max_y = DBL_MAX;
        }
        void inf_z()
        {
            min_z = -DBL_MAX;
            max_z = DBL_MAX;
        }
        void inf_xyz()
        {
            inf_x();
            inf_y();
            inf_z();
        }
    };
};
struct eigenvalue_t // Eigen Value ,lamada1 > lamada2 > lamada3;
{
    double lamada1;
    double lamada2;
    double lamada3;
};

struct eigenvector_t // the eigen vector corresponding to the eigen value
{
    Eigen::Vector3f principalDirection;
    Eigen::Vector3f middleDirection;
    Eigen::Vector3f normalDirection;
};

struct pca_feature_t // PCA
{
    eigenvalue_t values;
    eigenvector_t vectors;
    double curvature;
    double linear;
    double planar;
    double spherical;
    double linear_2;
    double planar_2;
    double spherical_2;
    double normal_diff_ang_deg;
    pcl::PointNormal pt;
    int ptId;
    int pt_num = 0;
    std::vector<int> neighbor_indices;
    std::vector<bool> close_to_query_point;
};

template <typename PointT>
class PrincipleComponentAnalysis
{
public:
    /**
     * \brief Estimate the normals of the input Point Cloud by PCL speeding up with OpenMP
     * \param[in] in_cloud is the input Point Cloud Pointer
     * \param[in] radius is the neighborhood search radius (m) for KD Tree
     * \param[out] normals is the normal of all the points from the Point Cloud
     */
    bool get_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                         float radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // Create the normal estimation class, and pass the input dataset to it;
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setNumberOfThreads(omp_get_max_threads()); // More threads sometimes would not speed up the procedure
        ne.setInputCloud(in_cloud);
        // Create an empty kd-tree representation, and pass it to the normal estimation object;
        typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

        ne.setSearchMethod(tree);
        // Use all neighbors in a sphere of radius;
        ne.setRadiusSearch(radius);
        // Compute the normal
        ne.compute(*normals);
        check_normal(normals);
        return true;
    }

    bool get_pc_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                            float radius,
                            pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        bool normal_ready = get_normal_pcar(in_cloud, radius, normals);
        if (normal_ready)
        {
            // Concatenate the XYZ and normal fields*
            pcl::concatenateFields(*in_cloud, *normals, *pointnormals);
            return true;
        }
        else
            return false;
    }

    bool get_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                         int K,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // Create the normal estimation class, and pass the input dataset to it;
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setNumberOfThreads(omp_get_max_threads()); // More threads sometimes would not speed up the procedure
        ne.setInputCloud(in_cloud);
        // Create an empty kd-tree representation, and pass it to the normal estimation object;
        typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setSearchMethod(tree);
        // Use all neighbors in a sphere of radius;
        ne.setKSearch(K);
        // Compute the normal
        ne.compute(*normals);
        check_normal(normals);
        return true;
    }

    bool get_pc_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                            int K,
                            pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        bool normal_ready = get_normal_pcak(in_cloud, K, normals);
        if (normal_ready)
        {
            // Concatenate the XYZ and normal fields*
            pcl::concatenateFields(*in_cloud, *normals, *pointnormals);
            return true;
        }
        else
            return false;
    }

    /**
     * \brief Principle Component Analysis (PCA) of the Point Cloud with fixed search radius
     * \param[in] in_cloud is the input Point Cloud (XYZI) Pointer
     * \param[in]     radius is the neighborhood search radius (m) for KD Tree
     * \param[out]features is the pca_feature_t vector of all the points from the Point Cloud
     */
    // R - K neighborhood (with already built-kd tree)
    // within the radius, we would select the nearest K points for calculating PCA

    bool get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                            std::vector<pca_feature_t> &features, typename pcl::KdTreeFLANN<PointT>::Ptr &tree,
                            float radius, int nearest_k, int min_k = 1)
    {
        // LOG(INFO) << "[" << in_cloud->points.size() << "] points used for PCA, pca down rate is [" << pca_down_rate << "]";
        features.resize(in_cloud->points.size());

        omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for                                  // Multi-thread
        for (int i = 0; i < in_cloud->points.size(); i++) // faster way
        {
            std::vector<int> search_indices_used; // points would be stored in sequence (from the closest point to the farthest point within the neighborhood)
            std::vector<int> search_indices;      // point index vector
            std::vector<float> squared_distances; // distance vector

            float neighborhood_r = radius;
            int neighborhood_k = nearest_k;

            tree->radiusSearch(i, neighborhood_r, search_indices, squared_distances, neighborhood_k); // TODO： 插入方法

            features[i].pt.x = in_cloud->points[i].x;
            features[i].pt.y = in_cloud->points[i].y;
            features[i].pt.z = in_cloud->points[i].z;
            features[i].ptId = i;
            features[i].pt_num = search_indices.size();

            // deprecated
            features[i].close_to_query_point.resize(search_indices.size());
            for (int j = 0; j < search_indices.size(); j++)
            {
                if (squared_distances[j] < 0.64 * radius * radius) // 0.5^(2/3)
                    features[i].close_to_query_point[j] = true;
                else
                    features[i].close_to_query_point[j] = false;
            }

            get_pca_feature(in_cloud, search_indices, features[i]); // 这步将修改featrus中的数据

            if (features[i].pt_num > min_k)
                assign_normal(in_cloud->points[i], features[i]); // 此处的点云注册  默认为平面点？

            std::vector<int>().swap(search_indices);
            std::vector<int>().swap(search_indices_used);
            std::vector<float>().swap(squared_distances);
        }
        //}
        return true;
    }

    void calculate_normal_inconsistency(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                        std::vector<pca_feature_t> &features)
    {
        for (int i = 0; i < in_cloud->points.size(); i++)
        {
            double n_x = 0, n_y = 0, n_z = 0;

            for (int j = 0; j < features[i].neighbor_indices.size(); j++)
            {
                n_x += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_x);
                n_y += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_y);
                n_z += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_z);
            }

            Eigen::Vector3d n_mean;
            Eigen::Vector3d n_self;
            n_mean << n_x / features[i].pt_num, n_y / features[i].pt_num, n_z / features[i].pt_num;
            n_mean.normalize();

            n_self << in_cloud->points[i].normal_x, in_cloud->points[i].normal_y, in_cloud->points[i].normal_z;

            features[i].normal_diff_ang_deg = 180.0 / M_PI * std::acos(std::abs(n_mean.dot(n_self))); // n1.norm()=n2.norm()=1
                                                                                                      // if (i % 10 == 0)
                                                                                                      // 	LOG(INFO) << features[i].normal_diff_ang_deg;
        }
    }

    /**
     * \brief Use PCL to accomplish the Principle Component Analysis (PCA)
     * of one point and its neighborhood
     * \param[in] in_cloud is the input Point Cloud Pointer
     * \param[in] search_indices is the neighborhood points' indices of the search point.
     * \param[out]feature is the pca_feature_t of the search point.
     */
    bool get_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                         std::vector<int> &search_indices,
                         pca_feature_t &feature)
    {
        int pt_num = search_indices.size();

        if (pt_num <= 3)
            return false;

        typename pcl::PointCloud<PointT>::Ptr selected_cloud(new pcl::PointCloud<PointT>());
        for (int i = 0; i < pt_num; ++i)
            selected_cloud->points.push_back(in_cloud->points[search_indices[i]]);

        pcl::PCA<PointT> pca_operator; // pcl点云库中的自带PCA方法
        pca_operator.setInputCloud(selected_cloud);

        // Compute eigen values and eigen vectors
        Eigen::Matrix3f eigen_vectors = pca_operator.getEigenVectors();
        Eigen::Vector3f eigen_values = pca_operator.getEigenValues();

        feature.vectors.principalDirection = eigen_vectors.col(0);
        feature.vectors.normalDirection = eigen_vectors.col(2);

        feature.vectors.principalDirection.normalize();
        feature.vectors.normalDirection.normalize();

        feature.values.lamada1 = eigen_values(0);
        feature.values.lamada2 = eigen_values(1);
        feature.values.lamada3 = eigen_values(2);

        if ((feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3) == 0)
            feature.curvature = 0;
        else
            feature.curvature = feature.values.lamada3 / (feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3);

        feature.linear_2 = ((feature.values.lamada1) - (feature.values.lamada2)) / (feature.values.lamada1);
        feature.planar_2 = ((feature.values.lamada2) - (feature.values.lamada3)) / (feature.values.lamada1);
        feature.spherical_2 = (feature.values.lamada3) / (feature.values.lamada1);

        search_indices.swap(feature.neighbor_indices);
        return true;
    }

    // is_palne_feature (true: assign point normal as pca normal vector, false: assign point normal as pca primary direction vector)
    bool assign_normal(PointT &pt, pca_feature_t &pca_feature, bool is_plane_feature = true)
    {
        if (is_plane_feature)
        {
            pt.normal_x = pca_feature.vectors.normalDirection.x();
            pt.normal_y = pca_feature.vectors.normalDirection.y();
            pt.normal_z = pca_feature.vectors.normalDirection.z();
            pt.normal[3] = pca_feature.planar_2; // planrity
        }
        else
        {
            pt.normal_x = pca_feature.vectors.principalDirection.x();
            pt.normal_y = pca_feature.vectors.principalDirection.y();
            pt.normal_z = pca_feature.vectors.principalDirection.z();
            pt.normal[3] = pca_feature.linear_2; // linarity
        }
        return true;
    }

protected:
private:
    /**
     * \brief Check the Normals (if they are finite)
     * \param normals is the input Point Cloud (XYZI)'s Normal Pointer
     */
    void check_normal(pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // It is advisable to check the normals before the call to compute()
        for (int i = 0; i < normals->points.size(); i++)
        {
            if (!pcl::isFinite<pcl::Normal>(normals->points[i]))
            {
                normals->points[i].normal_x = 0.577; // 1/ sqrt(3)
                normals->points[i].normal_y = 0.577;
                normals->points[i].normal_z = 0.577;
                // normals->points[i].curvature = 0.0;
            }
        }
    }
};

class groundSeg : public structList
{
public:
    groundSeg(ros::NodeHandle nh) : nh_(nh){};
    ~groundSeg(){};
    void groundInit(pointTypeCloud::Ptr &inputCloud, std_msgs::Header header)
    {
        groundSeginputCloudPtr.reset(new pointTypeCloud());
        groundCloudPtr.reset(new pointTypeCloud());
        nonGroundCloudPtr.reset(new pointTypeCloud());
        cloudHeader = header;
        groundSeginputCloudPtr = inputCloud;
    }

    void groundPubCloud()
    {
        sensor_msgs::PointCloud2 groundlaserCloudMsgOut;
        pcl::toROSMsg(*groundCloudPtr, groundlaserCloudMsgOut);
        groundlaserCloudMsgOut.header = cloudHeader;
        pubGround.publish(groundlaserCloudMsgOut);

        sensor_msgs::PointCloud2 nonGroundlaserCloudMsgOut;
        pcl::toROSMsg(*nonGroundCloudPtr, nonGroundlaserCloudMsgOut);
        nonGroundlaserCloudMsgOut.header = cloudHeader;
        pubNonGround.publish(nonGroundlaserCloudMsgOut);
    }
    bool random_downsample_pcl(typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in_out, int keep_number);
    bool random_downsample_pcl(typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in,
                               typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_out, int keep_number);

    bool ground_seg(pcl::PointCloud<pointType>::Ptr &cloud_in,
                    pcl::PointCloud<pointType>::Ptr &cloud_ground,
                    pcl::PointCloud<pointType>::Ptr &cloud_unground,
                    int min_grid_pt_num, float grid_resolution, float max_height_difference, float neighbor_height_diff,
                    float max_ground_height, float min_ground_height)
    {

        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

        bounds_t bounds;
        centerpoint_t center_pt;
        this->get_cloud_bbx_cpt<pointType>(cloud_in, bounds, center_pt); // Inherited from its parent class, use this->

        // Construct Grid
        int row, col, num_grid;
        row = ceil((bounds.max_y - bounds.min_y) / grid_resolution);
        col = ceil((bounds.max_x - bounds.min_x) / grid_resolution);
        num_grid = row * col;

        grid_t *grid = new grid_t[num_grid];

        // Each grid
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
            temp_id = temp_row * col + temp_col;
            if (temp_id >= 0 && temp_id < num_grid)
            {
                grid[temp_id].pts_count++;
                if (cloud_in->points[j].z > max_ground_height)
                    cloud_unground->points.push_back(cloud_in->points[j]);
                else
                {
                    grid[temp_id].point_id.push_back(j);
                    if (cloud_in->points[j].z < grid[temp_id].min_z && cloud_in->points[j].z > min_ground_height)
                    {
                        grid[temp_id].min_z = cloud_in->points[j].z;
                        grid[temp_id].neighbor_min_z = cloud_in->points[j].z;
                    }
                }
            }
        }

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
                        if (grid[m].neighbor_min_z > grid[m + j * col + k].min_z)
                            grid[m].neighbor_min_z = grid[m + j * col + k].min_z;
                    }
                }
            }
        }

        // For each grid
        for (int i = 0; i < num_grid; i++)
        {
            // Filtering some grids with too little points
            if (grid[i].pts_count >= min_grid_pt_num)
            {
                if (grid[i].min_z - grid[i].neighbor_min_z < neighbor_height_diff)
                {
                    for (int j = 0; j < grid[i].point_id.size(); j++)
                    {
                        if (cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z < max_height_difference &&
                            cloud_in->points[grid[i].point_id[j]].z > min_ground_height)
                            cloud_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]); // Add to ground points
                        else
                            cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); // Add to nonground points
                    }
                }
                else
                {
                    for (int j = 0; j < grid[i].point_id.size(); j++)
                        cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); // Add to nonground points
                }
            }
        }

        // free memory
        delete[] grid;

        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();

        std::cout << "Ground: [" << cloud_ground->points.size() << "] Unground: [" << cloud_unground->points.size() << "]." << std::endl;
        std::chrono::duration<double> ground_seg_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

        std::cout << "Ground segmentation done in [" << ground_seg_time.count() * 1000.0 << "] ms." << std::endl;

        return 1;
    }

    template <typename PointT>
    bool fast_ground_filter(const typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_in,
                            typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_ground,
                            typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_ground_down,
                            typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_unground,
                            typename pcl::PointCloud<pointTypeNormal>::Ptr &cloud_curb,
                            int min_grid_pt_num, float grid_resolution, float max_height_difference,
                            float neighbor_height_diff, float max_ground_height,
                            int ground_random_down_rate, int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
                            int estimate_ground_normal_method, float normal_estimation_radius, // estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
                            int distance_weight_downsampling_method, float standard_distance,  // standard distance: the distance where the distance_weight is 1
                            bool fixed_num_downsampling, int down_ground_fixed_num,
                            bool detect_curb_or_not, float intensity_thre,
                            bool apply_grid_wise_outlier_filter, float outlier_std_scale) // current intensity_thre is for kitti dataset (TODO: disable it)
        ;
    template <typename PointT>
    void get_cloud_bbx(const typename pcl::PointCloud<PointT>::Ptr &cloud, bounds_t &bound)
    {
        double min_x = DBL_MAX;
        double min_y = DBL_MAX;
        double min_z = DBL_MAX;
        double max_x = -DBL_MAX;
        double max_y = -DBL_MAX;
        double max_z = -DBL_MAX;

        for (int i = 0; i < cloud->points.size(); i++)
        {
            if (min_x > cloud->points[i].x)
                min_x = cloud->points[i].x;
            if (min_y > cloud->points[i].y)
                min_y = cloud->points[i].y;
            if (min_z > cloud->points[i].z)
                min_z = cloud->points[i].z;
            if (max_x < cloud->points[i].x)
                max_x = cloud->points[i].x;
            if (max_y < cloud->points[i].y)
                max_y = cloud->points[i].y;
            if (max_z < cloud->points[i].z)
                max_z = cloud->points[i].z;
        }
        bound.min_x = min_x;
        bound.max_x = max_x;
        bound.min_y = min_y;
        bound.max_y = max_y;
        bound.min_z = min_z;
        bound.max_z = max_z;
    }

    /**
     * 获取输入点云的： 包围盒范围+点云中心点
     */
    template <typename T>
    void get_cloud_bbx_cpt(const typename pcl::PointCloud<T>::Ptr &cloud, bounds_t &bound, centerpoint_t &cp)
    {
        get_cloud_bbx<T>(cloud, bound);
        cp.x = 0.5 * (bound.min_x + bound.max_x);
        cp.y = 0.5 * (bound.min_y + bound.max_y);
        cp.z = 0.5 * (bound.min_z + bound.max_z);
    }
    bool use_distance_adaptive_pca = false;
    int distance_inverse_sampling_method = 0; // distance_inverse_downsample; 0: disabled; 1: linear weight; 2: quadratic weight
    float standard_distance = 15.0;           // the distance where the weight is 1; only useful when distance_inverse_downsample is on
    int estimate_ground_normal_method = 3;    // estimate_ground_normal_method; 0: directly use (0;0;1); 1: estimate normal in fix radius neighborhood ; 2: estimate normal in k nearest neighborhood; 3: use ransac to estimate plane coeffs in a grid
    float normal_estimation_radius = 2.0;     // only when enabled when estimate_ground_normal_method = 1
    bool use_adpative_parameters = false;
    bool apply_scanner_filter = false;
    bool extract_curb_or_not = false;
    int extract_vertex_points_method = 2; // use the maximum curvature based keypoints
    int gf_grid_pt_num_thre = 8;
    int gf_reliable_neighbor_grid_thre = 0;
    int gf_down_down_rate_ground = 2;
    int pca_neighbor_k_min = 8;
    float intensity_thre = FLT_MAX; // default intensity_thre means that highly-reflective objects would not be prefered

    bool sharpen_with_nms_on = true;
    bool fixed_num_downsampling = false;
    int ground_down_fixed_num = 500;
    int pillar_down_fixed_num = 200;
    int facade_down_fixed_num = 800;
    int beam_down_fixed_num = 200;
    int roof_down_fixed_num = 200;
    int unground_down_fixed_num = 20000;

    float roof_height_min = 0.0;
    float approx_scanner_height = 2.0;
    float underground_thre = -7.0;
    float feature_pts_ratio_guess = 0.3;
    bool semantic_assisted = false;
    bool apply_roi_filtering = false;
    float roi_min_y = 0.0;
    float roi_max_y = 0.0;
    double gf_down_rate_ground = 15;
    double gf_downsample_rate_nonground = 3;
    double gf_max_ground_height = 5;
    double gf_min_ground_height = -5;
    double gf_neighbor_height_diff = 1.5;
    double gf_max_grid_height_diff = 0.3;
    double gf_grid_resolution = 3.0;

    ros::Publisher pubGround;
    ros::Publisher pubNonGround;
    std_msgs::Header cloudHeader;
    pointTypeCloud::Ptr groundSeginputCloudPtr;
    pointTypeCloud::Ptr groundCloudPtr;
    pointTypeCloud::Ptr nonGroundCloudPtr;

    ros::NodeHandle nh_;
};

class nongroundExtract
{
public:
    nongroundExtract(ros::NodeHandle nh) : nh_(nh){};
    ~nongroundExtract(){};
    void featureInit(pointTypeCloud::Ptr &inputCloud, std_msgs::Header header)
    {
        featureSeginputCloudPtr.reset(new pointTypeCloud());
        normalCloud.reset(new pointTypeCloudNormal());
        cloud_pillar.reset(new pointTypeCloudNormal());
        cloud_beam.reset(new pointTypeCloudNormal());
        cloud_facade.reset(new pointTypeCloudNormal());
        cloud_roof.reset(new pointTypeCloudNormal());
        cloudHeader = header;
        featureSeginputCloudPtr = inputCloud;
    }

    template <typename PointT1, typename PointT2>
    void pc2pc(typename PointT1::Ptr &cloud_in_anytype, typename PointT2::Ptr &cloud_out_normal)
    {
        for (auto p : *cloud_in_anytype)
        {
            pointTypeNormal pn;
            pn.x = p.x;
            pn.y = p.y;
            pn.z = p.z;
            cloud_out_normal->emplace_back(pn);
        }
    }

    template <typename PointT>
    void featureExtract(typename pcl::PointCloud<PointT>::Ptr &cloud_in)
    {
        index_with_feature.resize(cloud_in->points.size(), 0);

        std::vector<pca_feature_t> cloud_features;
        PrincipleComponentAnalysis<PointT> pca_estimator;
        typename pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
        tree->setInputCloud(cloud_in);

        pca_estimator.get_pc_pca_feature(cloud_in, cloud_features, tree, neighbor_searching_radius, neighbor_k, 1);

        for (int i = 0; i < cloud_in->points.size(); i++)
        {
            if (cloud_features[i].pt_num > neigh_k_min)
            {
                if (cloud_features[i].linear_2 > edge_thre)
                {
                    if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre)
                    {
                        pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
                        cloud_pillar->points.push_back(cloud_in->points[i]);
                        index_with_feature[i] = 1;
                    }
                    else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max && cloud_in->points[i].z > beam_height_min)
                    {
                        pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
                        cloud_beam->points.push_back(cloud_in->points[i]);
                        index_with_feature[i] = 2;
                    }
                }

                else if (cloud_features[i].planar_2 > planar_thre)
                {
                    if (std::abs(cloud_features[i].vectors.normalDirection.z()) < planar_vertical_sin_low_thre)
                    {
                        pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], true);
                        cloud_facade->points.push_back(cloud_in->points[i]);
                        index_with_feature[i] = 3;
                    }
                }
            }
        }
    };

    void pubFeatureCloud()
    {
        sensor_msgs::PointCloud2 beamCloudMsgOut;
        pcl::toROSMsg(*cloud_beam, beamCloudMsgOut);
        beamCloudMsgOut.header = cloudHeader;
        pubBeam.publish(beamCloudMsgOut);

        sensor_msgs::PointCloud2 pillarCloudMsgOut;
        pcl::toROSMsg(*cloud_pillar, pillarCloudMsgOut);
        pillarCloudMsgOut.header = cloudHeader;
        pubPillar.publish(pillarCloudMsgOut);

        sensor_msgs::PointCloud2 facadeCloudMsgOut;
        pcl::toROSMsg(*cloud_facade, facadeCloudMsgOut);
        facadeCloudMsgOut.header = cloudHeader;
        pubFacade.publish(facadeCloudMsgOut);
    }

    float neighbor_searching_radius = 1.0;

    int neighbor_k = 25;
    int neigh_k_min = 8;
    int pca_down_rate = 2; // one in ${pca_down_rate} unground points would be select as the query points for calculating pca, the else would only be used as neighborhood points
    float edge_thre = 0.65;
    float planar_thre = 0.65;
    float linear_vertical_sin_high_thre = 0.94;
    float linear_vertical_sin_low_thre = 0.17; // 70 degree (pillar); 10 degree (beam)
    float planar_vertical_sin_high_thre = 0.98;
    float planar_vertical_sin_low_thre = 0.34; // 80 degree (roof); 20 degree (facade)
    float beam_height_max = FLT_MAX;
    float beam_height_min = 0.5;

    std_msgs::Header cloudHeader;
    pointTypeCloud::Ptr featureSeginputCloudPtr;
    pointTypeCloudNormal::Ptr normalCloud;
    pointTypeCloudNormal::Ptr cloud_pillar;
    pointTypeCloudNormal::Ptr cloud_beam;
    pointTypeCloudNormal::Ptr cloud_facade;
    pointTypeCloudNormal::Ptr cloud_roof;
    std::vector<int> index_with_feature;
    ros::NodeHandle nh_;

    ros::Publisher pubBeam;
    ros::Publisher pubPillar;
    ros::Publisher pubFacade;
};
