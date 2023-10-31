
#ifndef _ODOM_ESTIMATION_CLASS_H_
#define _ODOM_ESTIMATION_CLASS_H_

// std lib
#include <string>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <algorithm>
#include <numeric>
// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// LOCAL LIB
#include "lidar.h"
#include "lidarOptimization.h"
#include <ros/ros.h>
#include "common.hpp"
typedef pcl::PointXYZRGB PointType;

class OdomBaseClass
{
public:
	OdomBaseClass(){};
	~OdomBaseClass(){};

	void pointAssociateToMap(PointType const *const pi, PointType *const po);
	void observeMean(std::vector<double> &observe_vec);
	void downSamplingToMap(const pcl::PointCloud<PointType>::Ptr &pc_in, pcl::PointCloud<PointType>::Ptr &pc_out, pcl::VoxelGrid<PointType> &downGrid);
	void extractstablepoint(pcl::PointCloud<PointType>::Ptr lasecloudMap_input, int k_new, float theta_p, int theta_max);
	pcl::PointCloud<PointType>::Ptr rgbds(pcl::PointCloud<PointType>::Ptr input, float dsleaf, unsigned int min_points_per_voxel_);

	// optimization variable
	double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
	Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(parameters);
	Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(parameters + 4);

	Eigen::Isometry3d odom;
	Eigen::Isometry3d last_odom;

	// local map
	pcl::CropBox<PointType> cropBoxFilter;
	float map_resolution;
	int k_new_surf;
	float theta_p_surf;
	int theta_max_surf;
	int k_new_edge;
	float theta_p_edge;
	int theta_max_edge;
	double weightType;

	// optimization count
	int optimization_count;
	/**
	 * 点云-体素 序号对
	 */
	struct cloud_point_index_idx
	{
		unsigned int idx;				// 体素号
		unsigned int cloud_point_index; // 点云号

		cloud_point_index_idx(unsigned int idx_, unsigned int cloud_point_index_) : idx(idx_), cloud_point_index(cloud_point_index_) {}
		bool operator<(const cloud_point_index_idx &p) const { return (idx < p.idx); }
	};

	struct edgeInfo
	{
		Eigen::Vector3d curr_point, point_a, point_b;
		float observe, round;
	};

	struct surfInfo
	{
		Eigen::Vector3d curr_point, norm;
		float negative_OA_dot_norm, observe, round;
	};
	void writefile(std::vector<double> &observe_vec, std::string filename)
	{
		std::ofstream ofs;													  // 输出流
		ofs.open(filename, std::ios::out | std::ios::app | std::ios::binary); // 若没有指定路径，默认情况下和当前项目的文件路径是一致的
		if (ofs.is_open())
		{
			for (double ele : observe_vec)
			{
				ofs << ele << ",";
			}
		}

		ofs.close();
	}

	void pointSparsityMean(std::vector<double> &pointSparsity)
	{
		//  writefile(pointSparsity, "/home/r/catkin_wss/pfilter_ws/src/PFilter-noetic/pointdis.dat");
		double min_element = *std::min_element(pointSparsity.begin(), pointSparsity.end());
		double max_element = *std::max_element(pointSparsity.begin(), pointSparsity.end());
		auto length = (max_element - min_element);
		if (length == 0)
			return;
		for (auto &ele : pointSparsity)
		{
			ele = (ele - min_element) / length;
			ele -= 1.0;
			ele = abs(ele);
			ele *= 2.0;
		}
	}

	void pointDisMean(std::vector<double> &pointDis)
	{
		for (auto &ele : pointDis)
		{
			ele /= 0.6;
			ele -= 1.0;
			ele = abs(ele);
			ele *= 2.0;
		}
	}
};

class Odom_ES_EstimationClass : public OdomBaseClass
{

public:
	Odom_ES_EstimationClass(){};
	~Odom_ES_EstimationClass(){};
	void init(lidar::Lidar lidar_param, double map_resolution_in, int k_new_para, float theta_p_para, int theta_max_para, double weightType_para);
	void getMap(pcl::PointCloud<PointType>::Ptr &laserCloudMap);
	void initMapWithPoints(const pcl::PointCloud<PointType>::Ptr &edge_in, const pcl::PointCloud<PointType>::Ptr &surf_in);
	void updatePointsToMap(const pcl::PointCloud<PointType>::Ptr &edge_in, const pcl::PointCloud<PointType>::Ptr &surf_in);

	pcl::PointCloud<PointType>::Ptr laserCloudCornerMap; // 存储体素点云质心的系数点云地图
	pcl::PointCloud<PointType>::Ptr laserCloudSurfMap;	 // 存储体素点云质心的系数点云地图

private:
	// kd-tree
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeEdgeMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap;

	// points downsampling before add to map
	pcl::VoxelGrid<PointType> downSizeFilterEdge;
	pcl::VoxelGrid<PointType> downSizeFilterSurf;

	// function
	void addEdgeCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function);
	void addSurfCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function);
	void addPointsToMap(const pcl::PointCloud<PointType>::Ptr &downsampledEdgeCloud, const pcl::PointCloud<PointType>::Ptr &downsampledSurfCloud);
};

class Odom_BPF_EstimationClass : public OdomBaseClass
{

public:
	Odom_BPF_EstimationClass(){};
	~Odom_BPF_EstimationClass(){};
	void init(lidar::Lidar lidar_param, double map_resolution_in, int k_new_para, float theta_p_para, int theta_max_para, double weightType_para);
	void getMap(pcl::PointCloud<PointType>::Ptr &laserCloudMap);
	void initMapWithPoints(const pcl::PointCloud<PointType>::Ptr &beam_in, const pcl::PointCloud<PointType>::Ptr &pillar_in, const pcl::PointCloud<PointType>::Ptr &facade_in);
	void updatePointsToMap(const pcl::PointCloud<PointType>::Ptr &beam_in, const pcl::PointCloud<PointType>::Ptr &pillar_in, const pcl::PointCloud<PointType>::Ptr &facade_in);

	pcl::PointCloud<PointType>::Ptr laserCloudBeamMap;	 // 存储体素点云质心的系数点云地图
	pcl::PointCloud<PointType>::Ptr laserCloudPillarMap; // 存储体素点云质心的系数点云地图
	pcl::PointCloud<PointType>::Ptr laserCloudFacadeMap; // 存储体素点云质心的系数点云地图
	pcl::PointCloud<PointType>::Ptr laserCloudGroundMap; // 存储体素点云质心的系数点云地图
	pcl::PointCloud<PointType>::Ptr laserCloudMergeMap; // 存储体素点云质心的系数点云地图
	
private:
	// kd-tree
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeBeamMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreePillarMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeFacadeMap;

	// points downsampling before add to map
	pcl::VoxelGrid<PointType> downSizeFilterBeam;
	pcl::VoxelGrid<PointType> downSizeFilterPillar;
	pcl::VoxelGrid<PointType> downSizeFilterFacade;

	// function
	void addBeamCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function);
	void addPillarCostFactor(const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function);
	void addFacadeCostFactor(const pcl::PointCloud<PointType>::Ptr &surf_cloud, const pcl::PointCloud<PointType>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function);
	void addPointsToMap(const pcl::PointCloud<PointType>::Ptr &downsampledBeamCloud, const pcl::PointCloud<PointType>::Ptr &downsampledPillarCloud, const pcl::PointCloud<PointType>::Ptr &downsampledFacadeCloud);
	void mergeFeatures(int mergeGround);

	
};

#endif // _ODOM_ESTIMATION_CLASS_H_
