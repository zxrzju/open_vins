/*
 * @Author: your name
 * @Date: 2020-08-21 17:11:54
 * @LastEditTime: 2020-09-08 19:33:22
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /open_vins-master/aloam/scanRegistration.h
 */
// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __SCANREGISTRATION_H

#define __SCANREGISTRATION_H

#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;


extern ros::Publisher pubLaserCloud;
extern ros::Publisher pubCornerPointsSharp;
extern ros::Publisher pubCornerPointsLessSharp;
extern ros::Publisher pubSurfPointsFlat;
extern ros::Publisher pubSurfPointsLessFlat;
extern ros::Publisher pubRemovePoints;
extern std::vector<ros::Publisher> pubEachScan;

extern bool PUB_EACH_LINE;
extern double MINIMUM_RANGE;

class EdgeFeature
{

public:
    size_t featid;
    Eigen::Vector3d p_L1;
    Eigen::Vector3d p_L1inL0;
    Eigen::Vector3d p_L0_a;
    Eigen::Vector3d p_L0_b;
    double timestamp0 = -1;
    double timestamp1 = -1;
    double res_estimation = 0.0;
};
// extern pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
// extern pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;   //(new pcl::KdTreeFLANN<pcl::PointXYZI>());

int aloamInit(ros::NodeHandle &nh);
int scan_registration(pcl::PointCloud<pcl::PointXYZ> &laserCloudIn, pcl::PointCloud<PointType>::Ptr &laserCloud, pcl::PointCloud<PointType> &cornerPointsSharp, pcl::PointCloud<PointType> &cornerPointsLessSharp, pcl::PointCloud<PointType> &surfPointsFlat, pcl::PointCloud<PointType> &surfPointsLessFlat, int N_SCANS);

void TransformToStart(PointType const *const pi, PointType *const po);

void TransformToEnd(PointType const *const pi, PointType *const po);


void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2);
void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2);
void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2);
void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2);
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2);

int test_odometry(ros::NodeHandle &nh, ros::Publisher &pubLaserCloudCornerLast, ros::Publisher &pubLaserCloudSurfLast, ros::Publisher &pubLaserCloudFullRes, ros::Publisher &pubLaserOdometry, ros::Publisher &pubLaserPath);

int get_correspondences(pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast, pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast, pcl::PointCloud<PointType>::Ptr &laserCloudCornerLast, pcl::PointCloud<PointType>::Ptr &laserCloudSurfLast, pcl::PointCloud<PointType>::Ptr &cornerPointsSharp, pcl::PointCloud<PointType>::Ptr &cornerPointsLessSharp, pcl::PointCloud<PointType>::Ptr &surfPointsFlat, pcl::PointCloud<PointType>::Ptr &surfPointsLessFlat, Eigen::Quaterniond q_last_curr, Eigen::Vector3d t_last_curr, std::vector<EdgeFeature> &edge_list, double time0, double time1);


#endif