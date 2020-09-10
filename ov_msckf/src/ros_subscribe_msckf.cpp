/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2019 Patrick Geneva
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2019 Guoquan Huang
 * Copyright (C) 2019 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "core/RosVisualizer.h"
#include "utils/dataset_reader.h"
#include "utils/parse_ros.h"

#include "aloam/scanRegistration.h"

using namespace ov_msckf;


VioManager* sys;
RosVisualizer* viz;

// Buffer data
double time_buffer = -1;
cv::Mat img0_buffer, img1_buffer;

// Time offset base IMU to lidar (t_imu = t_lidar + t_off)
std::unordered_map<size_t, double> _calib_dt_LIDARtoIMU;

std::unordered_map<size_t, std::string> _topic_lidar;

std::unordered_map<size_t, ros::Subscriber> _sublidar;

std::unordered_map<size_t, double> _lidar_time_buffer, _lidar_time_buffer_last;

std::unordered_map<size_t, pcl::PointCloud<pcl::PointXYZ>::Ptr> _lidar_pc_buffer;

std::unordered_map<size_t, pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr> _lidar_kdtree_corner, _lidar_kdtree_surface;

std::unordered_map<size_t, pcl::PointCloud<PointType>::Ptr> _laserCloudCornerLast, _laserCloudSurfLast;

std::unordered_map<size_t, Eigen::Quaterniond> _q_laser_last;

std::unordered_map<size_t, Eigen::Vector3d> _t_laser_last;

// pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
// pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// Callback functions
void callback_inertial(const sensor_msgs::Imu::ConstPtr& msg);
void callback_monocular(const sensor_msgs::ImageConstPtr& msg0);
void callback_stereo(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1);
void callback_laser(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg, std::string &topic, int scans);

ros::Publisher pubLaserCloudCornerLast;
ros::Publisher pubLaserCloudSurfLast;
ros::Publisher pubLaserCloudFullRes;
ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;
VioManagerOptions params;

// Main function
int main(int argc, char **argv)
{

    // Launch our ros node
    ros::init(argc, argv, "run_subscribe_msckf");
    ros::NodeHandle nh("~");

    // Create our VIO system
    // VioManagerOptions params = parse_ros_nodehandler(nh);
    params = parse_ros_nodehandler(nh);
    sys = new VioManager(params);
    viz = new RosVisualizer(nh, sys);

    // pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
    // pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    // pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    // pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    // pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    // ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    // ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    // ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    // ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    // ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);
    // pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    // pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    // pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    // pcl::PointCloud<PointType>::Ptr laserCloudInPtr(new pcl::PointCloud<PointType>());



    // // aloamInit(nh);
    for (int i = 0; i < params.state_options.num_lidars; i++)
    {
        std::string topic_lidar; 
        nh.param<std::string>("topic_lidar" + std::to_string(i), topic_lidar, "");
        ros::Subscriber sublidar = nh.subscribe<sensor_msgs::PointCloud2>(topic_lidar, 9999, boost::bind(&callback_laser, _1, topic_lidar, params.state_options.scans));

        _sublidar.insert({i, sublidar});
        _topic_lidar.insert({i, topic_lidar});
        _lidar_time_buffer.insert({i, -1});
        _lidar_time_buffer_last.insert({i, -1});

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        _lidar_pc_buffer.insert({i, pc});

        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        _lidar_kdtree_corner.insert({i, kdtreeCornerLast});

        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        _lidar_kdtree_surface.insert({i, kdtreeSurfLast});

        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
        _laserCloudCornerLast.insert({i, laserCloudCornerLast});

        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
        _laserCloudSurfLast.insert({i, laserCloudSurfLast});

        Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
        Eigen::Vector3d t_w_curr(0, 0, 0);
        _q_laser_last.insert({i, q_w_curr});
        _t_laser_last.insert({i, t_w_curr});
    }

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Our camera topics (left and right stereo)
    std::string topic_imu;
    std::string topic_camera0, topic_camera1;
    nh.param<std::string>("topic_imu", topic_imu, "/imu0");
    nh.param<std::string>("topic_camera0", topic_camera0, "/cam0/image_raw");
    nh.param<std::string>("topic_camera1", topic_camera1, "/cam1/image_raw");

    // Logic for sync stereo subscriber
    // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
    message_filters::Subscriber<sensor_msgs::Image> image_sub0(nh,topic_camera0.c_str(),1);
    message_filters::Subscriber<sensor_msgs::Image> image_sub1(nh,topic_camera1.c_str(),1);
    //message_filters::TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image> sync(image_sub0,image_sub1,5);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(5), image_sub0,image_sub1);

    // Create subscribers
    ros::Subscriber subimu = nh.subscribe(topic_imu.c_str(), 9999, callback_inertial);
    ros::Subscriber subcam;
    if(params.state_options.num_cameras == 1) {
        ROS_INFO("subscribing to: %s", topic_camera0.c_str());
        subcam = nh.subscribe(topic_camera0.c_str(), 1, callback_monocular);
    } else if(params.state_options.num_cameras == 2) {
        ROS_INFO("subscribing to: %s", topic_camera0.c_str());
        ROS_INFO("subscribing to: %s", topic_camera1.c_str());
        sync.registerCallback(boost::bind(&callback_stereo, _1, _2));
    } else {
        ROS_ERROR("INVALID MAX CAMERAS SELECTED!!!");
        std::exit(EXIT_FAILURE);
    }

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Spin off to ROS
    ROS_INFO("done...spinning to ros");
    ros::spin();

    // Final visualization
    viz->visualize_final();

    // Finally delete our system
    delete sys;
    delete viz;


    // Done!
    return EXIT_SUCCESS;
}

void callback_inertial(const sensor_msgs::Imu::ConstPtr& msg) {

    // convert into correct format
    double timem = msg->header.stamp.toSec();
    Eigen::Vector3d wm, am;
    wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    // send it to our VIO system
    sys->feed_measurement_imu(timem, wm, am);

    static long cnt = 0;
    // cnt++;
    if (cnt++ == 0)
    {
        sys->get_state()->_timestamp = timem;

        // cout << "propagate\n";
        // ov_msckf::State *state = sys->get_state();
        // ov_msckf::Propagator *propagator = sys->get_propagator();
        // propagator->propagate_and_clone(state, timem);
        // cout << sys->get_state()->_timestamp << endl;
    }

    viz->visualize_odometry(timem);

}



void callback_monocular(const sensor_msgs::ImageConstPtr& msg0) {

    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Fill our buffer if we have not
    if(img0_buffer.rows == 0) {
        time_buffer = cv_ptr->header.stamp.toSec();
        img0_buffer = cv_ptr->image.clone();
        return;
    }

    // send it to our VIO system
    sys->feed_measurement_monocular(time_buffer, img0_buffer, 0);
    viz->visualize();

    // move buffer forward
    time_buffer = cv_ptr->header.stamp.toSec();
    img0_buffer = cv_ptr->image.clone();

}


void callback_laser(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg, std::string &topic, int scans){
    ov_msckf::State *state = sys->get_state();
    // cout << "cloneIMUs: "<< state->_clones_IMU.size() << endl;
    
    double timem = laserCloudMsg->header.stamp.toSec();
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast, kdtreeSurfLast;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudInPtr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr lastLaserCloudPtr;

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    
    laserCloudInPtr = laserCloudIn.makeShared();

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType> cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat;
    pcl::PointCloud<PointType>::Ptr cornerPointsSharpPtr, cornerPointsLessSharpPtr, surfPointsFlatPtr, surfPointsLessFlatPtr;

    int num_lidars = _topic_lidar.size();
    int current_lidar = 0;
    for(int i = 0; i < num_lidars; i++)
        if(_topic_lidar.at(i) == topic){
            current_lidar = i;
            break;
        }
    // std::cout << current_lidar << std::endl;
    if(_lidar_time_buffer.at(current_lidar) == -1){
        _lidar_time_buffer.at(current_lidar) = 0;
        _lidar_time_buffer_last.at(current_lidar) = timem;
        _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;
        return ;
        }
    else if(_lidar_time_buffer.at(current_lidar) == 0){
        _lidar_time_buffer.at(current_lidar) = timem;
        lastLaserCloudPtr = _lidar_pc_buffer.at(current_lidar);
        scan_registration(*lastLaserCloudPtr, laserCloud, cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat, scans);
        cornerPointsSharpPtr = cornerPointsSharp.makeShared();
        cornerPointsLessSharpPtr = cornerPointsLessSharp.makeShared();
        surfPointsFlatPtr = surfPointsFlat.makeShared();
        surfPointsLessFlatPtr = surfPointsLessFlat.makeShared();

        _lidar_kdtree_corner.at(current_lidar)->setInputCloud(cornerPointsLessSharpPtr);
        _lidar_kdtree_surface.at(current_lidar)->setInputCloud(surfPointsLessFlatPtr);
        _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;
        _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
        _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;


        return;
    }
    kdtreeCornerLast = _lidar_kdtree_corner.at(current_lidar);
    kdtreeSurfLast = _lidar_kdtree_surface.at(current_lidar);
    lastLaserCloudPtr = _lidar_pc_buffer.at(current_lidar);

    scan_registration(*lastLaserCloudPtr, laserCloud, cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat, scans);
    cornerPointsSharpPtr = cornerPointsSharp.makeShared();
    cornerPointsLessSharpPtr = cornerPointsLessSharp.makeShared();
    surfPointsFlatPtr = surfPointsFlat.makeShared();
    surfPointsLessFlatPtr = surfPointsLessFlat.makeShared();

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast = _laserCloudCornerLast.at(current_lidar);
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast = _laserCloudSurfLast.at(current_lidar);

    std::vector<EdgeFeature> edge_list;

    Eigen::Quaterniond q_last_curr(1, 0, 0, 0);
    Eigen::Vector3d t_last_curr(0, 0, 0);    



    // lidar_feature_extraction();
    // do_lidar_propagate_update(timestamp, edge_list);
    double timestamp =  _lidar_time_buffer.at(current_lidar);


    if (state->_timestamp >= timestamp)
    {
        printf("%3f, %3f\n", timestamp, state->_timestamp);
        printf(YELLOW "lidar received out of order (prop dt = %3f)\n" RESET, (timestamp - state->_timestamp));
        return;
    }

    ov_msckf::Propagator *propagator = sys->get_propagator();
    propagator->propagate_and_clone(state, timestamp);

    if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5))
    {
        printf("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(), std::min(state->_options.max_clone_size, 5));

        _lidar_time_buffer_last.at(current_lidar) = _lidar_time_buffer.at(current_lidar);
        _lidar_time_buffer.at(current_lidar) = timem;
        _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
        _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;
        _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;
        return;
    }

    Eigen::Quaterniond q_last = _q_laser_last.at(current_lidar);
    Eigen::Vector3d t_last = _t_laser_last.at(current_lidar);

    Eigen::Matrix4d poseLast, poseCurr, poseDelta;
    poseLast.block<3, 3>(0, 0) = q_last.toRotationMatrix();
    poseLast.block<3, 1>(0, 3) = t_last;

    // propagator->propagate_and_clone(state, timestamp);

    poseCurr.block<3, 3>(0, 0) = state->_clones_IMU.at(_lidar_time_buffer.at(current_lidar))->Rot();
    poseCurr.block<3, 1>(0, 3) = state->_clones_IMU.at(_lidar_time_buffer.at(current_lidar))->pos();
        // std::cout << "after:\n" << (state->_imu->pos().transpose()) << std::endl;

    poseDelta = poseCurr*Inv_se3(poseLast);

    // Get calibration for our anchor camera
    Eigen::Matrix<double, 3, 3> R_ItoL = state->_calib_IMUtoLIDAR.at(current_lidar)->Rot();
    Eigen::Matrix<double, 3, 1> p_IinL = state->_calib_IMUtoLIDAR.at(current_lidar)->pos();

    // Eigen::Matrix<double, 3, 3> R_last = q_last.toRotationMatrix();//.transpose();
    Eigen::Matrix<double, 3, 3> R_last = state->_clones_IMU.at(_lidar_time_buffer_last.at(current_lidar))->Rot();

    // Eigen::Matrix<double, 3, 3> R_curr = state->_imu->Rot().transpose();
    Eigen::Matrix<double, 3, 3> R_curr = state->_clones_IMU.at(_lidar_time_buffer.at(current_lidar))->Rot();//.transpose();
    
    Eigen::Matrix<double, 3, 3> R_curr_to_last = R_ItoL * R_last * (R_ItoL * R_curr).transpose();

    // Eigen::Matrix<double, 3, 1> p_last = t_last;
    Eigen::Matrix<double, 3, 1> p_last = state->_clones_IMU.at(_lidar_time_buffer_last.at(current_lidar))->pos();

    // Eigen::Matrix<double, 3, 1> p_curr = state->_imu->pos();
    Eigen::Matrix<double, 3, 1> p_curr = state->_clones_IMU.at(_lidar_time_buffer.at(current_lidar))->pos();
    Eigen::Matrix<double, 3, 1> p_LinI = -R_ItoL.transpose() * p_IinL;
    Eigen::Matrix<double, 3, 1> p_CurrinLast = R_ItoL * R_last * (p_curr - p_last + R_curr.transpose() * p_LinI) + p_IinL;

    Eigen::Quaterniond deltaQ(R_curr_to_last);
    Eigen::Vector3d deltaT = p_CurrinLast;
    // std::cout<<"delta: "<<deltaT.transpose()<<" "<<std::endl;
    // std::cout<<timestamp<<"\n";

    get_correspondences(kdtreeCornerLast, kdtreeSurfLast, laserCloudCornerLast, laserCloudSurfLast, cornerPointsSharpPtr, cornerPointsLessSharpPtr, surfPointsFlatPtr, surfPointsLessFlatPtr, deltaQ, deltaT, edge_list, (_lidar_time_buffer_last.at(current_lidar)), (_lidar_time_buffer.at(current_lidar)));

    cout<<"clone pose: \n"<<p_last.transpose()<<endl<<p_curr.transpose()<<endl;
    // for (auto &edge : edge_list)
    // {
    //     Eigen::Vector3d num1, num2, dem1;

    //     num1 = skew_x(edge.p_L1 - edge.p_L0_a) * (edge.p_L1 - edge.p_L0_b);
    //     dem1 = edge.p_L0_a - edge.p_L0_b;
    //     cout << "distance: "<<sqrt(double(num1.transpose() * num1)/double(dem1.transpose() * dem1)) << endl;
    // }


    // // // UpdateMSCKF

    if((int)edge_list.size() > sys->get_state()->_options.max_loam_in_update)
        edge_list.erase(edge_list.begin(), edge_list.end() - state->_options.max_loam_in_update);

    std::vector<EdgeFeature> feature_vec = edge_list;
    {
        // Return if no features
        if (feature_vec.empty())
            return;
        // Start timing
        boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
        rT0 = boost::posix_time::microsec_clock::local_time();

        // Calculate the max possible measurement size
        size_t max_meas_size = 3*feature_vec.size();

        // Calculate max possible state size (i.e. the size of our covariance)
        // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
        size_t max_hx_size = state->max_covariance_size();

        // Large Jacobian and residual of *all* features for this update
        Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
        Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
        std::unordered_map<Type *, size_t> Hx_mapping;
        std::vector<Type *> Hx_order_big;
        size_t ct_jacob = 0;
        size_t ct_meas = 0;

        // 4. Compute linear system for each feature, nullspace project, and reject
        auto feat = feature_vec.begin();
        while (feat != feature_vec.end())
        {


            // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
            Eigen::MatrixXd H_f;
            Eigen::MatrixXd H_x;
            Eigen::VectorXd res;
            std::vector<Type *> Hx_order, x_order;

            // // get_feature_jacobian_full()
            {
                int total_hx = 0;
                std::unordered_map<Type *, size_t> map_hx;

                PoseJPL *calibration = state->_calib_IMUtoLIDAR.at(current_lidar);
                // If doing lidar extrinsics
                if (state->_options.do_calib_lidar_pose)
                {
                    map_hx.insert({calibration, total_hx});
                    x_order.push_back(calibration);
                    total_hx += calibration->size();
                }
                // Add this clone if it is not added already
                PoseJPL *clone_Ci = state->_clones_IMU.at(feat->timestamp0);
                if (map_hx.find(clone_Ci) == map_hx.end())
                {
                    map_hx.insert({clone_Ci, total_hx});
                    x_order.push_back(clone_Ci);
                    total_hx += clone_Ci->size();
                }

                clone_Ci = state->_clones_IMU.at(feat->timestamp1);
                if (map_hx.find(clone_Ci) == map_hx.end())
                {
                    map_hx.insert({clone_Ci, total_hx});
                    x_order.push_back(clone_Ci);
                    total_hx += clone_Ci->size();
                }

                // Allocate our residual and Jacobians
                int c = 0;
                int jacobsize = 3;

                res = Eigen::VectorXd::Zero(1 * 1);
                H_f = Eigen::MatrixXd::Zero(1 * 1, jacobsize);
                H_x = Eigen::MatrixXd::Zero(1 * 1, total_hx);



                Eigen::MatrixXd dpfg_dlambda;
                std::vector<Eigen::MatrixXd> dpfg_dx, H_x;
                std::vector<Type *> dpfg_dx_order, x_order;
                dpfg_dx = H_x;
                dpfg_dx_order = x_order;
                // // get_feature_jacobian_representation()
                {
                    Eigen::Matrix3d skew_b_a, temp0;
                    Eigen::Matrix<double, 1, 3> temp1;
                    Eigen::Vector3d axb;
                    skew_b_a.noalias() = skew_x(feat->p_L0_b - feat->p_L0_a);
                    axb = skew_x(feat->p_L0_a)*feat->p_L0_b;

                    temp0 = skew_b_a.transpose()*skew_b_a;
                    temp1 = axb.transpose()*skew_b_a;

                    Eigen::Matrix<double, 1, 3> dr_dpf = 2*feat->p_L1inL0.transpose()*temp0 + 2*temp1;

                    Eigen::Matrix3d dpf_dr0 = R_ItoL * skew_x(R_last * (R_curr.transpose() * R_ItoL * feat->p_L1 + p_curr - p_last + R_curr.transpose() * p_LinI));

                    Eigen::Matrix3d dpf_dp0 = -R_ItoL * R_last;

                    Eigen::Matrix3d dpf_dr1 = -R_ItoL * R_last * R_curr.transpose() * skew_x(R_ItoL.transpose() * feat->p_L1 + p_LinI);

                    Eigen::Matrix3d dpf_dp1 = R_ItoL * R_last;

                    Eigen::Matrix3d dpf_dril = skew_x(R_curr_to_last * (feat->p_L1 - p_IinL)) - R_curr_to_last * skew_x(feat->p_L1 - p_IinL);

                    Eigen::Matrix3d dpf_dpil = -R_curr_to_last + Eigen::Matrix3d::Identity();



                    // Jacobian for our anchor pose
                    Eigen::Matrix<double, 3, 6> H_anc0;
                    H_anc0.block(0, 0, 3, 3).noalias() = dr_dpf*dpf_dp0;
                    H_anc0.block(0, 3, 3, 3).noalias() = dr_dpf*dpf_dr0;

                    // Add anchor Jacobians to our return vector
                    x_order.push_back(state->_clones_IMU.at(feat->timestamp0));
                    H_x.push_back(H_anc0);

                    // Jacobian for our anchor pose
                    Eigen::Matrix<double, 3, 6> H_anc1;
                    H_anc1.block(0, 0, 3, 3).noalias() = dr_dpf*dpf_dp1;
                    H_anc1.block(0, 3, 3, 3).noalias() = dr_dpf*dpf_dr1;

                    // Add anchor Jacobians to our return vector
                    x_order.push_back(state->_clones_IMU.at(feat->timestamp1));
                    H_x.push_back(H_anc1);

                    // Get calibration Jacobians (for anchor clone)
                    if (state->_options.do_calib_lidar_pose)
                    {
                        Eigen::Matrix<double, 3, 6> H_calib;
                        H_calib.block(0, 0, 3, 3).noalias() = dr_dpf*dpf_dpil;
                        H_calib.block(0, 3, 3, 3).noalias() = dr_dpf*dpf_dril;
                        x_order.push_back(state->_calib_IMUtoLIDAR.at(feat->timestamp0));
                        H_x.push_back(H_calib);
                    }
                }
                dpfg_dx = H_x;
                dpfg_dx_order = x_order;

                // Assert that all the ones in our order are already in our local jacobian mapping
                for (auto &type : dpfg_dx_order)
                {
                    assert(map_hx.find(type) != map_hx.end());
                }
                
            }
            

            
            Hx_order = x_order;
            // // nullspace_place()

            /// Chi2 distance check
            Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
            Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
            // S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());

            S.diagonal() += 0 * Eigen::VectorXd::Ones(S.rows());
            double chi2 = res.dot(S.llt().solve(res));

            // Get our threshold (we precompute up to 500 but handle the case that it is more)
            double chi2_check;
            if (res.rows() < 500)
            {
                // chi2_check = chi_squared_table[res.rows()];
            }
            else
            {
                boost::math::chi_squared chi_squared_dist(res.rows());
                chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
                printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
            }

            // // Check if we should delete or not
            // if (chi2 > _options.chi2_multipler * chi2_check)
            // {
            //     // (*it2)->to_delete = true;
            //     it2 = feature_vec.erase(it2);
            //     //cout << "featid = " << feat.featid << endl;
            //     //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //     //cout << "res = " << endl << res.transpose() << endl;
            //     continue;
            // }

    //         // We are good!!! Append to our large H vector
    //         size_t ct_hx = 0;
    //         for (const auto &var : Hx_order)
    //         {

    //             // Ensure that this variable is in our Jacobian
    //             if (Hx_mapping.find(var) == Hx_mapping.end())
    //             {
    //                 Hx_mapping.insert({var, ct_jacob});
    //                 Hx_order_big.push_back(var);
    //                 ct_jacob += var->size();
    //             }

    //             // Append to our large Jacobian
    //             Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
    //             ct_hx += var->size();
    //         }
    //         // Append our residual and move forward
    //         res_big.block(ct_meas, 0, res.rows(), 1) = res;
    //         ct_meas += res.rows();
    //         it2++;

    //         // Return if we don't have anything and resize our matrices
    //         if (ct_meas < 1)
    //         {
    //             return;
    //         }
    //         assert(ct_meas <= max_meas_size);
    //         assert(ct_jacob <= max_hx_size);
    //         res_big.conservativeResize(ct_meas, 1);
    //         Hx_big.conservativeResize(ct_meas, ct_jacob);

    //         // 5. Perform measurement compression
    //         UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    //         if (Hx_big.rows() < 1)
    //         {
    //             return;
    //         }
    //         rT4 = boost::posix_time::microsec_clock::local_time();

    //         // Our noise is isotropic, so make it here after our compression
    //         // Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    //         Eigen::MatrixXd R_big = 0 * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    //         // 6. With all good features update the state
    //         StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    //         rT5 = boost::posix_time::microsec_clock::local_time();


        }

    }

    _q_laser_last.at(current_lidar) = Eigen::Quaterniond(poseCurr.block<3, 3>(0, 0));
    _t_laser_last.at(current_lidar) = state->_clones_IMU.at(_lidar_time_buffer.at(current_lidar))->pos();

    StateHelper::marginalize_old_clone(state);

    printf("sharp corner: %d, edge feature: %d\n", cornerPointsSharp.size(), edge_list.size());
    _lidar_time_buffer_last.at(current_lidar) = _lidar_time_buffer.at(current_lidar);
    _lidar_time_buffer.at(current_lidar) = timem;
    _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
    _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;
    _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;

    
    // printf("feature points: %d  %d  %d  %d \n", cornerPointsSharp.size(), cornerPointsLessSharp.size(), surfPointsFlat.size(), surfPointsLessFlat.size());
    // sensor_msgs::PointCloud2 laserCloudOutMsg;
    // pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    // laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    // laserCloudOutMsg.header.frame_id = "/camera_init";
    // pubLaserCloud.publish(laserCloudOutMsg);
    // sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    // pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    // cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    // cornerPointsSharpMsg.header.frame_id = "/camera_init";
    // pubCornerPointsSharp.publish(cornerPointsSharpMsg);
    // sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    // pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    // cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    // cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    // pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);
    // sensor_msgs::PointCloud2 surfPointsFlat2;
    // pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    // surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    // surfPointsFlat2.header.frame_id = "/camera_init";
    // pubSurfPointsFlat.publish(surfPointsFlat2);
    // sensor_msgs::PointCloud2 surfPointsLessFlat2;
    // pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    // surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    // surfPointsLessFlat2.header.frame_id = "/camera_init";
    // pubSurfPointsLessFlat.publish(surfPointsLessFlat2);
    // ROS_INFO(topic.c_str());

}

void callback_stereo(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1) {

    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr0;
    try {
        cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr1;
    try {
        cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    // Fill our buffer if we have not
    if(img0_buffer.rows == 0 || img1_buffer.rows == 0) {
        time_buffer = cv_ptr0->header.stamp.toSec();
        img0_buffer = cv_ptr0->image.clone();
        time_buffer = cv_ptr1->header.stamp.toSec();
        img1_buffer = cv_ptr1->image.clone();
        return;
    }

    // send it to our VIO system
    sys->feed_measurement_stereo(time_buffer, img0_buffer, img1_buffer, 0, 1);
    viz->visualize();

    // move buffer forward
    time_buffer = cv_ptr0->header.stamp.toSec();
    img0_buffer = cv_ptr0->image.clone();
    time_buffer = cv_ptr1->header.stamp.toSec();
    img1_buffer = cv_ptr1->image.clone();

}
