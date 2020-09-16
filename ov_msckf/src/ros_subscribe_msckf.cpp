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
std::map<int, double> chi_squared_table;

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

    for (int i = 1; i < 500; i++)
    {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }

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
        // propagator->propagate_and_clone_lidar(state, timem);
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

// double time_buf = 0;
void callback_laser(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg, std::string &topic, int scans){

    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 = rT1 = rT2 = rT3 = rT4 = rT5 = boost::posix_time::microsec_clock::local_time();

    ov_msckf::State *state = sys->get_state();
    // cout << "cloneIMUs: "<< state->_clones_IMU_lidar.size() << endl;

    double timem = laserCloudMsg->header.stamp.toSec();

    cout<<setprecision(18)<<"lidar time:"<<timem<<endl;


    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast, kdtreeSurfLast;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudInPtr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr lastLaserCloudPtr;

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    
    laserCloudInPtr = laserCloudIn.makeShared();

    pcl::PointCloud<PointType>::Ptr laserCloud;//(new pcl::PointCloud<PointType>());
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
        
        if(state->_timestamp > _lidar_time_buffer_last.at(current_lidar)){
            _lidar_time_buffer.at(current_lidar) = -1;
            return;
        }


        ov_msckf::Propagator *propagator = sys->get_propagator();
        propagator->propagate_and_clone_lidar(state, _lidar_time_buffer_last.at(current_lidar));


        _lidar_time_buffer.at(current_lidar) = timem;
        lastLaserCloudPtr = _lidar_pc_buffer.at(current_lidar);
        scan_registration(*lastLaserCloudPtr, laserCloud, cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat, scans);
        // cornerPointsSharpPtr = cornerPointsSharp.makeShared();
        cornerPointsLessSharpPtr = cornerPointsLessSharp.makeShared();
        // surfPointsFlatPtr = surfPointsFlat.makeShared();
        surfPointsLessFlatPtr = surfPointsLessFlat.makeShared();

        _lidar_kdtree_corner.at(current_lidar)->setInputCloud(cornerPointsLessSharpPtr);
        _lidar_kdtree_surface.at(current_lidar)->setInputCloud(surfPointsLessFlatPtr);
        _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;
        _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
        _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;

        StateHelper::marginalize_old_clone_lidar(state);

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

    rT1 = boost::posix_time::microsec_clock::local_time();

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast = _laserCloudCornerLast.at(current_lidar);
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast = _laserCloudSurfLast.at(current_lidar);

    std::vector<EdgeFeature> edge_list;

    Eigen::Quaterniond q_last_curr(1, 0, 0, 0);
    Eigen::Vector3d t_last_curr(0, 0, 0);    



    // // lidar_feature_extraction();
    // // do_lidar_propagate_update(timestamp, edge_list);
    double timestamp =  _lidar_time_buffer.at(current_lidar);


    if (state->_timestamp >= timestamp)
    {
        printf("%3f, %3f\n", timestamp, state->_timestamp);
        printf(YELLOW "lidar received out of order (prop dt = %3f)\n" RESET, (timestamp - state->_timestamp));

        _lidar_time_buffer.at(current_lidar) = -1;

        return;
    }

    ov_msckf::Propagator *propagator = sys->get_propagator();
    propagator->propagate_and_clone_lidar(state, timestamp);

    if ((int)state->_clones_IMU_lidar.size() < std::min(state->_options.max_clone_size, 5))
    {
        printf("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU_lidar.size(), std::min(state->_options.max_clone_size, 5));

        _lidar_time_buffer_last.at(current_lidar) = _lidar_time_buffer.at(current_lidar);
        _lidar_time_buffer.at(current_lidar) = timem;
        _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
        _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;
        _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;
        return;
    }

    Eigen::Quaterniond q_last = _q_laser_last.at(current_lidar);
    Eigen::Vector3d t_last = _t_laser_last.at(current_lidar);


    // Get calibration for our anchor camera
    Eigen::Matrix<double, 3, 3> R_ItoL = state->_calib_IMUtoLIDAR.at(current_lidar)->Rot();
    Eigen::Matrix<double, 3, 1> p_IinL = state->_calib_IMUtoLIDAR.at(current_lidar)->pos();

    // Eigen::Matrix<double, 3, 3> R_last = q_last.toRotationMatrix();//.transpose();
    Eigen::Matrix<double, 3, 3> R_last = state->_clones_IMU_lidar.at(_lidar_time_buffer_last.at(current_lidar))->Rot().transpose();

    // Eigen::Matrix<double, 3, 3> R_curr = state->_imu->Rot().transpose();
    Eigen::Matrix<double, 3, 3> R_curr = state->_clones_IMU_lidar.at(_lidar_time_buffer.at(current_lidar))->Rot().transpose();

    Eigen::Matrix<double, 3, 3> R_curr_to_last = R_ItoL * R_last * (R_ItoL * R_curr).transpose();

    // Eigen::Matrix<double, 3, 1> p_last = t_last;
    Eigen::Matrix<double, 3, 1> p_last = state->_clones_IMU_lidar.at(_lidar_time_buffer_last.at(current_lidar))->pos();

    // Eigen::Matrix<double, 3, 1> p_curr = state->_imu->pos();
    Eigen::Matrix<double, 3, 1> p_curr = state->_clones_IMU_lidar.at(_lidar_time_buffer.at(current_lidar))->pos();
    Eigen::Matrix<double, 3, 1> p_LinI = -R_ItoL.transpose() * p_IinL;
    Eigen::Matrix<double, 3, 1> p_CurrinLast = R_ItoL * R_last * (p_curr - p_last + R_curr.transpose() * p_LinI) + p_IinL;

    Eigen::Quaterniond deltaQ(R_curr_to_last);
    Eigen::Vector3d deltaT = p_CurrinLast;

    // Eigen::Quaterniond deltaQ(1, 0, 0, 0);
    // Eigen::Vector3d deltaT(0, 0, 0);

    // std::cout<<"delta: "<<deltaT.transpose()<<" "<<std::endl;
    // std::cout<<timestamp<<"\n";
    rT2 = boost::posix_time::microsec_clock::local_time();

    get_correspondences(kdtreeCornerLast, kdtreeSurfLast, laserCloudCornerLast, laserCloudSurfLast, cornerPointsSharpPtr, cornerPointsLessSharpPtr, surfPointsFlatPtr, surfPointsLessFlatPtr, deltaQ, deltaT, edge_list, (_lidar_time_buffer_last.at(current_lidar)), (_lidar_time_buffer.at(current_lidar)));

    rT3 = boost::posix_time::microsec_clock::local_time();

    // cout<<"clone pose: \n"<<p_last.transpose()<<endl<<p_curr.transpose()<<endl;


    // // // UpdateMSCKF

    if((int)edge_list.size() > sys->get_state()->_options.max_loam_in_update)
        edge_list.erase(edge_list.begin(), edge_list.end() - state->_options.max_loam_in_update);

    std::vector<EdgeFeature> feature_vec = edge_list;
    {
        // Return if no features
        if (feature_vec.empty())
            return;

        // Calculate the max possible measurement size
        // residual dim 1
        size_t max_meas_size = feature_vec.size();

        // Calculate max possible state size (i.e. the size of our covariance)
        // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
        size_t max_hx_size = state->max_covariance_size();

        // Large Jacobian and residual of *all* features for this update
        Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
        // cout<<"res_big: "<<(res_big.rows())<<" "<<(res_big.cols())<<endl;
        Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
        // cout << "Hx_big: " << (Hx_big.rows()) << " " << (Hx_big.cols()) << endl;

        // Our noise is isotropic, so make it here after our compression
        Eigen::MatrixXd R_big = params.sigma_pix_lidar * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
        Eigen::VectorXd n_big = Eigen::VectorXd::Zero(max_meas_size);

        
        std::unordered_map<Type *, size_t> Hx_mapping;
        std::vector<Type *> Hx_order_big;
        size_t ct_jacob = 0;
        size_t ct_meas = 0;

        // H_x only related to clone poses and calibration

        PoseJPL *calibration = state->_calib_IMUtoLIDAR.at(current_lidar);
        // If doing lidar extrinsics

        // Add this clone if it is not added already
        PoseJPL *clone_I0 = state->_clones_IMU_lidar.at(feature_vec[0].timestamp0);
        if (Hx_mapping.find(clone_I0) == Hx_mapping.end())
        {
            Hx_mapping.insert({clone_I0, ct_jacob});
            Hx_order_big.push_back(clone_I0);
            ct_jacob += clone_I0->size();
        }

        PoseJPL *clone_I1 = state->_clones_IMU_lidar.at(feature_vec[0].timestamp1);
        if (Hx_mapping.find(clone_I1) == Hx_mapping.end())
        {
            Hx_mapping.insert({clone_I1, ct_jacob});
            Hx_order_big.push_back(clone_I1);
            ct_jacob += clone_I1->size();
        }

        if (state->_options.do_calib_lidar_pose)
        {
            Hx_mapping.insert({calibration, ct_jacob});
            Hx_order_big.push_back(calibration);
            ct_jacob += calibration->size();
        }


        // 4. Compute linear system for each feature, nullspace project, and reject
        auto feat = feature_vec.begin();
        while (feat != feature_vec.end())
        {
            // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
            Eigen::MatrixXd H_f;
            Eigen::MatrixXd H_x;
            Eigen::MatrixXd R_x;
            Eigen::VectorXd res;

            int total_hx = ct_jacob;
            std::unordered_map<Type *, size_t> map_hx;


            // Allocate our residual and Jacobians
            int c = 0;
            int jacobsize = 3;

            R_x = Eigen::VectorXd::Zero(1 * 1);
            res = Eigen::VectorXd::Zero(1 * 1);
            H_f = Eigen::MatrixXd::Zero(1 * 1, jacobsize);
            H_x = Eigen::MatrixXd::Zero(1 * 1, total_hx);

            // // get_feature_jacobian_full()
            {
                // // get_feature_jacobian_representation()
                {
                    Eigen::Matrix3d skew_b_a, temp0;
                    Eigen::Matrix<double, 1, 3> temp1;
                    Eigen::Vector3d axb;

                    skew_b_a.noalias() = skew_x(feat->p_L0_b - feat->p_L0_a);

                    Eigen::Matrix<double, 3, 1> temp_m = skew_x(feat->p_L0_b - feat->p_L0_a) * feat->p_L1inL0 + skew_x(feat->p_L0_a) *feat->p_L0_b;

                    Eigen::Matrix<double, 3, 3> temp_dm = skew_x(feat->p_L0_b - feat->p_L0_a);

                    Eigen::Matrix<double, 1, 3> dr_dpf = 2*temp_m.transpose()*temp_dm;

                    Eigen::Matrix3d dpf_dr0 = R_ItoL * skew_x(R_last * (R_curr.transpose() * R_ItoL * feat->p_L1)) + R_ItoL * skew_x(R_last * (p_curr - p_last + R_curr.transpose() * p_LinI));

                    Eigen::Matrix3d dpf_dp0 = -R_ItoL * R_last;

                    Eigen::Matrix3d dpf_dr1 = -R_ItoL * R_last * R_curr.transpose() * skew_x(R_ItoL.transpose() * feat->p_L1 + p_LinI);

                    Eigen::Matrix3d dpf_dp1 = R_ItoL * R_last;

                    Eigen::Matrix3d dpf_dril = skew_x(R_curr_to_last * (feat->p_L1 - p_IinL)) - R_curr_to_last * skew_x(feat->p_L1 - p_IinL);

                    Eigen::Matrix3d dpf_dpil = -R_curr_to_last + Eigen::Matrix3d::Identity();



                    // Jacobian for our anchor pose
                    Eigen::Matrix<double, 1, 6> H_anc0;
                    H_anc0.block(0, 0, 1, 3).noalias() = dr_dpf*dpf_dr0;
                    H_anc0.block(0, 3, 1, 3).noalias() = dr_dpf*dpf_dp0;
                    H_x.block(0, Hx_mapping[clone_I0], 1, clone_I0->size()).noalias() += H_anc0;

                    // Jacobian for our anchor pose
                    Eigen::Matrix<double, 1, 6> H_anc1;
                    H_anc1.block(0, 0, 1, 3).noalias() = dr_dpf*dpf_dr1;
                    H_anc1.block(0, 3, 1, 3).noalias() = dr_dpf*dpf_dp1;
                    H_x.block(0, Hx_mapping[clone_I1], 1, clone_I1->size()).noalias() += H_anc1;

                    // Get calibration Jacobians (for anchor clone)
                    if (state->_options.do_calib_lidar_pose)
                    {
                        Eigen::Matrix<double, 1, 6> H_calib;
                        H_calib.block(0, 0, 1, 3).noalias() = dr_dpf*dpf_dril;
                        H_calib.block(0, 3, 1, 3).noalias() = dr_dpf*dpf_dpil;

                        H_x.block(0, Hx_mapping[calibration], 1, calibration->size()).noalias() += H_calib;

                        // Hx_order.push_back(state->_calib_IMUtoLIDAR.at(feat->timestamp0));
                        // H_x.push_back(H_calib);
                    }

                    Eigen::Matrix<double, 1, 3> Ji = dr_dpf * R_curr_to_last;

                    Eigen::Matrix<double, 3, 1> mm = skew_x(feat->p_L1inL0)*(feat->p_L0_a - feat->p_L0_b) + skew_x(feat->p_L0_a)*feat->p_L0_b;

                    Eigen::Matrix<double, 3, 1> nn = feat->p_L0_a - feat->p_L0_b;

                    Eigen::Matrix<double, 3, 3> dm_da = skew_x(feat->p_L1inL0 - feat->p_L0_b);
                    Eigen::Matrix<double, 3, 3> dm_db = skew_x(feat->p_L0_a - feat->p_L1inL0);
                    Eigen::Matrix<double, 3, 3> dn_da = Eigen::Matrix3d::Identity();
                    Eigen::Matrix<double, 3, 3> dn_db = -Eigen::Matrix3d::Identity();

                    double denominator = nn.transpose()*nn;

                    Eigen::Matrix<double, 1, 3> Jj = (2*mm.transpose()*dm_da*denominator - 2*mm.transpose()*mm*nn.transpose()*dn_da)/(denominator*denominator);
                    Eigen::Matrix<double, 1, 3> Jk = (2*mm.transpose()*dm_db*denominator - 2*mm.transpose()*mm*nn.transpose()*dn_db)/(denominator*denominator);

                    R_x = (params.sigma_pix_lidar * Ji * Ji.transpose() + params.sigma_pix_lidar * Jj * Jj.transpose() + params.sigma_pix_lidar * Jk * Jk.transpose());
                    // cout << " Ji:" << Ji * Ji.transpose() << endl;
                    // cout << " Jj:" << Jj * Jj.transpose() << endl;
                    // cout << " Jk:" << Jk * Jk.transpose() << endl;
                    // cout << "R_x:"<< params.sigma_pix_lidar * Ji * Ji.transpose() + params.sigma_pix_lidar * Jj * Jj.transpose() + params.sigma_pix_lidar * Jk * Jk.transpose()<<endl;
                }

                res(0) = feat->res_estimation;
                
            }

            // // // nullspace_place()  // --no need

            /// Chi2 distance check -- skip
            Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order_big);
            Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
            // S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());

            S.diagonal() += R_x(0) * Eigen::VectorXd::Ones(S.rows());
            double chi2 = res.dot(S.llt().solve(res));

            // Get our threshold (we precompute up to 500 but handle the case that it is more)
            double chi2_check;
            if (res.rows() < 500)
            {
                chi2_check = chi_squared_table[res.rows()];
            }
            else
            {
                boost::math::chi_squared chi_squared_dist(res.rows());
                chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
                printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
            }

            // Check if we should delete or not
            if (chi2 > 1 * chi2_check)
            {
                // (*it2)->to_delete = true;
                feat = feature_vec.erase(feat);
                //cout << "featid = " << feat.featid << endl;
                //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
                //cout << "res = " << endl << res.transpose() << endl;
                continue;
            }



            // We are good!!! Append to our large H vector
            size_t ct_hx = 0;
            for (const auto &var : Hx_order_big)
            {
                // cout << var->id()<<" ";
                Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
                ct_hx += var->size();
            }
            // Append our residual and move forward
            res_big.block(ct_meas, 0, res.rows(), 1) = res;
            // R_big.block(ct_meas, ct_meas, R_x.rows(), 1) = R_x;
            R_big(ct_meas, ct_meas) = sqrt(R_x(0, 0));
            n_big.block(ct_meas, 0, res.rows(), 1) = R_x;
            ct_meas += res.rows();

            feat++;
        }
        // cout<<endl;

        // Return if we don't have anything and resize our matrices
        if (ct_meas < 1)
        {
            return;
        }
        assert(ct_meas <= max_meas_size);
        assert(ct_jacob <= max_hx_size);
        // cout << "H_x  res:" << (Hx_big.rows()) << " " << (Hx_big.cols()) << " " << (res_big.rows()) << endl;
        // cout<<"ct_meas: "<<ct_meas<<" "<<ct_jacob<<endl;
        res_big.conservativeResize(ct_meas, 1);
        Hx_big.conservativeResize(ct_meas, ct_jacob);
        R_big.conservativeResize(ct_meas, ct_meas);

        // cout << "before H_x.rows() <= H_x.cols(): " << (Hx_big.rows()) << " " << (Hx_big.cols()) << " " << (res_big.rows()) << " " << (res_big.cols()) << " " << (R_big.rows()) << " " << (R_big.cols()) << endl;

        // cout<<"before\n"<<"H_x\n"<<Hx_big<<endl<<"res_big\n"<<res_big<<endl<<"R_big\n"<<R_big<<endl;

        // cout<<"after chi2: "<< ct_meas<<endl;
        // 5. Perform measurement compression

        // R_big = params.sigma_pix_lidar * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
        // cout << "R_big_before: " << (R_big.rows()) << " " << (R_big.cols()) << endl << R_big << endl;
        UpdaterHelper::measurement_compress_inplace(Hx_big, res_big, R_big);

        // cout << "R_big: " << (R_big.rows()) << " " << (R_big.cols()) << endl << R_big << endl;

        Eigen::MatrixXd RRT = R_big * R_big.transpose();
        // cout << "RRT: " << (RRT.rows())<<" "<<(RRT.cols())<<endl << RRT << endl;
        // cout << "H_x.rows() <= H_x.cols(): " << (Hx_big.rows()) << " " << (Hx_big.cols()) << " " << (res_big.rows()) << " " << (res_big.cols()) << " " << (R_big.rows()) << " " << (R_big.cols()) << endl;
        if (Hx_big.rows() < 1)
        {
            return;
        }
        rT4 = boost::posix_time::microsec_clock::local_time();


        // // Our noise is isotropic, so make it here after our compression
        // // Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
        // R_big = params.sigma_pix_lidar * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

        //         // 6. With all good features update the state
        // cout<<"H_x\n"<<Hx_big<<endl<<"res_big\n"<<res_big<<endl<<"R_big\n"<<R_big<<endl;
                StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, RRT);

                // UpdaterHelper::measurement_compress_inplace(Hx_big, res_big, n_big);
                // cout<<"n_big:" <<n_big.transpose()<<endl;

        rT5 = boost::posix_time::microsec_clock::local_time();

    }


    _q_laser_last.at(current_lidar) = Eigen::Quaterniond(R_curr);
    _t_laser_last.at(current_lidar) = state->_clones_IMU_lidar.at(_lidar_time_buffer.at(current_lidar))->pos();

    StateHelper::marginalize_old_clone_lidar(state);

    // printf("sharp corner: %d, edge feature: %d\n", cornerPointsSharp.size(), edge_list.size());
    _lidar_time_buffer_last.at(current_lidar) = _lidar_time_buffer.at(current_lidar);
    _lidar_time_buffer.at(current_lidar) = timem;
    _laserCloudCornerLast.at(current_lidar) = cornerPointsLessSharpPtr;
    _laserCloudSurfLast.at(current_lidar) = surfPointsLessFlatPtr;
    _lidar_pc_buffer.at(current_lidar) = laserCloudInPtr;

    rT6 = boost::posix_time::microsec_clock::local_time();

    printf("[TIME-LIDAR]: %.4f seconds for scan registration\n", (rT1 - rT0).total_microseconds() * 1e-6);
    printf("[TIME-LIDAR]: %.4f seconds for calculating matrix\n", (rT2 - rT1).total_microseconds() * 1e-6);
    printf("[TIME-LIDAR]: %.4f seconds for getting correspondence\n", (rT3 - rT2).total_microseconds() * 1e-6);
    printf("[TIME-LIDAR]: %.4f seconds for getting Hx_big\n", (rT4 - rT3).total_microseconds() * 1e-6);
    printf("[TIME-LIDAR]: %.4f seconds for ekf update\n", (rT5 - rT4).total_microseconds() * 1e-6);
    // if ((rT6 - rT0).total_microseconds() * 1e-6 > 0.008)
    printf("[TIME-LIDAR]: %.4f seconds for all lidar\n", (rT6 - rT0).total_microseconds() * 1e-6);
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
