#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <geometry_msgs/TransformStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/ColorRGBA.h>
#include <person_msgs/PersonCovList.h>
#include <person_msgs/Person2DList.h>
#include <skeleton_3d/fusion_body_parts.h>

#include <string>
#include <vector>
#include <map>
#include <functional>

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::map;
using person_msgs::PersonCovList;
using person_msgs::Person2DList;
using person_msgs::Person2D;

const string BASE_FRAME = "base";
const string g_cam_frame = "_color_optical_frame";
const string g_cam_info_topic = "/color/camera_info";
const string g_skel_pred_topic = "/skel_pred";
static vector<string> CAM_FRAMES = {"cam_1_color_optical_frame", "cam_2_color_optical_frame", "cam_3_color_optical_frame", "cam_4_color_optical_frame"};
static vector<string> CAM_INFO_TOPICS = {"cam_1/color/camera_info", "cam_2/color/camera_info", "cam_3/color/camera_info", "cam_4/color/camera_info"};
static vector<string> SKELETON_PRED_TOPICS = {"cam_1/skel_pred", "cam_2/skel_pred", "cam_3/skel_pred", "cam_4/skel_pred"};
static unsigned int NUM_CAMERAS = 4;

const string FUSED_SKELETON_TOPIC = "human_pose_estimation/persons3d_fused_pred";

static unsigned int NUM_KEYPOINTS = 17;

static string g_param_pose_method = "simple"; // "simple" or h36m

const int g_kp2kpFusion_idx_simple[17] = {FUSION_BODY_PARTS::Nose,
                                              FUSION_BODY_PARTS::LEye, FUSION_BODY_PARTS::REye, FUSION_BODY_PARTS::LEar, FUSION_BODY_PARTS::REar,
                                              FUSION_BODY_PARTS::LShoulder, FUSION_BODY_PARTS::RShoulder, FUSION_BODY_PARTS::LElbow, FUSION_BODY_PARTS::RElbow, FUSION_BODY_PARTS::LWrist, FUSION_BODY_PARTS::RWrist,
                                              FUSION_BODY_PARTS::LHip, FUSION_BODY_PARTS::RHip, FUSION_BODY_PARTS::LKnee, FUSION_BODY_PARTS::RKnee, FUSION_BODY_PARTS::LAnkle, FUSION_BODY_PARTS::RAnkle};
const int g_kp2kpFusion_idx_h36m[17] = {FUSION_BODY_PARTS::Nose, FUSION_BODY_PARTS::Head, FUSION_BODY_PARTS::Neck, FUSION_BODY_PARTS::Belly, FUSION_BODY_PARTS::MidHip,
                                              FUSION_BODY_PARTS::LShoulder, FUSION_BODY_PARTS::RShoulder, FUSION_BODY_PARTS::LElbow, FUSION_BODY_PARTS::RElbow, FUSION_BODY_PARTS::LWrist, FUSION_BODY_PARTS::RWrist,
                                              FUSION_BODY_PARTS::LHip, FUSION_BODY_PARTS::RHip, FUSION_BODY_PARTS::LKnee, FUSION_BODY_PARTS::RKnee, FUSION_BODY_PARTS::LAnkle, FUSION_BODY_PARTS::RAnkle};
static const int* g_kp2kpFusion_idx;

const int DIM = 3;
const int N_SAMPLES = 2*DIM+1;
typedef Eigen::Matrix<double, DIM, N_SAMPLES> SamplesMatType;
typedef Eigen::Matrix<double, 1, N_SAMPLES> SamplesWeightType;
typedef Eigen::Matrix<double, 2, N_SAMPLES> TransformedSampledMatType;

SamplesWeightType draw_sigma_points(SamplesMatType& sigma_points, Eigen::Vector3d mean, Eigen::Matrix3d cov){
  const double kappa = 0.5; // (scaling) equal weight on all samples // TODO: find best parameter (or use scaled transform, kappa=0, alpha = 1e-3, beta = 2)
  SamplesWeightType weights;
  weights << 2*kappa, Eigen::Matrix<double, 1, N_SAMPLES - 1>::Ones();
  weights /= (2.0 * (DIM + kappa));

  Eigen::Matrix3d mat = std::sqrt(DIM + kappa) * Eigen::Matrix3d::Identity();
  SamplesMatType samples_std_normal;
  samples_std_normal << Eigen::Vector3d::Zero(), -mat, mat;

  sigma_points = cov.llt().matrixL() * samples_std_normal + mean.replicate<1, N_SAMPLES>();

  return weights;
}

void getTransforms(map<string, geometry_msgs::TransformStamped>& transforms_cam, const tf2_ros::Buffer& tfBuffer){
  bool success = false;
  while (!success && ros::ok()){
    try {
      for (int i = 0; i < NUM_CAMERAS; ++i) {
        auto transform = tfBuffer.lookupTransform(CAM_FRAMES[i], BASE_FRAME, ros::Time(0));
        auto ret = transforms_cam.insert(std::pair<string, geometry_msgs::TransformStamped>(CAM_FRAMES[i], transform));
        if(!ret.second){
          ROS_ERROR("transform for frame %s already exists!", ret.first->first.c_str());
        }
      }
    } catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
        continue;
    }

    if(transforms_cam.size() == NUM_CAMERAS){
      success = true;
      ROS_INFO("Sucessfully retrieved camera extrinsic transforms.");
    }
    else {
      ROS_ERROR("Wrong number of camera transforms! (need to get %d transforms, but received %zu.", NUM_CAMERAS, transforms_cam.size());
    }
  }
}

void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr msg, sensor_msgs::CameraInfo& msg_out){
  msg_out = *msg;
}

void getIntrinsics(vector<sensor_msgs::CameraInfo>& intrinsics, ros::NodeHandle& nh){
  vector<ros::Subscriber> intrinsic_subs;
  intrinsics.resize(NUM_CAMERAS);
  for (size_t i = 0; i < NUM_CAMERAS; ++i) {
    auto sub = nh.subscribe<sensor_msgs::CameraInfo>(CAM_INFO_TOPICS[i], 1, std::bind(cameraInfoCallback, std::placeholders::_1, std::ref(intrinsics[i])));
    intrinsic_subs.push_back(sub);
  }

  ros::Rate rate(1.0);
  bool success = false;
  while(!success && ros::ok()){
    ROS_INFO("Spinning.. Waiting to receive camera intrinsics.");
    ros::spinOnce();
    rate.sleep();

    success = true;
    for (const auto& intr : intrinsics) {
      if (intr.D.size() == 0 && intr.distortion_model != "none"){ //D empty as no distortion modeled
        success = false;
        break;
      }
    }
  }

  intrinsic_subs.clear();
  ROS_INFO("intrinsics received.");
  for (const auto& intr : intrinsics) {
    cout << "fx: " << intr.K[0] << ", fy: " << intr.K[4] << ", cx: " << intr.K[2] << ", cy: " << intr.K[5] << endl;
  }
}

void fusedSkeletonCallback(const PersonCovList::ConstPtr& persons_msg, map<string, geometry_msgs::TransformStamped>& transforms_cam, const vector<sensor_msgs::CameraInfo>& intrinsics, const vector<ros::Publisher>& skel_pubs){
  if(persons_msg->header.frame_id != BASE_FRAME){
    ROS_ERROR("Fused person is not given in \"%s\" but in \"%s\". Aborting!", BASE_FRAME.c_str(), persons_msg->header.frame_id.c_str());
    return;
  }

  const int num_trasforms = transforms_cam.size();
  const int num_persons = persons_msg->persons.size();
  vector<Person2DList> persons_proj_vec(num_trasforms);
  vector<Eigen::Affine3d> to_cam_eigen(num_trasforms);
  vector<image_geometry::PinholeCameraModel> cam_intrinsics(num_trasforms);
  std::vector<std::vector<double> > min_x(num_trasforms), min_y(num_trasforms), max_x(num_trasforms, std::vector<double>(num_persons, 0)), max_y(num_trasforms, std::vector<double>(num_persons, 0));

  for (int i = 0; i < num_trasforms; ++i) {
    string cam_frame = intrinsics[i].header.frame_id;
    to_cam_eigen[i] = tf2::transformToEigen(transforms_cam[cam_frame]);
    cam_intrinsics[i].fromCameraInfo(intrinsics[i]);

    persons_proj_vec[i].header.frame_id = intrinsics[i].header.frame_id;
    persons_proj_vec[i].header.stamp = persons_msg->ts_per_cam[i];
    persons_proj_vec[i].fb_delay = persons_msg->fb_delay_per_cam[i];

    min_x[i] = std::vector<double>(num_persons, intrinsics[i].width);
    min_y[i] = std::vector<double>(num_persons, intrinsics[i].height);
  }

  for (int person_idx = 0; person_idx < num_persons; ++person_idx) {
    if(persons_msg->persons[person_idx].keypoints.size() != FUSION_BODY_PARTS::NUM_KEYPOINTS){
      ROS_ERROR("Fused person %d: Expected skeleton to have %d Keypoints, but got %zu. Aborting!", person_idx, FUSION_BODY_PARTS::NUM_KEYPOINTS, persons_msg->persons[person_idx].keypoints.size());
      continue;
    }

    std::vector<Person2D> person_in_cam(num_trasforms);
    std::vector<int> num_valid_kps_in_cam(num_trasforms, 0);
    for (int i = 0; i < num_trasforms; ++i) {
      person_in_cam[i].keypoints.resize(NUM_KEYPOINTS);
      person_in_cam[i].score = 1.0; // TODO, no per-person total score available
    }

    for (int kp_idx = 0; kp_idx < NUM_KEYPOINTS; ++kp_idx) {
      const auto& kp_3d = persons_msg->persons[person_idx].keypoints.at(g_kp2kpFusion_idx[kp_idx]);

      if(kp_3d.score <= 0.0f) //if(kp_3d.joint.x == 0.0 && kp_3d.joint.y == 0.0 && kp_3d.joint.z == 0.0)
        continue;

      Eigen::Matrix3d cov;
      cov << kp_3d.cov[0], kp_3d.cov[1], kp_3d.cov[2],
             kp_3d.cov[1], kp_3d.cov[3], kp_3d.cov[4],
             kp_3d.cov[2], kp_3d.cov[4], kp_3d.cov[5];
      Eigen::Vector3d joint_base(kp_3d.joint.x, kp_3d.joint.y, kp_3d.joint.z);
      SamplesMatType samples;
      SamplesWeightType weights = draw_sigma_points(samples, joint_base, cov); //draw sigma points.
      TransformedSampledMatType weights_replicated = weights.replicate<2,1>();

      for (int i = 0; i < num_trasforms; ++i) {
        TransformedSampledMatType samples_proj;
        for(int sample_idx = 0; sample_idx < N_SAMPLES; ++sample_idx){ //project sigma-points to image plane
          Eigen::Vector3d joint_cam = to_cam_eigen[i] * samples.col(sample_idx); // project joint to camera frame
          cv::Point2d joint_px = cam_intrinsics[i].project3dToPixel(cv::Point3d(joint_cam.x(), joint_cam.y(), joint_cam.z())); // project joint to image plane
          samples_proj(0, sample_idx) = joint_px.x;
          samples_proj(1, sample_idx) = joint_px.y;
        }

        Eigen::Vector2d joint_px = (samples_proj.cwiseProduct(weights_replicated)).rowwise().sum();
        TransformedSampledMatType samples_proj_centered = samples_proj - joint_px.replicate<1, N_SAMPLES>();
        Eigen::Matrix2d cov_2d = samples_proj_centered.cwiseProduct(weights_replicated) * samples_proj_centered.transpose();
        //cout << "Joint: " << kp_idx << ", camera: " << person_idx << ": pixel-covariance: " << endl << cov_2d << endl;

        if(joint_px.x() < 0 || joint_px.x() > intrinsics[i].width || joint_px.y() < 0 || joint_px.y() > intrinsics[i].height) // joint is outside of image plane
          continue;

        ++num_valid_kps_in_cam[i];
        person_in_cam[i].keypoints[kp_idx].x = static_cast<float>(joint_px.x()); // abs coordinates
        person_in_cam[i].keypoints[kp_idx].y = static_cast<float>(joint_px.y()); // abs coordinates
        person_in_cam[i].keypoints[kp_idx].score = kp_3d.score;
        person_in_cam[i].keypoints[kp_idx].cov[0] = static_cast<float>(cov_2d(0, 0));
        person_in_cam[i].keypoints[kp_idx].cov[1] = static_cast<float>(cov_2d(0, 1));
        person_in_cam[i].keypoints[kp_idx].cov[2] = static_cast<float>(cov_2d(1, 1));

        if (joint_px.x() < min_x[i][person_idx]){min_x[i][person_idx] = joint_px.x();}
        if (joint_px.y() < min_y[i][person_idx]){min_y[i][person_idx] = joint_px.y();}
        if (joint_px.x() > max_x[i][person_idx]){max_x[i][person_idx] = joint_px.x();}
        if (joint_px.y() > max_y[i][person_idx]){max_y[i][person_idx] = joint_px.y();}
      }
    }

    for (int i = 0; i < num_trasforms; ++i) {
      if(num_valid_kps_in_cam[i] > 0){
        person_in_cam[i].bbox = {(float)min_x[i][person_idx], (float)min_y[i][person_idx], (float)max_x[i][person_idx], (float)max_y[i][person_idx]};
        persons_proj_vec[i].persons.push_back(person_in_cam[i]);
      }
    }
  }

  for (int i = 0; i < num_trasforms; ++i)
    skel_pubs[i].publish(persons_proj_vec[i]);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "multi_skeleton_reprojection");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  nh_private.param<string>("pose_method", g_param_pose_method, "simple");
  std::vector<string> cam_names;
  nh_private.param("cameras", cam_names, std::vector<string>());
  if(cam_names.size() > 0){
    NUM_CAMERAS = cam_names.size();
    CAM_FRAMES.clear();
    CAM_INFO_TOPICS.clear();
    SKELETON_PRED_TOPICS.clear();
    for (int i = 0; i < NUM_CAMERAS; ++i) {
      CAM_FRAMES.push_back((string)cam_names[i] + g_cam_frame);
      CAM_INFO_TOPICS.push_back((string)cam_names[i] + g_cam_info_topic);
      SKELETON_PRED_TOPICS.push_back((string)cam_names[i] + g_skel_pred_topic);
    }
  }

  ROS_INFO("NUM_CAMERAS: %d, topics and frames: ", NUM_CAMERAS);
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    cout << "\t" << CAM_FRAMES[i] << ", " << CAM_INFO_TOPICS[i] << ", " << SKELETON_PRED_TOPICS[i] << endl;
  }

  if (g_param_pose_method == "h36m")
    g_kp2kpFusion_idx = g_kp2kpFusion_idx_h36m;
  else
    g_kp2kpFusion_idx = g_kp2kpFusion_idx_simple;

  NUM_KEYPOINTS = 17;
  ROS_INFO("Using pose estimation method: %s, and %d keypoints.", g_param_pose_method.c_str(), NUM_KEYPOINTS);

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  map<string, geometry_msgs::TransformStamped> transforms_cam;
  getTransforms(transforms_cam, tfBuffer);

  vector<sensor_msgs::CameraInfo> intrinsics;
  getIntrinsics(intrinsics, nh);

  vector<ros::Publisher> skeleton_pubs;
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    skeleton_pubs.push_back(nh.advertise<Person2DList>(SKELETON_PRED_TOPICS[i], 1));
  }

  ROS_INFO("Reprojecting into %zu / %d camera views", transforms_cam.size(), NUM_CAMERAS);

  if(transforms_cam.size() != intrinsics.size() || transforms_cam.size() != skeleton_pubs.size() || intrinsics.size() != skeleton_pubs.size()){
    ROS_ERROR("incoherent number of transforms, intrinsics and output heatmaps! Aborting!");
    return -1;
  }
  
  ros::Subscriber fused_skel_sub = nh.subscribe<PersonCovList>(FUSED_SKELETON_TOPIC, 1, std::bind(fusedSkeletonCallback, std::placeholders::_1, std::ref(transforms_cam), std::cref(intrinsics), std::cref(skeleton_pubs)), ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  ros::spin();
}
