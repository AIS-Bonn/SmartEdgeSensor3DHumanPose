#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include "my_message_filters/sync_policies/approximate_time_vec.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>

#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/ColorRGBA.h>
#include <image_geometry/pinhole_camera_model.h>
#include <person_msgs/PersonCovList.h>
#include <person_msgs/Person2DList.h>
#include <visualization_msgs/MarkerArray.h>
#include <skeleton_3d/fusion_body_parts.h>
#include <Hungarian.h>

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::map;
using person_msgs::Person2DList;
using person_msgs::KeypointWithCovariance;
using person_msgs::PersonCov;
using person_msgs::PersonCovList;

static constexpr int max_num_timings = 10;
static std::vector<double> g_timings(max_num_timings, 0.0);
static std::vector<int> g_timing_cnt(max_num_timings, 0);

const double MAX_COSTS = 1e6;

const string BASE_FRAME = "base";
const string g_cam_frame = "_color_optical_frame";
const string g_cam_info_topic = "/color/camera_info";
const string g_skel_2d_topic = "/human_joints";
static vector<string> CAM_FRAMES = {"cam_1_color_optical_frame", "cam_2_color_optical_frame", "cam_3_color_optical_frame", "cam_4_color_optical_frame"};
static vector<string> CAM_INFO_TOPICS = {"cam_1/color/camera_info", "cam_2/color/camera_info", "cam_3/color/camera_info", "cam_4/color/camera_info"};
static vector<string> SKELETON_2D_TOPICS = {"cam_1/human_joints", "cam_2/human_joints", "cam_3/human_joints", "cam_4/human_joints"};
static unsigned int NUM_CAMERAS = 4;
const string SKELETON_3D_TOPIC = "human_pose_estimation/skeleton_3d";
const string PERSON_3D_TOPIC = "human_pose_estimation/persons_3d";

static int NUM_KEYPOINTS = 17;
const int g_min_num_valid_keypoints = 9; // 5
static float g_triangulation_threshold = 0.30f; // threshold on confidence values for triangulation
const double g_reproj_error_max_acceptable = 0.050; // in normalized image coordinates (roughly 50 pixels (fx / fy is around 1000))
static double g_max_epipolar_error = 0.050; // in normalized image coordinates
const double g_max_joint_dist_to_root = 2.0; //joints can have a maximum distance of 2m to the root joint (mid-hip)
const double g_merge_dist_thresh = 0.20; // merge detections when average distance is smaller than threshold (m)
const double g_avg_delay = 0.10; // avg. pipeline delay for feedback, seconds
const double g_max_sync_diff = 0.067; // seconds

static std::vector<Person2DList::ConstPtr> g_skel_data;
static bool g_skel_data_updated = false;
static std::mutex g_skel_data_mutex;
static std::condition_variable g_skel_data_cv;

static const int* g_kpParent;
static const double* g_limbLength;
static const double* g_limbLSigma;
static const int* g_kp2kpFusion_idx;

static string g_param_pose_method = "simple"; // "simple" or "h36m"
static bool   g_param_vis_covariance = false;

static std::vector<std_msgs::ColorRGBA> g_colors;

struct EdgeTPU_BodyParts_Simple{
  static const int Nose = 0,
  RShoulder = 6,
  RElbow = 8,
  RWrist = 10,
  LShoulder = 5,
  LElbow = 7,
  LWrist = 9,
  RHip = 12,
  RKnee = 14,
  RAnkle = 16,
  LHip = 11,
  LKnee = 13,
  LAnkle = 15,
  REye = 2,
  LEye = 1,
  REar = 4,
  LEar = 3;
                                          // 0, 1,    2,    3,    4,     5,  6,  7,    8,    9,    10,   11,   12,   13,   14,   15,   16,
  static constexpr int kpParent[17]      = {-1, 0,    0,    1,    2,     0,  0,  5,    6,    7,    8,    5,    6,    11,   12,   13,   14};
  static constexpr double limbLength[17] = {-1, 0.05, 0.05, 0.10, 0.10, -1, -1,  0.28, 0.28, 0.25, 0.25, 0.50, 0.50, 0.45, 0.45, 0.446, 0.446}; // From H36M - Statistics (except 1-6)
  static constexpr double limbLSigma[17] = {-1, 0.05, 0.05, 0.05, 0.05, -1, -1,  0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10,  0.10};
  static constexpr double shoulderDist = 0.35, shoulderSigma = 0.15;
};

constexpr int EdgeTPU_BodyParts_Simple::kpParent[17];
constexpr double EdgeTPU_BodyParts_Simple::limbLength[17];
constexpr double EdgeTPU_BodyParts_Simple::limbLSigma[17];
constexpr double EdgeTPU_BodyParts_Simple::shoulderDist, EdgeTPU_BodyParts_Simple::shoulderSigma;

struct EdgeTPU_BodyParts_H36M{
  static const int Nose = 0,
  Head = 1,
  Neck = 2,
  Belly = 3,
  Root = 4,
  LShoulder = 5,
  RShoulder = 6,
  LElbow = 7,
  RElbow = 8,
  LWrist = 9,
  RWrist = 10,
  LHip = 11,
  RHip = 12,
  LKnee = 13,
  RKnee = 14,
  LAnkle = 15,
  RAnkle = 16;
                                          // 0, 1,     2,     3,     4,     5,     6,     7,    8,    9,    10,   11,    12,    13,    14,    15,    16,
  static constexpr int kpParent[17]      = {-1, 0,     0,     2,     3,     2,     2,     5,    6,    7,    8,    4,     4,     11,    12,    13,    14};
  static constexpr double limbLength[17] = {-1, 0.115, 0.116, 0.255, 0.238, 0.149, 0.149, 0.28, 0.28, 0.25, 0.25, 0.134, 0.134, 0.449, 0.449, 0.446, 0.446}; // From H36M - Statistics
  static constexpr double limbLSigma[17] = {-1, 0.07,  0.07,  0.15,  0.15,  0.10,  0.10,  0.15, 0.15, 0.15, 0.15, 0.10,  0.10,  0.20,  0.20,  0.20,  0.20};
};

constexpr int EdgeTPU_BodyParts_H36M::kpParent[17];
constexpr double EdgeTPU_BodyParts_H36M::limbLength[17];
constexpr double EdgeTPU_BodyParts_H36M::limbLSigma[17];

const int g_kp2kpFusion_idx_simple[17] = {FUSION_BODY_PARTS::Nose,
                                              FUSION_BODY_PARTS::LEye, FUSION_BODY_PARTS::REye, FUSION_BODY_PARTS::LEar, FUSION_BODY_PARTS::REar,
                                              FUSION_BODY_PARTS::LShoulder, FUSION_BODY_PARTS::RShoulder, FUSION_BODY_PARTS::LElbow, FUSION_BODY_PARTS::RElbow, FUSION_BODY_PARTS::LWrist, FUSION_BODY_PARTS::RWrist,
                                              FUSION_BODY_PARTS::LHip, FUSION_BODY_PARTS::RHip, FUSION_BODY_PARTS::LKnee, FUSION_BODY_PARTS::RKnee, FUSION_BODY_PARTS::LAnkle, FUSION_BODY_PARTS::RAnkle};
const int g_kp2kpFusion_idx_h36m[17] = {FUSION_BODY_PARTS::Nose, FUSION_BODY_PARTS::Head, FUSION_BODY_PARTS::Neck, FUSION_BODY_PARTS::Belly, FUSION_BODY_PARTS::MidHip,
                                              FUSION_BODY_PARTS::LShoulder, FUSION_BODY_PARTS::RShoulder, FUSION_BODY_PARTS::LElbow, FUSION_BODY_PARTS::RElbow, FUSION_BODY_PARTS::LWrist, FUSION_BODY_PARTS::RWrist,
                                              FUSION_BODY_PARTS::LHip, FUSION_BODY_PARTS::RHip, FUSION_BODY_PARTS::LKnee, FUSION_BODY_PARTS::RKnee, FUSION_BODY_PARTS::LAnkle, FUSION_BODY_PARTS::RAnkle};

static double g_min_sigmas_3d[3] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
static double g_max_sigmas_3d[3] = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
static double g_limbLCovOffsetSigma = 0.075; // standard deviation for noise offset due to limbe-length model

typedef Eigen::Matrix<float, 3, 4> Matrix34f;

struct PersonHypothesis{
  std::vector<std::vector<Eigen::Vector3f> > keypoints_normalized;
  std::vector<std::vector<Eigen::Matrix2f> > keypoints_cov_normalized;
  std::vector<Matrix34f> cameraExtrinsics;
  std::vector<int> cameraIDs;
  std::vector<float> score;
};

void getTransforms(map<string, Eigen::Affine3d>& transforms_cam, const tf2_ros::Buffer& tfBuffer){
  bool success = false;
  while (!success && ros::ok()){
    try {
      for (int i = 0; i < NUM_CAMERAS; ++i) {
        auto transform = tfBuffer.lookupTransform(CAM_FRAMES[i], BASE_FRAME, ros::Time(0));
        Eigen::Affine3d to_cam_eigen = tf2::transformToEigen(transform);
        auto ret = transforms_cam.insert(std::pair<string, Eigen::Affine3d>(CAM_FRAMES[i], to_cam_eigen));
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

  cout << "Transforms: " << endl;
  for (const auto & it : transforms_cam) {
    cout << it.first << ":" << endl;
    cout << it.second.matrix() << endl << endl;
  }
}

void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr msg, sensor_msgs::CameraInfo& msg_out){
  msg_out = *msg;
}

void getIntrinsics(vector<sensor_msgs::CameraInfo>& intrinsics, ros::NodeHandle& nh){
  vector<ros::Subscriber> intrinsic_subs;
  intrinsics.resize(NUM_CAMERAS);
  for (int i = 0; i < NUM_CAMERAS; ++i) {
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

inline void cross_prod_matrix(const Eigen::Vector3d& vec, Eigen::Matrix3d& mat_cross){
  mat_cross << 0, -vec(2), vec(1),
               vec(2), 0, -vec(0),
              -vec(1), vec(0), 0;
}

inline void pseudo_inv34d(const Eigen::Matrix<double,3,4>& mat, Eigen::Matrix<double,4,3>& mat_inv, double epsilon = std::numeric_limits<double>::epsilon()){
  auto svd = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(mat.cols(), mat.rows()) * svd.singularValues()[0];
  mat_inv = svd.matrixV() * Eigen::Vector3d( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();
}

int get_fundamental_idx(int i, int j){
  if(i >= j)
    return -1;
  if(i > NUM_CAMERAS - 2 || j > NUM_CAMERAS - 1)
    return -1;

  int start_idx = 0;
  for(int ii = 0; ii < i; ++ii)
    start_idx += NUM_CAMERAS - ii - 1;

  return start_idx + j - i - 1;
}

void setKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const Eigen::Matrix3d &cov){
  kp.cov[0] = cov(0, 0); // xx
  kp.cov[1] = cov(0, 1); // xy
  kp.cov[2] = cov(0, 2); // xz
  kp.cov[3] = cov(1, 1); // yy
  kp.cov[4] = cov(1, 2); // yz
  kp.cov[5] = cov(2, 2); // zz
}

void mergeKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const person_msgs::KeypointWithCovariance &kp1, const person_msgs::KeypointWithCovariance &kp2){
  kp.cov[0] = (kp1.cov[0] + kp2.cov[0]) / 2.0; // xx
  kp.cov[1] = (kp1.cov[1] + kp2.cov[1]) / 2.0; // xy
  kp.cov[2] = (kp1.cov[2] + kp2.cov[2]) / 2.0; // xz
  kp.cov[3] = (kp1.cov[3] + kp2.cov[3]) / 2.0; // yy
  kp.cov[4] = (kp1.cov[4] + kp2.cov[4]) / 2.0; // yz
  kp.cov[5] = (kp1.cov[5] + kp2.cov[5]) / 2.0; // zz
}

void addToKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const double& sigma){
  kp.cov[0] += sigma*sigma; //xx
  kp.cov[3] += sigma*sigma; //yy
  kp.cov[5] += sigma*sigma; //zz
}

void setMarkerPose(visualization_msgs::Marker& marker, const KeypointWithCovariance& kp) {
  marker.pose.position = kp.joint;

  Eigen::Matrix3d cov;
  cov << kp.cov[0], kp.cov[1], kp.cov[2],
         kp.cov[1], kp.cov[3], kp.cov[4],
         kp.cov[2], kp.cov[4], kp.cov[5];
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);

  Eigen::Quaterniond q_rot_cov;
  if(es.eigenvectors().determinant() > 0.0)
    q_rot_cov = Eigen::Quaterniond(es.eigenvectors());
  else
    q_rot_cov = Eigen::Quaterniond(-1.0 * es.eigenvectors()); // Determinant must be +1!

  marker.pose.orientation.w = q_rot_cov.w();
  marker.pose.orientation.x = q_rot_cov.x();
  marker.pose.orientation.y = q_rot_cov.y();
  marker.pose.orientation.z = q_rot_cov.z();

  marker.scale.x = 2.0 * 2.7955 * std::sqrt(es.eigenvalues().x()); //2.7955 = sqrt(chi2inv(0.95,3) -> 2-sigma interval
  marker.scale.y = 2.0 * 2.7955 * std::sqrt(es.eigenvalues().y());
  marker.scale.z = 2.0 * 2.7955 * std::sqrt(es.eigenvalues().z());

  if (es.eigenvalues().x() < g_min_sigmas_3d[0]){g_min_sigmas_3d[0] = es.eigenvalues().x();}
  if (es.eigenvalues().y() < g_min_sigmas_3d[1]){g_min_sigmas_3d[1] = es.eigenvalues().y();}
  if (es.eigenvalues().z() < g_min_sigmas_3d[2]){g_min_sigmas_3d[2] = es.eigenvalues().z();}

  if (es.eigenvalues().x() > g_max_sigmas_3d[0]){g_max_sigmas_3d[0] = es.eigenvalues().x();}
  if (es.eigenvalues().y() > g_max_sigmas_3d[1]){g_max_sigmas_3d[1] = es.eigenvalues().y();}
  if (es.eigenvalues().z() > g_max_sigmas_3d[2]){g_max_sigmas_3d[2] = es.eigenvalues().z();}
}

int normalize_keypoints(vector<Eigen::Vector3f>& keypoints_normalized, vector<Eigen::Matrix2f>& covs_normalized, const image_geometry::PinholeCameraModel& intr, const person_msgs::Person2D& human){
  int num_valid_kps = 0;
  const float fx = static_cast<float>(intr.fx());
  const float fy = static_cast<float>(intr.fy());
  const float cx = static_cast<float>(intr.cx());
  const float cy = static_cast<float>(intr.cy());

  for (int kp_idx = 0; kp_idx < NUM_KEYPOINTS; ++ kp_idx) {
    const auto& kp = human.keypoints[kp_idx]; // kp = [x, y, score]
    if(kp.score >= g_triangulation_threshold){
      keypoints_normalized[kp_idx] = Eigen::Vector3f((kp.x - cx) / fx, (kp.y - cy) / fy, kp.score);

      Eigen::Matrix2f cov;
      cov << kp.cov[0] / (fx*fx), kp.cov[1] / (fx*fy), // scale covariance to normalized camera coordinates
             kp.cov[1] / (fx*fy), kp.cov[2] / (fy*fy);
      covs_normalized[kp_idx] = cov;

      ++num_valid_kps;
    }
  }
  return num_valid_kps;
}

double calcCost(const PersonHypothesis& hyp, const vector<Eigen::Vector3f>& det_kps, int det_ID, const vector<Eigen::Matrix3f>& fundamental_matrices, bool& veto){
  double total_cost = 0.;
  int n_obs_used = 0;
  int n_obs_in_hyp = hyp.cameraIDs.size();
  if(n_obs_in_hyp == 0){
    veto = true;
    return MAX_COSTS;
  }

  veto = false; // if true we cannot join the detection with the hypothesis
  double tmp_veto = 0.0;
  const double tolerance = 1.0 - 1.0 / (2*n_obs_in_hyp), veto_delta = 1.0 / n_obs_in_hyp;
  for (int obs_idx = 0; obs_idx < n_obs_in_hyp; ++obs_idx) {
    double cost = 0.;
    int n_joints_used = 0;
    const Eigen::Matrix3f& F = fundamental_matrices[get_fundamental_idx(hyp.cameraIDs[obs_idx], det_ID)];
    const Eigen::Matrix3f  FT = F.transpose();
    const vector<Eigen::Vector3f>& hyp_kps = hyp.keypoints_normalized[obs_idx]; // kp = [x, y, conf]
    for (int kp_idx = 0; kp_idx < NUM_KEYPOINTS; ++ kp_idx) {
      if(hyp_kps[kp_idx].z() > g_triangulation_threshold && det_kps[kp_idx].z() > g_triangulation_threshold){ // z is confidence value! both keypoints are valid
        Eigen::Vector3f p1(hyp_kps[kp_idx].x(), hyp_kps[kp_idx].y(), 1.0f);
        Eigen::Vector3f p2(det_kps[kp_idx].x(), det_kps[kp_idx].y(), 1.0f);
        Eigen::Vector3f epipolar_line_1 = F * p1;
        Eigen::Vector3f epipolar_line_2 = FT * p2;
        //Point-to-line distance: abs(p2.dot(l)) / sqrt(l0²+l1²)
        float d1 = std::abs(p2.dot(epipolar_line_1)) / std::sqrt(epipolar_line_1(0) * epipolar_line_1(0) + epipolar_line_1(1) * epipolar_line_1(1));
        float d2 = std::abs(p1.dot(epipolar_line_2)) / std::sqrt(epipolar_line_2(0) * epipolar_line_2(0) + epipolar_line_2(1) * epipolar_line_2(1));
        cost += static_cast<double>(d1 + d2);
        ++n_joints_used;
      }
    }

    if(n_joints_used > 0){
      cost /= n_joints_used;
      total_cost += cost;
      ++n_obs_used;
      if(cost > g_max_epipolar_error && (hyp.score[obs_idx] > 0.5f || n_obs_in_hyp == 1)){
        tmp_veto += veto_delta;
      }
      else if(cost > 2 * g_max_epipolar_error && (hyp.score[obs_idx] > 0.5f || n_obs_in_hyp == 1)){
        tmp_veto += 1;
      }
    }
  }

  if(tmp_veto > tolerance)
    veto = true;

  if(n_obs_used > 0){
    return total_cost / n_obs_used;
  }
  else {
    veto = true;
    return MAX_COSTS;
  }
}

double calc_3D_dist(const PersonCov& p1, const PersonCov& p2){
  int num_joints_used = 0;
  double dist = 0;
  for (int kp_idx = 0; kp_idx < FUSION_BODY_PARTS::NUM_KEYPOINTS; ++kp_idx) { // For each joint of the skeleton.
    const auto& kp1 = p1.keypoints[kp_idx];
    const auto& kp2 = p2.keypoints[kp_idx];
    if(kp1.score > 0 && kp2.score > 0){ // valid measurement of joint exists for both persons
      dist += std::sqrt(std::pow(kp1.joint.x - kp2.joint.x, 2) + std::pow(kp1.joint.y - kp2.joint.y, 2) + std::pow(kp1.joint.z - kp2.joint.z, 2));
      ++num_joints_used;
    }
  }

  if(num_joints_used > 0)
    return dist / num_joints_used;
  else
    return MAX_COSTS;
}

void merge_persons(PersonCov& p1, const PersonCov& p2){
  for (int kp_idx = 0; kp_idx < FUSION_BODY_PARTS::NUM_KEYPOINTS; ++kp_idx) { // For each joint of the skeleton.
    auto& kp1 = p1.keypoints[kp_idx];
    const auto& kp2 = p2.keypoints[kp_idx];
    double total_score = static_cast<double>(kp1.score + kp2.score);
    if(total_score > 0.0){
      kp1.joint.x = ((double)kp1.score * kp1.joint.x + (double)kp2.score * kp2.joint.x) / total_score;
      kp1.joint.y = ((double)kp1.score * kp1.joint.y + (double)kp2.score * kp2.joint.y) / total_score;
      kp1.joint.z = ((double)kp1.score * kp1.joint.z + (double)kp2.score * kp2.joint.z) / total_score;
      kp1.score = std::max(kp1.score, kp2.score);
      mergeKeypointCovariance(kp1, kp1, kp2);
    }
  }
}

double calcReprojectionError(const Eigen::Vector3f& reconstructedPoint, const std::vector<Matrix34f>& cameraMatrices, const std::vector<Eigen::Vector3f>& pointsOnEachCamera){ // pointsOnEachCamera are [x, y, conf]
  double averageError = 0.;
  double norm = 0.;
  for (auto i = 0u ; i < cameraMatrices.size() ; i++)
  {
      Eigen::Vector2f imageX = (cameraMatrices[i] * reconstructedPoint.homogeneous()).eval().hnormalized();
      const float dx = imageX.x() - pointsOnEachCamera[i].x();
      const float dy = imageX.y() - pointsOnEachCamera[i].y();
      const float error = std::sqrt(dx*dx + dy*dy);
      averageError += static_cast<double>(pointsOnEachCamera[i].z() * error); // weight by confidence
      norm += static_cast<double>(pointsOnEachCamera[i].z());
  }
  return averageError / norm;
}

Eigen::Vector3f triangulate(const std::vector<Matrix34f>& cameraMatrices, const std::vector<Eigen::Vector3f>& jointOnEachCamera, bool weight_by_conf = false, double* reproj_error = nullptr){ // jointOnEachCamera are [x, y, conf]
  // Create and fill A for homogenous equation system Ax = 0: linear triangulation method to find a 3D point (DLT). For example, see Hartley & Zisserman section 12.2 (p.312).
  const int numCameras = cameraMatrices.size();
  Eigen::Matrix<float, -1, 4> A = Eigen::Matrix<float, -1, 4>::Zero(2 * numCameras, 4);
  for (auto i = 0 ; i < numCameras ; i++)
  {
      A.row(2*i) = jointOnEachCamera[i].x() * cameraMatrices[i].row(2) - cameraMatrices[i].row(0);
      A.row(2*i).normalize();
      A.row(2*i+1) = jointOnEachCamera[i].y() * cameraMatrices[i].row(2) - cameraMatrices[i].row(1);
      A.row(2*i+1).normalize();
      if(weight_by_conf){
        A.row(2*i) *= jointOnEachCamera[i].z(); // z = confidence
        A.row(2*i+1)*= jointOnEachCamera[i].z();
      }
  }

  const Eigen::Vector4f pt_homog = A.jacobiSvd(Eigen::ComputeThinV).matrixV().col(3);
  //const Eigen::Vector4f pt_homog = A.bdcSvd(Eigen::ComputeThinV).matrixV().col(3);

  const Eigen::Vector3f reconstructedPoint = pt_homog.hnormalized();

  if(reproj_error != nullptr)
    *reproj_error = calcReprojectionError(reconstructedPoint, cameraMatrices, jointOnEachCamera);

  return reconstructedPoint;
}

double calcJointDist(const geometry_msgs::Point& j1, const geometry_msgs::Point& j2){
  return std::sqrt((j1.x - j2.x) * (j1.x - j2.x) + (j1.y - j2.y) * (j1.y - j2.y) + (j1.z - j2.z) * (j1.z - j2.z));
}

inline void mod_samples(vector<vector<Eigen::Vector3f> >& sigma_points, const Eigen::Matrix2f& A, const float b, const int cam_idx){
  //Cholesky 2x2
  const float l11 = std::sqrt(A(0,0));
  const float l21 = A(1,0) / l11;
  const float l22 = sqrt(A(1,1) - l21*l21);

  const float dx1 = l11*b;
  const float dy1 = l21*b;
  const float dy2 = l22*b; //dx2 = 0

  sigma_points[4*cam_idx + 1][cam_idx].x() -= dx1;
  sigma_points[4*cam_idx + 1][cam_idx].y() -= dy1;
  sigma_points[4*cam_idx + 2][cam_idx].y() -= dy2; // dx2 = 0
  sigma_points[4*cam_idx + 3][cam_idx].x() += dx1;
  sigma_points[4*cam_idx + 3][cam_idx].y() += dy1;
  sigma_points[4*cam_idx + 4][cam_idx].y() += dy2; // dx2 = 0
}

Eigen::RowVectorXf draw_sigma_points(vector<vector<Eigen::Vector3f> >& sigma_points, const vector<Eigen::Vector3f>& jointOnEachCamera, const vector<Eigen::Matrix2f>& cov_mats){
  const int num_cameras = cov_mats.size();
  const int dim = 2*num_cameras;
  constexpr float kappa = 0.5; // (scaling) equal weight on all samples
  const int num_samples = 2 * dim + 1;
  Eigen::RowVectorXf weights(num_samples);
  weights << 2 * kappa, Eigen::RowVectorXf::Ones(num_samples - 1);
  weights /= (2.0f * (dim + kappa));

  sigma_points.resize(num_samples); //num_samples - 1
  std::fill(sigma_points.begin(), sigma_points.end(), jointOnEachCamera);
  float b = std::sqrt(dim + kappa);
  for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
    mod_samples(sigma_points, cov_mats[cam_idx], b, cam_idx);
  }

  return weights;
}

void calc_covariance(Eigen::Matrix3f& cov, const Eigen::Vector3f& mean_sample, const vector<Eigen::Vector3f>& jointOnEachCamera, const vector<Eigen::Matrix2f>& covarianceOnEachCamera, const vector<Matrix34f>& cameraMatrices){
  vector<vector<Eigen::Vector3f> > sigma_points;
  Eigen::RowVectorXf weights = draw_sigma_points(sigma_points, jointOnEachCamera, covarianceOnEachCamera);

  int num_samples = weights.cols();
  assert(sigma_points.size() == num_samples); // num_samples - 1

  Eigen::Matrix3Xf transformedSamples(3, num_samples);

  for (int sample_idx = 0; sample_idx < num_samples ; ++sample_idx) { //num_samples - 1
      transformedSamples.col(sample_idx) = triangulate(cameraMatrices, sigma_points[sample_idx]);
  }

  Eigen::Matrix3Xf transformedSamples_centered = transformedSamples.colwise() - mean_sample;
  cov = (transformedSamples_centered.array().rowwise() * weights.array()).matrix() * transformedSamples_centered.transpose();
}

void triangulate_persons(const vector<Person2DList::ConstPtr>& people, PersonCovList& persons3d_msg, visualization_msgs::MarkerArray& skel3d_msg,
                        const map<string, Matrix34f>& transforms_cam, const vector<Eigen::Matrix3f>& fundamental_matrices, const vector<sensor_msgs::CameraInfo>& intrinsics){ //, double delta_t
  //Prepare intrinsics, offsets and extrinsics per camera..
  unsigned int num_humans = 0;
  vector<int> cameraIDs;
  vector<Matrix34f> cameraExtrinsics;
  vector<image_geometry::PinholeCameraModel> cameraIntrinsics;
  vector<vector<person_msgs::Person2D>> humans;
  humans.reserve(NUM_CAMERAS);
  assert(people.size() == NUM_CAMERAS);
  assert(intrinsics.size() == NUM_CAMERAS);
  assert(transforms_cam.size() == NUM_CAMERAS);

  for (int i = 0; i < NUM_CAMERAS; ++i) { // This loop basically serves to sort out camera views with no detection at all.
    if(people[i]->persons.size() == 0){
      continue; // No human detected in camera view i.
    }

    humans.push_back(people[i]->persons);

    string cam_frame = intrinsics[i].header.frame_id;
    cameraExtrinsics.push_back(transforms_cam.at(cam_frame));

    image_geometry::PinholeCameraModel cam_intrinsics;
    cam_intrinsics.fromCameraInfo(intrinsics[i]);
    cameraIntrinsics.push_back(cam_intrinsics);

    cameraIDs.push_back(i);

    num_humans+=people[i]->persons.size();
  }

  int num_cams_with_det = cameraIDs.size();
  if( num_cams_with_det < 2){ // need at least two cameras with at least one detection
    return;
  }

  //Data association implemented from Tanke, Julian, and Juergen Gall. "Iterative greedy matching for 3d human pose tracking from multiple views." German Conference on Pattern Recognition, 2019.
  //https://github.com/jutanke/mv3dpose

  vector<PersonHypothesis> H;
  int cam_idx = 0; // add all detections in the first camera as initial hypotheses
  while(H.size() == 0 && cam_idx < num_cams_with_det){
    const auto& intr = cameraIntrinsics[cam_idx];
    for (int human_idx = 0; human_idx < humans[cam_idx].size(); ++human_idx) {
      PersonHypothesis hyp;
      hyp.cameraExtrinsics.push_back(cameraExtrinsics[cam_idx]);
      hyp.cameraIDs.push_back(cameraIDs[cam_idx]);
      hyp.score.push_back(humans[cam_idx][human_idx].score);

      vector<Eigen::Vector3f> keypoints_normalized(NUM_KEYPOINTS, Eigen::Vector3f(0.f, 0.f, -1.f)); // Keypoints in normalized image coordinates
      vector<Eigen::Matrix2f> covs_normalized(NUM_KEYPOINTS); // Covariances in normalized image coordinates
      int num_valid_kps = normalize_keypoints(keypoints_normalized, covs_normalized, intr, humans[cam_idx][human_idx]);

      if(num_valid_kps > NUM_KEYPOINTS / 2){ // only process hypotheses with at least half of the keypoints detected
        hyp.keypoints_normalized.push_back(keypoints_normalized); // Only save the hypothesis if there enough any valid keypoints
        hyp.keypoints_cov_normalized.push_back(covs_normalized);
        H.push_back(hyp);
      }
    }
    ++cam_idx;
  }

  for (; cam_idx < num_cams_with_det; ++cam_idx) {
    vector<vector<Eigen::Vector3f> > dets_keypoints_normalized;
    vector<vector<Eigen::Matrix2f> > dets_covs_normalized;
    vector<float> dets_score;
    const auto& intr = cameraIntrinsics[cam_idx];

    for (int det_idx = 0; det_idx < humans[cam_idx].size(); ++det_idx) {
      vector<Eigen::Vector3f> keypoints_normalized(NUM_KEYPOINTS, Eigen::Vector3f(0., 0., -1.)); // Keypoints in normalized image coordinates
      vector<Eigen::Matrix2f> covs_normalized(NUM_KEYPOINTS); // Covariances in normalized image coordinates
      int num_valid_kps = normalize_keypoints(keypoints_normalized, covs_normalized, intr, humans[cam_idx][det_idx]);

      if(num_valid_kps > NUM_KEYPOINTS / 2){ // only process hypotheses with at least half of the keypoints detected
        dets_keypoints_normalized.push_back(keypoints_normalized);
        dets_covs_normalized.push_back(covs_normalized);
        dets_score.push_back(humans[cam_idx][det_idx].score);
      }
    }

    int n_hyp = H.size();
    int n_det = dets_keypoints_normalized.size();
    if(n_det == 0)
      continue;

    Eigen::MatrixXd C(n_hyp, n_det); // Cost-Matrix in ColumnMajor(!) order, n_hyp = rows, n_det = cols
    double cost = 0.0;
    Eigen::VectorXi assignment = -Eigen::VectorXi::Ones(n_hyp);
    Eigen::Matrix<bool, -1, -1> mask = Eigen::Matrix<bool, -1, -1>::Zero(n_hyp, n_det);

    for (int det_idx = 0; det_idx < n_det; ++det_idx) {
      for (int hyp_idx = 0; hyp_idx < n_hyp; ++ hyp_idx) {
        bool veto;
        C(hyp_idx, det_idx) = calcCost(H[hyp_idx], dets_keypoints_normalized[det_idx], cameraIDs[cam_idx], fundamental_matrices, veto);
        if(!veto && C(hyp_idx, det_idx) < g_max_epipolar_error){
          mask(hyp_idx, det_idx) = true;
          assignment[hyp_idx] = det_idx;
        }

      }
    }

    if((mask.array().colwise().count() > 1).any() || (mask.array().rowwise().count() > 1).any()){ // more than one feasible association. run hungarian.
      //auto t_start = std::chrono::high_resolution_clock::now();
      HungarianAlgorithm::assignmentoptimal(assignment.data(), &cost, C.data(), n_hyp, n_det);
      //auto t_end = std::chrono::high_resolution_clock::now();
      //auto duration_assing = std::chrono::duration_cast<std::chrono::microseconds>( t_end - t_start ).count();
      //cout << "Hungarian assignment duration: " << duration_assing / 1000. << "ms." << endl;
    }

    vector<bool> det_handled(n_det, false);
    for (int hyp_idx = 0; hyp_idx < n_hyp; ++hyp_idx) {
      int det_idx = assignment(hyp_idx);
      if(det_idx >= 0){ // this hypothesis was assigned a detection
        det_handled[det_idx] = true;
        if(!mask(hyp_idx, det_idx)){ // even the closest other person has too large epipolar distance (> threshold) -> new hypothesis
          PersonHypothesis hyp;
          hyp.cameraExtrinsics.push_back(cameraExtrinsics[cam_idx]);
          hyp.cameraIDs.push_back(cameraIDs[cam_idx]);
          hyp.score.push_back(dets_score[det_idx]);
          hyp.keypoints_normalized.push_back(dets_keypoints_normalized[det_idx]);
          hyp.keypoints_cov_normalized.push_back(dets_covs_normalized[det_idx]);
          H.push_back(hyp);
          //cout << "Camera " << cam_idx << " (ID: " << hyp.cameraIDs.back() << "): added new hypothesis with score: " << hyp.score.back() << endl;
        }
        else { // add detection to hypothesis
          H[hyp_idx].cameraExtrinsics.push_back(cameraExtrinsics[cam_idx]);
          H[hyp_idx].cameraIDs.push_back(cameraIDs[cam_idx]);
          H[hyp_idx].score.push_back(dets_score[det_idx]);
          H[hyp_idx].keypoints_normalized.push_back(dets_keypoints_normalized[det_idx]);
          H[hyp_idx].keypoints_cov_normalized.push_back(dets_covs_normalized[det_idx]);
          //cout << "Camera " << cam_idx << " (ID: " << H[hyp_idx].cameraIDs.back() << "): added detection with score: " << H[hyp_idx].score.back() << " to hypothesis " << hyp_idx << endl;
        }
      }
    }

    for (int det_idx = 0; det_idx < n_det; ++det_idx) {
      if(!det_handled[det_idx]){ // this detection has not been assigned to any hypothesis -> initialize a new one
        PersonHypothesis hyp;
        hyp.cameraExtrinsics.push_back(cameraExtrinsics[cam_idx]);
        hyp.cameraIDs.push_back(cameraIDs[cam_idx]);
        hyp.score.push_back(dets_score[det_idx]);
        hyp.keypoints_normalized.push_back(dets_keypoints_normalized[det_idx]);
        hyp.keypoints_cov_normalized.push_back(dets_covs_normalized[det_idx]);
        H.push_back(hyp);
        //cout << "Camera " << cam_idx << " (ID: " << hyp.cameraIDs.back() << "): added new hypothesis with score: " << hyp.score.back() << endl;
      }
    }
  } // end for all cameras with detections

  #pragma omp parallel num_threads(H.size())
  {
    vector<PersonCov> persons3d_private;
    vector<visualization_msgs::Marker> skel3d_markers_private;
    #pragma omp for nowait
    for (int hyp_idx = 0; hyp_idx < H.size(); ++hyp_idx) {
      const auto& hyp = H[hyp_idx];
      int n_obs = hyp.cameraIDs.size();
      if(n_obs >= 2){ // Triangulate each joint
        PersonCov person_3d; // output message single person
        person_3d.keypoints.resize(FUSION_BODY_PARTS::NUM_KEYPOINTS);

        //##### visualization messages single person ###########
        visualization_msgs::Marker skel3d_single;
        skel3d_single.header = persons3d_msg.header;
        skel3d_single.lifetime = ros::Duration(2.0);
        skel3d_single.pose.position.x = 0;
        skel3d_single.pose.position.y = 0;
        skel3d_single.pose.position.z = 0;
        skel3d_single.pose.orientation.w = 1;
        skel3d_single.pose.orientation.x = 0;
        skel3d_single.pose.orientation.y = 0;
        skel3d_single.pose.orientation.z = 0;
        skel3d_single.type = visualization_msgs::Marker::LINE_LIST;
        skel3d_single.scale.x = .05;
        skel3d_single.ns = "joints";
        skel3d_single.id = hyp_idx;
        skel3d_single.color.r = 1.0;
        skel3d_single.color.a = 1.0;

        visualization_msgs::Marker skel3d_joints;
        skel3d_joints.header = skel3d_single.header;
        skel3d_joints.lifetime = ros::Duration(2.0);
        skel3d_joints.pose = skel3d_single.pose;
        skel3d_joints.type = visualization_msgs::Marker::SPHERE_LIST;
        skel3d_joints.scale.x = skel3d_joints.scale.y = skel3d_joints.scale.z = 0.07;
        skel3d_joints.ns = "joint_spheres";
        skel3d_joints.id = hyp_idx;
        skel3d_joints.color.r = skel3d_joints.color.g = 0.5;
        skel3d_joints.color.a = 1.0;

        int num_valid_keypoints = 0;
        for (int kp_idx = 0; kp_idx < NUM_KEYPOINTS; ++ kp_idx) {
          vector<Eigen::Vector3f> jointOnEachCamera; // x, y, conf
          vector<Eigen::Matrix2f> covarianceOnEachCamera;
          vector<Matrix34f> cameraMatrices;
          vector<int> cameraIdx_joint;
          float avg_score = 0;
          for (int obs_idx = 0; obs_idx < n_obs; ++obs_idx) {
            if(hyp.keypoints_normalized[obs_idx][kp_idx].z() >= g_triangulation_threshold){ // only use keypoints with confidence larger than a minimum threshold
              jointOnEachCamera.push_back(hyp.keypoints_normalized[obs_idx][kp_idx]);
              covarianceOnEachCamera.push_back(hyp.keypoints_cov_normalized[obs_idx][kp_idx]);
              cameraMatrices.push_back(hyp.cameraExtrinsics[obs_idx]);
              cameraIdx_joint.push_back(hyp.cameraIDs[obs_idx]);
              avg_score += hyp.keypoints_normalized[obs_idx][kp_idx].z();
            }
          }

          int numCameras = cameraMatrices.size();
          if(numCameras < 2)
            continue; // less than 2 observations for current joint - cannot triangulate!

          avg_score /= numCameras;

          // Implementation of triangulation based on OpenPose 3D library:
          // https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/3d_reconstruction_module.md
          // https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/src/openpose/3d
          // Cao, Zhe, et al. "OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields." IEEE transactions on pattern analysis and machine intelligence 43.1 (2019): 172-186.

          double reprojError;
          Eigen::Vector3f reconstructedPoint = triangulate(cameraMatrices, jointOnEachCamera, true, &reprojError);

          if (reprojError > g_reproj_error_max_acceptable && numCameras == 3){
            int bestIndex = -1;
            float bestDist = static_cast<float>(reprojError * reprojError); // seems to be a good initialization..
            for (int i = 0; i < numCameras; ++i){
                // Set initial values
                auto cameraMatricesSubset = cameraMatrices;
                auto pointsOnEachCameraSubset = jointOnEachCamera;
                auto cameraIdxSubset = cameraIdx_joint;

                // Remove camera i
                cameraMatricesSubset.erase(cameraMatricesSubset.begin() + i);
                pointsOnEachCameraSubset.erase(pointsOnEachCameraSubset.begin() + i);
                cameraIdxSubset.erase(cameraIdxSubset.begin() + i);

                //Point-to-line distance: abs(p2.dot(l)) / sqrt(l0²+l1²)
                Eigen::Vector3f p1(pointsOnEachCameraSubset[0].x(), pointsOnEachCameraSubset[0].y(), 1.0f);
                Eigen::Vector3f p2(pointsOnEachCameraSubset[1].x(), pointsOnEachCameraSubset[1].y(), 1.0f);
                const Eigen::Matrix3f& F = fundamental_matrices[get_fundamental_idx(cameraIdxSubset[0], cameraIdxSubset[1])];
                Eigen::Vector3f epipolar_line_1 = F * p1;
                Eigen::Vector3f epipolar_line_2 = F.transpose() * p2;
                float numerator_1 = p2.dot(epipolar_line_1);
                float numerator_2 = p1.dot(epipolar_line_2);
                float squared_symm_dist =  numerator_1 * numerator_1 / (epipolar_line_1(0) * epipolar_line_1(0) + epipolar_line_1(1) * epipolar_line_1(1))
                                         + numerator_2 * numerator_2 / (epipolar_line_2(0) * epipolar_line_2(0) + epipolar_line_2(1) * epipolar_line_2(1));
                //cout << "\t epipolar dist removing camera " << cameraIdx_joint[i] << " (F_" << cameraIdxSubset[0] << cameraIdxSubset[1] << " (idx " << get_fundamental_idx(cameraIdxSubset[0], cameraIdxSubset[1]) << ")): " << std::sqrt(squared_symm_dist) << endl;

                if(squared_symm_dist < bestDist){
                  bestDist = squared_symm_dist;
                  bestIndex = i;
                }
            }

            if(bestIndex != -1){
                // Remove camera i
                cameraMatrices.erase(cameraMatrices.begin() + bestIndex);
                jointOnEachCamera.erase(jointOnEachCamera.begin() + bestIndex);
                covarianceOnEachCamera.erase(covarianceOnEachCamera.begin() + bestIndex);
                cameraIdx_joint.erase(cameraIdx_joint.begin() + bestIndex);

                // Get new triangulation results
                reconstructedPoint = triangulate(cameraMatrices, jointOnEachCamera, true, &reprojError);
                avg_score = (jointOnEachCamera[0].z() + jointOnEachCamera[1].z()) / 2.0f; // update score as avg of the two left points.
                //cout << "Joint " << kp_idx << ": avg reprojection error removing camera " << cameraIdx_joint[bestIndex] << ": " << reprojError << " (approx. " << reprojError * cameraIntrinsics[0].fx() << "px), score: " << avg_score << endl;
            }
          }
          else if (reprojError > g_reproj_error_max_acceptable && numCameras >= 4){ //Run with all but 1 camera for each camera and use the one with minimum average reprojection error.
            // Find best projection
            double bestReprojection = reprojError;
            int bestReprojectionIndex = -1; // -1 means with all camera views
            float bestScore = avg_score;
            Eigen::Vector3f bestReconstructedPoint;
            for (int i = 0; i < numCameras; ++i){
                // Set initial values
                auto cameraMatricesSubset = cameraMatrices;
                auto pointsOnEachCameraSubset = jointOnEachCamera;

                // Remove camera i
                cameraMatricesSubset.erase(cameraMatricesSubset.begin() + i);
                pointsOnEachCameraSubset.erase(pointsOnEachCameraSubset.begin() + i);

                // Get new triangulation results
                double projectionErrorSubset;
                Eigen::Vector3f reconstructedPointSubset = triangulate(cameraMatricesSubset, pointsOnEachCameraSubset, true, &projectionErrorSubset);

                // If projection doesn't change much, this point is inlier (or all points are bad). Thus, save new best results only if considerably better
                if (bestReprojection > projectionErrorSubset && projectionErrorSubset < 0.9*reprojError){
                    bestReprojection = projectionErrorSubset;
                    bestReprojectionIndex = i;
                    bestReconstructedPoint = reconstructedPointSubset;

                    float tmpScore = 0.;
                    for (const auto& pt : pointsOnEachCameraSubset) {
                      tmpScore += pt.z();
                    }
                    bestScore = tmpScore / pointsOnEachCameraSubset.size();
                }
            }
            // Remove noisy camera
            if (bestReprojectionIndex != -1){
                // Remove noisy camera
                cameraMatrices.erase(cameraMatrices.begin() + bestReprojectionIndex);
                jointOnEachCamera.erase(jointOnEachCamera.begin() + bestReprojectionIndex);
                covarianceOnEachCamera.erase(covarianceOnEachCamera.begin() + bestReprojectionIndex);
                cameraIdx_joint.erase(cameraIdx_joint.begin() + bestReprojectionIndex);
                // Update reconstructedPoint & projectionError
                reconstructedPoint = bestReconstructedPoint;
                reprojError = bestReprojection;
                avg_score = bestScore;
                //cout << "Joint " << kp_idx << ": avg reprojection error removing camera " << bestReprojectionIndex << ": " << reprojError << " (approx. " << reprojError * cameraIntrinsics[0].fx() << "px), score: " << avg_score << endl;
            }
          }

          if (reprojError > g_reproj_error_max_acceptable){ // if reprojection error is still large, down-weight the measurement
            avg_score *= (g_reproj_error_max_acceptable / reprojError);
            if(reprojError > 2 * g_reproj_error_max_acceptable)
              cout << "Person " << hyp_idx << ", Joint " << kp_idx << ": large reprojection error of " << reprojError << " (approx. " << reprojError * cameraIntrinsics[0].fx() << "px)! Downweight score by factor: " << g_reproj_error_max_acceptable / reprojError << endl;
          }

          Eigen::Matrix3f covariance;
          calc_covariance(covariance, reconstructedPoint, jointOnEachCamera, covarianceOnEachCamera, cameraMatrices);

          geometry_msgs::Point joint_3d; // in base-frame
          joint_3d.x = static_cast<double>(reconstructedPoint.x());
          joint_3d.y = static_cast<double>(reconstructedPoint.y());
          joint_3d.z = static_cast<double>(reconstructedPoint.z());

          person_3d.keypoints[g_kp2kpFusion_idx[kp_idx]].joint = joint_3d;
          person_3d.keypoints[g_kp2kpFusion_idx[kp_idx]].score = avg_score;
          setKeypointCovariance(person_3d.keypoints[g_kp2kpFusion_idx[kp_idx]], covariance.cast<double>());
          ++num_valid_keypoints;

        } // End for each Keypoint

        std::vector<int> kpIdx2msgIdx(NUM_KEYPOINTS, -1);
        for (int kp_idx = 0; kp_idx < NUM_KEYPOINTS; ++ kp_idx) {
          auto& kp = person_3d.keypoints[g_kp2kpFusion_idx[kp_idx]];
          if(kp.score <= 0)
            continue;

          int parent_idx = g_kpParent[kp_idx];
          // Increase covariance in this case when limb-length is incoherent with model
          if(parent_idx >= 0){ // parent is well-defined
            const auto& parent_kp = person_3d.keypoints[g_kp2kpFusion_idx[parent_idx]];
            if(parent_kp.score > 0 && g_limbLength[kp_idx] > 0){ // if parent keypoint exists and limb-length is well-defined
              double joint_dist = calcJointDist(kp.joint, parent_kp.joint);
              addToKeypointCovariance(kp, g_limbLCovOffsetSigma * (joint_dist - g_limbLength[kp_idx]) / g_limbLSigma[kp_idx]); // joint_dist - g_limbLength[kp_idx] // add offset to covariance (sigma^2 * I)
            }
            else if(g_param_pose_method == "simple" && kp_idx == EdgeTPU_BodyParts_Simple::RShoulder){ // Special case for shoulders in Simple-Baseline model as there is no neck-keypoint
              auto& kpLSh = person_3d.keypoints[g_kp2kpFusion_idx[EdgeTPU_BodyParts_Simple::LShoulder]];
              if(kpLSh.score > 0){ // both shoulders exist.
                double joint_dist = calcJointDist(kp.joint, kpLSh.joint);
                addToKeypointCovariance(kp, g_limbLCovOffsetSigma * (joint_dist - EdgeTPU_BodyParts_Simple::shoulderDist) / EdgeTPU_BodyParts_Simple::shoulderSigma); // add offset to covariance (sigma^2 * I)
                addToKeypointCovariance(kpLSh, g_limbLCovOffsetSigma * (joint_dist - EdgeTPU_BodyParts_Simple::shoulderDist) / EdgeTPU_BodyParts_Simple::shoulderSigma); // add offset to covariance (sigma^2 * I)
              }
            }
          }

          skel3d_joints.points.push_back(kp.joint);
          skel3d_joints.colors.push_back(g_colors[g_kp2kpFusion_idx[kp_idx]]);

          if(g_param_vis_covariance && g_kp2kpFusion_idx[kp_idx] < 15){
            visualization_msgs::Marker joint_3d_cov;
            joint_3d_cov.header = skel3d_joints.header;
            joint_3d_cov.lifetime = ros::Duration(5.0);
            joint_3d_cov.type = visualization_msgs::Marker::SPHERE;
            joint_3d_cov.ns = "joint_cov_3d";
            joint_3d_cov.id = FUSION_BODY_PARTS::NUM_KEYPOINTS * hyp_idx + kp_idx;
            joint_3d_cov.color = g_colors[g_kp2kpFusion_idx[kp_idx]];
            joint_3d_cov.color.a = 0.50f;
            setMarkerPose(joint_3d_cov, kp);
            skel3d_markers_private.push_back(joint_3d_cov);
          }

          if(parent_idx >= 0){
            if (kpIdx2msgIdx[parent_idx] != -1 && kpIdx2msgIdx[parent_idx] < skel3d_single.points.size())
              skel3d_single.points.push_back(skel3d_single.points[kpIdx2msgIdx[parent_idx]]); // add the parent joint
            else
              skel3d_single.points.push_back(kp.joint);

            skel3d_single.points.push_back(kp.joint);

            skel3d_single.colors.push_back(g_colors[g_kp2kpFusion_idx[kp_idx]]);
            skel3d_single.colors.push_back(g_colors[g_kp2kpFusion_idx[kp_idx]]);
          }
          else{
            skel3d_single.points.push_back(kp.joint);
            skel3d_single.points.push_back(kp.joint);

            skel3d_single.colors.push_back(g_colors[g_kp2kpFusion_idx[kp_idx]]);
            skel3d_single.colors.push_back(g_colors[g_kp2kpFusion_idx[kp_idx]]);
          }

          kpIdx2msgIdx[kp_idx] = skel3d_single.points.size() - 1;
        } // End for each keypoint

        // Check for outliers by distance to root
        KeypointWithCovariance root;
        if(person_3d.keypoints[FUSION_BODY_PARTS::MidHip].score > 0)
          root = person_3d.keypoints[FUSION_BODY_PARTS::MidHip];
        else if (person_3d.keypoints[FUSION_BODY_PARTS::LHip].score > 0 && person_3d.keypoints[FUSION_BODY_PARTS::RHip].score > 0){
          geometry_msgs::Point root_joint;
          root_joint.x = (person_3d.keypoints[FUSION_BODY_PARTS::LHip].joint.x + person_3d.keypoints[FUSION_BODY_PARTS::RHip].joint.x) / 2.;
          root_joint.y = (person_3d.keypoints[FUSION_BODY_PARTS::LHip].joint.y + person_3d.keypoints[FUSION_BODY_PARTS::RHip].joint.y) / 2.;
          root_joint.z = (person_3d.keypoints[FUSION_BODY_PARTS::LHip].joint.z + person_3d.keypoints[FUSION_BODY_PARTS::RHip].joint.z) / 2.;
          float root_score = (person_3d.keypoints[FUSION_BODY_PARTS::LHip].score + person_3d.keypoints[FUSION_BODY_PARTS::RHip].score)/ 2.f;
          root.joint = root_joint;
          root.score = root_score;
        }

        if(root.score > 0){
          for (int kp_idx = 0; kp_idx < FUSION_BODY_PARTS::NUM_KEYPOINTS; ++kp_idx) {
            auto& kp = person_3d.keypoints[kp_idx];
            if(kp.score > 0){
              double joint_dist = calcJointDist(root.joint, kp.joint);
              if(joint_dist > g_max_joint_dist_to_root){
                //cout << "WARNING: resetting keypoint " << kp_idx << " as it has large distance to root: " << joint_dist << "m." << endl;
                kp = KeypointWithCovariance();
                --num_valid_keypoints;
              }
            }
            else {
              kp = KeypointWithCovariance();
              --num_valid_keypoints;
            }
          }
        }

        //Check for implausible feet height
        double feet_height = 0.0;
        if(person_3d.keypoints[FUSION_BODY_PARTS::LAnkle].score > 0 && person_3d.keypoints[FUSION_BODY_PARTS::RAnkle].score > 0)
          feet_height = (person_3d.keypoints[FUSION_BODY_PARTS::LAnkle].joint.z + person_3d.keypoints[FUSION_BODY_PARTS::RAnkle].joint.z) / 2.0;
        else if(person_3d.keypoints[FUSION_BODY_PARTS::LAnkle].score > 0)
          feet_height = person_3d.keypoints[FUSION_BODY_PARTS::LAnkle].joint.z;
        else if(person_3d.keypoints[FUSION_BODY_PARTS::RAnkle].score > 0)
          feet_height = person_3d.keypoints[FUSION_BODY_PARTS::RAnkle].joint.z;
        if(std::abs(feet_height) > 0.50){ // Max tolerance of 50cm above or below ground
          cout << "WARNING: removing person detection " << hyp_idx << " due to implausible feet-height of " << feet_height << "m." << endl;
          num_valid_keypoints = 0;
        }

        if(num_valid_keypoints > g_min_num_valid_keypoints){
          persons3d_private.push_back(person_3d);

          skel3d_markers_private.push_back(skel3d_single);
          skel3d_markers_private.push_back(skel3d_joints);
        }
      } // End if n_obs > 2
    } // End for each hypothesis

    #pragma omp critical
    {
      persons3d_msg.persons.insert(persons3d_msg.persons.end(), persons3d_private.begin(), persons3d_private.end());
      skel3d_msg.markers.insert(skel3d_msg.markers.end(), skel3d_markers_private.begin(), skel3d_markers_private.end());
    }
  } // end omp parallel

  //check for closeby skeleton end eventually merge
  for (int i = 0; i < persons3d_msg.persons.size(); ++i) {
    for (int j = i+1; j < persons3d_msg.persons.size();) {
      if(calc_3D_dist(persons3d_msg.persons[i], persons3d_msg.persons[j]) < g_merge_dist_thresh){ // merge person_3d with p
        merge_persons(persons3d_msg.persons[i], persons3d_msg.persons[j]);

        persons3d_msg.persons.erase(persons3d_msg.persons.begin() + j);
        skel3d_msg.markers.erase(skel3d_msg.markers.begin() + 2*j, skel3d_msg.markers.begin() + 2* (j+1));
      }
      else
        ++j;
    }
  }
}

void skeletonCallback(const vector<Person2DList::ConstPtr>& people){
  {
    std::lock_guard<std::mutex> lck (g_skel_data_mutex);
    g_skel_data = people;
    g_skel_data_updated = true;
  }
  g_skel_data_cv.notify_one();
}

void skeletonThreadCallback(const map<string, Matrix34f>& transforms_cam, const vector<Eigen::Matrix3f>& fundamental_matrices, const vector<sensor_msgs::CameraInfo>& intrinsics,
                      const ros::Publisher& pers3d_pub, const ros::Publisher& skel3d_pub){
  double last_stamp = 0;
  vector<Person2DList::ConstPtr> people;
  vector<Person2DList::Ptr> dummy_msgs(NUM_CAMERAS);
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    dummy_msgs[i] = boost::make_shared<Person2DList>();
  }

  while(ros::ok()){
    //ROS_INFO("Waiting for new data from synchronizer...");
    std::unique_lock<std::mutex> lck(g_skel_data_mutex);
    g_skel_data_cv.wait(lck, []{return g_skel_data_updated;});

    people = g_skel_data;
    g_skel_data_updated = false;

    lck.unlock();

    assert(people.size() == NUM_CAMERAS);

    double t_max = 0.0; // most recent message (pivot element)
    int t_max_idx = -1;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
      if(people[i]->header.stamp.toSec() > t_max){
        t_max = people[i]->header.stamp.toSec();
        t_max_idx = i;
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double delta_t = people[t_max_idx]->header.stamp.toSec() - last_stamp;
    if(delta_t > 0.17){
        cout << "WARNING: Large frame delay delta_t = " << delta_t << "s (should be < 0.17s)" << endl;
    }
    if(delta_t <= 0.0){
      cout << "WARNING: re-using message or jumped backwards in time: delta_t = " << delta_t << "s. Will not process this message!" << endl;
      continue;
    }
    last_stamp = people[t_max_idx]->header.stamp.toSec();

    for (int i = 0; i < NUM_CAMERAS; ++i) {
      double dt = t_max - people[i]->header.stamp.toSec();
      if(dt > g_max_sync_diff){
        dummy_msgs[i]->header = people[i]->header;
        dummy_msgs[i]->fb_delay = people[i]->fb_delay;
        people[i] = dummy_msgs[i];
        cout << "WARNING: sync time diff of msg " << i << " larger than " << g_max_sync_diff * 1000 << "ms: " << dt * 1000 << "ms (w.r.t. pivot msg " << t_max_idx << "). REMOVING." << endl;
      }
    }

    PersonCovList persons3d_msg; // output data structure
    persons3d_msg.header = people[t_max_idx]->header;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
      persons3d_msg.ts_per_cam.push_back(people[i]->header.stamp);
      persons3d_msg.fb_delay_per_cam.push_back(people[i]->fb_delay);
    }
    persons3d_msg.header.frame_id = BASE_FRAME;

    visualization_msgs::MarkerArray skel3d_msg; // visualization message

    triangulate_persons(people, persons3d_msg, skel3d_msg, transforms_cam, fundamental_matrices, intrinsics); //, delta_t

    pers3d_pub.publish(persons3d_msg);

    if(skel3d_msg.markers.size() > 0)
      skel3d_pub.publish(skel3d_msg);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    //cout << "Triangulation: " << persons3d_msg.persons.size() << " detections, duration: " << duration / 1000. << "ms." << endl;
    g_timings[0] += duration / 1000.;
    ++g_timing_cnt[0];
    if(persons3d_msg.persons.size() > 0 && persons3d_msg.persons.size() < max_num_timings){
      g_timings[persons3d_msg.persons.size()] += duration / 1000.;
      ++g_timing_cnt[persons3d_msg.persons.size()];
    }
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "skeleton_singlePerson_3d");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  nh_private.param<string>("pose_method", g_param_pose_method, "simple");
  nh_private.param<bool>("vis_cov", g_param_vis_covariance, false);
  nh_private.param<double>("max_epi_dist", g_max_epipolar_error, 0.050);
  NUM_KEYPOINTS = 17;
  g_triangulation_threshold = 0.30f;

  if(g_param_pose_method == "h36m"){
    g_kpParent = EdgeTPU_BodyParts_H36M::kpParent;
    g_limbLength = EdgeTPU_BodyParts_H36M::limbLength;
    g_limbLSigma = EdgeTPU_BodyParts_H36M::limbLSigma;
    g_kp2kpFusion_idx = g_kp2kpFusion_idx_h36m;
  }
  else{
    g_kpParent = EdgeTPU_BodyParts_Simple::kpParent;
    g_limbLength = EdgeTPU_BodyParts_Simple::limbLength;
    g_limbLSigma = EdgeTPU_BodyParts_Simple::limbLSigma;
    g_kp2kpFusion_idx = g_kp2kpFusion_idx_simple;
  }

  std::vector<string> cam_names;
  nh_private.param("cameras", cam_names, std::vector<string>());
  if(cam_names.size() > 0){
    NUM_CAMERAS = cam_names.size();
    CAM_FRAMES.clear();
    CAM_INFO_TOPICS.clear();
    SKELETON_2D_TOPICS.clear();
    for (int i = 0; i < NUM_CAMERAS; ++i) {
      CAM_FRAMES.push_back((string)cam_names[i] + g_cam_frame);
      CAM_INFO_TOPICS.push_back((string)cam_names[i] + g_cam_info_topic);
      SKELETON_2D_TOPICS.push_back((string)cam_names[i] + g_skel_2d_topic);
    }
  }

  ROS_INFO("NUM_CAMERAS: %d, topics and frames: ", NUM_CAMERAS);
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    cout << "\t" << CAM_FRAMES[i] << ", " << CAM_INFO_TOPICS[i] << ", " << SKELETON_2D_TOPICS[i] << endl;
  }

  if(NUM_CAMERAS < 2){
    ROS_ERROR("Need at least 2 cameras for triangulation. Aborting!");
    return -1;
  }

  ROS_INFO("Using pose estimation method: %s, with %d keypoints, triangulation-threshold %f, max reprojection error: %f and max epipolar dist: %f.", g_param_pose_method.c_str(), NUM_KEYPOINTS, g_triangulation_threshold, g_reproj_error_max_acceptable, g_max_epipolar_error);

  //Define Colors...
  std_msgs::ColorRGBA color; color.a = 1.0;
  color.r = 1.0;             color.g = 0.0; color.b = 0.0; g_colors.push_back(color); // Nose -> (255, 0, 0)
  color.r = 85.f/255.f; color.g = 170.f/255.f; color.b=0.; g_colors.push_back(color); // Neck -> (85, 170, 0)
  color.r = 0.0;             color.g = 1.0; color.b = 0.0; g_colors.push_back(color); // Rshld -> (0, 255, 0)
  color.r = 0.0; color.g = 1.0; color.b = 170.0f / 255.0f; g_colors.push_back(color); // RElb -> (0, 255, 170)
  color.r = 0.0; color.g = 170.0f / 255.0f; color.b = 1.0; g_colors.push_back(color); // RWr -> (0, 170, 255)
  color.r = 85.0f / 255.0f;  color.g = 1.0; color.b = 0.0; g_colors.push_back(color); // LShld -> (85, 255, 0)
  color.r = 0.0; color.g = 1.0;  color.b = 85.0f / 255.0f; g_colors.push_back(color); // LElb -> (0, 255, 85)
  color.r = 0.0; color.g = 1.0; color.b = 1.0;             g_colors.push_back(color); // LWr ->  (0, 255, 255)
  color.r = 0.f; color.g = 85.f/255.f;color.b=170.f/255.f; g_colors.push_back(color); // MidHip / Root -> (0, 85, 170)
  color.r = 0.0; color.g = 0.0;             color.b = 1.0; g_colors.push_back(color); // RHip -> (0, 0, 255)
  color.r = 100.0f / 255.0f; color.g = 0.0; color.b = 1.0; g_colors.push_back(color); // RKnee -> (100, 0, 255)
  color.r = 1.0;             color.g = 0.0; color.b = 1.0; g_colors.push_back(color); // RAnk -> (255, 0, 255)
  color.r = 0.0;  color.g = 85.0f / 255.0f; color.b = 1.0; g_colors.push_back(color); // LHip -> (0, 85, 255)
  color.r = 50.0f / 255.0f;  color.g = 0.0; color.b = 1.0; g_colors.push_back(color); // LKnee -> (50, 0, 255)
  color.r = 170.0f / 255.0f; color.g = 0.0; color.b = 1.0; g_colors.push_back(color); // LAnk -> (170, 0, 255)
  color.r = 1.0; color.g = 170.0f / 255.0f; color.b = 0.0; g_colors.push_back(color); // REye -> (255, 170, 0)
  color.r = 1.0; color.g =  85.0f / 255.0f; color.b = 0.0; g_colors.push_back(color); // LEye -> (255, 85, 0)
  color.r = 170.0f / 255.0f; color.g = 1.0; color.b = 0.0; g_colors.push_back(color); // REar -> (170, 255, 0)
  color.r = 1.0; color.g = 1.0f; color.b = 0.0;            g_colors.push_back(color); // LEar -> (255, 255, 0)
  color.r = 1.0; color.g = 150.0f / 255.0f; color.b = 0.0; g_colors.push_back(color); // Head -> (255, 150, 0)
  color.r = 42.f/255.f; color.g=0.5; color.b = 85.f/255.f; g_colors.push_back(color); // Belly -> (42, 128, 85)
  color.r = 50.0f / 255.0f;  color.g = 0.0; color.b = 1.0; g_colors.push_back(color);
  color.r = 100.0f / 255.0f; color.g = 0.0; color.b = 1.0; g_colors.push_back(color);
  color.r = 150.0f / 255.0f; color.g = 0.0; color.b = 1.0; g_colors.push_back(color);
  color.r = 200.0f / 255.0f; color.g = 0.0; color.b = 1.0; g_colors.push_back(color);
  color.r = 1.0; color.g = 0.0; color.b = 200.0f / 255.0f; g_colors.push_back(color);
  color.r = 1.0; color.g = 0.0; color.b = 150.0f / 255.0f; g_colors.push_back(color);
  color.r = 1.0; color.g = 0.0; color.b = 100.0f / 255.0f; g_colors.push_back(color);
  color.r = 1.0; color.g = 0.0; color.b =  50.0f / 255.0f; g_colors.push_back(color);

  ros::Publisher pub_pers3d = nh.advertise<PersonCovList>(PERSON_3D_TOPIC, 1);

  ros::Publisher pub_skel3d = nh.advertise<visualization_msgs::MarkerArray>(SKELETON_3D_TOPIC, 1);

  std::vector<message_filters::Subscriber<Person2DList>> skel_subs(NUM_CAMERAS);
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    skel_subs[i].subscribe(nh, SKELETON_2D_TOPICS[i], 1, ros::TransportHints().tcpNoDelay());
  }

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  map<string, Eigen::Affine3d> transforms_cam;
  getTransforms(transforms_cam, tfBuffer);

  vector<Eigen::Matrix3f> fundamental_matrices;
  vector<Eigen::Matrix<double, 3, 4> > Ps;
  vector<Eigen::Vector4d> Cs;
  for (int i = 0; i < NUM_CAMERAS; ++i) {
    Cs.push_back(transforms_cam.at(CAM_FRAMES[i]).inverse().matrix().col(3));
    Ps.push_back(transforms_cam.at(CAM_FRAMES[i]).matrix().block<3,4>(0,0)); // 3 x 4 Projection Matrices
  }

  for (int i = 0; i < NUM_CAMERAS; ++i) {
    for(int j = i+1; j < NUM_CAMERAS; ++j){
      Eigen::Vector3d e_ij = Ps[j] * Cs[i];
      Eigen::Matrix3d e_ij_cross;
      cross_prod_matrix(e_ij, e_ij_cross);
      Eigen::Matrix<double,4,3> Pinv;
      pseudo_inv34d(Ps[i], Pinv);
      fundamental_matrices.push_back((e_ij_cross * Ps[j] * Pinv).cast<float>());
    }
  }

  ROS_INFO("Synchronizing %d cameras, calculated %zu fundamental matrices", NUM_CAMERAS, fundamental_matrices.size());

  map<string, Matrix34f> camera_matrices;
  for (const auto & it : transforms_cam) {
    camera_matrices.insert(std::pair<string, Matrix34f>(it.first, it.second.matrix().block<3,4>(0,0).cast<float>()));
  }

  vector<sensor_msgs::CameraInfo> intrinsics;
  getIntrinsics(intrinsics, nh);

  std::thread skel_data_thread(std::bind(skeletonThreadCallback, std::cref(camera_matrices), std::cref(fundamental_matrices), std::cref(intrinsics), std::cref(pub_pers3d), std::cref(pub_skel3d)));

  typedef message_filters::sync_policies::ApproximateTimeVec<Person2DList> mySyncPolicy;
  mySyncPolicy syncPolicy(std::max(3u, 1 + NUM_CAMERAS / 4), NUM_CAMERAS);
  syncPolicy.setInterMessageLowerBound(ros::Duration(0.020));
  syncPolicy.setAgePenalty(2.0);
  message_filters::SynchronizerVec<mySyncPolicy> sync((mySyncPolicy)syncPolicy, skel_subs);
  sync.registerCallback(skeletonCallback);
  ros::spin();

  // Wake up skel_data thread to make it exit cleanly.
  {
    std::lock_guard<std::mutex> lck (g_skel_data_mutex);
    g_skel_data_updated = true; // will re-use last message
  }
  g_skel_data_cv.notify_one();
  skel_data_thread.join();

  for (int i = 0; i < max_num_timings; ++i) {
    if(g_timing_cnt[i] > 0){
      cout << "Triangulation: ";
      if(i > 0)
        cout << i << " detections: ";
      cout << "avg runtime: " << g_timings[i] / g_timing_cnt[i] << "ms" << endl;
    }
  }

  if(g_param_vis_covariance){
    cout << "Sigmas 3D [mm]:" << endl;
    cout << "min: x: " << std::sqrt(g_min_sigmas_3d[0]) * 1000 << "mm, y: " << std::sqrt(g_min_sigmas_3d[1]) * 1000 << "mm, z: " << std::sqrt(g_min_sigmas_3d[2]) * 1000 << "mm." << endl;
    cout << "max: x: " << std::sqrt(g_max_sigmas_3d[0]) * 1000 << "mm, y: " << std::sqrt(g_max_sigmas_3d[1]) * 1000 << "mm, z: " << std::sqrt(g_max_sigmas_3d[2]) * 1000 << "mm." << endl;
  }
}
