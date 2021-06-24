#include <ros/ros.h>
#include <Eigen/Core>

#include <person_msgs/PersonCovList.h>
#include <skeleton_3d/fusion_body_parts.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#include <Hungarian.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Marginals.h>

#include <string>
#include <functional>
#include <vector>

#include <chrono>

using namespace gtsam;

using std::string;
using std::cout;
using std::endl;
using person_msgs::PersonCovList;
using person_msgs::PersonCov;

static constexpr int max_num_timings = 10;
static std::vector<double> g_timings(max_num_timings, 0.0);
static std::vector<int> g_timing_cnt(max_num_timings, 0);

const string PERSON_TOPIC = "human_pose_estimation/persons_3d";

static string g_param_pose_method = "simple";
static bool   g_param_normalize_by_height = false;
static bool   g_param_vis_covariance = false;
constexpr double FUSION_BODY_PARTS::vel_sigmas[FUSION_BODY_PARTS::NUM_KEYPOINTS];

static std::vector<std_msgs::ColorRGBA> g_colors;

static double g_limbLSigmaFactor = 1.0; // standard deviation multiplicator for limb lengths in skeleton model
static double g_predNoiseSigma = 0.12; //0.124; // standard deviation for prediction noise covariance
const double g_defaultResSigma = 0.10; // 10cm standard deviation for result when marginal covariance cannot be used

const float g_min_score = 0.10f;
const double g_avg_delay = 0.10;
const double g_root_sigma_factor = 100.0;
const int g_n_mov_avg = 3;
static std::vector<double> g_fb_delay_buffer(g_n_mov_avg, g_avg_delay);

static ros::Publisher g_pub_fusion, g_pub_fusion_pred, g_pub_fusion_marker;

static double g_t_prev;
static int g_next_id = 0;
static int g_frame_nr = 0;

const double g_t_max_unobserved = 1.0; // keep hypothesis alive for 1 second without observations
const double g_dist_threshold = 5.0; // maximum distance for association of detection to track
const double g_merge_dist_thresh = 0.20; // merge detections when average distance is smaller than threshold (m)
const double MAX_DIST = 1e6;
const int g_min_num_obs_track = 10; // min. number of observations before publishing

class TrackingHypothesis{
public:
  Values prevEstimate;
  std::vector<std::vector<Eigen::Vector3d>> velBuffer;
  double t_prev;
  int num_obs;
  int id;
  double height_prev;
  Point3 root_prev;

public:
  TrackingHypothesis(int id) :
    velBuffer(std::vector<std::vector<Eigen::Vector3d>>(FUSION_BODY_PARTS::NUM_KEYPOINTS, std::vector<Eigen::Vector3d>(g_n_mov_avg, Eigen::Vector3d::Zero()))),
    num_obs(0), id(id), height_prev(-1.0)
  {}

  double calc_normed_dist(const PersonCov& person, double t){
    double delta_t = t - t_prev;
    int num_joints_used = 0;
    double dist = 0;
    for (int kp_idx = 0; kp_idx < FUSION_BODY_PARTS::NUM_KEYPOINTS; ++kp_idx) { // For each joint of the skeleton.
      const auto& kp = person.keypoints[kp_idx];
      if(kp.score > g_min_score && prevEstimate.exists(kp_idx)){ // valid measurement of joint and joint exists
        const Point3& prev_kp = prevEstimate.at<Point3>(kp_idx) * height_prev + root_prev;
        dist += std::sqrt(std::pow(kp.joint.x - prev_kp.x(), 2) + std::pow(kp.joint.y - prev_kp.y(), 2) + std::pow(kp.joint.z - prev_kp.z(), 2)) / (FUSION_BODY_PARTS::vel_sigmas[kp_idx] * delta_t);
        ++num_joints_used;
      }
    }

    if(num_joints_used > 0)
      return dist / num_joints_used;
    else
      return MAX_DIST;
  }

  double calc_3d_dist(const TrackingHypothesis& other){
    int num_joints_used = 0;
    double dist = 0;
    for(const Values::ConstFiltered<Point3>::KeyValuePair it: prevEstimate.filter<Point3>()){
      if(other.prevEstimate.exists(it.key)){
        const Point3& kp = it.value * height_prev + root_prev;
        const Point3& other_kp = other.prevEstimate.at<Point3>(it.key) * other.height_prev + other.root_prev;
        dist += std::sqrt(std::pow(kp.x() - other_kp.x(), 2) + std::pow(kp.y() - other_kp.y(), 2) + std::pow(kp.z() - other_kp.z(), 2));
        ++num_joints_used;
      }
    }

    if(num_joints_used > 0)
      return dist / num_joints_used;
    else
      return MAX_DIST;
  }

};

static std::vector<TrackingHypothesis> g_tracks;

//gtsam measurement factor
class UnaryFactor : public NoiseModelFactor1<Point3>{
  double mx_, my_, mz_;

public:
  typedef boost::shared_ptr<UnaryFactor> shared_ptr;

  UnaryFactor(Key j, double x, double y, double z, const SharedNoiseModel& model):
      NoiseModelFactor1<Point3>(model, j), mx_(x), my_(y), mz_(z) {}

  virtual ~UnaryFactor() {}

  Vector evaluateError(const Point3& q, boost::optional<Matrix&> H = boost::none) const {
      if (H) (*H) = (Matrix(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();
      return (Vector(3) << q.x() - mx_, q.y() - my_, q.z() - mz_).finished();
  }

  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new UnaryFactor(*this))); }
}; // UnaryFactor


void define_colors(){
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
}

void reset(){
    ROS_INFO("Received reset signal. Resetting.");

    g_tracks.clear();
    g_fb_delay_buffer = std::vector<double>(g_n_mov_avg, g_avg_delay);
    g_next_id = 0;
    g_frame_nr = 0;
}

void remove_old_tracks(visualization_msgs::MarkerArray& vis_markers, const std_msgs::Header& header, double t){
  for(const auto& track : g_tracks){
    if(t - track.t_prev > g_t_max_unobserved){
      // Delete marker for skeleton and joints
      visualization_msgs::Marker del_skel;
      del_skel.header = header;
      del_skel.ns = "skeleton_fused";
      del_skel.id = track.id;
      del_skel.action = visualization_msgs::Marker::DELETE;
      vis_markers.markers.push_back(del_skel);

      visualization_msgs::Marker del_joints;
      del_joints.header = header;
      del_joints.ns = "joints_fused";
      del_joints.id = track.id;
      del_joints.action = visualization_msgs::Marker::DELETE;
      vis_markers.markers.push_back(del_joints);
    }
  }
  g_tracks.erase(std::remove_if(g_tracks.begin(), g_tracks.end(), [t](const TrackingHypothesis& track){return t - track.t_prev > g_t_max_unobserved;}), g_tracks.end());
}

void setKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const Eigen::Matrix3d &cov){
  kp.cov[0] = cov(0, 0); // xx
  kp.cov[1] = cov(0, 1); // xy
  kp.cov[2] = cov(0, 2); // xz
  kp.cov[3] = cov(1, 1); // yy
  kp.cov[4] = cov(1, 2); // yz
  kp.cov[5] = cov(2, 2); // zz
}

void addToKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const double& sigma){
  kp.cov[0] += sigma*sigma; //xx
  kp.cov[3] += sigma*sigma; //yy
  kp.cov[5] += sigma*sigma; //zz
}

void mergeKeypointCovariance(person_msgs::KeypointWithCovariance &kp, const person_msgs::KeypointWithCovariance &kp1, const person_msgs::KeypointWithCovariance &kp2){
  kp.cov[0] = (kp1.cov[0] + kp2.cov[0]) / 2.0; // xx
  kp.cov[1] = (kp1.cov[1] + kp2.cov[1]) / 2.0; // xy
  kp.cov[2] = (kp1.cov[2] + kp2.cov[2]) / 2.0; // xz
  kp.cov[3] = (kp1.cov[3] + kp2.cov[3]) / 2.0; // yy
  kp.cov[4] = (kp1.cov[4] + kp2.cov[4]) / 2.0; // yz
  kp.cov[5] = (kp1.cov[5] + kp2.cov[5]) / 2.0; // zz
}

void setMarkerPose(visualization_msgs::Marker& marker, const geometry_msgs::Point &joint, const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>& es) {
  marker.pose.position = joint;

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
}

void setMarkerPose(visualization_msgs::Marker &marker, const Point3 &joint, const Eigen::Vector3d &sigmas, const Eigen::Matrix3d &rot) {
  marker.pose.position.x = joint.x();
  marker.pose.position.y = joint.y();
  marker.pose.position.z = joint.z();

  Eigen::Quaterniond q_rot_cov(rot);

  marker.pose.orientation.w = q_rot_cov.w();
  marker.pose.orientation.x = q_rot_cov.x();
  marker.pose.orientation.y = q_rot_cov.y();
  marker.pose.orientation.z = q_rot_cov.z();

  marker.scale.x = 2.0 * 2.7955 * sigmas.x(); //2.7955 = sqrt(chi2inv(0.95,3) -> 2-sigma interval
  marker.scale.y = 2.0 * 2.7955 * sigmas.y();
  marker.scale.z = 2.0 * 2.7955 * sigmas.z();
}

bool addJointToSkeleton(int kp_idx, const geometry_msgs::Point& joint_fused, visualization_msgs::Marker& skel_fused, std::vector<int>& kpIdx2msgIdx){
  switch (kp_idx) {
  case FUSION_BODY_PARTS::Nose:
    skel_fused.points.push_back(joint_fused);
    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::Head:
    if(kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Nose]]); // add the Nose joint
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::RElbow: case FUSION_BODY_PARTS::RWrist: case FUSION_BODY_PARTS::LElbow: case FUSION_BODY_PARTS::LWrist: case FUSION_BODY_PARTS::RKnee: case FUSION_BODY_PARTS::RAnkle: case FUSION_BODY_PARTS::LKnee: case FUSION_BODY_PARTS::LAnkle:
    if (kpIdx2msgIdx[kp_idx - 1] != -1 && kpIdx2msgIdx[kp_idx - 1] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[kp_idx - 1]]); // add the previous joint
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::RShoulder: case FUSION_BODY_PARTS::LShoulder: case FUSION_BODY_PARTS::MidHip:
    if (kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Neck]]); // add the neck joint (idx 1)
    else if(kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Nose]]); // add the nose joint (idx 0) when neck is not observed.
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::Belly:
    if(kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Neck]]); // add the Neck joint when Belly is not observed
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);

    if(kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip]]); // add the Neck joint when Belly is not observed
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::RHip: case FUSION_BODY_PARTS::LHip:
    if (kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::MidHip]]); // add the hip joint (idx 8)
    else if (kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Neck] < skel_fused.points.size())
       skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Neck]]); // add the neck joint (idx 1)
    else if (kpIdx2msgIdx[kp_idx - 7] != -1 && kpIdx2msgIdx[kp_idx - 7] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[kp_idx - 7]]); // add the resp. shoulders
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::REye: case FUSION_BODY_PARTS::LEye: case FUSION_BODY_PARTS::Neck:
    if (kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] != -1 && kpIdx2msgIdx[FUSION_BODY_PARTS::Nose] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[FUSION_BODY_PARTS::Nose]]); // add the nose joint (idx 0)
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;
  case FUSION_BODY_PARTS::REar: case FUSION_BODY_PARTS::LEar:
    if (kpIdx2msgIdx[kp_idx - 2] != -1 && kpIdx2msgIdx[kp_idx - 2] < skel_fused.points.size())
      skel_fused.points.push_back(skel_fused.points[kpIdx2msgIdx[kp_idx - 2]]); // add the pre-previous joint (the resp. eye)
    else
      skel_fused.points.push_back(joint_fused);

    skel_fused.points.push_back(joint_fused);

    skel_fused.colors.push_back(g_colors[kp_idx]);
    skel_fused.colors.push_back(g_colors[kp_idx]);
    break;

  default:
    return false;
  }

  kpIdx2msgIdx[kp_idx] = skel_fused.points.size() - 1;
  return true;
}

void addBinaryFactors(NonlinearFactorGraph& graph, const std::vector<bool>& joints_measured){
  if(g_param_normalize_by_height){
    if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::RHip])
      graph.emplace_shared<RangeFactor<Point3>>( 8,  9, 0.17, noiseModel::Isotropic::Sigma(1, 0.062 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Right-Hip
    if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::LHip])
      graph.emplace_shared<RangeFactor<Point3>>( 8, 12, 0.17, noiseModel::Isotropic::Sigma(1, 0.062 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Left-Hip
    if(joints_measured[FUSION_BODY_PARTS::RHip] && joints_measured[FUSION_BODY_PARTS::RKnee])
      graph.emplace_shared<RangeFactor<Point3>>( 9, 10, 0.694, noiseModel::Isotropic::Sigma(1, 0.111 * g_limbLSigmaFactor)); // Right-Hip <-> Right Knee
    if(joints_measured[FUSION_BODY_PARTS::RKnee] && joints_measured[FUSION_BODY_PARTS::RAnkle])
      graph.emplace_shared<RangeFactor<Point3>>(10, 11, 0.708, noiseModel::Isotropic::Sigma(1, 0.097 * g_limbLSigmaFactor)); // Right-Knee <-> Right Ankle
    if(joints_measured[FUSION_BODY_PARTS::LHip] && joints_measured[FUSION_BODY_PARTS::LKnee])
      graph.emplace_shared<RangeFactor<Point3>>(12, 13, 0.694, noiseModel::Isotropic::Sigma(1, 0.111 * g_limbLSigmaFactor)); // Left-Hip <-> Left Knee
    if(joints_measured[FUSION_BODY_PARTS::LKnee] && joints_measured[FUSION_BODY_PARTS::LAnkle])
      graph.emplace_shared<RangeFactor<Point3>>(13, 14, 0.708, noiseModel::Isotropic::Sigma(1, 0.097 * g_limbLSigmaFactor)); // Left-Knee <-> Left Ankle
    if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::Nose])
      graph.emplace_shared<RangeFactor<Point3>>( 1,  0, 0.33, noiseModel::Isotropic::Sigma(1, 0.050 * g_limbLSigmaFactor)); // Neck <-> Nose
    if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::RShoulder])
      graph.emplace_shared<RangeFactor<Point3>>( 1,  2, 0.262, noiseModel::Isotropic::Sigma(1, 0.092 * g_limbLSigmaFactor)); // Neck <-> Right Shoulder
    if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::LShoulder])
      graph.emplace_shared<RangeFactor<Point3>>( 1,  5, 0.262, noiseModel::Isotropic::Sigma(1, 0.092 * g_limbLSigmaFactor)); // Neck <-> Left Shoulder
    if(joints_measured[FUSION_BODY_PARTS::RShoulder] && joints_measured[FUSION_BODY_PARTS::RElbow])
      graph.emplace_shared<RangeFactor<Point3>>( 2,  3, 0.515, noiseModel::Isotropic::Sigma(1, 0.071 * g_limbLSigmaFactor)); // Right Shoulder <-> Right Elbow
    if(joints_measured[FUSION_BODY_PARTS::RElbow] && joints_measured[FUSION_BODY_PARTS::RWrist])
      graph.emplace_shared<RangeFactor<Point3>>( 3,  4, 0.444, noiseModel::Isotropic::Sigma(1, 0.084 * g_limbLSigmaFactor)); // Right Elbow <-> Right Wrist
    if(joints_measured[FUSION_BODY_PARTS::LShoulder] && joints_measured[FUSION_BODY_PARTS::LElbow])
      graph.emplace_shared<RangeFactor<Point3>>( 5,  6, 0.515, noiseModel::Isotropic::Sigma(1, 0.071 * g_limbLSigmaFactor)); // Left Shoulder <-> Left Elbow
    if(joints_measured[FUSION_BODY_PARTS::LElbow] && joints_measured[FUSION_BODY_PARTS::LWrist])
      graph.emplace_shared<RangeFactor<Point3>>( 6,  7, 0.444, noiseModel::Isotropic::Sigma(1, 0.084 * g_limbLSigmaFactor)); // Left Elbow <-> Left Wrist

    //The following pairs occur in H36M Model only
    if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::Belly])
      graph.emplace_shared<RangeFactor<Point3>>( 8, 20, 0.49, noiseModel::Isotropic::Sigma(1, 0.05 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Belly
    if(joints_measured[FUSION_BODY_PARTS::Belly] && joints_measured[FUSION_BODY_PARTS::Neck])
      graph.emplace_shared<RangeFactor<Point3>>(20,  1, 0.51, noiseModel::Isotropic::Sigma(1, 0.05 * g_limbLSigmaFactor)); // Belly <-> Neck
    if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::Head])
      graph.emplace_shared<RangeFactor<Point3>>( 0, 19, 0.23, noiseModel::Isotropic::Sigma(1, 0.05 * g_limbLSigmaFactor)); // Nose <-> Head

    //The following pairs occur in Simple-Baselines model only
    if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::Neck] && !joints_measured[FUSION_BODY_PARTS::Belly])
      graph.emplace_shared<RangeFactor<Point3>>( 8,  1, 1.000, noiseModel::Isotropic::Sigma(1, 0.02 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Neck
    if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::REye])
      graph.emplace_shared<RangeFactor<Point3>>( 0, 15, 0.085, noiseModel::Isotropic::Sigma(1, 0.06 * g_limbLSigmaFactor)); // Nose <-> right Eye
    if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::LEye])
      graph.emplace_shared<RangeFactor<Point3>>( 0, 16, 0.085, noiseModel::Isotropic::Sigma(1, 0.06 * g_limbLSigmaFactor)); // Nose <-> left Eye
    if(joints_measured[FUSION_BODY_PARTS::REye] && joints_measured[FUSION_BODY_PARTS::REar])
      graph.emplace_shared<RangeFactor<Point3>>(15, 17, 0.167, noiseModel::Isotropic::Sigma(1, 0.08 * g_limbLSigmaFactor)); // right Eye <-> right Ear
    if(joints_measured[FUSION_BODY_PARTS::LEye] && joints_measured[FUSION_BODY_PARTS::LEar])
      graph.emplace_shared<RangeFactor<Point3>>(16, 18, 0.167, noiseModel::Isotropic::Sigma(1, 0.08 * g_limbLSigmaFactor)); // left Eye <-> left Ear
  }
  else{ // absolute bone lengths
  if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::RHip])
    graph.emplace_shared<RangeFactor<Point3>>( 8,  9, 0.134, noiseModel::Isotropic::Sigma(1, 0.033 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Right-Hip
  if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::LHip])
    graph.emplace_shared<RangeFactor<Point3>>( 8, 12, 0.134, noiseModel::Isotropic::Sigma(1, 0.033 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Left-Hip
  if(joints_measured[FUSION_BODY_PARTS::RHip] && joints_measured[FUSION_BODY_PARTS::RKnee])
    graph.emplace_shared<RangeFactor<Point3>>( 9, 10, 0.449, noiseModel::Isotropic::Sigma(1, 0.051 * g_limbLSigmaFactor)); // Right-Hip <-> Right Knee
  if(joints_measured[FUSION_BODY_PARTS::RKnee] && joints_measured[FUSION_BODY_PARTS::RAnkle])
    graph.emplace_shared<RangeFactor<Point3>>(10, 11, 0.446, noiseModel::Isotropic::Sigma(1, 0.051 * g_limbLSigmaFactor)); // Right-Knee <-> Right Ankle
  if(joints_measured[FUSION_BODY_PARTS::LHip] && joints_measured[FUSION_BODY_PARTS::LKnee])
    graph.emplace_shared<RangeFactor<Point3>>(12, 13, 0.449, noiseModel::Isotropic::Sigma(1, 0.051 * g_limbLSigmaFactor)); // Left-Hip <-> Left Knee
  if(joints_measured[FUSION_BODY_PARTS::LKnee] && joints_measured[FUSION_BODY_PARTS::LAnkle])
    graph.emplace_shared<RangeFactor<Point3>>(13, 14, 0.446, noiseModel::Isotropic::Sigma(1, 0.051 * g_limbLSigmaFactor)); // Left-Knee <-> Left Ankle
  if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::Nose])
    graph.emplace_shared<RangeFactor<Point3>>( 1,  0, 0.20, noiseModel::Isotropic::Sigma(1, 0.025 * g_limbLSigmaFactor)); // Neck <-> Nose
  if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::RShoulder])
    graph.emplace_shared<RangeFactor<Point3>>( 1,  2, 0.15, noiseModel::Isotropic::Sigma(1, 0.042 * g_limbLSigmaFactor)); // Neck <-> Right Shoulder
  if(joints_measured[FUSION_BODY_PARTS::Neck] && joints_measured[FUSION_BODY_PARTS::LShoulder])
    graph.emplace_shared<RangeFactor<Point3>>( 1,  5, 0.15, noiseModel::Isotropic::Sigma(1, 0.042 * g_limbLSigmaFactor)); // Neck <-> Left Shoulder
  if(joints_measured[FUSION_BODY_PARTS::RShoulder] && joints_measured[FUSION_BODY_PARTS::RElbow])
    graph.emplace_shared<RangeFactor<Point3>>( 2,  3, 0.28, noiseModel::Isotropic::Sigma(1, 0.045 * g_limbLSigmaFactor)); // Right Shoulder <-> Right Elbow
  if(joints_measured[FUSION_BODY_PARTS::RElbow] && joints_measured[FUSION_BODY_PARTS::RWrist])
    graph.emplace_shared<RangeFactor<Point3>>( 3,  4, 0.25, noiseModel::Isotropic::Sigma(1, 0.063 * g_limbLSigmaFactor)); // Right Elbow <-> Right Wrist
  if(joints_measured[FUSION_BODY_PARTS::LShoulder] && joints_measured[FUSION_BODY_PARTS::LElbow])
    graph.emplace_shared<RangeFactor<Point3>>( 5,  6, 0.28, noiseModel::Isotropic::Sigma(1, 0.045 * g_limbLSigmaFactor)); // Left Shoulder <-> Left Elbow
  if(joints_measured[FUSION_BODY_PARTS::LElbow] && joints_measured[FUSION_BODY_PARTS::LWrist])
    graph.emplace_shared<RangeFactor<Point3>>( 6,  7, 0.25, noiseModel::Isotropic::Sigma(1, 0.063 * g_limbLSigmaFactor)); // Left Elbow <-> Left Wrist

  //The following pairs occur in H36M Model only
  if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::Belly])
    graph.emplace_shared<RangeFactor<Point3>>( 8, 20, 0.23846, noiseModel::Isotropic::Sigma(1, 0.071 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Belly
  if(joints_measured[FUSION_BODY_PARTS::Belly] && joints_measured[FUSION_BODY_PARTS::Neck])
    graph.emplace_shared<RangeFactor<Point3>>(20,  1, 0.25534, noiseModel::Isotropic::Sigma(1, 0.035 * g_limbLSigmaFactor)); // Belly <-> Neck
  if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::Head])
    graph.emplace_shared<RangeFactor<Point3>>( 0, 19, 0.11500, noiseModel::Isotropic::Sigma(1, 0.035 * g_limbLSigmaFactor)); // Nose <-> Head

  //The following pairs occur in Simple-Baselines model only
  if(joints_measured[FUSION_BODY_PARTS::MidHip] && joints_measured[FUSION_BODY_PARTS::Neck] && !joints_measured[FUSION_BODY_PARTS::Belly])
    graph.emplace_shared<RangeFactor<Point3>>( 8,  1, 0.50, noiseModel::Isotropic::Sigma(1, 0.071 * g_limbLSigmaFactor)); // Mid-Hip (root) <-> Neck
  if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::REye])
    graph.emplace_shared<RangeFactor<Point3>>( 0, 15, 0.05, noiseModel::Isotropic::Sigma(1, 0.035 * g_limbLSigmaFactor)); // Nose <-> right Eye
  if(joints_measured[FUSION_BODY_PARTS::Nose] && joints_measured[FUSION_BODY_PARTS::LEye])
    graph.emplace_shared<RangeFactor<Point3>>( 0, 16, 0.05, noiseModel::Isotropic::Sigma(1, 0.035 * g_limbLSigmaFactor)); // Nose <-> left Eye
  if(joints_measured[FUSION_BODY_PARTS::REye] && joints_measured[FUSION_BODY_PARTS::REar])
    graph.emplace_shared<RangeFactor<Point3>>(15, 17, 0.10, noiseModel::Isotropic::Sigma(1, 0.05 * g_limbLSigmaFactor)); // right Eye <-> right Ear
  if(joints_measured[FUSION_BODY_PARTS::LEye] && joints_measured[FUSION_BODY_PARTS::LEar])
    graph.emplace_shared<RangeFactor<Point3>>(16, 18, 0.10, noiseModel::Isotropic::Sigma(1, 0.05 * g_limbLSigmaFactor)); // left Eye <-> left Ear
  }
}

void setInitialState(const Values& measurements, TrackingHypothesis& track, std::vector<bool>& use_velocity){
  std::vector<int> keys_to_remove;
  for(const Values::ConstFiltered<Point3>::KeyValuePair it: track.prevEstimate.filter<Point3>()){
    if(!measurements.exists(it.key)){ // joint was observed in last timestep but is not in current measurement. -> remove it from the initial values, to keep them coherent with the graph.
      keys_to_remove.push_back(it.key);
    }
  }
  for (const int& key : keys_to_remove) {
    track.prevEstimate.erase(key);
    track.velBuffer[key] = std::vector<Eigen::Vector3d>(g_n_mov_avg, Eigen::Vector3d::Zero()); // reset velocity buffer for non-observed joint
  }

  for(const Values::ConstFiltered<Point3>::KeyValuePair it: measurements.filter<Point3>()){
    if(!track.prevEstimate.exists(it.key)){
      track.prevEstimate.insert(it.key, it.value); // joint was not measured in last timestep and thus is not present in prevEstimate. -> add current measurement as initial value for the optimization
    }
    else{
      use_velocity[it.key] = true; // joint was measured in last timestamp and is currently observed -> velocity estimate can be calculated.
    }
  }
}

void skeletonCallback(const person_msgs::PersonCovList::ConstPtr &persons){
  double t = persons->header.stamp.toSec();
  auto t1 = std::chrono::high_resolution_clock::now();

  if((t - g_t_prev) > 0.17){
    ROS_WARN("Large frame delay delta_t = %fs (should be < 0.17s)", (t - g_t_prev));
  }

  double curr_avg_delay = 0.0; // determine average delay of feedback between all cameras
  int n_valid_delay_meas = 0;
  for (const auto& delay : persons->fb_delay_per_cam) {
    if(delay > 0.0f){ // delay = -1.0 if no delay could be measured on sensor (no feedback used)
      curr_avg_delay += static_cast<double>(delay);
      ++n_valid_delay_meas;
    }
  }
  if(n_valid_delay_meas > 0)
    curr_avg_delay /= n_valid_delay_meas;
  else
    curr_avg_delay = g_avg_delay;
  g_fb_delay_buffer[g_frame_nr % g_n_mov_avg] = curr_avg_delay; // save current delay in moving-average buffer
  double pred_delta_t = std::accumulate(g_fb_delay_buffer.begin(), g_fb_delay_buffer.end(), 0.0) / g_n_mov_avg;

  PersonCovList persons_fused;
  persons_fused.header = persons->header;
  persons_fused.ts_per_cam = persons->ts_per_cam;
  persons_fused.fb_delay_per_cam = std::vector<float>(persons->fb_delay_per_cam.size(), (float)pred_delta_t); // store the predicted latency in message //TODO: handle differences between cameras!
  PersonCovList persons_fused_pred = persons_fused;
  visualization_msgs::MarkerArray vis_markers;

  int n_hyp = g_tracks.size(), n_det = persons->persons.size();

  if(n_det == 0){
    remove_old_tracks(vis_markers, persons_fused.header, t);
    if(!vis_markers.markers.empty())
      g_pub_fusion_marker.publish(vis_markers);
    g_pub_fusion.publish(persons_fused);
    g_pub_fusion_pred.publish(persons_fused_pred);

    g_t_prev = t;
    return;
  }

  int *assignment = nullptr;

  if(n_hyp > 0){
    double *C = new double[n_det * n_hyp]; // Cost-Matrix in ColumnMajor (!) order, n_det = rows, n_hyp = cols
    assignment = new int[n_det];
    double cost = 0.0;
    for (int track_idx = 0; track_idx < n_hyp; ++track_idx) {
      for (int person_idx = 0; person_idx < n_det; ++person_idx) {
        C[person_idx + n_det * track_idx] = g_tracks[track_idx].calc_normed_dist(persons->persons[person_idx], t);
        //cout << "track " << track_idx << " <-> det " << person_idx << ": dist: " << C[person_idx + n_det * track_idx] << endl;
      }
    }

    HungarianAlgorithm::assignmentoptimal(assignment, &cost, C, n_det, n_hyp);
    for (int i = 0; i < n_det; ++i) {
      if(assignment[i] >= 0 && C[i + n_det * assignment[i]] > g_dist_threshold) // Costs are larger than a threshold -> don't associate
        assignment[i] = -1;
    }

    delete[] C;
  }

  std::vector<int> track_ids(n_det);
  for (int person_idx = 0; person_idx < n_det; ++person_idx) {
    if(assignment != nullptr && assignment[person_idx] >= 0)
      track_ids[person_idx] = assignment[person_idx];
    else{
      g_tracks.push_back(TrackingHypothesis(g_next_id));
      track_ids[person_idx] = g_tracks.size() - 1;
      cout << "detection " << person_idx << " initialized new track " << g_next_id << endl;
      ++g_next_id;
    }
  }

  #pragma omp parallel num_threads(n_det)
  {
    std::vector<PersonCov> persons_fused_private, persons_fused_pred_private;
    std::vector<visualization_msgs::Marker> vis_markers_private;
    #pragma omp for nowait
    for (int person_idx = 0; person_idx < n_det; ++person_idx) {
      const auto& person = persons->persons[person_idx];

      TrackingHypothesis& curr_track = g_tracks[track_ids[person_idx]];

      PersonCov person_fused, person_fused_pred;
      person_fused.keypoints.resize(FUSION_BODY_PARTS::NUM_KEYPOINTS);
      person_fused_pred.keypoints.resize(FUSION_BODY_PARTS::NUM_KEYPOINTS);
      person_fused.id = curr_track.id;
      person_fused_pred.id = curr_track.id;

      visualization_msgs::Marker skel_fused;
      skel_fused.header = persons_fused.header;
      //skel_fused.lifetime = ros::Duration(g_t_max_unobserved);
      skel_fused.pose.position.x = 0;
      skel_fused.pose.position.y = 0;
      skel_fused.pose.position.z = 0;
      skel_fused.pose.orientation.w = 1;
      skel_fused.pose.orientation.x = 0;
      skel_fused.pose.orientation.y = 0;
      skel_fused.pose.orientation.z = 0;
      skel_fused.type = visualization_msgs::Marker::LINE_LIST;
      skel_fused.scale.x = .05;
      skel_fused.ns = "skeleton_fused";
      skel_fused.id = curr_track.id;
      skel_fused.color.r = 1.0;
      skel_fused.color.a = 1.0;

      visualization_msgs::Marker joints_fused;
      joints_fused.header = persons_fused.header;
      //joints_fused.lifetime = ros::Duration(g_t_max_unobserved);
      joints_fused.pose = skel_fused.pose;
      joints_fused.type = visualization_msgs::Marker::SPHERE_LIST;
      joints_fused.scale.x = joints_fused.scale.y = joints_fused.scale.z = 0.07;
      joints_fused.ns = "joints_fused";
      joints_fused.id = curr_track.id;
      joints_fused.color.r = joints_fused.color.g = 0.5;
      joints_fused.color.a = 1.0;

      int num_meas = 0;
      std::vector<bool> joints_measured(FUSION_BODY_PARTS::NUM_KEYPOINTS, false), use_velocity(FUSION_BODY_PARTS::NUM_KEYPOINTS, false);
      NonlinearFactorGraph graph;
      Values measurements;

      person_msgs::KeypointWithCovariance root, neck;
      double height = 1.0;
      if(g_param_pose_method == "h36m"){
        root = person.keypoints[FUSION_BODY_PARTS::MidHip];
        neck  = person.keypoints[FUSION_BODY_PARTS::Neck];
      }
      else if(g_param_pose_method == "simple"){ // Add root as mean of both hips
        const auto& kpHp_left = person.keypoints[FUSION_BODY_PARTS::LHip];
        const auto& kpHp_right = person.keypoints[FUSION_BODY_PARTS::RHip];
        if(kpHp_left.score > 0.0f && kpHp_right.score > 0.0f){
          root.joint.x = (kpHp_left.joint.x + kpHp_right.joint.x) / 2.0;
          root.joint.y = (kpHp_left.joint.y + kpHp_right.joint.y) / 2.0;
          root.joint.z = (kpHp_left.joint.z + kpHp_right.joint.z) / 2.0;
          root.score   = (kpHp_left.score   + kpHp_right.score)   / 2.0f;
        }

        // Add Neck as mean of both shoulders
        const auto& kpSh_left = person.keypoints[FUSION_BODY_PARTS::LShoulder];
        const auto& kpSh_right = person.keypoints[FUSION_BODY_PARTS::RShoulder];
        if(kpSh_left.score > 0.0f && kpSh_right.score > 0.0f){
          neck.joint.x = (kpSh_left.joint.x + kpSh_right.joint.x) / 2.0;
          neck.joint.y = (kpSh_left.joint.y + kpSh_right.joint.y) / 2.0;
          neck.joint.z = (kpSh_left.joint.z + kpSh_right.joint.z) / 2.0;
          neck.score   = (kpSh_left.score   + kpSh_right.score)   / 2.0f;
        }
      }

      if(root.score > g_min_score){
        if(g_param_normalize_by_height){
          if(neck.score > g_min_score){
            //height = || neck - root ||
            height = Eigen::Vector3d(neck.joint.x - root.joint.x, neck.joint.y - root.joint.y, neck.joint.z - root.joint.z).norm();
          }
          else{
            ROS_WARN("Neck joint is not observed. Cannot define height (neck - root), defaulting to 0.6m");
            height = 0.60;
          }
        }

        Eigen::Matrix3d root_cov;
        if(g_param_pose_method == "h36m"){
          root_cov << root.cov[0], root.cov[1], root.cov[2],
                      root.cov[1], root.cov[3], root.cov[4],
                      root.cov[2], root.cov[4], root.cov[5];
        }
        else if(g_param_pose_method == "simple"){
          const auto& kpHp_left = person.keypoints[FUSION_BODY_PARTS::LHip];
          const auto& kpHp_right = person.keypoints[FUSION_BODY_PARTS::RHip];
          Eigen::Matrix3d cov_left, cov_right;
          cov_left << kpHp_left.cov[0], kpHp_left.cov[1], kpHp_left.cov[2],
                      kpHp_left.cov[1], kpHp_left.cov[3], kpHp_left.cov[4],
                      kpHp_left.cov[2], kpHp_left.cov[4], kpHp_left.cov[5];
          cov_right << kpHp_right.cov[0], kpHp_right.cov[1], kpHp_right.cov[2],
                       kpHp_right.cov[1], kpHp_right.cov[3], kpHp_right.cov[4],
                       kpHp_right.cov[2], kpHp_right.cov[4], kpHp_right.cov[5];
          root_cov = (cov_left + cov_right) / 2.0;
        }

        // Set root-joint measurement to origin.
        graph.emplace_shared<UnaryFactor>((int)(FUSION_BODY_PARTS::MidHip), 0.0, 0.0, 0.0, noiseModel::Gaussian::Covariance(root_cov / (height * height) / (g_root_sigma_factor*g_root_sigma_factor))); // For root joint (MidHip) decrease the unary covariance to fix its global position
        measurements.insert((int)(FUSION_BODY_PARTS::MidHip), Point3(0.0, 0.0, 0.0));
        ++num_meas;
        joints_measured[FUSION_BODY_PARTS::MidHip] = true;
      }
      else{
        //ROS_WARN("Root joint of person %d is not defined. Cannot center skeleton model", person_idx);
      }

      if(curr_track.height_prev < 0.0){ //initialize prev_height and prev_root
        curr_track.height_prev = height;
        curr_track.root_prev = Point3(root.joint.x, root.joint.y, root.joint.z);
      }

      for (int kp_idx = 0; kp_idx < FUSION_BODY_PARTS::NUM_KEYPOINTS; ++kp_idx) { // For each joint of the skeleton.
        if(kp_idx == FUSION_BODY_PARTS::MidHip)
          continue; // already  handled above..

        const auto& kp = person.keypoints[kp_idx];
        if(kp.score > g_min_score){ // There is a valid measurement of the resp. joint.
          Eigen::Matrix3d cov;
          cov << kp.cov[0], kp.cov[1], kp.cov[2],
                 kp.cov[1], kp.cov[3], kp.cov[4],
                 kp.cov[2], kp.cov[4], kp.cov[5];
          graph.emplace_shared<UnaryFactor>(kp_idx, (kp.joint.x - root.joint.x) / height, (kp.joint.y - root.joint.y) / height, (kp.joint.z - root.joint.z) / height, noiseModel::Gaussian::Covariance(cov / (height * height)));
          measurements.insert(kp_idx, Point3((kp.joint.x - root.joint.x) / height, (kp.joint.y - root.joint.y) / height, (kp.joint.z - root.joint.z) / height));
          ++num_meas;
          joints_measured[kp_idx] = true;
        }
      }

      if(g_param_pose_method == "simple"){ // Add Neck as mean of both shoulders
        if(neck.score > g_min_score){
          const auto& kpSh_left = person.keypoints[FUSION_BODY_PARTS::LShoulder];
          const auto& kpSh_right = person.keypoints[FUSION_BODY_PARTS::RShoulder];
          Eigen::Matrix3d cov_left, cov_right;
          cov_left << kpSh_left.cov[0], kpSh_left.cov[1], kpSh_left.cov[2],
                      kpSh_left.cov[1], kpSh_left.cov[3], kpSh_left.cov[4],
                      kpSh_left.cov[2], kpSh_left.cov[4], kpSh_left.cov[5];
          cov_right << kpSh_right.cov[0], kpSh_right.cov[1], kpSh_right.cov[2],
                       kpSh_right.cov[1], kpSh_right.cov[3], kpSh_right.cov[4],
                       kpSh_right.cov[2], kpSh_right.cov[4], kpSh_right.cov[5];
          graph.emplace_shared<UnaryFactor>((int)(FUSION_BODY_PARTS::Neck), (neck.joint.x - root.joint.x) / height, (neck.joint.y - root.joint.y) / height, (neck.joint.z - root.joint.z) / height, noiseModel::Gaussian::Covariance((cov_left + cov_right) / 2.0 / (height * height)));
          measurements.insert((int)(FUSION_BODY_PARTS::Neck), Point3((neck.joint.x - root.joint.x) / height, (neck.joint.y - root.joint.y) / height, (neck.joint.z - root.joint.z) / height));
          ++num_meas;
          joints_measured[FUSION_BODY_PARTS::Neck] = true;
        }
      }

      if(num_meas == 0){ // no joints measured for person -> continue with next person
        continue;
      }

      setInitialState(measurements, curr_track, use_velocity);
      addBinaryFactors(graph, joints_measured);

      LevenbergMarquardtOptimizer optimizer(graph, curr_track.prevEstimate);
      Values result;
      try{
        result = optimizer.optimize();
      }
      catch(const std::exception& e){
        ROS_WARN("Optimization failed! \n%s\n\n", e.what());

        curr_track.prevEstimate.print();
        graph.print();

        result = measurements; // Optimization failed. output measurements.
      }

      Marginals marginals;
      bool use_marginals = false;
      try {
        marginals = Marginals(graph, result);
        use_marginals = true;
      } catch (gtsam::IndeterminantLinearSystemException e){
        ROS_WARN("Cannot use marginal covariance as Linear System is indetermined.");
      }

      std::vector<int> kpIdx2msgIdx(FUSION_BODY_PARTS::NUM_KEYPOINTS, -1);
      for(const Values::ConstFiltered<Point3>::KeyValuePair it: result.filter<Point3>()){
        int kp_idx = it.key;

        geometry_msgs::Point joint_fused;
        joint_fused.x = it.value.x() * height + root.joint.x;
        joint_fused.y = it.value.y() * height + root.joint.y;
        joint_fused.z = it.value.z() * height + root.joint.z;
        person_fused.keypoints[kp_idx].joint = joint_fused;

        if(kp_idx == FUSION_BODY_PARTS::MidHip)
          person_fused.keypoints[kp_idx].score = std::max(g_min_score, root.score);
        else if(kp_idx == FUSION_BODY_PARTS::Neck)
          person_fused.keypoints[kp_idx].score = std::max(g_min_score, neck.score);
        else
          person_fused.keypoints[kp_idx].score = std::max(g_min_score, person.keypoints[kp_idx].score);

        Eigen::Matrix3d cov;
        if(use_marginals){
          try{
            cov = marginals.marginalCovariance(kp_idx) * height * height;
          }catch (gtsam::IndeterminantLinearSystemException e){
            ROS_WARN("Cannot use marginal covariance of joint %d as Linear System is indetermined.", kp_idx);
            cov = g_defaultResSigma * g_defaultResSigma * Eigen::Matrix3d::Identity();
          }

          if(g_param_vis_covariance && kp_idx < 15){
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
            visualization_msgs::Marker joint_fused_cov;
            joint_fused_cov.header = persons_fused.header;
            joint_fused_cov.lifetime = ros::Duration(g_t_max_unobserved);
            joint_fused_cov.type = visualization_msgs::Marker::SPHERE;
            joint_fused_cov.ns = "joint_cov_fused";
            joint_fused_cov.id = FUSION_BODY_PARTS::NUM_KEYPOINTS * person_idx + kp_idx;
            joint_fused_cov.color = g_colors[kp_idx];
            joint_fused_cov.color.a = 0.50f;
            setMarkerPose(joint_fused_cov, joint_fused, es);
            vis_markers_private.push_back(joint_fused_cov);
          }
        }
        else {
          cov = g_defaultResSigma * g_defaultResSigma * Eigen::Matrix3d::Identity();
        }

        if(kp_idx == FUSION_BODY_PARTS::MidHip) // re-scale covariance to compensate sigma_factor for root joint
          cov *= (g_root_sigma_factor * g_root_sigma_factor);

        setKeypointCovariance(person_fused.keypoints[kp_idx], cov);

        geometry_msgs::Point joint_fused_pred = joint_fused;
        if(use_velocity[kp_idx]){
          curr_track.velBuffer[kp_idx][g_frame_nr % g_n_mov_avg] = (it.value * height + Point3(root.joint.x, root.joint.y, root.joint.z)
                                                                - (curr_track.prevEstimate.at<Point3>(kp_idx) * curr_track.height_prev + curr_track.root_prev) ) / (t - g_t_prev); //add / replace velocity of joint in moving average buffer
          Eigen::Vector3d pred_delta = std::accumulate(curr_track.velBuffer[kp_idx].begin(), curr_track.velBuffer[kp_idx].end(), Eigen::Vector3d::Zero().eval()) / g_n_mov_avg * pred_delta_t; //g_avg_delay
          joint_fused_pred.x += pred_delta.x();
          joint_fused_pred.y += pred_delta.y();
          joint_fused_pred.z += pred_delta.z();
        }

        person_fused_pred.keypoints[kp_idx].joint = joint_fused_pred;
        person_fused_pred.keypoints[kp_idx].score = person_fused.keypoints[kp_idx].score;
        person_fused_pred.keypoints[kp_idx].cov   = person_fused.keypoints[kp_idx].cov;
        addToKeypointCovariance(person_fused_pred.keypoints[kp_idx], g_predNoiseSigma); // Add prediction noise

        joints_fused.points.push_back(joint_fused);
        joints_fused.colors.push_back(g_colors[kp_idx]);

        addJointToSkeleton(kp_idx, joint_fused, skel_fused, kpIdx2msgIdx);
      }

      curr_track.t_prev = t;
      curr_track.prevEstimate.swap(result);
      curr_track.height_prev = height;
      curr_track.root_prev = Point3(root.joint.x, root.joint.y, root.joint.z);
      ++curr_track.num_obs;

      if(curr_track.num_obs > g_min_num_obs_track){
        persons_fused_private.push_back(person_fused);
        persons_fused_pred_private.push_back(person_fused_pred);
      }
      if(joints_fused.points.size() > 0 && curr_track.num_obs > g_min_num_obs_track){
        vis_markers_private.push_back(joints_fused);
        vis_markers_private.push_back(skel_fused);
      }
    } // end for all persons

    #pragma omp critical
    {
      persons_fused.persons.insert(persons_fused.persons.end(), persons_fused_private.begin(), persons_fused_private.end());
      persons_fused_pred.persons.insert(persons_fused_pred.persons.end(), persons_fused_pred_private.begin(), persons_fused_pred_private.end());
      vis_markers.markers.insert(vis_markers.markers.end(), vis_markers_private.begin(), vis_markers_private.end());
    }
  } // end omp parallel

  if(assignment != nullptr)
    delete[] assignment;

  //remove old tracks..
  remove_old_tracks(vis_markers, persons_fused.header, t);

  //check for tracks that are close to each other and merge them
  for (int i = 0; i < g_tracks.size(); ++i) {
    for(int j = i+1; j < g_tracks.size();){
      if(g_tracks[i].calc_3d_dist(g_tracks[j]) < g_merge_dist_thresh){
        cout << "Removing track " << g_tracks[j].id << " because of close overlap with track " << g_tracks[i].id << endl;
        int id_to_remove = g_tracks[j].id;
        g_tracks.erase(g_tracks.begin() + j);

        // Delete marker for skeleton and joints // TODO: actually, ID-s would need to be re-assigned for the current timestep..
        visualization_msgs::Marker del_skel;
        del_skel.header = persons_fused.header;
        del_skel.ns = "skeleton_fused";
        del_skel.id = id_to_remove;
        del_skel.action = visualization_msgs::Marker::DELETE;
        vis_markers.markers.push_back(del_skel);

        visualization_msgs::Marker del_joints;
        del_joints.header = persons_fused.header;
        del_joints.ns = "joints_fused";
        del_joints.id = id_to_remove;
        del_joints.action = visualization_msgs::Marker::DELETE;
        vis_markers.markers.push_back(del_joints);

        // re-assign id in persons message
        for (int person_idx = 0; person_idx < persons_fused.persons.size(); ++ person_idx) {
          if(persons_fused.persons[person_idx].id == id_to_remove){
            persons_fused.persons[person_idx].id = g_tracks[i].id;
            persons_fused_pred.persons[person_idx].id = g_tracks[i].id;
          }
        }
      }
      else
        ++j;
    }
  }

  g_pub_fusion_marker.publish(vis_markers);
  g_pub_fusion.publish(persons_fused);
  g_pub_fusion_pred.publish(persons_fused_pred);

  g_t_prev = t;
  ++g_frame_nr;

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  //cout << "Skeleton Model: " << persons_fused.persons.size() << " detections, duration: " << duration / 1000. << "ms." << endl;
  g_timings[0] += duration / 1000.;
  ++g_timing_cnt[0];
  if(persons_fused.persons.size() > 0 && persons_fused.persons.size() < max_num_timings){
    g_timings[persons_fused.persons.size()] += duration / 1000.;
    ++g_timing_cnt[persons_fused.persons.size()];
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pose_prior");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  nh_private.param<string>("pose_method", g_param_pose_method, "simple");
  nh_private.param<bool>  ("norm_height", g_param_normalize_by_height, false);
  nh_private.param<bool>("vis_cov", g_param_vis_covariance, false);

  if(!g_param_normalize_by_height)
    g_limbLSigmaFactor = 1.0;
  else
    g_limbLSigmaFactor = 2.0;

  ROS_INFO("Using pose method %s (norm_height = %d) and skeleton model with limb-length sigma factor of %f.", g_param_pose_method.c_str(), g_param_normalize_by_height, g_limbLSigmaFactor);

  define_colors();

  g_pub_fusion = nh.advertise<PersonCovList>("human_pose_estimation/persons3d_fused", 1);
  g_pub_fusion_pred = nh.advertise<PersonCovList>("human_pose_estimation/persons3d_fused_pred", 1);
  g_pub_fusion_marker = nh.advertise<visualization_msgs::MarkerArray>("human_pose_estimation/skeleton3d_fused", 1);

  ros::Subscriber sub_person3d = nh.subscribe<PersonCovList>(PERSON_TOPIC, 1, skeletonCallback, ros::TransportHints().tcpNoDelay());
  //ros::Subscriber sub_reset = nh.subscribe<edgetpu_ros_pose_estimation::StringStamped>("/human_pose_estimation/reset_filter", 1, reset); //TODO

  ros::spin();

  for (int i = 0; i < max_num_timings; ++i) {
    if(g_timing_cnt[i] > 0){
      cout << "Skeleton Model: ";
      if(i > 0)
        cout << i << " detections: ";
      cout << "avg runtime: " << g_timings[i] / g_timing_cnt[i] << "ms" << endl;
    }
  }
}
