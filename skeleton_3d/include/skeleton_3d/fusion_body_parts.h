#pragma once

struct FUSION_BODY_PARTS {
     static const int Nose = 0,
     Neck = 1,
     RShoulder = 2,
     RElbow = 3,
     RWrist = 4,
     LShoulder = 5,
     LElbow = 6,
     LWrist = 7,
     MidHip = 8,
     RHip = 9,
     RKnee = 10,
     RAnkle = 11,
     LHip = 12,
     LKnee = 13,
     LAnkle = 14,
     REye = 15,
     LEye = 16,
     REar = 17,
     LEar = 18,
     Head = 19,
     Belly = 20,
     NUM_KEYPOINTS = 21;
                                                       // 0, 1,    2,    3,    4,    5,    6,    7,    8,    9,    10,    11,   12,   13,   14,    15,   16,   17,   18,   19,   20
    static constexpr int kpParent[NUM_KEYPOINTS]      = {-1, 0,    1,    2,    3,    1,    5,    6,    20,   1,    9,     10,   1,    12,   13,    0,    0,    15,   16,   0,    1};
    static constexpr double limbLength[NUM_KEYPOINTS] = {-1, 0.20, 0.15, 0.28, 0.25, 0.15, 0.28, 0.25, 0.24, 0.48, 0.45, 0.445, 0.48, 0.45, 0.445, 0.05, 0.05, 0.10, 0.10, 0.12, 0.26}; // From H36M - Statistics (except 14 - 17) (https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
    static constexpr double limbLThresh[NUM_KEYPOINTS]= {-1, 0.20, 0.15, 0.25, 0.25, 0.15, 0.25, 0.25, 0.25, 0.40, 0.40, 0.40,  0.40, 0.40, 0.40,  0.10, 0.10, 0.15, 0.15, 0.15, 0.25};
    static constexpr double hipDist = 0.27, hipDThresh = 0.25;
    //For people, the σ's are .026, .025, .035, .079, .072, .062, .107, .087, & .089 for the nose, eyes, ears, shoulders, elbows, wrists, hips, knees, & ankles, respectively.
    static constexpr double oks_sigmas[NUM_KEYPOINTS] = {0.026, 0.079, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062, 0.107, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089, 0.025, 0.025, 0.035, 0.035, 0.079, 0.107};
    static constexpr double vel_sigmas[NUM_KEYPOINTS] = {2., 1., 1., 2., 3., 1., 2., 3., 1., 1., 2., 3., 1., 2., 3., 2., 2., 2., 2., 2., 1.};
};
