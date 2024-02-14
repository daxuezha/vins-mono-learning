#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

// 一个进行sfm的特征
struct SFMFeature
{
    bool state;   // 是否已三角化
    int id;
    vector<pair<int,Vector2d>> observation;  // 存储该特征在不同帧下的坐标
    double position[3];
    double depth;
};

// ceres 的重投影误差，ceres的规定，必须重载一个类和（）运算符，然后在这个类里面写重投影误差的计算
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);	// 旋转这个点
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];	// 这其实就是Rcw * pw + tcw
		// 得到该相机坐标系下的3d坐标
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];	// 归一化处理
		// 跟现有观测形成残差
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}
	// 重载一个静态函数，用于生成costfunction，这样才可以在ceres中使用
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction< // 基类指针，其括号内是末班的实例化，也叫模版的参数列表。
	          ReprojectionError3D, 2, 4, 3, 3>( // 派生类，2是残差的维度，4是相机的位姿维度，3是3d点的维度
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};