#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/**
 * @brief 对特征点三角化，手写
 * 
 * @param[in] Pose0 两帧位姿，已知
 * @param[in] Pose1 
 * @param[in] point0 特征点在两帧下的观测，已知
 * @param[in] point1 
 * @param[out] point_3d 三角化结果，未知
 */

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	// 通过奇异值分解求解一个Ax = 0得到
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	// 齐次向量归一化
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * @brief 根据上一帧的位姿通过pnp求解当前帧的位姿，使用cv库里的函数
 * 
 * @param[in] R_initial 上一帧的位姿
 * @param[in] P_initial 
 * @param[in] i 	当前帧的索引
 * @param[in] sfm_f 	所有特征点的信息
 * @return true 
 * @return false 
 */

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true) // 是false就是没有被三角化，pnp是3d到2d求解，因此需要3d点
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);  // 
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;
}

/**
 * @brief 根据两帧索引和位姿计算对应特征点的三角化位置，自己写了一个接口
 * 
 * @param[in] frame0 
 * @param[in] Pose0 
 * @param[in] frame1 
 * @param[in] Pose1 
 * @param[in] sfm_f 
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)	// feature_num是特征点总数
	{
		if (sfm_f[j].state == true)	// 已经三角化过了，就没必要再三角化了
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		// 遍历该特征点的观测，也就是检查特征点所在的帧，看看是不能两帧都能看到
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;	// 取出在该帧的观测
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)	// 如果都能被看到
		{
			Vector3d point_3d;
			// 将这个特征点进行三角化，这个三角化没有调用opencv接口，输入两帧的位姿和对应的观测归一化坐标，输出3d点
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;	// 标志位置true
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * @brief 根据已有的枢纽帧和最后一帧的位姿变换，得到各帧位姿和3d点坐标，最后通过ceres进行优化
 * 
 * @param[in] frame_num 滑窗内KF总数 
 * @param[out] q  恢复出来的滑窗中各个姿态
 * @param[out] T  恢复出来的滑窗中各个平移
 * @param[in] l 	枢纽帧的idx
 * @param[in] relative_R 	枢纽帧和最后一帧的旋转
 * @param[in] relative_T 	枢纽帧和最后一帧的平移
 * @param[in] sfm_f 	用来做sfm的特征点集合
 * @param[out] sfm_tracked_points 恢复出来的地图点
 * @return true 
 * @return false 
 */

bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	// 枢纽帧设置为单位帧，也可以理解为当前滑窗内重新设定的世界系原点
	q[l].w() = 1;  // q是指针，当数组用
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();

	// 求得最后一帧的位姿
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	// 由于纯视觉slam处理都是Tcw(世界向相机),包括orbslam，但是vins里这里的枢纽帧和最后一帧的位姿是是Twc相反的，因此下面把Twc转成Tcw
	// rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4]; // 以四元数形式存储
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	// 将枢纽帧和最后一帧Twc转成Tcw，包括四元数，旋转矩阵，平移向量和增广矩阵
	// ! 这里涉及一个坐标系反向推导的公式
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);

	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

	// 以上准备工作做好后开始具体实现

	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	// Step 1 求解枢纽帧到最后一帧之间帧和最后一帧的位相对姿及对应特征点的三角化处理
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			// ! 这是依次求解，因此通过PnP求解的上一帧的位姿是刚在循环中求出来就做为初值，
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			// 通过位姿赋值当前的位姿为下一帧的求解做准备
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// ! 当前帧和最后一帧进行三角化处理，和最后一帧的共视特征点在当前滑窗设定的世界坐标下的3d坐标
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	// Step 2 考虑有些特征点不能被最后一帧看到，因此，fix枢纽帧，遍历枢纽帧到最后一帧进行特征点三角化
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	// Step 3 处理完枢纽帧到最后一帧，开始处理枢纽帧之前的帧，和前面的步骤一致
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		// 这种情况就是后一帧先求解出来
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		// ! 当前帧和枢纽帧进行三角化处理，和枢纽帧的共视特征点在当前滑窗设定的世界坐标下的3d坐标，这个和前面输入的参数不一致
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	// Step 4 得到了所有关键帧的位姿，遍历没有被三角化的特征点，进行三角化
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true) // 三角化过了就不用再三角化了
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)	// 只有被两个以上的KF观测到才可以三角化
		{
			Vector2d point0, point1;
			// 取当前被遍历到的特征点的存在的首尾两个KF，尽量保证两KF之间足够位移
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/

	// > full BA，滑窗内的所有帧和3d点进行全局BA优化
	// Step 5 求出了所有的位姿和3d点之后，进行一个视觉slam的global BA，gobal BA指的是对滑窗内的所以kf进行BA，非global BA指的是针对相邻两帧
	// 可能需要介绍一下ceres  http://ceres-solver.org/
	// 数值求导的速度会快于自动求导，如果说在实时恢复两帧位姿的情况下去BA那么肯定数值求导最好，只不过自己计算出来比较繁琐；但是对于像初始化这种对实时性要求不高的情况，可以选择自动求导
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	// cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		// 这些都是待优化的参数块，ceres需要的是double数组，因此需要提前准备好
		// 平移
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		// 旋转
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		// 添加参数块，参数：参数块，维度，参数化（四元数是单独的参数化方式）
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3); // 平移是广义上的加法，不用假定

		// 由于是单目视觉slam，有七个自由度不可观，因此，fix一些参数块避免在零空间漂移
		// 可观性/能观性用最通俗和直观的方式来描述就是：状态一变，测量就变
		// 单目slam存在7自由度不可观：3旋转，3平移，尺度皆不可观
		// 单目+IMU存在4自由度不可观：Yaw与3平移，pitch与roll因重力而可观，尺度因加速度计而可观
		// 因此在调试对应slam系统中，如果出现问题，一定要考虑到可观性/能观性问题

		// ! fix设置的世界坐标系第l帧的位姿，同时fix最后一帧的位移用来fix尺度；这样保证尺度的一致性，维持自由度和可观性
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}
	// 只有视觉重投影构成约束，因此遍历所有的特征点，构建约束
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)	// 必须是三角化之后的
			continue;
		// 遍历所有的观测帧，对这些帧逐个建立约束
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first; // 观测到该特征点的帧
			// cost_function是重投影误差，这个是ceres的一个类，用来构建误差项
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// 约束了这一帧位姿和3d地图点
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
			// 三个参数：代价函数（误差计算方式），NULL（核函数，这里不用），待优化参数块。。。
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR; //舒尔补的线性求解器
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2; // 200ms最大的计算时间
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	// 优化结束，把double数组的值返回成对应类型的值
	// 同时Tcw -> Twc
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse(); // 因为要实现Rwc，所以取逆
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	// 调整平移，因为是Rwc，所以twc = -Rcw * tcw
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	// sfm_tracked_points用于存储经过优化后的三角化后的3d点
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

