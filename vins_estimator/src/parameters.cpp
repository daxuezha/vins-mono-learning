#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file; // config文件路径
    config_file = readParam<std::string>(n, "config_file"); // 从ros参数服务器中读取config_file参数
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ); // 用opencv读取config文件yaml
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    // 将config文件中的imu_topic参数，也就是IMU数据（测量值）读取到IMU_TOPIC中，可以直接在在config文件中修改
    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"]; // 单词最大求解时间
    NUM_ITERATIONS = fsSettings["max_num_iterations"]; // 单次优化最大迭代次数
    MIN_PARALLAX = fsSettings["keyframe_parallax"]; // 关键帧视差
    // VINS判断是否为关键帧的条件是同一个特征点在两个图像中的视差是否大于关键帧视察，大于就是关键帧
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH; // 关键帧视察在虚拟相机下的表示

    std::string OUTPUT_PATH; // 输出路径
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists，如果输出路径不存在就创建
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());
    // 利用ofstream在输出路径下创建一个空文件
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    // imu和图像的各种参数
    ACC_N = fsSettings["acc_n"]; // 静态加速度计噪声标准差
    ACC_W = fsSettings["acc_w"]; // 静态加速度计随机游走噪声标准差
    GYR_N = fsSettings["gyr_n"]; // 静态陀螺仪噪声标准差
    GYR_W = fsSettings["gyr_w"]; // 静态陀螺仪随机游走噪声标准差
    G.z() = fsSettings["g_norm"]; // 重力加速度
    ROW = fsSettings["image_height"]; // 图像高度
    COL = fsSettings["image_width"]; // 图像宽度
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    // 外参（旋转外参）
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"]; // ！ 外参标志位，是否在线标定外参
    if (ESTIMATE_EXTRINSIC == 2) // 2表示没有外参先验，需要初始化
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity()); // 单位矩阵最为初始值
        TIC.push_back(Eigen::Vector3d::Zero()); // 零向量最为初始值
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1) // 1表示有外参先验，需要优化
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0) // 0表示固定外参，不需要优化
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R; // 旋转矩阵，从相机坐标系到IMU坐标系
        fsSettings["extrinsicTranslation"] >> cv_T; // 平移向量，从相机坐标系到IMU坐标系
        // VINS里面的参数都是eigen类型的，所以这里需要转换一下
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0; // 特征点深度初始值
    BIAS_ACC_THRESHOLD = 0.1; // 加速度计偏差阈值
    BIAS_GYR_THRESHOLD = 0.1; // 陀螺仪偏差阈值

    // 传感器的时间偏差
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
