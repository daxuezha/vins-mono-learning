#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

// 图片的回调函数
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag) // 对第一帧图像的基本操作
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec(); // 记录第一帧图像的时间戳，单位是秒
        last_image_time = img_msg->header.stamp.toSec(); // 记录上一帧图像的时间戳，由于是第一帧，所以就是第一帧的时间戳
        return;
    }
    // detect unstable camera stream
    // 检查时间戳是否正常，这里认为当前时间戳和上一帧时间戳差值超过一秒或者时间戳顺序错乱就异常，直接重启
    // 图像时间差太多光流追踪就会失败，这里没有描述子匹配，因此对时间戳要求就高
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        // 一些常规的reset操作，包括重置计数器，重置时间戳，重置标志位，重置发布频率
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);  // 告诉其他模块要重启了
        return;
    }
    last_image_time = img_msg->header.stamp.toSec(); // 记录当前时间戳
    // frequency control
    // 控制一下发给后端的频率，相机帧数太快因此后端跟不上，这里控制一下，定义10帧，在yaml文件中定义。
    // pub_count是计数器，记录发给后端的帧数，PUB_THIS_FRAME是一个标志位，代表这一帧是否要发给后端
    // 1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) 当前的频率
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)    // 保证发给后端的不超过这个频率
    {
        PUB_THIS_FRAME = true; // 频率在10hz以内，就可以发布这一帧
        // reset the frequency control
        // 这段时间的频率和预设频率十分接近，就认为这段时间很棒，重启一下，避免delta t太大
        // 因为pub_count始终在增加，img_msg->header.stamp.toSec() - first_image_time也在增加，所以这个频率是在变化的
        // 随着时间的推移，由于分母增加，相对应的时间戳增加量如果是恒定的，相当于相同时间戳间隔所接受的pub_count增加量会逐渐增大
        // 以至于会出现1s内接受了10帧，然后1s内接受了20帧，然后1s内接受了30帧。。。这样的向后端发送帧数是后端无法处理的
        // 因此当当前频率和预设频率相差太小的时候，就重启一下，避免delta t太大
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // 即使不发布也是正常做光流追踪的！光流对图像的变化要求尽可能小
    // 光流的特性，连续两帧图像的变化尽可能小，因此不管是否发布，都要做光流追踪

    cv_bridge::CvImageConstPtr ptr; // 专用于ros和opencv之间的转换
    // 把ros message转成cv::Mat
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) // 如果不是第一张图像，或者不是双目的情况下，读取图像信息
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }
    
    // 这部分对光流追踪的结果进行更新，就是id的更新
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);    // 单目的情况下可以直接用=号
        if (!completed)
            break;
    }
    // 给后端喂数据
   if (PUB_THIS_FRAME)
   {
        pub_count++;    // 计数器更新
        // 发布的信息是一个sensor_msgs::PointCloud类型的消息，这个消息包含了很多信息，包括id，像素坐标，归一化坐标，速度等
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;   // 去畸变的归一化相机坐标系
            auto &cur_pts = trackerData[i].cur_pts; // 像素坐标
            auto &ids = trackerData[i].ids; // id
            auto &pts_velocity = trackerData[i].pts_velocity;   // 归一化坐标下的速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                // 只发布追踪大于1的，因为等于1表示没追踪到，没法构成重投影约束，也没法三角化
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);   // 这个并没有用到
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    // > 利用这个ros消息的格式进行信息存储
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        // 信息的存储，存储的是id，像素坐标，归一化坐标，速度
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // 一轮回调函数结束，发布这个消息
        // skip the first image; since no optical speed on frist image，如果是第一帧图像，就不发布
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);    // 前端得到的信息通过这个publisher发布出去

        // 可视化相关操作
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img); // 这里一定要注释掉，否则程序不往下运行了
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");   // ros节点初始化
    ros::NodeHandle n("~"); // 声明一个句柄，～代表这个节点的命名空间
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);    // 设置ros log级别
    readParameters(n); // 读取配置文件

    // 下面的for循环是为了读取每个相机的内参，为了fusion多相机做准备
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);    // 获得每个相机的内参，包括畸变参数

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // 这个向roscore注册订阅这个topic，收到一次message就执行一次回调函数，本节点的关键是回调函数：图像处理
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback); // 参数分别是topic名，队列长度，回调函数
    // 注册一些publisher，也就是发布一些信息给与其他模块
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000); // 实际发出去的是 /feature_tracker/feature
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();    // spin代表这个节点开始循环查询topic是否接收，也就是前面注册的subscriber执行回调函数
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?