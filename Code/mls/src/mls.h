#ifndef MLS_H
#define MLS_H

#include <opencv2/opencv.hpp>

class MLS
{
public:
    MLS();

    ~MLS();

    void detect_keypoints(cv::Mat mask, cv::Point anchor, std::vector<cv::Point2d> *p = nullptr, std::vector<cv::Point2d> *q = nullptr, double ratio = 0.5);

    void similarity_deform(cv::Mat src, cv::OutputArray dst, std::vector<cv::Point2d> p, std::vector<cv::Point2d> q, const double alpha = 1.0, const int grid_size = 1);

    void rigid_deform(cv::Mat src, cv::OutputArray dst, std::vector<cv::Point2d> &p, std::vector<cv::Point2d> &q, const double alpha = 1.0, const int grid_size = 1);

    void meshgrid(cv::Mat xgv, cv::Mat ygv, cv::OutputArray X, cv::OutputArray Y);

    double bilinear_interpolate(double u, double v, double v11, double v12, double v21, double v22);

    void interpolate(cv::Mat src, cv::OutputArray dst, cv::Mat_<double> X, cv::Mat_<double> Y, const int grid_size = 1);

};



#endif