#include "mls.h"
#include <math.h>

#define PI  3.141592654
#define DEG_PI  (180 / PI)

double angle(cv::Point2d a, cv::Point2d b, cv::Point2d anchor, bool full=false)
{
    double dx_ab = a.x - b.x, dy_ab = a.y - b.y;
    double dx_ac = a.x - anchor.x, dy_ac = a.y - b.y;
    double dx_bc = b.x - anchor.x, dy_bc = b.y - anchor.y;
    double dist_a_2 = dx_bc * dx_bc + dy_bc * dy_bc;
    double dist_b_2 = dx_ac * dx_ac + dy_ac * dy_ac;
    double dist_c_2 = dx_ab * dx_ab + dy_ab * dy_ab;
    double dist_ab = std::sqrt(dist_a_2 * dist_b_2);
    double cos_c = (dist_a_2 + dist_b_2 - dist_c_2) / (2 * dist_ab);
    if (full && dy_ac < 0)
        return 360 - std::acos(cos_c) * DEG_PI;
    else
        return std::acos(cos_c) * DEG_PI;
}

MLS::MLS()
{
}

MLS::~MLS()
{
}

void MLS::detect_keypoints(cv::Mat mask_t21, cv::Point anchor, std::vector<cv::Point2d>* p, std::vector<cv::Point2d> *q, double ratio)
{
    if (mask_t21.empty())
        return;
    double dratio = 1 - ratio;
    const int w_border = mask_t21.cols - 1, h_border = mask_t21.rows - 1;
    cv::Mat mask_hairend(mask_t21.size(), mask_t21.type(), cv::Scalar::all(0));
    mask_hairend.setTo(255, mask_t21 == 128);
    //mask_hairend.setTo(255, mask_t21 == 4);
    //mask_hairend.setTo(255, mask_t21 == 2);
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // 开运算，不直接用cv::morphologyEx，因为不能消除锚点
    //cv::erode(mask_hairend, mask_hairend, kernel, cv::Point(-1, -1), 1);
    //cv::dilate(mask_hairend, mask_hairend, kernel, cv::Point(-1, -1), 3);
    //cv::morphologyEx(mask_hairend, mask_hairend, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);
    double dist_max = std::numeric_limits<double>::min();
    double dist_min = std::numeric_limits<double>::max();
    cv::Point pt_min, pt_max;
    cv::Mat mask_dist(mask_hairend.size(), CV_64FC1, cv::Scalar(-1));
    for (int r = 0; r < anchor.y; ++r) {
        double *ptr_dist = mask_dist.ptr<double>(r);
        const uchar* ptr_mask = mask_hairend.ptr<uchar>(r);
        for (int c = 0; c < mask_hairend.cols; ++c) {
            if (ptr_mask[c] == 255) {
                double dx = anchor.x - c;
                double dy = anchor.y - r;
                double dist = dx * dx + dy * dy;
                ptr_dist[c] = dist;
                if (dist > dist_max) {
                    dist_max = dist;
                    pt_max.x = c;
                    pt_max.y = r;
                }
                if (dist < dist_min) {
                    dist_min = dist;
                    pt_min.x = c;
                    pt_min.y = r;
                }
            }
        }
    }
    dist_min = std::sqrt(dist_min);
    dist_max = std::sqrt(dist_max);
    const double step = 15;
    std::vector<std::pair<double, cv::Point>> angle_pts_far, angle_pts_near;
    std::vector<cv::Point> pts_far, pts_near;
    bool contain_far = false, contain_near = false;
    for (double angle = 210; angle <= 330; angle += step) {
        cv::Point pt_far(-1,-1), pt_near(-1,-1), ptnone(-1,-1);
        double min_dist = std::numeric_limits<double>::max();
        double max_dist = std::numeric_limits<double>::min();
        for (double radius = dist_min; radius <= dist_max; ++radius) {
            double radian = angle / DEG_PI;
            cv::Point2d Far(anchor.x + std::cos(radian) * radius, anchor.y + std::sin(radian) * radius);
            int x = Far.x;
            int y = Far.y;
            if (x < 0)
                x = 0;
            if (x > w_border)
                x = w_border;
            if (y < 0)
                y = 0;
            if (y > h_border)
                y = h_border;
            double dist = mask_dist.at<double>(y, x);
            if (dist < 0)  
                continue;
            if (dist < min_dist) {
                min_dist = dist;
                pt_near.x = x;
                pt_near.y = y;
                if (pt_near == pt_min)
                    contain_near = true;
            }
            if (dist > max_dist) {
                max_dist = dist;
                pt_far.x = x;
                pt_far.y = y;
                if (pt_far == pt_max)
                    contain_far = true;
            }
        }
        if (pt_far != ptnone && pt_near != ptnone) {
            //pts_far.emplace_back(pt_far);
            //pts_near.emplace_back(pt_near);
            angle_pts_far.emplace_back(angle, pt_far);
            angle_pts_near.emplace_back(angle, pt_near);
        }
        
    }
    
    if (!contain_far) {
        double ang = angle(pt_max, cv::Point2d(anchor.x + 5, anchor.y), anchor, true);
        double radian_x = (pt_max.x - anchor.x) / dist_max, radian_y = (pt_max.y - anchor.y) / dist_max;
        for (double radius = dist_min; radius < dist_max; ++radius) {
            int x = anchor.x + radian_x * radius;
            int y = anchor.y + radian_y * radius;
            if (x < 0)
                x = 0;
            if (x > w_border)
                x = w_border;
            if (y < 0)
                y = 0;
            if (y > h_border)
                y = h_border;
            double dist = mask_dist.at<double>(y, x);
            if (dist < 0) 
                continue;
            else {
                angle_pts_near.emplace_back(ang, cv::Point(x,y));
                angle_pts_far.emplace_back(ang, pt_max);
                break;
            }
        }
    }

    size_t ss = angle_pts_far.size();
    for (size_t i = 0; i < ss; ++i) {
        std::pair<double, cv::Point> angle_pt_far = angle_pts_far[i];
        std::pair<double, cv::Point> angle_pt_near = angle_pts_near[i];
        double angle = angle_pt_far.first;
        cv::Point pt_far = angle_pt_far.second, pt_near = angle_pt_near.second;
        //if (angle <= 241 || angle >= 299) {
        //    p->emplace_back(pt_far);
        //    p->emplace_back(pt_near);
        //    q->emplace_back(0.75 * pt_far.x + 0.25 * pt_near.x, 0.75 * pt_far.y + 0.25 * pt_near.y);
        //    q->emplace_back(pt_near);
        //} else {
        //    p->emplace_back(pt_far);
        //    p->emplace_back(pt_near);
        //    q->emplace_back(ratio * pt_far.x + dratio * pt_near.x, ratio * pt_far.y + dratio * pt_near.y);
        //    q->emplace_back(pt_near);
        //}
        double r = std::fabs(angle - 270) / 90 + 0.1;
        double dr = 1 - r;
        p->emplace_back(pt_far);
        p->emplace_back(pt_near);
        q->emplace_back(r * pt_far.x + dr * pt_near.x, r * pt_far.y + dr * pt_near.y);
        q->emplace_back(pt_near);
    }
}


void MLS::similarity_deform(cv::Mat src, cv::OutputArray dst, std::vector<cv::Point2d> p, std::vector<cv::Point2d> q, const double alpha, const int grid_size)
{
    if (src.empty() || p.empty() || (p.size() != q.size()))
        return;
    p.swap(q);
    cv::Mat _dst(src.size(), src.type(), cv::Scalar::all(0));
    cv::Mat Dx(src.size(), CV_64FC1, cv::Scalar(0));
    cv::Mat Dy(src.size(), CV_64FC1, cv::Scalar(0));
    const int height = src.rows;
    const int width = src.cols;
    const size_t n_ctrls = p.size();

    // w[i] = 1 / |p_i - v| ^ 2a
    double *w = new double[n_ctrls];

    cv::Point2d phat, qhat;
    // pstar = sum(w_i * p_i) / sum(w_i)  qstar = sum(w_i * q_i) / sum(w_i) 
    cv::Point2d pstar, qstar;
    // v_pstar = v - pstar;
    cv::Point2d v_pstar;
    // phat_perp 为垂直于phat的点, v_pstar_perp 为垂直于 v_pstar 的点
    cv::Point2d phat_perp, v_pstar_perp;
    // 变形点 fv = sum(qhat_i * 1/miu_s * A_i) + qstar
    cv::Point2d fv;
    // miu_s = 1 / (sum(phat_i.T * w_i * phat_i))
    double miu_s;
    int i;
    for (int r = 0; r < height; r += grid_size) {
        double *ptr_dx_r = Dx.ptr<double>(r);
        double *ptr_dy_r = Dy.ptr<double>(r);
        for (int c = 0; c < width; c += grid_size) {
            // sum_w = sum(w_i)
            double sum_w = 0;
            // sum_w_p = sum(w_i*p_i) 
            cv::Point2d sum_w_p(0, 0);
            // sum_w_q = sum(w_i*q_i)
            cv::Point2d sum_w_q(0, 0);
            // grid coordinate
            cv::Point2d v(c, r);
            for (i = 0; i < n_ctrls; ++i) {
                // 跳过控制点，直接使用q_i
                if (r == static_cast<int>(p[i].y) && c == static_cast<int>(p[i].x)) break;
                double dx = p[i].x - v.x;
                double dy = p[i].y - v.y;
                //w[i] = 1 / (std::pow(dx * dx + dy * dy, alpha));
                w[i] = 1 / (dx * dx + dy * dy);
                sum_w += w[i];
                sum_w_p += w[i] * p[i];
                sum_w_q += w[i] * q[i];
            }
            if (i == n_ctrls) {
                pstar = sum_w_p / sum_w;
                qstar = sum_w_q / sum_w;

                miu_s = 0;
                for (i = 0; i < n_ctrls; ++i) {
                    // 跳过控制点, w_i值无效
                    if (r == static_cast<int>(p[i].y) && c == static_cast<int>(p[i].x)) continue;
                    phat = p[i] - pstar;
                    miu_s += w[i] * phat.dot(phat);
                }
                
                fv.x = 0, fv.y = 0;
                v_pstar = v - pstar;
                v_pstar_perp.x = -v_pstar.y, v_pstar_perp.y = v_pstar.x;
                cv::Point2d temp;
                for (i = 0; i < n_ctrls; ++i) {
                    // 跳过控制点, w_i值无效
                    if (r == static_cast<int>(p[i].y) && c == static_cast<int>(p[i].x)) continue;
                    phat = p[i] - pstar;
                    qhat = q[i] - qstar;
                    phat_perp = cv::Point2d(-phat.y, phat.x);
                    temp.x = qhat.x * phat.dot(v_pstar) - qhat.y * phat_perp.dot(v_pstar);
                    temp.y = -qhat.x * phat.dot(v_pstar_perp) + qhat.y * phat_perp.dot(v_pstar_perp);
                    fv += w[i] * temp;
                }
                fv = fv / miu_s + qstar;
            } else {
                fv = q[i];
            }
            //Dx.at<double>(r,c) = fv.x - c;
            //Dy.at<double>(r,c) = fv.y - r;
            *(ptr_dx_r + c) = fv.x - c;
            *(ptr_dy_r + c) = fv.y - r;
        }
    }
    delete[] w;
    interpolate(src, dst, Dx, Dy, grid_size);
}

void MLS::rigid_deform(cv::Mat src, cv::OutputArray dst, std::vector<cv::Point2d> &p, std::vector<cv::Point2d> &q, const double alpha, const int grid_size)
{
}

void MLS::meshgrid(cv::Mat xgv, cv::Mat ygv, cv::OutputArray X, cv::OutputArray Y)
{
    cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
    cv::repeat(ygv.reshape(1, 1).t(), xgv.total(), 1, Y);
}

inline double MLS::bilinear_interpolate(double u, double v, double v11, double v12, double v21, double v22)
{
    return (v11 * (1 - v) + v12 * v) * (1 - u) + (v21 * (1 - v) + v22 * v) * u;
}

void MLS::interpolate(cv::Mat src, cv::OutputArray dst, cv::Mat_<double> DX, cv::Mat_<double> DY, const int grid_size)
{
    if (src.empty())
        return;
    cv::Mat _dst(src.size(), CV_8UC3, cv::Scalar::all(0));
    double dx, dy;
    int height = src.rows, width = src.cols;
    int border_h = height - 1, border_w = width - 1;
    double w, h, nx, ny;
    int channels = src.channels();
    for (int r = 0; r < height; r += grid_size) {
        for (int c = 0; c < width; c += grid_size) {
            int gr = grid_size + r, gc = grid_size + c;
            w = grid_size, h = grid_size;
            if (gr >= height) {
                gr = border_h;
                h = gr - r + 1;
            }
            if (gc >= width) {
                gc = border_w;
                w = gc - c + 1;
            }
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    // 双线性插值计算栅格内的坐标
                    //std::cout << "height: " << height << " width: " << width << " i: " << i << " j: " << j << " r: " << r << " c: " << c << " gr: " << gr << " gc: " << gc << std::endl;
                    dx = bilinear_interpolate(i / h, j / w, DX(r, c), DX(r, gc), DX(gr, c), DX(gr, gc));
                    dy = bilinear_interpolate(i / h, j / w, DY(r, c), DY(r, gc), DY(gr, c), DY(gr, gc));

                    nx = c + j + dx;
                    ny = r + i + dy;
                    if (nx > border_w)
                        nx = border_w;
                    if (nx < 0)
                        nx = 0;
                    if (ny > border_h)
                        ny = border_h;
                    if (ny < 0)
                        ny = 0;
                    int x1 = static_cast<int>(nx);
                    int y1 = static_cast<int>(ny);
                    int x2 = x1 + 1;
                    int y2 = y1 + 1;
                    if (x2 < 0 || y2 < 0) {
                        std::cout << "height: " << height << " width: " << width << std::endl;
                        std::cout << "w: " << w << " h: " << h << std::endl;
                        std::cout << " i: " << i << " j: " << j << " r: " << r << " c: " << c << " gr: " << gr << " gc: " << gc << std::endl;
                        std::cout << nx << " " << ny << std::endl;
                        std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
                    }
                        
                    double u = ny - y1;
                    double v = nx - x1;
                    
                    //std::cout << "height: " << height << " width: " << width << " x1: " << x1 << " y1: " << y1 << " x2: " << x2 << " y2: " << y2 << std::endl;
                    // 双线性插值计算坐标(c+j, r+i)处像素值
                    if (channels == 3) {
                        cv::Vec3b V11 = src.at<cv::Vec3b>(y1, x1);
                        cv::Vec3b V12 = src.at<cv::Vec3b>(y1, x2);
                        cv::Vec3b V21 = src.at<cv::Vec3b>(y2, x1);
                        cv::Vec3b V22 = src.at<cv::Vec3b>(y2, x2);
                        
                        uchar vb = bilinear_interpolate(u, v, V11[0], V12[0], V21[0], V22[0]);
                        uchar vg = bilinear_interpolate(u, v, V11[1], V12[1], V21[1], V22[1]);
                        uchar vr = bilinear_interpolate(u, v, V11[2], V12[2], V21[2], V22[2]);

                        _dst.at<cv::Vec3b>(r + i, c + j) = cv::Vec3b(vb, vg, vr);
                    }
                    else if (channels == 1) {
                        uchar V11 = src.at<uchar>(y1, x1);
                        uchar V12 = src.at<uchar>(y1, x2);
                        uchar V21 = src.at<uchar>(y2, x1);
                        uchar V22 = src.at<uchar>(y2, x2);

                        uchar value = bilinear_interpolate(u, v, V11, V12, V21, V22);
                        _dst.at<uchar>(r + i, c + j) = value;
                    }
                }
            }
        }
    }
    _dst.copyTo(dst);
}
