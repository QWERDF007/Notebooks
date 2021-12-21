#include <array>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <json/json.h>

#include "../../utils/comm/scp.h"


void test_demo1(const int grid_size = 10)
{
    std::string img_path = "E:/xingfu/data/test/1/7a6a3562acde992a0e50615849766336.jpg";
    MLS mls;
    cv::Mat src = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat dst;
    std::vector<cv::Point2d> p = { {220, 115}, {381, 63}, {428, 74}, {512, 113},
        {343, 198},{250, 211},{425, 207},{351, 825}, {279, 358},{423, 353},{352, 431},{356, 504},{155, 602},{630, 667},{349, 589},{174, 365},{520, 365},{480, 488},{187, 255},{504, 220},
        {0, 0},{713, 0},{0, 949},{713, 949},{357, 949}
    };
    std::vector<cv::Point2d> q = { {220, 145},{381, 87},{428, 102},{470, 142},
        {343, 198},{250, 211},{425, 207},{351, 825},{279, 358},{423, 353},{352, 431},{356, 504},{155, 602},{630, 667},{349, 589},{174, 365},{520, 365},{480, 488},{187, 255},{504, 220},
        {0, 0},{713, 0},{0, 949},{713, 949},{357, 949}
    };
    double start = cv::getTickCount();
    //for (int i = 0; i < 100; ++i)
        mls.similarity_deform(src, dst, p, q, 1.0, grid_size);
    double end = cv::getTickCount();
    std::cout << "elapsed time: " << (end - start) / cv::getTickFrequency() << std::endl;
    
    if (!dst.empty()) {
        cv::Mat mat2show(src.rows, src.cols * 3, src.type(), cv::Scalar::all(0));
        src.copyTo(mat2show(cv::Rect(0, 0, src.cols, src.rows)));
        dst.copyTo(mat2show(cv::Rect(src.cols, 0, src.cols, src.rows)));
        cv::Mat d = dst - src;
        d.copyTo(mat2show(cv::Rect(2*src.cols, 0, src.cols, src.rows)));
        cv::imshow("dst", mat2show);
    }
    else {
        cv::imshow("src", src);
    }
    cv::waitKey();
}

void test_demo2()
{
    std::string data_folder = "E:/xingfu/data/hairend_deform/t509/";
    std::string json_file = "E:/xingfu/data/hairend_deform/509.json";
    //std::string json_file = data_folder + "abc.json";
    std::string src_folder = data_folder + "0/";
    std::string mask_T21_folder = data_folder + "2/";
    std::string trimap_folder = data_folder + "1/";
    std::string output_folder = data_folder + "deformed2/";
    std::ifstream fs(json_file);
    if (!fs.is_open()) {
        spdlog::error("can't open json file");
        return;
    }
    utils::Scp scp("192.168.0.131", 22, "xftrain", "123456");
    std::string doc;
    std::getline(fs, doc, '\n');
    Json::CharReaderBuilder rbuilder;
    std::string errs;
    MLS mls;
    while (!fs.eof()) {
        Json::Value root;
        std::stringstream stream(doc);
        Json::parseFromStream(rbuilder, stream, &root, &errs);
        if (!root.isMember("id")) {
            std::getline(fs, doc, '\n');
            continue;
        }
        std::string id = root["id"].asString();
        spdlog::info("id: {:<25}", id);
        std::string name = id + ".jpg";
        std::string src_path = src_folder + name;
        std::string mask_T21_path = mask_T21_folder + name;
        std::string trimap_path = trimap_folder + name;


        cv::Mat src = cv::imread(src_path, cv::IMREAD_COLOR);
        cv::Mat mask_T21 = cv::imread(mask_T21_path, cv::IMREAD_GRAYSCALE);
        cv::Mat trimap = cv::imread(trimap_path, cv::IMREAD_GRAYSCALE);
        if (trimap.empty()) {
            spdlog::error("Can't open {}", trimap_path);
            std::getline(fs, doc, '\n');
            continue;
        }
        if (src.empty()) {
            spdlog::error("Can't open {}.jpg", src_path);
            std::getline(fs, doc, '\n');
            continue;
        }

        cv::Mat tmp_src(src.rows + 10, src.cols, src.type(), cv::Scalar::all(255));
        src.copyTo(tmp_src(cv::Rect(0, 10, src.cols, src.rows)));

        cv::Mat tmp_trimap(trimap.rows + 10, trimap.cols, trimap.type(), cv::Scalar::all(0));
        trimap.copyTo(tmp_trimap(cv::Rect(0, 10, trimap.cols, trimap.rows)));

        if (trimap.size() != src.size()) {
            spdlog::error("id: {} trimap size != src size", id);
            std::getline(fs, doc, '\n');
            continue;
        }

        const int w_border = src.cols - 1, h_border = src.rows - 1;

        if (!root.isMember("facial")) {
            spdlog::error("id: {} not facial", id);
            std::getline(fs, doc, '\n');
            continue;
        }
        Json::Value facial = root["facial"];
        if (!facial.isMember("leye")) {
            spdlog::error("id: {} no leye", id);
            std::getline(fs, doc, '\n');
            continue;
        }
        Json::Value j_leye = facial["leye"];
        if (!facial.isMember("reye")) {
            spdlog::error("id: {} no reye", id);
            std::getline(fs, doc, '\n');
            continue;
        }
        Json::Value j_reye = facial["reye"];

        cv::Point leye(j_leye[0].asInt(), j_leye[1].asInt());
        cv::Point reye(j_reye[0].asInt(), j_reye[1].asInt());
        
        cv::Point2d eye_center(leye + reye);
        eye_center /= 2;
        
        cv::Point anchor = eye_center;
        double start = cv::getTickCount();
        std::vector<cv::Point2d> p, q;
        cv::Rect deform_region(0, 0, tmp_src.cols, anchor.y);
        mls.detect_keypoints(tmp_trimap(deform_region), anchor, &p, &q, 0.1);

        p.emplace_back(0, 0);
        p.emplace_back(w_border, 0);
        p.emplace_back(w_border, anchor.y);
        p.emplace_back(0, anchor.y);
        p.emplace_back(leye.x, leye.y+10);
        p.emplace_back(reye.x, reye.y+10);
        
        q.emplace_back(0, 0);
        q.emplace_back(w_border, 0);
        q.emplace_back(w_border, anchor.y);
        q.emplace_back(0, anchor.y);
        q.emplace_back(leye.x, leye.y+10);
        q.emplace_back(reye.x, reye.y+10);

        cv::Mat dst;
        mls.similarity_deform(tmp_src(deform_region), dst, p, q, 1, 100);
        double end = cv::getTickCount();
        double elapsed = (end - start) / cv::getTickFrequency();
        spdlog::info("elapsed time: {}", elapsed);

        cv::Mat tmp = tmp_src.clone();
        dst.copyTo(tmp(deform_region));
        tmp(cv::Rect(0, 10, src.cols, src.rows)).copyTo(dst);
        cv::Mat d = dst - src;
        for (auto & pt : p) {
            cv::circle(src, cv::Point(pt.x, pt.y - 10), 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }
        for (auto & pt : q) {
            cv::circle(src, cv::Point(pt.x, pt.y - 10), 3, cv::Scalar(255, 0, 0), cv::FILLED);
        }
        //for (auto & pt : p) {
        //    cv::circle(dst, pt, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        //}
        //for (auto & pt : q) {
        //    cv::circle(dst, pt, 3, cv::Scalar(255, 0, 0), cv::FILLED);
        //}

        //cv::normalize(mask_T21, mask_T21, 0, 255, cv::NORM_MINMAX);
        //cv::imshow("maskT21", mask_T21);
        //cv::imshow("src", src);
        //cv::imshow("dst", dst);
        //cv::imshow("d", d);
        cv::Mat toshow(src.rows, src.cols * 3, CV_8UC3, cv::Scalar::all(0));
        src.copyTo(toshow(cv::Rect(0, 0, src.cols, src.rows)));
        dst.copyTo(toshow(cv::Rect(src.cols, 0, src.cols, src.rows)));
        d.copyTo(toshow(cv::Rect(2* src.cols, 0, src.cols, src.rows)));
        cv::imwrite(output_folder + name, toshow);
        //cv::imshow("res", toshow);
        //cv::imshow("trimap", trimap);
        //cv::waitKey();
        std::getline(fs, doc, '\n');
    }
}


int main(int argc, char **argv)
{       
    spdlog::set_pattern("[%Y/%m/%d %H:%M:%S %e] [%n] [%^%L%$] [%t] %v");
    //test_demo1(5);
    test_demo2();
    system("pause");
    return 0;
}