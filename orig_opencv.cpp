/*
Compilation:
g++ -o orig_opencv orig_opencv.cpp -std=c++17 `pkg-config --cflags --libs opencv4`

Execution:
./orig_opencv <imagename> <size>
Example: ./orig_opencv snow.png 778x1036
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 解析尺寸字符串 (如 "778x1036")
bool parseSize(const std::string& sizeStr, int& width, int& height) {
    size_t xPos = sizeStr.find('x');
    if (xPos == std::string::npos) {
        return false;
    }
    
    try {
        width = std::stoi(sizeStr.substr(0, xPos));
        height = std::stoi(sizeStr.substr(xPos + 1));
        return (width > 0 && height > 0);
    } catch (const std::exception&) {
        return false;
    }
}

int main(int argc, char *argv[])
{
    // 處理命令行參數
    if (argc < 2) {
        std::cout << "使用方法: ./orig_opencv <imagename> [size]" << std::endl;
        std::cout << "範例:" << std::endl;
        std::cout << "  ./orig_opencv snow.png             - 使用原始尺寸" << std::endl;
        std::cout << "  ./orig_opencv snow.png 778x1036    - 調整為 778x1036" << std::endl;
        return EXIT_FAILURE;
    }
    
    // 設置輸入圖像路徑
    std::string inputImagePath = argv[1];
    std::cout << "處理圖像: " << inputImagePath << std::endl;
    
    // 讀取輸入圖像
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_UNCHANGED);
    
    // 檢查圖像是否成功讀取
    if (inputImage.empty()) {
        std::cout << "無法讀取圖像: " << inputImagePath << std::endl;
        return EXIT_FAILURE;
    }

    // 解析大小參數並調整圖像大小（如果有指定）
    cv::Mat resizedImage = inputImage;
    std::string sizeStr = "original";
    
    if (argc >= 3) {
        int width, height;
        if (parseSize(argv[2], width, height)) {
            sizeStr = argv[2];
            cv::resize(inputImage, resizedImage, cv::Size(width, height));
            std::cout << "調整輸入圖像為 " << width << "x" << height << std::endl;
        } else {
            std::cout << "無效的尺寸格式。使用原始尺寸: " 
                      << inputImage.cols << "x" << inputImage.rows << std::endl;
        }
    } else {
        std::cout << "使用原始尺寸: " << inputImage.cols << "x" << inputImage.rows << std::endl;
    }
    
    // 萃取檔案名稱（不含路徑和副檔名）
    std::string filename = inputImagePath;
    size_t last_slash_pos = filename.find_last_of("/\\");
    if (last_slash_pos != std::string::npos) {
        filename = filename.substr(last_slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    std::string basename = filename.substr(0, dot_pos);
    std::string extension = filename.substr(dot_pos);
    
    // 確保 results 目錄存在
    std::filesystem::create_directory("results");
    
    // 設置CSV結果文件
    std::string resultpath = "results/" + basename + "_" + sizeStr + "_opencv.csv";
    std::ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << std::endl;
    
    // 定義迭代次數
    int numIter = 10; // 減少迭代次數以加快處理
    
    // 輸出檔案名稱
    std::string outputFilename = "results/" + basename + "_" + sizeStr + "_opencv" + extension;
    
    // 創建時間評估變量
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto rgbToHsvTime = std::chrono::duration<double, std::milli>::zero();
    auto gaussianBlurTime = std::chrono::duration<double, std::milli>::zero();
    auto subtractTime = std::chrono::duration<double, std::milli>::zero();
    auto addTime = std::chrono::duration<double, std::milli>::zero();
    auto equalizeHistTime = std::chrono::duration<double, std::milli>::zero();
    auto hsvToRgbTime = std::chrono::duration<double, std::milli>::zero();
    
    // STEP 1 - RGB TO HSV CONVERSION
    cv::Mat inputImageHsv;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::cvtColor(resizedImage, inputImageHsv, cv::COLOR_BGR2HSV);
        end = std::chrono::high_resolution_clock::now();
        rgbToHsvTime += (end - start);
    }
    rgbToHsvTime /= numIter;
    
    std::cout << "RGB 轉 HSV 時間: " << rgbToHsvTime.count() << " ms" << std::endl;
    
    // 分離 HSV 通道
    std::vector<cv::Mat> channels;
    cv::split(inputImageHsv, channels);
    cv::Mat inputImageH = channels[0];
    cv::Mat inputImageS = channels[1];
    cv::Mat inputImageV = channels[2];
    
    // STEP 2 - LOCAL ENHANCEMENT
    // 1. 圖像模糊
    cv::Mat blurredImage;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(inputImageV, blurredImage, cv::Size(5, 5), 0, 0);
        end = std::chrono::high_resolution_clock::now();
        gaussianBlurTime += (end - start);
    }
    gaussianBlurTime /= numIter;
    
    std::cout << "圖像模糊時間: " << gaussianBlurTime.count() << " ms" << std::endl;
    
    // 2. 圖像相減
    cv::Mat imageMask;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::subtract(inputImageV, blurredImage, imageMask);
        end = std::chrono::high_resolution_clock::now();
        subtractTime += (end - start);
    }
    subtractTime /= numIter;
    
    std::cout << "圖像相減時間: " << subtractTime.count() << " ms" << std::endl;
    
    // 3. 圖像銳化 (添加掩碼)
    int weight = 10;
    cv::Mat sharpenedImage;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::addWeighted(inputImageV, 1.0, imageMask, weight, 0, sharpenedImage);
        end = std::chrono::high_resolution_clock::now();
        addTime += (end - start);
    }
    addTime /= numIter;
    
    std::cout << "圖像銳化時間: " << addTime.count() << " ms" << std::endl;
    
    // STEP 3 - GLOBAL ENHANCEMENT (直方圖均衡化)
    cv::Mat globallyEnhancedImage;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::equalizeHist(sharpenedImage, globallyEnhancedImage);
        end = std::chrono::high_resolution_clock::now();
        equalizeHistTime += (end - start);
    }
    equalizeHistTime /= numIter;
    
    std::cout << "直方圖均衡化時間: " << equalizeHistTime.count() << " ms" << std::endl;
    
    // 合併 HSV 通道
    cv::Mat outputHSV;
    std::vector<cv::Mat> outChannels = {inputImageH, inputImageS, globallyEnhancedImage};
    cv::merge(outChannels, outputHSV);
    
    // HSV 到 RGB 轉換
    cv::Mat outputImage;
    
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cv::cvtColor(outputHSV, outputImage, cv::COLOR_HSV2BGR);
        end = std::chrono::high_resolution_clock::now();
        hsvToRgbTime += (end - start);
    }
    hsvToRgbTime /= numIter;
    
    std::cout << "HSV 轉 RGB 時間: " << hsvToRgbTime.count() << " ms" << std::endl;
    
    // 計算總時間和局部增強時間
    auto localEnhanceTime = gaussianBlurTime + subtractTime + addTime;
    auto totalTime = rgbToHsvTime + localEnhanceTime + equalizeHistTime + hsvToRgbTime;
    
    std::cout << "總處理時間: " << totalTime.count() << " ms" << std::endl;
    
    // 寫入結果圖像
    cv::imwrite(outputFilename, outputImage);
    std::cout << "增強後的圖像已儲存為: " << outputFilename << std::endl;
    
    // 寫入結果到CSV文件
    outFile << basename << ","
            << resizedImage.cols << "x" << resizedImage.rows << ","
            << "1" << ","
            << rgbToHsvTime.count() << ","
            << gaussianBlurTime.count() << ","
            << subtractTime.count() << ","
            << addTime.count() << ","
            << equalizeHistTime.count() << ","
            << hsvToRgbTime.count() << ","
            << totalTime.count() << std::endl;
    
    outFile.close();
    
    return 0;
}