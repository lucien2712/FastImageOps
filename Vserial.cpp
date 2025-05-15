/*
OpenCV (C++) Implemantation of "Image enhancement with the application of local and global enhancement methods for dark images" by
Singh et. al https://ieeexplore.ieee.org/document/8071892
This implementation only includes CPU version using OpenCV built-in functions
Author: Batuhan HANGÜN

Compilation:
g++ -g -Wall -o Vserial Vserial.cpp `pkg-config --cflags --libs opencv4`

Execution:
./Vserial <imagename> <size>
Example: ./Vserial snow.png 778x1036
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <thread>  // 添加 thread 頭文件用於 sleep_for

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

// 計算圖像直方圖 - 與 main.cpp 中的 histogramCalcPar 對應
void histogramCalc(const cv::Mat& inputImage, unsigned int imHistogram[256]) {
    std::fill(imHistogram, imHistogram + 256, 0);
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            imHistogram[inputImage.at<uchar>(i, j)]++;
        }
    }
}

// 使用5x5卷積核進行高斯模糊 - 與 main.cpp 中的 imageBlurPar 對應
/**
 * @brief   對單通道灰階影像做 5×5 加權平均 (Gaussian-like) 模糊
 *
 * @param inputImage   輸入的灰階影像 (CV_8UC1)
 * @param outputImage  模糊後的灰階影像 (CV_8UC1)，函式內 clone 與寫入
 */
void imageBlur(cv::Mat& inputImage, cv::Mat& outputImage) {
    // 1. 定義與 main.cpp 中相同的 5×5 濾波器 (近似 Gaussian 核)
    const int Filter[5][5] = {
        { 1,  4,  6,  4, 1},
        { 4, 16, 24, 16, 4},
        { 6, 24, 36, 24, 6},
        { 4, 16, 24, 16, 4},
        { 1,  4,  6,  4, 1}
    };
    // 濾波器所有元素之和，用於正規化
    const int sumOfElementsInFilter = 256;

    // 2. 複製原影像結構到輸出，稍後逐像素寫入新值
    outputImage = inputImage.clone();

    // 3. 對每一個像素 (i, j) 計算濾波後的強度
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            int accum = 0;  // 累加區域加權和

            // 3.1 掃描以 (i, j) 為中心的 5×5 區域
            for (int dy = -2; dy <= 2; ++dy) {
                int y = i + dy;
                // 邊界檢查：若超出影像範圍，跳過
                if (y < 0 || y >= inputImage.rows)
                    continue;

                for (int dx = -2; dx <= 2; ++dx) {
                    int x = j + dx;
                    if (x < 0 || x >= inputImage.cols)
                        continue;

                    // 累加：區域像素強度 × 對應的 Filter 權重
                    int weight = Filter[dy + 2][dx + 2];
                    accum += inputImage.at<uchar>(y, x) * weight;
                }
            }

            // 3.2 正規化加權和，並寫入輸出影像
            //     新強度 = ⌊ accum / sumOfElementsInFilter ⌋
            outputImage.at<uchar>(i, j) = static_cast<uchar>(accum / sumOfElementsInFilter);
        }
    }
}


// 圖像相減 - 與 main.cpp 中的 subtractImagePar 對應
/**
 * @brief   圖像相減 (與 main.cpp 中的 subtractImagePar 平行版對應)
 * @details 對兩張單通道灰階影像做逐像素差分，並將負值裁剪為 0，避免出現無效的負強度。
 *
 * @param inputImage1  第一張輸入灰階影像 (CV_8UC1)
 * @param inputImage2  第二張輸入灰階影像 (CV_8UC1)
 * @param outputImage  輸出差分影像 (CV_8UC1)，會在函式內被重新建立
 */
void subtractImage(cv::Mat& inputImage1,
                   cv::Mat& inputImage2,
                   cv::Mat& outputImage) {
    // 1. 建立輸出影像：與 inputImage1 同樣的尺寸與類型
    outputImage = cv::Mat(inputImage1.size(), inputImage1.type());

    // 2. 逐像素計算差分並裁剪負值
    for (int i = 0; i < inputImage1.rows; ++i) {
        for (int j = 0; j < inputImage1.cols; ++j) {
            // 讀取兩張影像在 (i, j) 的灰階值，轉成 int 以避免溢位
            int val1 = static_cast<int>( inputImage1.at<uchar>(i, j) );
            int val2 = static_cast<int>( inputImage2.at<uchar>(i, j) );
            // 計算差分
            int diff = val1 - val2;
            // 負值裁剪：若 diff < 0，則設為 0
            outputImage.at<uchar>(i, j) = static_cast<uchar>( (diff < 0) ? 0 : diff );
        }
    }
}

// 圖像銳化 - 與 main.cpp 中的 sharpenImagePar 對應
/**
 * @brief   圖像銳化 (與 main.cpp 中的 sharpenImagePar 平行版對應)
 * @details 使用高斯模糊後的影像建立高通掩膜，將此掩膜加權後疊加到原始影像，以強化邊緣細節。
 *
 * @param inputImage1  原始灰階影像 (CV_8UC1)
 * @param inputImage2  銳化掩膜 (CV_8UC1)
 *                      - 通常是由 `subtractImage(original, blurred, mask)` 得到  
 *                      - 也就是原始影像減去模糊後影像的結果，保留高頻細節
 * @param outputImage  輸出銳化後的灰階影像 (CV_8UC1)，在函式內重新建立
 */
void sharpenImage(cv::Mat& inputImage1,
                  cv::Mat& inputImage2,
                  cv::Mat& outputImage) {
    // 1. 定義銳化強度權重 w（越大邊緣增強越明顯）
    double weight = 10.0;

    // 2. 建立輸出影像：與 inputImage1 同樣的尺寸與資料類型
    outputImage = cv::Mat(inputImage1.size(), inputImage1.type());

    // 3. 逐像素計算：O = I1 + w * mask，並使用 saturate_cast 限制在 [0,255]
    for (int y = 0; y < inputImage1.rows; ++y) {
        for (int x = 0; x < inputImage1.cols; ++x) {
            int orig = static_cast<int>( inputImage1.at<uchar>(y, x) );
            int mask = static_cast<int>( inputImage2.at<uchar>(y, x) );
            int val  = orig + static_cast<int>(weight * mask);
            // uchar 是 OpenCV 中表示無符號 8 位元整數的類型（範圍：0 到 255）
            outputImage.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
        }
    }
}

// RGB到HSV轉換 - 與 main.cpp 中的 rgbToHsvPar 對應
/**
 * @brief   將 BGR 彩色影像轉換為 HSV 色彩空間
 * @details 與 main.cpp 中的 rgbToHsvPar 平行版對應，針對 CV_8UC3 (BGR) 影像，
 *          輸出範圍 H: [0,180], S: [0,255], V: [0,255]
 *
 * @param inputImage   輸入的 BGR 彩色影像 (CV_8UC3)
 * @param outputImage  輸出 HSV 影像 (CV_8UC3)，
 *                     通道 0 = H (0–180), 1 = S (0–255), 2 = V (0–255)
 */
void rgbToHsv(cv::Mat& inputImage, cv::Mat& outputImage) {
    // 1. 重新建立輸出影像：同樣大小與型態 (CV_8UC3)
    outputImage = cv::Mat(inputImage.size(), inputImage.type());
    
    // 2. 對每個像素執行轉換
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            // 2.1 讀取 BGR 分量並正規化到 [0,1]
            double blueSc  = inputImage.at<cv::Vec3b>(i, j)[0] / 255.0;
            double greenSc = inputImage.at<cv::Vec3b>(i, j)[1] / 255.0;
            double redSc   = inputImage.at<cv::Vec3b>(i, j)[2] / 255.0;
            
            // 2.2 找出最大值 cmax、最小值 cmin 及差值 delta
            double cmin  = std::min({ redSc, greenSc, blueSc });
            double cmax  = std::max({ redSc, greenSc, blueSc });
            double delta = cmax - cmin;
            
            double h = 0.0, s = 0.0, v = 0.0;
            
            // 2.3 計算 Hue, Saturation, Value
            if (delta < 1e-6) {
                // 如果所有通道相等，視為灰階，H=0, S=0, V=cmax
                h = 0.0;
                s = 0.0;
                v = cmax * 255.0;
            } else {
                // 2.3.1 Hue 計算 (°)
                if (cmax == redSc) {
                    h = 60.0 * ((greenSc - blueSc) / delta);
                } else if (cmax == greenSc) {
                    h = 120.0 + 60.0 * ((blueSc - redSc) / delta);
                } else {  // cmax == blueSc
                    h = 240.0 + 60.0 * ((redSc - greenSc) / delta);
                }
                // 保證 H >= 0
                if (h < 0.0) h += 360.0;
                // OpenCV H 範圍為 [0,180]，因此除以 2
                h = h / 2.0;
                
                // 2.3.2 Saturation 計算
                s = (cmax > 0.0) ? ((delta / cmax) * 255.0) : 0.0;
                
                // 2.3.3 Value 直接為最大值
                v = cmax * 255.0;
            }
            
            // 2.4 寫回到 outputImage (使用 saturate_cast 防止溢位)
            outputImage.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(h);
            outputImage.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(s);
            outputImage.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(v);
        }
    }
}

// HSV到RGB轉換 - 與 main.cpp 中的 hsvToRgbPar 對應
/**
 * @brief   將 HSV 影像轉換回 BGR 彩色影像
 * @details 與 main.cpp 中的 hsvToRgbPar 平行版對應，針對 CV_8UC3 (HSV) 影像，
 *          輸出 CV_8UC3 (BGR) 影像。
 *
 * @param inputImage   輸入的 HSV 影像 (CV_8UC3)
 *                       - 通道 0 = H (0–180 對應 0–360°)
 *                       - 通道 1 = S (0–255 對應 0.0–1.0)
 *                       - 通道 2 = V (0–255 對應 0.0–1.0)
 * @param outputImage  輸出 BGR 彩色影像 (CV_8UC3)
 */
void hsvToRgb(cv::Mat& inputImage, cv::Mat& outputImage) {
    // 1. 建立輸出影像：同樣尺寸與型態 (CV_8UC3)
    outputImage = cv::Mat(inputImage.size(), inputImage.type());
    
    // 2. 逐像素轉換
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            // 2.1 讀取並標準化 H, S, V
            double h = inputImage.at<cv::Vec3b>(i, j)[0] * 2.0;      // H: [0,180] → [0,360)
            double s = inputImage.at<cv::Vec3b>(i, j)[1] / 255.0;   // S: [0,255] → [0,1]
            double v = inputImage.at<cv::Vec3b>(i, j)[2] / 255.0;   // V: [0,255] → [0,1]
            
            // 2.2 計算中間變數：C, X, m
            double C = s * v;                                                   // 色度 (chroma)
            double X = C * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));          // 第二大色度分量
            double m = v - C;                                                   // 亮度補償
            
            // 2.3 根據 Hue 所在扇區選擇 (r', g', b')
            double r1=0, g1=0, b1=0;
            if      (h <  60) { r1 = C; g1 = X; b1 = 0; }
            else if (h < 120) { r1 = X; g1 = C; b1 = 0; }
            else if (h < 180) { r1 = 0; g1 = C; b1 = X; }
            else if (h < 240) { r1 = 0; g1 = X; b1 = C; }
            else if (h < 300) { r1 = X; g1 = 0; b1 = C; }
            else              { r1 = C; g1 = 0; b1 = X; }
            
            // 2.4 加上 m 並映射回 [0,255]
            double r = (r1 + m) * 255.0;
            double g = (g1 + m) * 255.0;
            double b = (b1 + m) * 255.0;
            
            // 2.5 寫回 BGR (注意 Vec3b 順序：B,G,R)
            outputImage.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(b);
            outputImage.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(g);
            outputImage.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(r);
        }
    }
}

// 合併的直方圖計算和均衡化函數
/**
 * @brief   計算直方圖並對灰階影像做直方圖均衡化
 * @details 依序完成以下步驟：
 *            1. 計算原影像的直方圖  
 *            2. 計算累積分布函數 (CDF)  
 *            3. 建立灰階映射查找表 (LUT)  
 *            4. 套用 LUT，輸出均衡化後影像  
 *
 * @param inputImage   輸入的灰階影像 (CV_8UC1)
 * @param outputImage  均衡化後的灰階影像 (CV_8UC1)，在函式內 clone 並寫入
 */
void histogramCalcAndEqual(const cv::Mat& inputImage, cv::Mat& outputImage) {
    // 1. 計算直方圖：histogram[i] = 灰階值 i 的像素數
    unsigned int histogram[256] = {0};
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            uchar val = inputImage.at<uchar>(y, x);
            histogram[val]++;
        }
    }
    
    // 2. 計算累積分布函數 (CDF)：cdf[i] = sum_{j=0..i} histogram[j]
    int totalPixels = inputImage.rows * inputImage.cols;
    int cdf[256] = {0};
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
    
    // 3. 建立查找表 (LUT)：LUT[i] = round( (L-1) * CDF[i] / MN )
    //    scale = 255.0f / totalPixels
    uchar lut[256];
    float scale = 255.0f / static_cast<float>(totalPixels);
    for (int i = 0; i < 256; ++i) {
        lut[i] = cv::saturate_cast<uchar>(scale * cdf[i]);
    }
    
    // 4. 套用 LUT：對每個像素 r，輸出 lut[r]
    outputImage = inputImage.clone();
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            uchar r = inputImage.at<uchar>(y, x);
            outputImage.at<uchar>(y, x) = lut[r];
        }
    }
}

int main(int argc, char *argv[])
{
    // 處理命令行參數
    if (argc < 2) {
        std::cout << "Usage: ./serial <imagename> [size]" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  ./serial snow.png             - Use original size" << std::endl;
        std::cout << "  ./serial snow.png 778x1036    - Resize to 778x1036" << std::endl;
        return EXIT_FAILURE;
    }
    
    // 設置輸入圖像路徑
    std::string inputImagePath = argv[1];
    std::cout << "Processing image: " << inputImagePath << std::endl;
    
    // 讀取輸入圖像
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_UNCHANGED);
    
    // 檢查圖像是否成功讀取
    if (inputImage.empty()) {
        std::cout << "Couldn't read the image: " << inputImagePath << std::endl;
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
            std::cout << "Resized input image to " << width << "x" << height << std::endl;
        } else {
            std::cout << "Invalid size format. Using original size: " 
                      << inputImage.cols << "x" << inputImage.rows << std::endl;
        }
    } else {
        std::cout << "Using original size: " << inputImage.cols << "x" << inputImage.rows << std::endl;
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
    
    // 創建結果目錄
    std::filesystem::create_directory("results");
    
    // 設置CSV結果文件
    std::string resultpath = "results/" + basename + "_" + sizeStr + "_Vserial.csv";
    std::ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << std::endl;
    
    // 輸出檔案名稱
    std::string outputFilename = "results/" + basename + "_" + sizeStr + "_Vserial" + extension;
    
    // =========== 修改: 統一計時架構 ===========
    // 1. 預先分配所有需要的矩陣
    cv::Mat inputImageHsv(resizedImage.size(), resizedImage.type());
    cv::Mat blurredImage(resizedImage.size(), CV_8UC1);
    cv::Mat imageMask(resizedImage.size(), CV_8UC1);
    cv::Mat sharpenedImage(resizedImage.size(), CV_8UC1);
    cv::Mat globallyEnhancedImage(resizedImage.size(), CV_8UC1);
    cv::Mat outputHSV(resizedImage.size(), resizedImage.type());
    cv::Mat outputImage(resizedImage.size(), resizedImage.type());
    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> outChannels(3);
    
    // 2. 創建時間評估變量
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto rgbToHsvTime = std::chrono::duration<double, std::milli>::zero();
    auto blurTime = std::chrono::duration<double, std::milli>::zero();
    auto subtractTime = std::chrono::duration<double, std::milli>::zero();
    auto sharpenTime = std::chrono::duration<double, std::milli>::zero();
    auto histogramTotalTime = std::chrono::duration<double, std::milli>::zero();
    auto hsvToRgbTime = std::chrono::duration<double, std::milli>::zero();
    
    // 3. 定義統一的迭代次數
    const int warmupIter = 3;  // 熱身迭代次數
    const int numIter = 10;    // 計時迭代次數
    
    // =========== STEP 1: RGB TO HSV CONVERSION ===========
    // 熱身迭代（不計時）
    for (int i = 0; i < warmupIter; ++i) {
        rgbToHsv(resizedImage, inputImageHsv);
    }
    
    // 強制系統同步
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // 計時迭代
    rgbToHsvTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        rgbToHsv(resizedImage, inputImageHsv);
        end = std::chrono::high_resolution_clock::now();
        rgbToHsvTime += (end - start);
    }
    rgbToHsvTime /= numIter;
    
    std::cout << "RGB to HSV Conversion Time: " << rgbToHsvTime.count() << " ms" << std::endl;
    
    // 分離 HSV 通道 - 不計入時間
    cv::split(inputImageHsv, channels);
    cv::Mat& inputImageH = channels[0]; 
    cv::Mat& inputImageS = channels[1];
    cv::Mat& inputImageV = channels[2];
    
    // =========== STEP 2-1: 圖像模糊 ===========
    // 熱身迭代
    for (int i = 0; i < warmupIter; ++i) {
        imageBlur(inputImageV, blurredImage); 
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    blurTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        imageBlur(inputImageV, blurredImage);
        end = std::chrono::high_resolution_clock::now();
        blurTime += (end - start);
    }
    blurTime /= numIter;
    
    std::cout << "Image Blur Time: " << blurTime.count() << " ms" << std::endl;
    
    // =========== STEP 2-2: 圖像相減 ===========
    // 熱身迭代
    for (int i = 0; i < warmupIter; ++i) {
        subtractImage(inputImageV, blurredImage, imageMask);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    subtractTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        subtractImage(inputImageV, blurredImage, imageMask);
        end = std::chrono::high_resolution_clock::now();
        subtractTime += (end - start);
    }
    subtractTime /= numIter;
    
    std::cout << "Image Subtraction Time: " << subtractTime.count() << " ms" << std::endl;
    
    // =========== STEP 2-3: 圖像銳化 ===========
    // 熱身迭代
    for (int i = 0; i < warmupIter; ++i) {
        sharpenImage(inputImageV, imageMask, sharpenedImage);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    sharpenTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        sharpenImage(inputImageV, imageMask, sharpenedImage);
        end = std::chrono::high_resolution_clock::now();
        sharpenTime += (end - start);
    }
    sharpenTime /= numIter;
    
    std::cout << "Image Sharpening Time: " << sharpenTime.count() << " ms" << std::endl;
    
    // =========== STEP 3: 直方圖處理 ===========
    // 熱身迭代
    for (int i = 0; i < warmupIter; ++i) {
        histogramCalcAndEqual(sharpenedImage, globallyEnhancedImage);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    histogramTotalTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        histogramCalcAndEqual(sharpenedImage, globallyEnhancedImage);
        end = std::chrono::high_resolution_clock::now();
        histogramTotalTime += (end - start);
    }
    histogramTotalTime /= numIter;

    std::cout << "Combined Histogram Calculation and Equalization Time: " 
              << histogramTotalTime.count() << " ms" << std::endl;
    
    // 合併 HSV 通道 - 不計入時間
    outChannels[0] = inputImageH;
    outChannels[1] = inputImageS; 
    outChannels[2] = globallyEnhancedImage;
    cv::merge(outChannels, outputHSV);
    
    // =========== STEP 4: HSV to RGB 轉換 ===========
    // 熱身迭代
    for (int i = 0; i < warmupIter; ++i) {
        hsvToRgb(outputHSV, outputImage);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    hsvToRgbTime = std::chrono::duration<double, std::milli>::zero();
    for (int i = 0; i < numIter; ++i) {
        start = std::chrono::high_resolution_clock::now();
        hsvToRgb(outputHSV, outputImage);
        end = std::chrono::high_resolution_clock::now();
        hsvToRgbTime += (end - start);
    }
    hsvToRgbTime /= numIter;
    
    std::cout << "HSV to RGB Conversion Time: " << hsvToRgbTime.count() << " ms" << std::endl;
    
    // 計算總時間
    auto totalTime = rgbToHsvTime + blurTime + subtractTime + sharpenTime + 
                     histogramTotalTime + hsvToRgbTime;
    
    std::cout << "Total Processing Time: " << totalTime.count() << " ms" << std::endl;
    
    // 寫入結果圖像
    cv::imwrite(outputFilename, outputImage);
    std::cout << "Enhanced image saved as: " << outputFilename << std::endl;
    
    // 寫入結果到CSV文件
    outFile << basename << ","
            << resizedImage.cols << "x" << resizedImage.rows << ","
            << "1" << ","  // 添加 Threads 欄位，設為 1
            << rgbToHsvTime.count() << ","
            << blurTime.count() << ","
            << subtractTime.count() << ","
            << sharpenTime.count() << ","
            << histogramTotalTime.count() << ","
            << hsvToRgbTime.count() << ","
            << totalTime.count() << std::endl;
    
    outFile.close();
    
    return 0;
}