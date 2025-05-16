/*
優化版本 - Singh et. al 圖像增強演算法實現
https://ieeexplore.ieee.org/document/8071892

編譯命令:
g++ -g -Wall -Ofast -flto -o Vproposed Vproposed.cpp `pkg-config --cflags --libs opencv4` -fopenmp -march=native -funroll-loops -ftree-vectorize

執行方式:
./Vproposed <imagename> <threads> <size>
例如: ./Vproposed snow.png 4 778x1036
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <new>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <math.h>       
#include <limits>
#include <vector>
#include <immintrin.h> // 支援SIMD指令
#include <sstream>
#include <filesystem>
#include <thread>

// 添加命名空間
using namespace std;
using namespace cv;

#define FILTERSIZE 5
// #define MAX_THREADS 128 // 限制最大執行緒數避免過度分配 (已註釋)

// 模糊濾波核
alignas(64) const float Filter[FILTERSIZE][FILTERSIZE] = {
    {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f},
    {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
    {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
    {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
    {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f}
};

// 提前計算1D版本的濾波核，加速存取
alignas(64) float Filter1D[FILTERSIZE*FILTERSIZE];

// 優化後的並行實現函數原型
void rgbToHsvParOpt(const Mat& inputImage, Mat& outputImage, int blockSize);
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage);
void subtractImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage);
void sharpenImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage, double weight);
void histogramCalcParOpt(const Mat& inputImage, unsigned int imHistogram[], int nThreads);
void histogramEqualParOpt(const Mat& inputImage, Mat& outputImage, const unsigned int imHistogram[]);
void hsvToRgbParOpt(const Mat& inputImage, Mat& outputImage, int blockSize);
string getexepath();
void histogramCalcAndEqualParOpt(const Mat& inputImage, Mat& outputImage, int nThreads);

// 計算理想的塊大小
int calculateOptimalBlockSize(const Mat& image) {
    // 根據影像大小調整塊大小
    int pixelCount = image.rows * image.cols;
    
    if (pixelCount < 500000) {      // 小於 500K 像素
        return 16;
    } else if (pixelCount < 2000000) { // 小於 2M 像素
        return 32;
    } else if (pixelCount < 8000000) { // 小於 8M 像素
        return 64;
    } else {                        // 超大影像
        return 128;
    }
}

// 檢查圖像是否有效
bool isImageValid(const Mat& image) {
    return !image.empty() && image.rows > 0 && image.cols > 0;
}

// 解析尺寸字符串 (如 "778x1036")
bool parseSize(const string& sizeStr, int& width, int& height) {
    size_t xPos = sizeStr.find('x');
    if (xPos == string::npos) {
        return false;
    }
    
    try {
        width = stoi(sizeStr.substr(0, xPos));
        height = stoi(sizeStr.substr(xPos + 1));
        return (width > 0 && height > 0);
    } catch (const exception&) {
        return false;
    }
}

int main(int argc, char const *argv[])
{
    // 將濾波核初始化為一維陣列，提高快取友好度
    for(int i = 0; i < FILTERSIZE; i++) {
        for(int j = 0; j < FILTERSIZE; j++) {
            Filter1D[i*FILTERSIZE + j] = Filter[i][j];
        }
    }

    if (argc < 2) {
        cout << "Usage: ./test_optimized <imagename> [threads] [size]" << endl;
        cout << "Examples:" << endl;
        cout << "  ./test_optimized snow.png                  - Run with default threads (all) and original size" << endl;
        cout << "  ./test_optimized snow.png 4                - Run with 4 threads and original size" << endl;
        cout << "  ./test_optimized snow.png 4 778x1036       - Run with 4 threads and resize to 778x1036" << endl;
        exit(EXIT_FAILURE);
    }

    // 解析執行緒數量參數
    int initial_omp_max_threads = omp_get_max_threads(); // 在任何 omp_set_num_threads 前獲取
    cout << "[診斷] 系統初始 omp_get_max_threads(): " << initial_omp_max_threads << endl;

    int numThreads = initial_omp_max_threads; // 預設使用系統最大執行緒數
    
    if (argc >= 3) {
        string thread_arg_str = argv[2];
        try {
            // 檢查參數是否為純數字，以區分執行緒數和尺寸參數
            bool is_numeric_arg = !thread_arg_str.empty() && 
                                  thread_arg_str.find_first_not_of("0123456789") == string::npos;

            if (is_numeric_arg) {
                int requested_threads = stoi(thread_arg_str);
                cout << "[診斷] 從命令列 argv[2] 讀取到請求的執行緒數: " << requested_threads << endl;
                if (requested_threads <= 0) {
                    cout << "[診斷] 請求的執行緒數 " << requested_threads << " 無效。將使用系統最大值: " << initial_omp_max_threads << endl;
                    numThreads = initial_omp_max_threads;
                } else {
                    numThreads = requested_threads; // 使用請求的執行緒數
                }
            } else {
                 // argv[2] 不是純數字，可能是尺寸參數（當執行緒數未提供時）
                 cout << "[診斷] 命令列 argv[2] ('" << thread_arg_str << "') 非執行緒數。將使用系統最大值: " << initial_omp_max_threads << endl;
                 numThreads = initial_omp_max_threads;
            }
        } catch (const std::exception& e) {
            // stoi 轉換失敗或其他異常
            cout << "[診斷] 解析執行緒數參數 argv[2] ('" << thread_arg_str << "') 失敗。錯誤: " << e.what() 
                      << " 將使用系統最大值: " << initial_omp_max_threads << endl;
            numThreads = initial_omp_max_threads;
        }
    } else {
        cout << "[診斷] 未從命令列提供執行緒數參數。將使用系統最大值: " << initial_omp_max_threads << endl;
        numThreads = initial_omp_max_threads; 
    }
    
    omp_set_num_threads(numThreads); // 設定OpenMP執行緒數
    // --- 結束新增的診斷訊息 ---

    cout << "Current working directory: " << getexepath() << endl;

    // 讀取輸入圖像
    Mat originalImage = imread(argv[1], IMREAD_UNCHANGED);
    
    // 檢查圖像是否正確讀取
    if (!isImageValid(originalImage)) {
        cerr << "Error: Could not read image " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }
    
    // 解析大小參數並調整圖像大小（如果有指定）
    Mat inputImage = originalImage;
    string sizeStr = "original";
    
    if (argc >= 4) {
        int width, height;
        if (parseSize(argv[3], width, height)) {
            sizeStr = argv[3];
            resize(originalImage, inputImage, Size(width, height));
            cout << "Resized input image to " << width << "x" << height << endl;
        } else {
            cout << "Invalid size format, using original size: " 
                      << originalImage.cols << "x" << originalImage.rows << endl;
        }
    } else {
        cout << "Using original size: " << originalImage.cols << "x" << originalImage.rows << endl;
    }

    // 計算最佳的執行緒數量和塊大小
    int optimalBlockSize = calculateOptimalBlockSize(inputImage);
    
    // 設置執行緒數量 (omp_set_num_threads 實際已在上方診斷區塊呼叫)
    // omp_set_num_threads(numThreads); // 這行重複，可移除或註釋，因為前面已經設定
    cout << "[最終設定] 設定給 omp_set_num_threads 的執行緒數: " << numThreads << endl;
    cout << "[最終設定] 呼叫 omp_set_num_threads 後，omp_get_max_threads() 回報: " << omp_get_max_threads() << endl;
    cout << "[最終設定] 使用的 BlockSize: " << optimalBlockSize << endl;

    // 優化路徑處理
    string filename = argv[1];
    size_t last_slash_pos = filename.find_last_of("/\\");
    if (last_slash_pos != string::npos) {
        filename = filename.substr(last_slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    string basename = filename.substr(0, dot_pos);
    string extension = filename.substr(dot_pos);
    
    // 確保results目錄存在
    filesystem::create_directory("results");
    
    string sizemodifier = to_string(inputImage.rows) + "x" + to_string(inputImage.cols);
    string threadModifier = "_t" + to_string(numThreads);
    string outimagename = "results/" + basename + "_" + sizeStr + "_Vproposed" + threadModifier + extension;
    cout << "Output image file name: " << outimagename << endl;

    // 創建CSV結果文件
    string resultpath = "results/" + basename + "_" + sizeStr + "_Vproposed_threads" + to_string(numThreads) + ".csv";
    ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,BlockSize,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << endl;
    
    // =========== 修改: 統一計時架構 ===========
    // 1. 預先分配所有需要的記憶體
    Mat inputImageHsv(inputImage.size(), inputImage.type());
    Mat blurredImage(inputImage.size(), CV_8UC1);
    Mat imageMask(inputImage.size(), CV_8UC1);
    Mat sharpenedImage(inputImage.size(), CV_8UC1);
    Mat globallyEnhancedImage(inputImage.size(), CV_8UC1);
    Mat outputHSV(inputImage.size(), inputImage.type());
    Mat finalOutput(inputImage.size(), inputImage.type());
    
    vector<Mat> inputImageHsvChannels(3);
    vector<Mat> channels(3);
    
    // 2. 創建時間評估變量
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    auto timeEllap1 = chrono::duration<double, milli>::zero();
    auto timeEllap2 = chrono::duration<double, milli>::zero();
    auto timeEllap3 = chrono::duration<double, milli>::zero();
    auto timeEllap4 = chrono::duration<double, milli>::zero();
    auto timeEllap7 = chrono::duration<double, milli>::zero();
    auto timeEllapHistCombined = chrono::duration<double, milli>::zero();
    
    // 3. 定義統一的迭代次數
    const int warmupIter = 3;  // 熱身迭代次數
    const int numIter = 10;    // 計時迭代次數，確保與其他版本相同

    // =========== STEP 1: RGB到HSV轉換 ===========
    // 熱身迭代（不計時）
    for(int i = 0; i < warmupIter; ++i) {
        rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
    }
    
    // 強制系統同步
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // 計時迭代
    timeEllap1 = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
        end = chrono::high_resolution_clock::now();
        timeEllap1 += (end - start);
    }
    timeEllap1 /= numIter;
    
    cout << "RGB to HSV Conversion: " << timeEllap1.count() << " ms" << endl;

    // 分離HSV通道 - 不計入時間
    split(inputImageHsv, inputImageHsvChannels);
    Mat& inputImageH = inputImageHsvChannels[0];
    Mat& inputImageS = inputImageHsvChannels[1];
    Mat& inputImageV = inputImageHsvChannels[2];

    // =========== STEP 2-1: 圖像模糊 ===========
    // 熱身迭代
    for(int i = 0; i < warmupIter; ++i) {
        imageBlurParOpt(inputImageV, blurredImage);
    }
    
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    timeEllap2 = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        imageBlurParOpt(inputImageV, blurredImage);
        end = chrono::high_resolution_clock::now();
        timeEllap2 += (end - start);
    }
    timeEllap2 /= numIter;

    cout << "Image Blur: " << timeEllap2.count() << " ms" << endl;

    // =========== STEP 2-2: 圖像相減 ===========
    // 熱身迭代
    for(int i = 0; i < warmupIter; ++i) {
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
    }
    
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    timeEllap3 = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
        end = chrono::high_resolution_clock::now();
        timeEllap3 += (end - start);
    }
    timeEllap3 /= numIter;

    cout << "Image Subtraction: " << timeEllap3.count() << " ms" << endl;

    // =========== STEP 2-3: 圖像銳化 ===========
    double weight = 10.0;
    
    // 熱身迭代
    for(int i = 0; i < warmupIter; ++i) {
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
    }
    
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    timeEllap4 = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
        end = chrono::high_resolution_clock::now();
        timeEllap4 += (end - start);
    }
    timeEllap4 /= numIter;

    cout << "Image Sharpening: " << timeEllap4.count() << " ms" << endl;

    // =========== STEP 3: 直方圖處理 ===========
    // 熱身迭代
    for(int i = 0; i < warmupIter; ++i) {
        histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, numThreads);
    }
    
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // 計時合併操作
    timeEllapHistCombined = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, numThreads);
        end = chrono::high_resolution_clock::now();
        timeEllapHistCombined += (end - start);
    }
    timeEllapHistCombined /= numIter;

    cout << "Combined Histogram Processing: " << timeEllapHistCombined.count() << " ms" << endl;

    // 合併HSV通道 - 不計入時間
    channels[0] = inputImageH;
    channels[1] = inputImageS;
    channels[2] = globallyEnhancedImage;
    merge(channels, outputHSV);
    
    // =========== STEP 4: HSV到RGB轉換 ===========
    // 熱身迭代
    for(int i = 0; i < warmupIter; ++i) {
        hsvToRgbParOpt(outputHSV, finalOutput, optimalBlockSize);
    }
    
    #pragma omp barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    timeEllap7 = chrono::duration<double, milli>::zero();
    for(int i = 0; i < numIter; ++i) {
        start = chrono::high_resolution_clock::now();
        hsvToRgbParOpt(outputHSV, finalOutput, optimalBlockSize);
        end = chrono::high_resolution_clock::now();
        timeEllap7 += (end - start);
    }
    timeEllap7 /= numIter;

    cout << "HSV to RGB Conversion: " << timeEllap7.count() << " ms" << endl;

    // 保存結果圖像
    imwrite(outimagename, finalOutput);
    
    // 計算總處理時間
    auto totalTime = timeEllap1 + timeEllap2 + timeEllap3 + timeEllap4 + timeEllapHistCombined + timeEllap7;

    cout << "Total Processing Time: " << totalTime.count() << " ms" << endl;

    // 寫入結果到CSV文件
    outFile << basename << "," 
           << inputImage.cols << "x" << inputImage.rows << "," 
           << numThreads << "," 
           << optimalBlockSize << ","
           << timeEllap1.count() << "," 
           << timeEllap2.count() << "," 
           << timeEllap3.count() << "," 
           << timeEllap4.count() << "," 
           << timeEllapHistCombined.count() << "," 
           << timeEllap7.count() << "," 
           << totalTime.count() << endl;

    outFile.close();
    return 0;
}

// 增強 RGB 到 HSV 轉換的向量化
void rgbToHsvParOpt(const Mat& inputImage, Mat& outputImage, int blockSize)
{
    // 使用 _mm256 指令組合
    #ifdef __AVX2__
    // AVX2 優化部分... (目前為空, 保留結構供未來擴展)
    #endif
    
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const float inv255_scalar = 1.0f / 255.0f; // Scalar precompute for remainder loop
    
    #pragma omp parallel
    {
        #ifdef __AVX2__
        const __m256 _zero_ps = _mm256_setzero_ps();
        const __m256 _one_ps = _mm256_set1_ps(1.0f);
        const __m256 _half_ps = _mm256_set1_ps(0.5f);
        const __m256 _2_ps = _mm256_set1_ps(2.0f);
        const __m256 _4_ps = _mm256_set1_ps(4.0f);
        const __m256 _60_ps = _mm256_set1_ps(60.0f);
        const __m256 _255_ps = _mm256_set1_ps(255.0f);
        const __m256 _360_ps = _mm256_set1_ps(360.0f);
        const __m256 _epsilon_ps = _mm256_set1_ps(1e-6f);
        const __m256 _inv255_ps = _mm256_set1_ps(1.0f / 255.0f);
        const __m128i _zero_xmm = _mm_setzero_si128();
        #endif

        #pragma omp for schedule(guided)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);
                
                for(int i = ii; i < blockEndI; ++i) {
                    const Vec3b* inRow = inputImage.ptr<Vec3b>(i);
                    Vec3b* outRow = outputImage.ptr<Vec3b>(i);
                    
                    int j = jj;
#ifdef __AVX2__
                    for (; j <= blockEndJ - 8; j += 8) {
                        unsigned char b_tmp[8], g_tmp[8], r_tmp[8];
                        for (int k_load = 0; k_load < 8; ++k_load) {
                            b_tmp[k_load] = inRow[j + k_load][0];
                            g_tmp[k_load] = inRow[j + k_load][1];
                            r_tmp[k_load] = inRow[j + k_load][2];
                        }

                        __m128i b_u8_xmm = _mm_loadl_epi64((__m128i*)b_tmp);
                        __m128i g_u8_xmm = _mm_loadl_epi64((__m128i*)g_tmp);
                        __m128i r_u8_xmm = _mm_loadl_epi64((__m128i*)r_tmp);

                        __m256i b_epi32 = _mm256_cvtepu8_epi32(b_u8_xmm);
                        __m256i g_epi32 = _mm256_cvtepu8_epi32(g_u8_xmm);
                        __m256i r_epi32 = _mm256_cvtepu8_epi32(r_u8_xmm);

                        __m256 b_ps_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(b_epi32), _inv255_ps);
                        __m256 g_ps_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(g_epi32), _inv255_ps);
                        __m256 r_ps_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(r_epi32), _inv255_ps);
                        
                        // Cmax, Cmin, Delta (using normalized R,G,B)
                        __m256 cmax_ps = _mm256_max_ps(r_ps_norm, _mm256_max_ps(g_ps_norm, b_ps_norm));
                        __m256 cmin_ps = _mm256_min_ps(r_ps_norm, _mm256_min_ps(g_ps_norm, b_ps_norm));
                        __m256 delta_ps = _mm256_sub_ps(cmax_ps, cmin_ps);

                        __m256 v_ps = cmax_ps; // V = Cmax (scaled 0-1)

                        __m256 s_ps = _mm256_div_ps(delta_ps, cmax_ps);
                        __m256 cmax_is_small_mask = _mm256_cmp_ps(cmax_ps, _epsilon_ps, _CMP_LT_OQ);
                        s_ps = _mm256_blendv_ps(s_ps, _zero_ps, cmax_is_small_mask);

                        __m256 delta_is_small_mask = _mm256_cmp_ps(delta_ps, _epsilon_ps, _CMP_LT_OQ);
                        s_ps = _mm256_blendv_ps(s_ps, _zero_ps, delta_is_small_mask);

                        __m256 h_ps = _mm256_setzero_ps();
                        __m256 inv_delta_ps = _mm256_div_ps(_one_ps, delta_ps);
                        inv_delta_ps = _mm256_blendv_ps(inv_delta_ps, _zero_ps, delta_is_small_mask);

                        __m256 r_is_cmax_mask = _mm256_cmp_ps(r_ps_norm, cmax_ps, _CMP_EQ_OQ);
                        __m256 g_is_cmax_mask = _mm256_andnot_ps(r_is_cmax_mask, _mm256_cmp_ps(g_ps_norm, cmax_ps, _CMP_EQ_OQ));

                        __m256 h_r_case = _mm256_mul_ps(_mm256_sub_ps(g_ps_norm, b_ps_norm), inv_delta_ps);
                        __m256 h_g_case = _mm256_add_ps(_2_ps, _mm256_mul_ps(_mm256_sub_ps(b_ps_norm, r_ps_norm), inv_delta_ps));
                        __m256 h_b_case = _mm256_add_ps(_4_ps, _mm256_mul_ps(_mm256_sub_ps(r_ps_norm, g_ps_norm), inv_delta_ps));
                        
                        h_ps = _mm256_blendv_ps(h_b_case, h_g_case, g_is_cmax_mask);
                        h_ps = _mm256_blendv_ps(h_ps, h_r_case, r_is_cmax_mask);
                        h_ps = _mm256_blendv_ps(h_ps, _zero_ps, delta_is_small_mask); // If delta is small, H is 0

                        h_ps = _mm256_mul_ps(h_ps, _60_ps);
                        __m256 h_neg_mask = _mm256_cmp_ps(h_ps, _zero_ps, _CMP_LT_OQ);
                        h_ps = _mm256_add_ps(h_ps, _mm256_and_ps(h_neg_mask, _360_ps));
                        
                        // Final OpenCV scaling
                        h_ps = _mm256_mul_ps(h_ps, _half_ps);   // H: [0, 360) -> [0, 180)
                        s_ps = _mm256_mul_ps(s_ps, _255_ps);    // S: [0, 1]   -> [0, 255]
                        v_ps = _mm256_mul_ps(v_ps, _255_ps);    // V: [0, 1]   -> [0, 255]

                        // Convert to uchar and store
                        __m256i h_epi32_final = _mm256_cvttps_epi32(h_ps);
                        __m256i s_epi32_final = _mm256_cvttps_epi32(s_ps);
                        __m256i v_epi32_final = _mm256_cvttps_epi32(v_ps);

                        __m128i h_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(h_epi32_final), _mm256_extracti128_si256(h_epi32_final, 1));
                        __m128i s_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(s_epi32_final), _mm256_extracti128_si256(s_epi32_final, 1));
                        __m128i v_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(v_epi32_final), _mm256_extracti128_si256(v_epi32_final, 1));

                        __m128i h_u8_xmm = _mm_packus_epi16(h_u16_xmm, _zero_xmm);
                        __m128i s_u8_xmm = _mm_packus_epi16(s_u16_xmm, _zero_xmm);
                        __m128i v_u8_xmm = _mm_packus_epi16(v_u16_xmm, _zero_xmm);

                        unsigned char h_u[8], s_u[8], v_u[8];
                        _mm_storel_epi64((__m128i*)h_u, h_u8_xmm);
                        _mm_storel_epi64((__m128i*)s_u, s_u8_xmm);
                        _mm_storel_epi64((__m128i*)v_u, v_u8_xmm);

                        for (int k_store = 0; k_store < 8; ++k_store) {
                            outRow[j + k_store][0] = h_u[k_store];
                            outRow[j + k_store][1] = s_u[k_store];
                            outRow[j + k_store][2] = v_u[k_store];
                        }
                    }
#endif
                    for (; j < blockEndJ; ++j) { // Scalar loop for remainder
                        uchar blue_s = inRow[j][0];
                        uchar green_s = inRow[j][1];
                        uchar red_s = inRow[j][2];
                        
                        float redScaled_s = red_s * inv255_scalar;
                        float greenScaled_s = green_s * inv255_scalar;
                        float blueScaled_s = blue_s * inv255_scalar;
                        
                        float cmin_s = min(min(redScaled_s, greenScaled_s), blueScaled_s);
                        float cmax_s = max(max(redScaled_s, greenScaled_s), blueScaled_s);
                        float delta_s = cmax_s - cmin_s;
                        
                        float h_s = 0.0f, s_s = 0.0f, v_s = cmax_s;

                        if(delta_s < 1e-6f) {
                            s_s = 0.0f;
                            h_s = 0.0f;
                        } else {
                            s_s = (cmax_s > 1e-6f) ? (delta_s / cmax_s) : 0.0f;
                            if(abs(cmax_s - redScaled_s) < 1e-6f) h_s = (greenScaled_s - blueScaled_s) / delta_s;
                            else if(abs(cmax_s - greenScaled_s) < 1e-6f) h_s = 2.0f + (blueScaled_s - redScaled_s) / delta_s;
                            else h_s = 4.0f + (redScaled_s - greenScaled_s) / delta_s;
                            h_s *= 60.0f;
                            if(h_s < 0.0f) h_s += 360.0f;
                        }
                        outRow[j][0] = saturate_cast<uchar>(h_s * 0.5f);
                        outRow[j][1] = saturate_cast<uchar>(s_s * 255.0f);
                        outRow[j][2] = saturate_cast<uchar>(v_s * 255.0f);
                    }
                }
            }
        }
    }
}

// 輔助函數：載入8個uchar並轉換為__m256 (8個float)
// 將此函數定義為 static inline，使其在 imageBlurParOpt 內部可見
#ifdef __AVX2__
static inline __m256 load_and_convert_uchar_to_float_avx2(const unsigned char* p) {
    __m128i u8_vec = _mm_loadl_epi64((const __m128i*)p); // 載入8個uchar到xmm的低64位元
    __m256i i32_vec = _mm256_cvtepu8_epi32(u8_vec);      // 轉換為8個int32到ymm
    return _mm256_cvtepi32_ps(i32_vec);                  // 轉換為8個float到ymm
}
#endif

// 大幅優化的模糊處理 - 使用可分離濾波器
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int halfSize = FILTERSIZE / 2;
    
    // 1D 高斯核 (標準化後，和為1)
    alignas(64) const float GaussianKernel1D_F[FILTERSIZE] = {
        1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f
    };

    // 中間影像，用於儲存水平模糊的結果
    Mat tempImage(rows, cols, CV_32FC1);

    // --- 水平遍歷 ---
    Mat paddedInputHorizontal;
    // 僅對左右進行邊界擴展，用於水平卷積
    copyMakeBorder(inputImage, paddedInputHorizontal, 0, 0, halfSize, halfSize, BORDER_REFLECT_101);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uchar* srcRow = paddedInputHorizontal.ptr<uchar>(i);
        float* tempRow = tempImage.ptr<float>(i);
        
        int j = 0;
#ifdef __AVX2__
        const __m256 gk0_v = _mm256_set1_ps(GaussianKernel1D_F[0]);
        const __m256 gk1_v = _mm256_set1_ps(GaussianKernel1D_F[1]);
        const __m256 gk2_v = _mm256_set1_ps(GaussianKernel1D_F[2]);
        const __m256 gk3_v = _mm256_set1_ps(GaussianKernel1D_F[3]);
        const __m256 gk4_v = _mm256_set1_ps(GaussianKernel1D_F[4]);

        for (; j <= cols - 8; j += 8) {
            // srcRow + j 指向當前輸出tempRow[j]對應的濾波窗口在paddedInput中的起始位置
            // v_xN 載入的是卷積核第N個系數所對應的輸入數據段
            // 例如，v_x0 包含 srcRow[j] 到 srcRow[j+7]
            // v_x1 包含 srcRow[j+1] 到 srcRow[j+8] 等
            __m256 v_x0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 0);
            __m256 v_x1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 1);
            __m256 v_x2 = load_and_convert_uchar_to_float_avx2(srcRow + j + 2);
            __m256 v_x3 = load_and_convert_uchar_to_float_avx2(srcRow + j + 3);
            __m256 v_x4 = load_and_convert_uchar_to_float_avx2(srcRow + j + 4);

            // 卷積:
            // tempRow[j+idx] = srcRow[j+idx+0]*gk0 + srcRow[j+idx+1]*gk1 + ... + srcRow[j+idx+4]*gk4
            // 這是通過將每個 v_xN（代表移位的輸入數據）與對應的廣播核係數 gkN_v 相乘並累加來實現的
            __m256 sum_v = _mm256_mul_ps(v_x0, gk0_v);
            sum_v = _mm256_fmadd_ps(v_x1, gk1_v, sum_v);
            sum_v = _mm256_fmadd_ps(v_x2, gk2_v, sum_v);
            sum_v = _mm256_fmadd_ps(v_x3, gk3_v, sum_v);
            sum_v = _mm256_fmadd_ps(v_x4, gk4_v, sum_v);
            
            _mm256_storeu_ps(tempRow + j, sum_v);
        }
#endif
        // 處理剩餘的像素 (純量迴圈)
        for (; j < cols; ++j) {
            const uchar* pIn = srcRow + j; 
            float sum = 0.0f;
            sum += static_cast<float>(pIn[0]) * GaussianKernel1D_F[0];
            sum += static_cast<float>(pIn[1]) * GaussianKernel1D_F[1];
            sum += static_cast<float>(pIn[2]) * GaussianKernel1D_F[2];
            sum += static_cast<float>(pIn[3]) * GaussianKernel1D_F[3];
            sum += static_cast<float>(pIn[4]) * GaussianKernel1D_F[4];
            tempRow[j] = sum;
        }
    }

    // --- 垂直遍歷 ---
    Mat paddedTempImage;
    copyMakeBorder(tempImage, paddedTempImage, halfSize, halfSize, 0, 0, BORDER_REFLECT_101);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        uchar* outRow = outputImage.ptr<uchar>(i);
        
        int j = 0;
#ifdef __AVX2__
        const __m256 k0_v = _mm256_set1_ps(GaussianKernel1D_F[0]);
        const __m256 k1_v = _mm256_set1_ps(GaussianKernel1D_F[1]);
        const __m256 k2_v = _mm256_set1_ps(GaussianKernel1D_F[2]);
        const __m256 k3_v = _mm256_set1_ps(GaussianKernel1D_F[3]);
        const __m256 k4_v = _mm256_set1_ps(GaussianKernel1D_F[4]);
        const __m128i zero_xmm_v = _mm_setzero_si128(); // Renamed to avoid conflict if used elsewhere

        for (; j <= cols - 8; j += 8) {
            __m256 in0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 0) + j);
            __m256 in1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 1) + j);
            __m256 in2 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 2) + j);
            __m256 in3 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 3) + j);
            __m256 in4 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 4) + j);

            __m256 sum_ps = _mm256_mul_ps(in0, k0_v);
            sum_ps = _mm256_fmadd_ps(in1, k1_v, sum_ps);
            sum_ps = _mm256_fmadd_ps(in2, k2_v, sum_ps);
            sum_ps = _mm256_fmadd_ps(in3, k3_v, sum_ps);
            sum_ps = _mm256_fmadd_ps(in4, k4_v, sum_ps);
            
            __m256i sum_epi32 = _mm256_cvttps_epi32(sum_ps);
            __m128i sum_u16_lo = _mm_packus_epi32(_mm256_castsi256_si128(sum_epi32), _mm256_extracti128_si256(sum_epi32, 1));
            __m128i sum_u8 = _mm_packus_epi16(sum_u16_lo, zero_xmm_v);
            _mm_storel_epi64((__m128i*)(outRow + j), sum_u8);
        }
        // Remainder loop for vertical pass
        for (; j < cols; ++j) {
            float sum = 0.0f;
            sum += paddedTempImage.ptr<float>(i + 0)[j] * GaussianKernel1D_F[0];
            sum += paddedTempImage.ptr<float>(i + 1)[j] * GaussianKernel1D_F[1];
            sum += paddedTempImage.ptr<float>(i + 2)[j] * GaussianKernel1D_F[2];
            sum += paddedTempImage.ptr<float>(i + 3)[j] * GaussianKernel1D_F[3];
            sum += paddedTempImage.ptr<float>(i + 4)[j] * GaussianKernel1D_F[4];
            outRow[j] = saturate_cast<uchar>(sum);
        }
#else
        for (int j_scalar = 0; j_scalar < cols; ++j_scalar) {
            float sum = 0.0f;
            sum += paddedTempImage.ptr<float>(i + 0)[j_scalar] * GaussianKernel1D_F[0];
            sum += paddedTempImage.ptr<float>(i + 1)[j_scalar] * GaussianKernel1D_F[1];
            sum += paddedTempImage.ptr<float>(i + 2)[j_scalar] * GaussianKernel1D_F[2];
            sum += paddedTempImage.ptr<float>(i + 3)[j_scalar] * GaussianKernel1D_F[3];
            sum += paddedTempImage.ptr<float>(i + 4)[j_scalar] * GaussianKernel1D_F[4];
            outRow[j_scalar] = saturate_cast<uchar>(sum);
        }
#endif
    }
}

// 改進的影像相減 - 使用SIMD指令
void subtractImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage)
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;
    
    // 檢查所有影像的記憶體是否連續
    if(inputImage1.isContinuous() && inputImage2.isContinuous() && outputImage.isContinuous()) {
        const size_t total = rows * cols;
        const uchar* in1 = inputImage1.ptr<uchar>(0);
        const uchar* in2 = inputImage2.ptr<uchar>(0);
        uchar* out = outputImage.ptr<uchar>(0);
        
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total; i += 16) {
            size_t remaining = min(size_t(16), total - i);
            
            // 對齊處理：SIMD 或 標量
            if(remaining == 16) {
                #ifdef __AVX2__
                // AVX2 實現 - 每次處理 16 個位元組
                __m128i a = _mm_loadu_si128((__m128i*)(in1 + i));
                __m128i b = _mm_loadu_si128((__m128i*)(in2 + i));
                __m128i result = _mm_subs_epu8(a, b); // 飽和減法，小於 0 的結果自動變為 0
                _mm_storeu_si128((__m128i*)(out + i), result);
                #else
                // 標量實現
                for(size_t j = 0; j < 16; ++j) {
                    out[i+j] = (in1[i+j] > in2[i+j]) ? (in1[i+j] - in2[i+j]) : 0;
                }
                #endif
            } else {
                // 處理剩餘的部分 (不足16個的部分)
                for(size_t j = 0; j < remaining; ++j) {
                    out[i+j] = (in1[i+j] > in2[i+j]) ? (in1[i+j] - in2[i+j]) : 0;
                }
            }
        }
    } else {
        // 不連續記憶體的實現
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* in1Row = inputImage1.ptr<uchar>(i);
            const uchar* in2Row = inputImage2.ptr<uchar>(i);
            uchar* outRow = outputImage.ptr<uchar>(i);
            
            #pragma omp simd
            for(int j = 0; j < cols; ++j) {
                outRow[j] = (in1Row[j] > in2Row[j]) ? (in1Row[j] - in2Row[j]) : 0;
            }
        }
    }
}

// 改進的銳化操作
void sharpenImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage, double weight)
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < rows; ++i) {
        const uchar* in1Row = inputImage1.ptr<uchar>(i);
        const uchar* in2Row = inputImage2.ptr<uchar>(i);
        uchar* outRow = outputImage.ptr<uchar>(i);
        
        // 嘗試向量化循環
        #pragma omp simd
        for(int j = 0; j < cols; ++j) {
            // 用 saturate_cast 確保結果在 [0,255] 之間
            int val = in1Row[j] + static_cast<int>(weight * in2Row[j]);
            outRow[j] = saturate_cast<uchar>(val);
        }
    }
}

// 改進的直方圖計算
void histogramCalcParOpt(const Mat& inputImage, unsigned int imHistogram[], int nThreads)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    
    // 使用 std::vector 安全地處理執行緒本地直方圖
    std::vector<std::vector<unsigned int>> localHistData(nThreads, std::vector<unsigned int>(256, 0));
    
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        // localHistData[tid] 是此執行緒的直方圖，已經被初始化為0
        
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            
            // 使用向量化循環處理每行
            for(int j = 0; j < cols; ++j) {
                localHistData[tid][row[j]]++;
            }
        }
    }
    
    // 合併所有線程的直方圖
    memset(imHistogram, 0, 256 * sizeof(unsigned int));
    for(int t = 0; t < nThreads; ++t) {
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHistData[t][j];
        }
        // std::vector 會自動管理記憶體，不需要手動 delete
    }
}

// 優化的直方圖均衡化
void histogramEqualParOpt(const Mat& inputImage, Mat& outputImage, const unsigned int imHistogram[])
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int totalPixels = rows * cols;
    
    // 使用一次性遍歷計算 CDF (累積分布函數)
    int cdf[256] = {0};
    cdf[0] = imHistogram[0];
    
    for(int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + imHistogram[i];
    }
    
    // 建立查找表 (LUT)
    uchar lut[256];
    float scale = 255.0f / totalPixels;
    
    #pragma omp parallel for simd schedule(static)
    for(int i = 0; i < 256; ++i) {
        lut[i] = saturate_cast<uchar>(scale * cdf[i]);
    }
    
    // 並行處理每行
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < rows; ++i) {
        const uchar* inRow = inputImage.ptr<uchar>(i);
        uchar* outRow = outputImage.ptr<uchar>(i);
        
        // 使用向量化循環
        #pragma omp simd
        for(int j = 0; j < cols; ++j) {
            outRow[j] = lut[inRow[j]];
        }
    }
}

// 實現合併的直方圖計算和均衡化函數
void histogramCalcAndEqualParOpt(const Mat& inputImage, Mat& outputImage, int nThreads)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int totalPixels = rows * cols;
    
    // 1. 使用 std::vector 安全地處理執行緒本地直方圖
    std::vector<std::vector<unsigned int>> localHistData(nThreads, std::vector<unsigned int>(256, 0));
    
    // 2. 並行計算直方圖
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        // localHistData[tid] 是此執行緒的直方圖，已經被初始化為0
        
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            
            for(int j = 0; j < cols; ++j) {
                localHistData[tid][row[j]]++;
            }
        }
    }
    
    // 3. 合併所有線程的直方圖到此函數的局部 imHistogram
    unsigned int imHistogram[256] = {0};
    for(int t = 0; t < nThreads; ++t) {
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHistData[t][j];
        }
        // std::vector 會自動管理記憶體，不需要手動 delete
    }
    
    // 4. 計算 CDF (累積分布函數)
    int cdf[256] = {0};
    cdf[0] = imHistogram[0];
    
    for(int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + imHistogram[i];
    }
    
    // 5. 建立查找表 (LUT)
    uchar lut[256];
    float scale = 255.0f / totalPixels;
    
    #pragma omp parallel for simd schedule(static)
    for(int i = 0; i < 256; ++i) {
        lut[i] = saturate_cast<uchar>(scale * cdf[i]);
    }
    
    // 6. 應用 LUT 到輸出影像
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < rows; ++i) {
        const uchar* inRow = inputImage.ptr<uchar>(i);
        uchar* outRow = outputImage.ptr<uchar>(i);
        
        #pragma omp simd
        for(int j = 0; j < cols; ++j) {
            outRow[j] = lut[inRow[j]];
        }
    }
}

// 大幅優化的 HSV 到 RGB 轉換
void hsvToRgbParOpt(const Mat& inputImage, Mat& outputImage, int blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const float inv255_scalar = 1.0f / 255.0f;
    const float inv60_scalar = 1.0f / 60.0f;
    
    #pragma omp parallel
    {
        #ifdef __AVX2__
        const __m256 _zero_ps = _mm256_setzero_ps();
        const __m256 _one_ps = _mm256_set1_ps(1.0f);
        const __m256 _255_ps = _mm256_set1_ps(255.0f);
        const __m256 _inv255_ps = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 _two_ps = _mm256_set1_ps(2.0f);
        const __m256 _inv60_ps = _mm256_set1_ps(1.0f / 60.0f);
        const __m256 _360_ps = _mm256_set1_ps(360.0f);
        const __m256 _epsilon_ps = _mm256_set1_ps(1e-6f); 
        const __m128i _zero_xmm = _mm_setzero_si128();
        
        // For floor operation (alternative to _mm256_floor_ps if not available by default with AVX2 includes)
        // const __m256 _round_abs_mask = _mm256_set1_ps(-0.0f); // Used with _mm256_round_ps for floor
        #endif

        #pragma omp for schedule(guided)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);
                
                for(int i = ii; i < blockEndI; ++i) {
                    const Vec3b* inRow = inputImage.ptr<Vec3b>(i);
                    Vec3b* outRow = outputImage.ptr<Vec3b>(i);
                    int j = jj;

#ifdef __AVX2__
                    for (; j <= blockEndJ - 8; j += 8) {
                        unsigned char h_tmp[8], s_tmp[8], v_tmp[8];
                        for (int k_load = 0; k_load < 8; ++k_load) {
                            h_tmp[k_load] = inRow[j + k_load][0];
                            s_tmp[k_load] = inRow[j + k_load][1];
                            v_tmp[k_load] = inRow[j + k_load][2];
                        }

                        __m128i h_u8_xmm = _mm_loadl_epi64((__m128i*)h_tmp);
                        __m128i s_u8_xmm = _mm_loadl_epi64((__m128i*)s_tmp);
                        __m128i v_u8_xmm = _mm_loadl_epi64((__m128i*)v_tmp);

                        __m256i h_epi32 = _mm256_cvtepu8_epi32(h_u8_xmm);
                        __m256i s_epi32 = _mm256_cvtepu8_epi32(s_u8_xmm);
                        __m256i v_epi32 = _mm256_cvtepu8_epi32(v_u8_xmm);

                        __m256 h_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(h_epi32), _two_ps); // H * 2.0 (0-360 range)
                        __m256 s_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(s_epi32), _inv255_ps); // S (0-1 range)
                        __m256 v_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(v_epi32), _inv255_ps); // V (0-1 range)

                        __m256 r_ps, g_ps, b_ps;

                        // Mask for S <= epsilon (saturation is zero or very small)
                        __m256 s_is_zero_mask = _mm256_cmp_ps(s_ps, _epsilon_ps, _CMP_LE_OQ);
                        
                        // Case: S is zero (grayscale) -> R=G=B=V
                        __m256 r_gray = v_ps; __m256 g_gray = v_ps; __m256 b_gray = v_ps;

                        // Case: S is not zero (color)
                        // Ensure H is in [0, 360)
                        __m256 h_ge_360_mask = _mm256_cmp_ps(h_ps, _360_ps, _CMP_GE_OQ);
                        h_ps = _mm256_blendv_ps(h_ps, _zero_ps, h_ge_360_mask); // if h >= 360, h = 0
                        h_ps = _mm256_mul_ps(h_ps, _inv60_ps); // h_ps = h_ps / 60.0f; -> sector [0, 6)
                        
                        // hi = floor(h_ps)
                        // AVX2 does not have a direct _mm256_floor_ps, need to emulate or use _mm256_round_ps with specific mode if available.
                        // A common way for floor with AVX (or if _mm256_floor_ps is not reliably available/performant):
                        // Convert to int (truncates), then back to float. If negative and different, subtract 1.
                        // Or, for positive numbers, truncate is floor.
                        // Since h_ps is [0,6), truncate is fine for floor.
                        __m256i hi_epi32 = _mm256_cvttps_epi32(h_ps); // Truncate to get integer part
                        // __m256 hi_ps = _mm256_cvtepi32_ps(hi_epi32); // hi as float if needed later, usually not
                        
                        __m256 f_ps = _mm256_sub_ps(h_ps, _mm256_cvtepi32_ps(hi_epi32)); // f = h - hi (fractional part)

                        __m256 p_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, s_ps));
                        __m256 q_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_ps, f_ps)));
                        __m256 t_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_ps, _mm256_sub_ps(_one_ps, f_ps))));

                        // Logic for the switch statement (hi is int, from 0 to 5)
                        // We need to compare hi_epi32 with 0, 1, 2, 3, 4, 5
                        // This is the most complex part to vectorize directly with good performance.
                        // It often involves calculating all branches and blending.

                        // Pre-calculate R,G,B for all 6 cases for each of the 8 float elements
                        __m256 r_case0 = v_ps, g_case0 = t_ps, b_case0 = p_ps;
                        __m256 r_case1 = q_ps, g_case1 = v_ps, b_case1 = p_ps;
                        __m256 r_case2 = p_ps, g_case2 = v_ps, b_case2 = t_ps;
                        __m256 r_case3 = p_ps, g_case3 = q_ps, b_case3 = v_ps;
                        __m256 r_case4 = t_ps, g_case4 = p_ps, b_case4 = v_ps;
                        __m256 r_case5 = v_ps, g_case5 = p_ps, b_case5 = q_ps; // default case also hi=5

                        // Create masks for hi values
                        __m256i hi_eq_0 = _mm256_cmpeq_epi32(hi_epi32, _mm256_setzero_si256());
                        __m256i hi_eq_1 = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(1));
                        __m256i hi_eq_2 = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(2));
                        __m256i hi_eq_3 = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(3));
                        __m256i hi_eq_4 = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(4));
                        // hi_eq_5 can be the remainder (not 0,1,2,3,4)

                        // Blend R values. Start with case 5 (default)
                        r_ps = r_case5;
                        r_ps = _mm256_blendv_ps(r_ps, r_case4, (__m256)hi_eq_4);
                        r_ps = _mm256_blendv_ps(r_ps, r_case3, (__m256)hi_eq_3);
                        r_ps = _mm256_blendv_ps(r_ps, r_case2, (__m256)hi_eq_2);
                        r_ps = _mm256_blendv_ps(r_ps, r_case1, (__m256)hi_eq_1);
                        r_ps = _mm256_blendv_ps(r_ps, r_case0, (__m256)hi_eq_0);
                        
                        // Blend G values
                        g_ps = g_case5;
                        g_ps = _mm256_blendv_ps(g_ps, g_case4, (__m256)hi_eq_4);
                        g_ps = _mm256_blendv_ps(g_ps, g_case3, (__m256)hi_eq_3);
                        g_ps = _mm256_blendv_ps(g_ps, g_case2, (__m256)hi_eq_2);
                        g_ps = _mm256_blendv_ps(g_ps, g_case1, (__m256)hi_eq_1);
                        g_ps = _mm256_blendv_ps(g_ps, g_case0, (__m256)hi_eq_0);

                        // Blend B values
                        b_ps = b_case5;
                        b_ps = _mm256_blendv_ps(b_ps, b_case4, (__m256)hi_eq_4);
                        b_ps = _mm256_blendv_ps(b_ps, b_case3, (__m256)hi_eq_3);
                        b_ps = _mm256_blendv_ps(b_ps, b_case2, (__m256)hi_eq_2);
                        b_ps = _mm256_blendv_ps(b_ps, b_case1, (__m256)hi_eq_1);
                        b_ps = _mm256_blendv_ps(b_ps, b_case0, (__m256)hi_eq_0);
                        
                        // Final selection: if S was zero, use grayscale values
                        r_ps = _mm256_blendv_ps(r_ps, r_gray, s_is_zero_mask);
                        g_ps = _mm256_blendv_ps(g_ps, g_gray, s_is_zero_mask);
                        b_ps = _mm256_blendv_ps(b_ps, b_gray, s_is_zero_mask);

                        // Scale to 0-255
                        r_ps = _mm256_mul_ps(r_ps, _255_ps);
                        g_ps = _mm256_mul_ps(g_ps, _255_ps);
                        b_ps = _mm256_mul_ps(b_ps, _255_ps);

                        // Convert to uchar and store (BGR order for OpenCV)
                        __m256i r_epi32_final = _mm256_cvttps_epi32(r_ps);
                        __m256i g_epi32_final = _mm256_cvttps_epi32(g_ps);
                        __m256i b_epi32_final = _mm256_cvttps_epi32(b_ps);

                        __m128i r_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(r_epi32_final), _mm256_extracti128_si256(r_epi32_final, 1));
                        __m128i g_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(g_epi32_final), _mm256_extracti128_si256(g_epi32_final, 1));
                        __m128i b_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(b_epi32_final), _mm256_extracti128_si256(b_epi32_final, 1));

                        __m128i r_u8_xmm = _mm_packus_epi16(r_u16_xmm, _zero_xmm);
                        __m128i g_u8_xmm = _mm_packus_epi16(g_u16_xmm, _zero_xmm);
                        __m128i b_u8_xmm = _mm_packus_epi16(b_u16_xmm, _zero_xmm);

                        unsigned char r_u[8], g_u[8], b_u[8];
                        _mm_storel_epi64((__m128i*)r_u, r_u8_xmm);
                        _mm_storel_epi64((__m128i*)g_u, g_u8_xmm);
                        _mm_storel_epi64((__m128i*)b_u, b_u8_xmm);

                        for (int k_store = 0; k_store < 8; ++k_store) {
                            outRow[j + k_store][0] = b_u[k_store]; // Store B
                            outRow[j + k_store][1] = g_u[k_store]; // Store G
                            outRow[j + k_store][2] = r_u[k_store]; // Store R
                        }
                    }
#endif
                    // Scalar loop for remainder
                    for (; j < blockEndJ; ++j) {
                        float h_s = inRow[j][0] * 2.0f;
                        float s_s = inRow[j][1] * inv255_scalar;
                        float v_s = inRow[j][2] * inv255_scalar;
                        
                        float r_f, g_f, b_f;
                        
                        if(s_s <= 1e-6f) {
                            r_f = g_f = b_f = v_s;
                        } else {
                            if (h_s >= 360.0f) h_s = 0.0f;
                            h_s *= inv60_scalar;
                            int hi_s = static_cast<int>(floor(h_s));
                            float f_s = h_s - hi_s;
                            float p_s = v_s * (1.0f - s_s);
                            float q_s = v_s * (1.0f - s_s * f_s);
                            float t_s = v_s * (1.0f - s_s * (1.0f - f_s));
                            switch(hi_s) {
                                case 0: r_f = v_s; g_f = t_s; b_f = p_s; break;
                                case 1: r_f = q_s; g_f = v_s; b_f = p_s; break;
                                case 2: r_f = p_s; g_f = v_s; b_f = t_s; break;
                                case 3: r_f = p_s; g_f = q_s; b_f = v_s; break;
                                case 4: r_f = t_s; g_f = p_s; b_f = v_s; break;
                                default:r_f = v_s; g_f = p_s; b_f = q_s; break;
                            }
                        }
                        outRow[j][0] = saturate_cast<uchar>(b_f * 255.0f);
                        outRow[j][1] = saturate_cast<uchar>(g_f * 255.0f);
                        outRow[j][2] = saturate_cast<uchar>(r_f * 255.0f);
                    }
                }
            }
        }
    }
}

// 獲取當前執行的路徑
string getexepath()
{
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return string(result, (count > 0) ? count : 0);
}