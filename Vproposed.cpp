/*
優化版本 - Singh et. al 圖像增強演算法實現
https://ieeexplore.ieee.org/document/8071892

編譯命令:
g++ -g -Wall -Ofast -flto -o Vproposed Vproposed.cpp `pkg-config --cflags --libs opencv4` -fopenmp -march=native -funroll-loops -ftree-vectorize

執行方式:
./Vproposed <imagename> <threads> <size>
例如: ./Vproposed chaewon.png 4 778x1036
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
#include <algorithm> // Required for std::min

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
void subtractImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage, int blockSize);
void sharpenImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage, double weight);
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
        cout << "Usage: ./Vproposed <imagename> [threads] [size]" << endl;
        cout << "Examples:" << endl;
        cout << "  ./Vproposed snow.png                  - Run with default threads (all) and original size" << endl;
        cout << "  ./Vproposed snow.png 4                - Run with 4 threads and original size" << endl;
        cout << "  ./Vproposed snow.png 4 778x1036       - Run with 4 threads and resize to 778x1036" << endl;
        exit(EXIT_FAILURE);
    }

    int initial_omp_max_threads = omp_get_max_threads();
    int numThreads = initial_omp_max_threads; 
    
    // Argument parsing logic (simplified for brevity, assuming it's correct from previous version)
    string size_arg_str; // To store the size argument if present
    if (argc >= 3) {
        string arg2 = argv[2];
        bool arg2_is_numeric = !arg2.empty() && arg2.find_first_not_of("0123456789") == string::npos;

        if (arg2_is_numeric) {
            int requested_threads = stoi(arg2);
            if (requested_threads > 0) {
                numThreads = requested_threads;
            }
            if (argc >= 4) { // Size is argv[3]
                size_arg_str = argv[3];
            }
        } else { // argv[2] is likely size
            size_arg_str = arg2;
            // numThreads remains default
        }
    }
    omp_set_num_threads(numThreads);


    Mat originalImage = imread(argv[1], IMREAD_UNCHANGED);
    if (!isImageValid(originalImage)) {
        cerr << "Error: Could not read image " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }
    
    Mat inputImage = originalImage;
    string sizeStr = "original";
    
    if (!size_arg_str.empty()) {
        int width, height;
        if (parseSize(size_arg_str, width, height)) {
            sizeStr = size_arg_str;
            resize(originalImage, inputImage, Size(width, height));
            // cout << "Resized input image to " << width << "x" << height << endl;
        } else {
            // cout << "Invalid size format '" << size_arg_str << "', using original size." << endl;
        }
    }
    // if (sizeStr == "original") {
    //     cout << "Using original size: " << inputImage.cols << "x" << inputImage.rows << endl;
    // }


    int optimalBlockSize = calculateOptimalBlockSize(inputImage);
    
    cout << "[設定] 使用執行緒數: " << numThreads << endl;
    cout << "[設定] 使用 BlockSize: " << optimalBlockSize << endl;

    string filename_str = argv[1];
    filesystem::path imagePath(filename_str);
    string basename = imagePath.stem().string();
    string extension = imagePath.extension().string();
    
    filesystem::create_directory("results");
    
    string outimagename = "results/" + basename + "_" + sizeStr + "_Vproposed_t" + to_string(numThreads) + extension;
    string resultpath = "results/" + basename + "_" + sizeStr + "_Vproposed_t" + to_string(numThreads) + ".csv";
    ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,BlockSize,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << endl;
    
    Mat inputImageHsv(inputImage.size(), inputImage.type());
    Mat blurredImage(inputImage.size(), CV_8UC1); 
    Mat imageMask(inputImage.size(), CV_8UC1);
    Mat sharpenedImage(inputImage.size(), CV_8UC1);
    Mat globallyEnhancedImage(inputImage.size(), CV_8UC1);
    Mat outputHSV(inputImage.size(), inputImage.type());
    Mat finalOutput(inputImage.size(), inputImage.type());
    
    vector<Mat> inputImageHsvChannels(3);
    vector<Mat> mergedChannels(3); 
    
    auto start_time = chrono::high_resolution_clock::now(); 
    auto end_time = chrono::high_resolution_clock::now(); 
    auto timeEllapRGBtoHSV = chrono::duration<double, milli>::zero();
    auto timeEllapBlur = chrono::duration<double, milli>::zero();
    auto timeEllapSubtract = chrono::duration<double, milli>::zero();
    auto timeEllapSharpen = chrono::duration<double, milli>::zero();
    auto timeEllapHSVtoRGB = chrono::duration<double, milli>::zero();
    auto timeEllapHistCombined = chrono::duration<double, milli>::zero();
    
    const int warmupIter = 3; 
    const int numIter = 10;   

    // =========== STEP 1: RGB到HSV轉換 ===========
    for(int i = 0; i < warmupIter; ++i) {
        rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
        end_time = chrono::high_resolution_clock::now();
        timeEllapRGBtoHSV += (end_time - start_time);
    }
    timeEllapRGBtoHSV /= numIter;
    cout << "RGB to HSV Conversion: " << timeEllapRGBtoHSV.count() << " ms" << endl;

    split(inputImageHsv, inputImageHsvChannels);
    Mat& inputImageH = inputImageHsvChannels[0];
    Mat& inputImageS = inputImageHsvChannels[1];
    Mat& inputImageV = inputImageHsvChannels[2];

    // =========== STEP 2-1: 圖像模糊 ===========
    for(int i = 0; i < warmupIter; ++i) {
        imageBlurParOpt(inputImageV, blurredImage);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        imageBlurParOpt(inputImageV, blurredImage);
        end_time = chrono::high_resolution_clock::now();
        timeEllapBlur += (end_time - start_time);
    }
    timeEllapBlur /= numIter;
    cout << "Image Blur: " << timeEllapBlur.count() << " ms" << endl;

    // =========== STEP 2-2: 圖像相減 ===========
    for(int i = 0; i < warmupIter; ++i) {
        subtractImageParOpt(inputImageV, blurredImage, imageMask, optimalBlockSize);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        subtractImageParOpt(inputImageV, blurredImage, imageMask, optimalBlockSize);
        end_time = chrono::high_resolution_clock::now();
        timeEllapSubtract += (end_time - start_time);
    }
    timeEllapSubtract /= numIter;
    cout << "Image Subtraction: " << timeEllapSubtract.count() << " ms" << endl;

    // =========== STEP 2-3: 圖像銳化 ===========
    double weight = 10.0;
    for(int i = 0; i < warmupIter; ++i) {
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
        end_time = chrono::high_resolution_clock::now();
        timeEllapSharpen += (end_time - start_time);
    }
    timeEllapSharpen /= numIter;
    cout << "Image Sharpening: " << timeEllapSharpen.count() << " ms" << endl;

    // =========== STEP 3: 直方圖處理 ===========
    for(int i = 0; i < warmupIter; ++i) {
        histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, numThreads);
    }
    #pragma omp barrier

    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, numThreads);
        end_time = chrono::high_resolution_clock::now();
        timeEllapHistCombined += (end_time - start_time);
    }
    timeEllapHistCombined /= numIter;
    cout << "Combined Histogram Processing: " << timeEllapHistCombined.count() << " ms" << endl;

    mergedChannels[0] = inputImageH;
    mergedChannels[1] = inputImageS;
    mergedChannels[2] = globallyEnhancedImage;
    merge(mergedChannels, outputHSV);
    
    // =========== STEP 4: HSV到RGB轉換 ===========
    for(int i = 0; i < warmupIter; ++i) {
        hsvToRgbParOpt(outputHSV, finalOutput, optimalBlockSize);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        hsvToRgbParOpt(outputHSV, finalOutput, optimalBlockSize);
        end_time = chrono::high_resolution_clock::now();
        timeEllapHSVtoRGB += (end_time - start_time);
    }
    timeEllapHSVtoRGB /= numIter;
    cout << "HSV to RGB Conversion: " << timeEllapHSVtoRGB.count() << " ms" << endl;

    imwrite(outimagename, finalOutput);
    cout << "增強後的圖像已儲存為: " << outimagename << endl;
    
    auto totalTime = timeEllapRGBtoHSV + timeEllapBlur + timeEllapSubtract + timeEllapSharpen + timeEllapHistCombined + timeEllapHSVtoRGB;
    cout << "Total Processing Time: " << totalTime.count() << " ms" << endl;

    outFile << basename << "," 
           << inputImage.cols << "x" << inputImage.rows << "," 
           << numThreads << "," 
           << optimalBlockSize << ","
           << timeEllapRGBtoHSV.count() << "," 
           << timeEllapBlur.count() << "," 
           << timeEllapSubtract.count() << "," 
           << timeEllapSharpen.count() << "," 
           << timeEllapHistCombined.count() << "," 
           << timeEllapHSVtoRGB.count() << "," 
           << totalTime.count() << endl;

    outFile.close();
    return 0;
}

// rgbToHsvParOpt 函數：將 RGB 圖像轉換為 HSV 色彩空間的並行優化版本
// (Comments and implementation as in the previous version, no changes here for this request)
/*
目標：將一張 RGB 彩色影像並行且向量化地轉換到 HSV 色彩空間。

核心優化：
    OpenMP 多執行緒：#pragma omp parallel + for collapse(2) 將整張影像切成多個 block 同時處理。
    AVX2 SIMD：一次同時計算 8 個像素（8 請注意 Vec3b 三通道）的 H/S/V，極大提升吞吐量。
    Block 分塊：避免巨量工作集影響快取命中率，將影像分成 blockSize × blockSize 小區塊依序處理。
    殘餘像素回落到純量計算：對不足 8 的尾端像素採純量（scalar）程式碼，確保正確性。
*/
void rgbToHsvParOpt(const Mat& inputImage,
                    Mat&       outputImage,
                    int        blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    // 常數：1/255，用來將 uchar [0..255] 正規化成 [0..1]
    const float inv255_scalar = 1.0f / 255.0f; 
    
    // 啟動 OpenMP 多線程區塊
    #pragma omp parallel
    {
    #ifdef __AVX2__
        // 以下都是 AVX2 用到的常數向量 (256-bit)，可一次裝 8 個 float
        const __m256 _zero_ps    = _mm256_setzero_ps();          // 全部元素 = 0.0f
        const __m256 _one_ps     = _mm256_set1_ps(1.0f);         // 全部元素 = 1.0f
        const __m256 _half_ps    = _mm256_set1_ps(0.5f);         // 全部元素 = 0.5f
        const __m256 _2_ps       = _mm256_set1_ps(2.0f);         // 全部元素 = 2.0f
        const __m256 _60_ps      = _mm256_set1_ps(60.0f);        // 全部元素 = 60.0f
        const __m256 _255_ps     = _mm256_set1_ps(255.0f);       // 全部元素 = 255.0f
        const __m256 _360_ps     = _mm256_set1_ps(360.0f);       // 全部元素 = 360.0f
        const __m256 _epsilon_ps = _mm256_set1_ps(1e-6f);        // 全部元素 = 1e-6f，用於防除以零
        const __m256 _inv255_ps  = _mm256_set1_ps(inv255_scalar);// 全部元素 = 1/255
        const __m128i _zero_xmm  = _mm_setzero_si128();          // 128-bit 全 0，用在 pack 時塞空白
        const __m256 _4_ps     = _mm256_set1_ps(4.0f);  // 用於 B 通道主導的色調計算
    #endif

        // collapse(2)：OpenMP 針對 ii / jj 兩層迴圈一起做分段並行，guided 均衡負載
        #pragma omp for collapse(2) schedule(guided)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);

                // 逐行遍歷這個 block
                for(int i = ii; i < blockEndI; ++i) {
                    const Vec3b* inRow  = inputImage.ptr<Vec3b>(i);  // 取得第 i 行指標
                    Vec3b*       outRow = outputImage.ptr<Vec3b>(i);// 輸出行指標

                    int j = jj;
    #ifdef __AVX2__
                    // —— AVX2 主迴圈：一次處理 8 個像素，共 8 個 Vec3b —— 
                    for (; j <= blockEndJ - 8; j += 8) {
                        // 1) 暫存原始 BGR 值到暫存陣列
                        unsigned char b_tmp[8], g_tmp[8], r_tmp[8];
                        for(int k = 0; k < 8; ++k) {
                            b_tmp[k] = inRow[j+k][0];
                            g_tmp[k] = inRow[j+k][1];
                            r_tmp[k] = inRow[j+k][2];
                        }

                        // 2) 將 8 個 uchar → 128-bit SSE register
                        //    _mm_loadl_epi64: 從記憶體讀 64 bits (8×uchar) into lower half of 128-bit
                        __m128i b_u8 = _mm_loadl_epi64((__m128i*)b_tmp);
                        __m128i g_u8 = _mm_loadl_epi64((__m128i*)g_tmp);
                        __m128i r_u8 = _mm_loadl_epi64((__m128i*)r_tmp);

                        // 3) 轉成 256-bit 整數向量 (zero-extend uchar → int32)
                        //    _mm256_cvtepu8_epi32: 8 × uint8 → 8 × int32
                        __m256i b_i32 = _mm256_cvtepu8_epi32(b_u8);
                        __m256i g_i32 = _mm256_cvtepu8_epi32(g_u8);
                        __m256i r_i32 = _mm256_cvtepu8_epi32(r_u8);

                        // 4) 再轉成 256-bit 浮點向量並正規化到 [0,1]
                        //    _mm256_cvtepi32_ps: 8 × int32 → 8 × float
                        //    _mm256_mul_ps: 對應元素乘以 inv255
                        __m256 b_f = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i32), _inv255_ps);
                        __m256 g_f = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i32), _inv255_ps);
                        __m256 r_f = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i32), _inv255_ps);

                        // 5) c_max = max(r,g,b), c_min = min(r,g,b), δ = c_max - c_min
                        __m256 cmax  = _mm256_max_ps(r_f, _mm256_max_ps(g_f, b_f));
                        __m256 cmin  = _mm256_min_ps(r_f, _mm256_min_ps(g_f, b_f));
                        __m256 delta = _mm256_sub_ps(cmax, cmin);

                        // 6) V = c_max
                        __m256 v_ps = cmax;

                        // 7) S = δ/c_max，若 c_max≈0 或 δ≈0，則 S=0
                        __m256 s_ps = _mm256_div_ps(delta, cmax);
                        //    _mm256_cmp_ps: 比較產生 mask
                        //    _mm256_or_ps: 合併 cmax<eps 或 delta<eps
                        //    _mm256_blendv_ps: mask 為 true 時拿第二個參數 (_zero_ps)
                        s_ps = _mm256_blendv_ps(
                                  s_ps, _zero_ps,
                                  _mm256_or_ps(
                                    _mm256_cmp_ps(cmax, _epsilon_ps, _CMP_LT_OQ),
                                    _mm256_cmp_ps(delta, _epsilon_ps, _CMP_LT_OQ)
                                  )
                               );

                        // 8) 計算 H 前置：invδ = (δ<ε ? 0 : 1/δ)
                        __m256 inv_delta = _mm256_blendv_ps(
                                              _mm256_div_ps(_one_ps, delta), // 1/δ
                                              _zero_ps,                       // 否則 0
                                              _mm256_cmp_ps(delta, _epsilon_ps, _CMP_LT_OQ)
                                           );

                        // 9) 根據 c_max 選不同公式計算 Hbase
                        __m256 mask_r = _mm256_cmp_ps(r_f, cmax, _CMP_EQ_OQ);
                        __m256 mask_g = _mm256_andnot_ps(mask_r,
                                         _mm256_cmp_ps(g_f, cmax, _CMP_EQ_OQ));
                        //    if cmax==r: (g - b)*invδ
                        __m256 h_r = _mm256_mul_ps(_mm256_sub_ps(g_f, b_f), inv_delta);
                        //    if cmax==g: 2 + (b - r)*invδ
                        __m256 h_g = _mm256_add_ps(_2_ps,
                                       _mm256_mul_ps(_mm256_sub_ps(b_f, r_f), inv_delta));
                        //    if cmax==b: 4 + (r - g)*invδ
                        __m256 h_b = _mm256_add_ps(_4_ps,
                                       _mm256_mul_ps(_mm256_sub_ps(r_f, g_f), inv_delta));
                        //    _mm256_blendv_ps: 根據 mask_g, mask_r 選擇分支
                        __m256 h_ps = _mm256_blendv_ps(
                                        _mm256_blendv_ps(h_b, h_g, mask_g),
                                        h_r, mask_r
                                      );

                        // 10) 若 δ≈0，則強制 H=0
                        h_ps = _mm256_blendv_ps(
                                 h_ps, _zero_ps,
                                 _mm256_cmp_ps(delta, _epsilon_ps, _CMP_LT_OQ)
                               );

                        // 11) *60°
                        h_ps = _mm256_mul_ps(h_ps, _60_ps);
                        //     <0 則 +360°
                        __m256 neg = _mm256_cmp_ps(h_ps, _zero_ps, _CMP_LT_OQ);
                        h_ps = _mm256_add_ps(h_ps, _mm256_and_ps(neg, _360_ps));
                        //     最終存回時取 H/2（範圍縮到 [0..180]）
                        h_ps = _mm256_mul_ps(h_ps, _half_ps);

                        // 12) H/S/V pack→int→pack→uchar，寫回 outputImage
                        __m256i h_i = _mm256_cvttps_epi32(h_ps);
                        __m256i s_i = _mm256_cvttps_epi32(_mm256_mul_ps(s_ps, _255_ps));
                        __m256i v_i = _mm256_cvttps_epi32(_mm256_mul_ps(v_ps, _255_ps));

                        //    _mm256_cvttps_epi32: 8×float → 8×int32 (truncate)
                        //    _mm256_castsi256_si128 + _mm256_extracti128_si256:
                        //      取出前後半段 128-bit
                        //    _mm_packus_epi32: 32→16 bit pack, saturate
                        //    _mm_packus_epi16: 16→8 bit pack, saturate
                        __m128i h_p = _mm_packus_epi16(
                                        _mm_packus_epi32(
                                          _mm256_castsi256_si128(h_i),
                                          _mm256_extracti128_si256(h_i,1)
                                        ), _zero_xmm);
                        __m128i s_p = _mm_packus_epi16(
                                        _mm_packus_epi32(
                                          _mm256_castsi256_si128(s_i),
                                          _mm256_extracti128_si256(s_i,1)
                                        ), _zero_xmm);
                        __m128i v_p = _mm_packus_epi16(
                                        _mm_packus_epi32(
                                          _mm256_castsi256_si128(v_i),
                                          _mm256_extracti128_si256(v_i,1)
                                        ), _zero_xmm);

                        // 最後把 8 個 uchar 元素從 128-bit register 寫到陣列
                        unsigned char h_u[8], s_u[8], v_u[8];
                        _mm_storel_epi64((__m128i*)h_u, h_p);
                        _mm_storel_epi64((__m128i*)s_u, s_p);
                        _mm_storel_epi64((__m128i*)v_u, v_p);

                        for(int k=0; k<8; ++k) {
                            outRow[j+k][0] = h_u[k];
                            outRow[j+k][1] = s_u[k];
                            outRow[j+k][2] = v_u[k];
                        }
                    }
    #endif
                    // —— 純量回落：處理不足 8 個尾端像素 —— 
                    for(; j < blockEndJ; ++j) {
                        uchar B = inRow[j][0], G = inRow[j][1], R = inRow[j][2];
                        float r = R * inv255_scalar,
                              g = G * inv255_scalar,
                              b = B * inv255_scalar;
                        // c_max, c_min, δ
                        float cmax = max({r,g,b}),
                              cmin = min({r,g,b}),
                              d    = cmax - cmin;
                        float H=0, S=0, V=cmax;
                        if(d > 1e-6f) {
                            S = d / cmax;  // S=δ/cmax
                            if   (fabs(cmax-r)<1e-6f)      H=(g-b)/d;
                            else if(fabs(cmax-g)<1e-6f)    H=2+(b-r)/d;
                            else                           H=4+(r-g)/d;
                            H = H*60; if(H<0) H+=360; H*=0.5f;
                        }
                        outRow[j][0] = saturate_cast<uchar>(H);
                        outRow[j][1] = saturate_cast<uchar>(S * 255.0f);
                        outRow[j][2] = saturate_cast<uchar>(V * 255.0f);
                    }
                }
            }
        }
    } // omp parallel end
}



#ifdef __AVX2__
// 將 8 個 uchar 轉成 8 個 float，並一次載入成 256-bit AVX 向量
// 參數 p 指向 8 個連續的 unsigned char 像素強度值
static inline __m256 load_and_convert_uchar_to_float_avx2(const unsigned char* p) {
    // _mm_loadl_epi64: 從 p 載入 64 bits (8×uchar) 到 128-bit SSE register 低半部
    __m128i u8_vec = _mm_loadl_epi64((const __m128i*)p);
    // _mm256_cvtepu8_epi32: 將 8 個無符號 8-bit 擴展為 8 個 32-bit 整數
    __m256i i32_vec = _mm256_cvtepu8_epi32(u8_vec);
    // _mm256_cvtepi32_ps: 將 8 個 32-bit 整數轉為 8 個 32-bit 浮點
    return _mm256_cvtepi32_ps(i32_vec);
}
#endif

// imageBlurParOpt 函數：對單通道影像做 5×5 高斯模糊 (分離為水平 + 垂直兩階段)，使用 AVX2 + OpenMP 加速
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage)
{
    const int rows     = inputImage.rows;
    const int cols     = inputImage.cols;
    const int halfSize = FILTERSIZE / 2; 

    // 一維 Gaussian kernel 權重 (float) 並對齊到 64 bytes，方便向量化存取
    alignas(64) const float GaussianKernel1D_F[FILTERSIZE] = {
        1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f
    };

    // 暫存中介結果：水平模糊後的浮點圖
    Mat tempImage(rows, cols, CV_32FC1);

    // --- 水平遍歷階段 (Separable Gaussian Row Convolution) ---
    // 對左右邊界加反射邊界 (BORDER_REFLECT_101) 以便水平濾波時不出界
    Mat paddedInputHorizontal;
    copyMakeBorder(inputImage, paddedInputHorizontal,
                   0, 0, halfSize, halfSize,
                   BORDER_REFLECT_101);

    // OpenMP 並行處理每一行
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uchar* srcRow  = paddedInputHorizontal.ptr<uchar>(i);  // 取得第 i 行 + 邊界後的 uchar 指標
        float*       tempRow = tempImage.ptr<float>(i);             // 取得第 i 行的暫存結果指標
        
        int j = 0;
    #ifdef __AVX2__
        // 事先載入 kernel 權重到 256-bit 向量，可一次處理 8 個 float
        const __m256 gk0_v = _mm256_set1_ps(GaussianKernel1D_F[0]);
        const __m256 gk1_v = _mm256_set1_ps(GaussianKernel1D_F[1]);
        const __m256 gk2_v = _mm256_set1_ps(GaussianKernel1D_F[2]);
        const __m256 gk3_v = _mm256_set1_ps(GaussianKernel1D_F[3]);
        const __m256 gk4_v = _mm256_set1_ps(GaussianKernel1D_F[4]);

        // SIMD 展開：一次處理 16 個像素（拆成兩組各 8 個）
        // 目標是加速水平捲積，每次讀取 5 個相鄰位置並乘以對應權重
        for (; j <= cols - 16; j += 16) {
            // 第一組 8 像素
            __m256 v0_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 0);
            __m256 v1_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 1);
            __m256 v2_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 2);
            __m256 v3_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 3);
            __m256 v4_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 4);
            // _mm256_mul_ps + _mm256_fmadd_ps: 向量化乘法與累加
            __m256 sum0 = _mm256_mul_ps(v0_0, gk0_v);
            sum0 = _mm256_fmadd_ps(v1_0, gk1_v, sum0);
            sum0 = _mm256_fmadd_ps(v2_0, gk2_v, sum0);
            sum0 = _mm256_fmadd_ps(v3_0, gk3_v, sum0);
            sum0 = _mm256_fmadd_ps(v4_0, gk4_v, sum0);
            // _mm256_storeu_ps: 將 8 個浮點值直接寫回 tempRow
            _mm256_storeu_ps(tempRow + j, sum0);

            // 第二組 8 像素 (偏移量 +8)
            __m256 v0_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 0);
            __m256 v1_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 1);
            __m256 v2_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 2);
            __m256 v3_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 3);
            __m256 v4_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 4);
            __m256 sum1 = _mm256_mul_ps(v0_1, gk0_v);
            sum1 = _mm256_fmadd_ps(v1_1, gk1_v, sum1);
            sum1 = _mm256_fmadd_ps(v2_1, gk2_v, sum1);
            sum1 = _mm256_fmadd_ps(v3_1, gk3_v, sum1);
            sum1 = _mm256_fmadd_ps(v4_1, gk4_v, sum1);
            _mm256_storeu_ps(tempRow + j + 8, sum1);
        }
    #endif
        // 標量迴圈處理剩餘不足 16 的尾端像素
        for (; j < cols; ++j) {
            const uchar* p = srcRow + j;
            float sum = 0.0f;
            // 逐點乘以 kernel 權重
            sum += float(p[0]) * GaussianKernel1D_F[0];
            sum += float(p[1]) * GaussianKernel1D_F[1];
            sum += float(p[2]) * GaussianKernel1D_F[2];
            sum += float(p[3]) * GaussianKernel1D_F[3];
            sum += float(p[4]) * GaussianKernel1D_F[4];
            tempRow[j] = sum;
        }
    }

    // --- 垂直遍歷階段 (Separable Gaussian Column Convolution) ---
    // 對上下邊界進行相同類型的反射填充
    Mat paddedTempImage;
    copyMakeBorder(tempImage, paddedTempImage,
                   halfSize, halfSize, 0, 0,
                   BORDER_REFLECT_101);

    // OpenMP 並行處理每一行
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        uchar* outRow = outputImage.ptr<uchar>(i);

        int j = 0;
    #ifdef __AVX2__
        // 同樣載入 kernel 權重為 256-bit 向量
        const __m256 k0_v = _mm256_set1_ps(GaussianKernel1D_F[0]);
        const __m256 k1_v = _mm256_set1_ps(GaussianKernel1D_F[1]);
        const __m256 k2_v = _mm256_set1_ps(GaussianKernel1D_F[2]);
        const __m256 k3_v = _mm256_set1_ps(GaussianKernel1D_F[3]);
        const __m256 k4_v = _mm256_set1_ps(GaussianKernel1D_F[4]);
        const __m128i zero_xmm = _mm_setzero_si128();  // 用於 pack saturate

        // SIMD 展開：一次處理 16 像素 (2 組各 8)
        for (; j <= cols - 16; j += 16) {
            // 第一組 8 像素：讀取相鄰 5 行的浮點值
            __m256 r0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+0) + j);
            __m256 r1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+1) + j);
            __m256 r2 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+2) + j);
            __m256 r3 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+3) + j);
            __m256 r4 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+4) + j);
            // 進行垂直捲積：sum = Σ row[k]*kernel[k]
            __m256 s0 = _mm256_mul_ps(r0, k0_v);
            s0 = _mm256_fmadd_ps(r1, k1_v, s0);
            s0 = _mm256_fmadd_ps(r2, k2_v, s0);
            s0 = _mm256_fmadd_ps(r3, k3_v, s0);
            s0 = _mm256_fmadd_ps(r4, k4_v, s0);
            // 將 8 個 float → 8 個 uchar
            __m256i i32_0 = _mm256_cvttps_epi32(s0);  // truncate float→int32
            __m128i i16_0 = _mm_packus_epi32(
                               _mm256_castsi256_si128(i32_0),
                               _mm256_extracti128_si256(i32_0,1)
                           );                // pack 32→16
            __m128i i8_0 = _mm_packus_epi16(i16_0, zero_xmm);  // pack 16→8
            _mm_storel_epi64((__m128i*)(outRow + j), i8_0);// store 8 uchar

            // 第二組 8 像素 (偏移 +8)
            __m256 r0b = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+0) + j+8);
            __m256 r1b = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+1) + j+8);
            __m256 r2b = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+2) + j+8);
            __m256 r3b = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+3) + j+8);
            __m256 r4b = _mm256_loadu_ps(paddedTempImage.ptr<float>(i+4) + j+8);
            __m256 s1 = _mm256_mul_ps(r0b, k0_v);
            s1 = _mm256_fmadd_ps(r1b, k1_v, s1);
            s1 = _mm256_fmadd_ps(r2b, k2_v, s1);
            s1 = _mm256_fmadd_ps(r3b, k3_v, s1);
            s1 = _mm256_fmadd_ps(r4b, k4_v, s1);
            __m256i i32_1 = _mm256_cvttps_epi32(s1);
            __m128i i16_1 = _mm_packus_epi32(
                               _mm256_castsi256_si128(i32_1),
                               _mm256_extracti128_si256(i32_1,1)
                           );
            __m128i i8_1 = _mm_packus_epi16(i16_1, zero_xmm);
            _mm_storel_epi64((__m128i*)(outRow + j + 8), i8_1);
        }
    #endif
        // 處理尾端不足 16 的像素
        for (; j < cols; ++j) {
            float sum = 0.0f;
            sum += paddedTempImage.ptr<float>(i+0)[j] * GaussianKernel1D_F[0];
            sum += paddedTempImage.ptr<float>(i+1)[j] * GaussianKernel1D_F[1];
            sum += paddedTempImage.ptr<float>(i+2)[j] * GaussianKernel1D_F[2];
            sum += paddedTempImage.ptr<float>(i+3)[j] * GaussianKernel1D_F[3];
            sum += paddedTempImage.ptr<float>(i+4)[j] * GaussianKernel1D_F[4];
            outRow[j] = saturate_cast<uchar>(sum);
        }
    }
}


// subtractImageParOpt：並行且向量化地將兩張單通道 8-bit 影像做像素相減，結果飽和到 [0,255]
void subtractImageParOpt(const Mat& inputImage1,
                         const Mat& inputImage2,
                         Mat&       outputImage,
                         int        blockSize)  // 新增 blockSize 參數
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;

    // 若三張 Mat 資料連續 (isContinuous)，可一次性視作 1D 緩衝區處理
    if (inputImage1.isContinuous() &&
        inputImage2.isContinuous() &&
        outputImage.isContinuous()) {
        
        const size_t totalPixels = size_t(rows) * cols;
        const uchar* in1_ptr = inputImage1.ptr<uchar>(0);  // 指向開頭
        const uchar* in2_ptr = inputImage2.ptr<uchar>(0);
        uchar*       out_ptr = outputImage.ptr<uchar>(0);

        // OpenMP 並行，static 分配連續區段
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < totalPixels / blockSize * blockSize; i += blockSize) {
#ifdef __AVX2__
            // ------- AVX2 向量化區 -------
            // _mm256_loadu_si256: 讀取 32 bytes (32×uchar) 到 256-bit 向量
            __m256i a = _mm256_loadu_si256((__m256i*)(in1_ptr + i));
            __m256i b = _mm256_loadu_si256((__m256i*)(in2_ptr + i));
            // _mm256_subs_epu8: 無符號 saturating subtract：
            // 對應元素做 (a - b)，若結果 <0 就取 0；>255 自動裁切
            __m256i result = _mm256_subs_epu8(a, b);
            // _mm256_storeu_si256: 將 256-bit 結果寫回記憶體
            _mm256_storeu_si256((__m256i*)(out_ptr + i), result);
#else
            // 若不支援 AVX2，就退回到 blockSize 元素純量迴圈
            for (size_t k = 0; k < blockSize; ++k) {
                int diff = int(in1_ptr[i + k]) - int(in2_ptr[i + k]);
                // saturate_cast<uchar>：保證 clamp 到 [0,255]
                out_ptr[i + k] = saturate_cast<uchar>(diff);
            }
    #endif
        }

        // 處理剩餘像素
        for (size_t i = totalPixels / blockSize * blockSize; i < totalPixels; ++i) {
            int diff = int(in1_ptr[i]) - int(in2_ptr[i]);
            out_ptr[i] = saturate_cast<uchar>(diff);
        }

    } else {
        // 若資料非連續，逐行處理，每行再做 SIMD hint
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const uchar* in1Row = inputImage1.ptr<uchar>(i);
            const uchar* in2Row = inputImage2.ptr<uchar>(i);
            uchar*       outRow = outputImage.ptr<uchar>(i);
            // #pragma omp simd：提示編譯器為此迴圈生成向量化指令
            #pragma omp simd
            for (int j = 0; j < cols; ++j) {
                int diff = int(in1Row[j]) - int(in2Row[j]);
                outRow[j] = saturate_cast<uchar>(diff);
            }
        }
    }
}


// sharpenImageParOpt：對兩張單通道影像做高提升銳化 (high-boost filtering)，並使用 AVX2 + OpenMP 加速
// 輸入：
//   inputImage1 = 原始影像 V 通道 (uchar[0..255])
//   inputImage2 = 掩碼影像 M 通道 (uchar[0..255])
//   outputImage = 輸出銳化後影像 (uchar[0..255])
//   weight      = 高提升倍率 w
void sharpenImageParOpt(const Mat& inputImage1,
                        const Mat& inputImage2,
                        Mat&       outputImage,
                        double     weight)
{
    const int rows     = inputImage1.rows;
    const int cols     = inputImage1.cols;
    // 將 double weight 轉為 float，以供 AVX2 向量運算
    const float weight_f = static_cast<float>(weight);

    // OpenMP 並行：每行由不同執行緒處理
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uchar* in1Row = inputImage1.ptr<uchar>(i);  // 原始像素行
        const uchar* in2Row = inputImage2.ptr<uchar>(i);  // 掩碼像素行
        uchar*       outRow = outputImage.ptr<uchar>(i);  // 輸出像素行
        
        int j = 0;
    #ifdef __AVX2__
        // 事先載入常數向量
        // weight_ps：8 × weight_f
        const __m256  weight_ps    = _mm256_set1_ps(weight_f);
        // zero_epi32：8 × 0 (for clamp lower bound)
        const __m256i zero_epi32   = _mm256_setzero_si256();
        // c255_epi32：8 × 255 (for clamp upper bound)
        const __m256i c255_epi32   = _mm256_set1_epi32(255);
        // zero_xmm_pack：128-bit zeros 用在最後 pack 時填充
        const __m128i zero_xmm_pack= _mm_setzero_si128();

        // AVX2 向量化迴圈：一次處理 8 個像素
        for (; j <= cols - 8; j += 8) {
            // 1) 載入原始像素 (8×uchar) → 128-bit register
            __m128i in1_u8 = _mm_loadl_epi64((__m128i*)(in1Row + j));
            //    _mm256_cvtepu8_epi32：8×uchar → 8×int32
            __m256i in1_i32 = _mm256_cvtepu8_epi32(in1_u8);

            // 2) 載入掩碼像素 8×uchar → int32 → float
            __m128i in2_u8 = _mm_loadl_epi64((__m128i*)(in2Row + j));
            __m256i in2_i32 = _mm256_cvtepu8_epi32(in2_u8);
            //    _mm256_cvtepi32_ps：int32 → float
            __m256  in2_ps  = _mm256_cvtepi32_ps(in2_i32);

            // 3) 計算 product = weight_f * mask
            //    _mm256_mul_ps：向量乘法 (8×float)
            __m256  prod_ps = _mm256_mul_ps(in2_ps, weight_ps);

            // 4) 浮點轉 int32 (truncate) → sum = orig + product
            //    _mm256_cvttps_epi32：8×float → 8×int32
            __m256i prod_i32 = _mm256_cvttps_epi32(prod_ps);
            //    _mm256_add_epi32：8×int32 加法
            __m256i sum_i32  = _mm256_add_epi32(in1_i32, prod_i32);

            // 5) 飽和 clamp 到 [0,255]
            //    _mm256_max_epi32：lower bound clamp at 0
            sum_i32 = _mm256_max_epi32(zero_epi32, sum_i32);
            //    _mm256_min_epi32：upper bound clamp at 255
            sum_i32 = _mm256_min_epi32(c255_epi32, sum_i32);

            // 6) Pack int32 → int16 → uint8
            //    _mm_packus_epi32：兩組 4×int32 → 8×uint16 (飽和)
            __m128i sum_i16 = _mm_packus_epi32(
                                  _mm256_castsi256_si128(sum_i32),
                                  _mm256_extracti128_si256(sum_i32,1)
                              );
            //    _mm_packus_epi16：8×uint16 → 8×uint8 (飽和)
            __m128i sum_u8  = _mm_packus_epi16(sum_i16, zero_xmm_pack);

            // 7) 寫回 8×uchar
            _mm_storel_epi64((__m128i*)(outRow + j), sum_u8);
        }
    #endif

        // 純量迴圈：處理尾端不足 8 像素
        for (; j < cols; ++j) {
            // high-boost filter: out = in1 + w * in2
            float val = float(in1Row[j]) + weight_f * float(in2Row[j]);
            // saturate_cast<uchar> 自動 clamp 到 [0,255]
            outRow[j] = saturate_cast<uchar>(val);
        }
    }
}


// histogramCalcAndEqualParOpt：並行計算直方圖並做直方圖等化（Histogram Equalization）
// inputImage  : 單通道灰階影像 (uchar)
// outputImage : 等化後輸出影像 (uchar)
// nThreads    : 使用的執行緒數
void histogramCalcAndEqualParOpt(const Mat& inputImage,
                                 Mat&       outputImage,
                                 int        nThreads)
{
    const int rows        = inputImage.rows;
    const int cols        = inputImage.cols;
    const int totalPixels = rows * cols;

    // 若影像為空，直接輸出全零圖
    if (totalPixels == 0) {
        if (outputImage.size() != inputImage.size() ||
            outputImage.type() != inputImage.type()) {
            outputImage = Mat::zeros(inputImage.size(), inputImage.type());
        } else {
            outputImage.setTo(Scalar(0));
        }
        return;
    }

    // --------------------------------------------------------------------------------
    // 1) 局部直方圖計算 (Per-thread local histograms)
    //    使用 nThreads 個 vector，各自累加自己分到的列
    //    localHistData[t][k] 最終存第 t 號執行緒對灰階值 k 的計數
    // --------------------------------------------------------------------------------
    std::vector<std::vector<unsigned int>> localHistData(
        nThreads, std::vector<unsigned int>(256, 0));

    // OpenMP 並行地對每一行像素做累加
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();               // 取得執行緒編號
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            for (int j = 0; j < cols; ++j) {
                // 累加 local histogram
                localHistData[tid][ row[j] ]++;
            }
        }
    }

    // --------------------------------------------------------------------------------
    // 2) 合併所有執行緒的直方圖 (Reduce local histograms)
    //    imHistogram[k] = Σ_t localHistData[t][k]
    // --------------------------------------------------------------------------------
    unsigned int imHistogram[256] = {0};
    for (int t = 0; t < nThreads; ++t) {
        for (int k = 0; k < 256; ++k) {
            imHistogram[k] += localHistData[t][k];
        }
    }

    // --------------------------------------------------------------------------------
    // 3) 計算累積分布函數 CDF (Cumulative Distribution Function)
    //    cdf[0] = H(0)
    //    cdf[i] = cdf[i-1] + H(i), i=1..255
    // --------------------------------------------------------------------------------
    int cdf[256] = {0};
    cdf[0] = imHistogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + imHistogram[i];
    }

    // --------------------------------------------------------------------------------
    // 4) 建立映射表 LUT (Lookup Table)
    //    t(r) = round( (L-1) * cdf[r] / (M*N) ), 其中 L=256
    //    scale = 255.0 / totalPixels
    // --------------------------------------------------------------------------------
    uchar lut[256];
    const float scale = 255.0f / float(totalPixels);

    // #pragma omp simd：提示編譯器用 SIMD 指令向量化此迴圈
    #pragma omp parallel for simd schedule(static)
    for (int r = 0; r < 256; ++r) {
        // roundf：四捨五入
        lut[r] = saturate_cast<uchar>( roundf(scale * float(cdf[r])) );
    }

    // --------------------------------------------------------------------------------
    // 5) 依 LUT 做像素重映射 (Apply mapping to image)
    //    out[i,j] = lut[ in[i,j] ]
    // --------------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uchar* inRow  = inputImage.ptr<uchar>(i);
        uchar*       outRow = outputImage.ptr<uchar>(i);
        // #pragma omp simd：提示向量化每行內的像素重映射
        #pragma omp simd
        for (int j = 0; j < cols; ++j) {
            outRow[j] = lut[ inRow[j] ];
        }
    }
}


// hsvToRgbParOpt：並行且向量化地將 HSV 單通道影像轉回 RGB
// inputImage  : 3 通道 HSV 影像 (uchar 格式，H∈[0..180], S,V∈[0..255])
// outputImage : 3 通道 RGB 影像 (uchar 格式)
// blockSize   : 分塊大小，用於 OpenMP collapse(2) 分割工作
void hsvToRgbParOpt(const Mat& inputImage,
                    Mat&       outputImage,
                    int        blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    // 用於正規化／反正規化
    const float inv255_scalar = 1.0f / 255.0f;  // S,V 轉 [0,1]
    const float inv60_scalar  = 1.0f / 60.0f;   // H 轉區間單位

    // 啟動 OpenMP 並行區
    #pragma omp parallel
    {
    #ifdef __AVX2__
        // AVX2 常用常數向量 (256-bit = 8×float)
        const __m256 _zero_ps    = _mm256_setzero_ps();    // 0.0f
        const __m256 _one_ps     = _mm256_set1_ps(1.0f);   // 1.0f
        const __m256 _255_ps     = _mm256_set1_ps(255.0f); // 255.0f
        const __m256 _inv255_ps  = _mm256_set1_ps(inv255_scalar); 
        const __m256 _two_ps     = _mm256_set1_ps(2.0f);   // 2.0f
        const __m256 _inv60_ps   = _mm256_set1_ps(inv60_scalar);
        const __m256 _360_ps     = _mm256_set1_ps(360.0f); 
        const __m256 _eps_ps     = _mm256_set1_ps(1e-6f);  // 防除零
        const __m128i _zero_xmm  = _mm_setzero_si128();    // 128-bit zeros
    #endif

        // collapse(2)：對 ii/jj 兩層分塊並行
        #pragma omp for collapse(2) schedule(guided)
        for (int ii = 0; ii < rows; ii += blockSize) {
            for (int jj = 0; jj < cols; jj += blockSize) {
                int iEnd = min(ii + blockSize, rows);
                int jEnd = min(jj + blockSize, cols);

                for (int i = ii; i < iEnd; ++i) {
                    const Vec3b* inRow  = inputImage.ptr<Vec3b>(i);   // HSV 行指標
                    Vec3b*       outRow = outputImage.ptr<Vec3b>(i); // RGB 行指標
                    int j = jj;

    #ifdef __AVX2__
                    // 一次處理 8 個像素
                    for (; j <= jEnd - 8; j += 8) {
                        // 1) 暫存 H,S,V 各 8 值至暫存陣列
                        unsigned char h_tmp[8], s_tmp[8], v_tmp[8];
                        for (int k = 0; k < 8; ++k) {
                            h_tmp[k] = inRow[j+k][0];  // H∈[0..180]
                            s_tmp[k] = inRow[j+k][1];  // S∈[0..255]
                            v_tmp[k] = inRow[j+k][2];  // V∈[0..255]
                        }

                        // 2) 載入 8×uchar → 128-bit SSE
                        __m128i h_u8 = _mm_loadl_epi64((__m128i*)h_tmp);
                        __m128i s_u8 = _mm_loadl_epi64((__m128i*)s_tmp);
                        __m128i v_u8 = _mm_loadl_epi64((__m128i*)v_tmp);

                        // 3) zero-extend uchar → int32 → float
                        __m256 h_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(h_u8)), _two_ps);
                        __m256 s_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(s_u8)), _inv255_ps);
                        __m256 v_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v_u8)), _inv255_ps);

                        // 4) 飽和度 S≈0 時直接輸出灰階
                        __m256 gray_mask = _mm256_cmp_ps(s_f, _eps_ps, _CMP_LE_OQ);
                        __m256 r_ps = v_f, g_ps = v_f, b_ps = v_f;  // 灰階 = V

                        // 5) 確保 H<360，否則歸 0
                        __m256 h_ge360 = _mm256_cmp_ps(h_f, _360_ps, _CMP_GE_OQ);
                        h_f = _mm256_blendv_ps(h_f, _zero_ps, h_ge360);

                        // 6) H→[0..6) 區間單位
                        h_f = _mm256_mul_ps(h_f, _inv60_ps);
                        // 取整 hi = floor(H), f = H – hi
                        __m256i hi = _mm256_cvttps_epi32(h_f);
                        __m256  f  = _mm256_sub_ps(h_f, _mm256_cvtepi32_ps(hi));

                        // 7) 計算 p,q,t 向量
                        __m256 p = _mm256_mul_ps(v_f, _mm256_sub_ps(_one_ps, s_f));
                        __m256 q = _mm256_mul_ps(v_f, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_f, f)));
                        __m256 t = _mm256_mul_ps(v_f, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_f, _mm256_sub_ps(_one_ps, f))));

                        // 8) 各區段對應 r_cX,g_cX,b_cX
                        __m256 r_c0 = v_f, g_c0 = t,    b_c0 = p;
                        __m256 r_c1 = q,    g_c1 = v_f, b_c1 = p;
                        __m256 r_c2 = p,    g_c2 = v_f, b_c2 = t;
                        __m256 r_c3 = p,    g_c3 = q,   b_c3 = v_f;
                        __m256 r_c4 = t,    g_c4 = p,   b_c4 = v_f;
                        __m256 r_c5 = v_f,  g_c5 = p,   b_c5 = q;

                        // 9) 根據 hi 值選擇分支
                        __m256i m0 = _mm256_cmpeq_epi32(hi, _mm256_setzero_si256());
                        __m256i m1 = _mm256_cmpeq_epi32(hi, _mm256_set1_epi32(1));
                        __m256i m2 = _mm256_cmpeq_epi32(hi, _mm256_set1_epi32(2));
                        __m256i m3 = _mm256_cmpeq_epi32(hi, _mm256_set1_epi32(3));
                        __m256i m4 = _mm256_cmpeq_epi32(hi, _mm256_set1_epi32(4));
                        // blendv_ps 根據 mask 選擇對應通道
                        r_ps = _mm256_blendv_ps(r_c5, r_c4, (__m256)m4);
                        r_ps = _mm256_blendv_ps(r_ps, r_c3, (__m256)m3);
                        r_ps = _mm256_blendv_ps(r_ps, r_c2, (__m256)m2);
                        r_ps = _mm256_blendv_ps(r_ps, r_c1, (__m256)m1);
                        r_ps = _mm256_blendv_ps(r_ps, r_c0, (__m256)m0);
                        
                        g_ps = _mm256_blendv_ps(g_c5, g_c4, (__m256)m4);
                        g_ps = _mm256_blendv_ps(g_ps, g_c3, (__m256)m3);
                        g_ps = _mm256_blendv_ps(g_ps, g_c2, (__m256)m2);
                        g_ps = _mm256_blendv_ps(g_ps, g_c1, (__m256)m1);
                        g_ps = _mm256_blendv_ps(g_ps, g_c0, (__m256)m0);

                        b_ps = _mm256_blendv_ps(b_c5, b_c4, (__m256)m4);
                        b_ps = _mm256_blendv_ps(b_ps, b_c3, (__m256)m3);
                        b_ps = _mm256_blendv_ps(b_ps, b_c2, (__m256)m2);
                        b_ps = _mm256_blendv_ps(b_ps, b_c1, (__m256)m1);
                        b_ps = _mm256_blendv_ps(b_ps, b_c0, (__m256)m0);

                        // 若 S≈0，覆蓋成灰階
                        r_ps = _mm256_blendv_ps(r_ps, v_f, gray_mask);
                        g_ps = _mm256_blendv_ps(g_ps, v_f, gray_mask);
                        b_ps = _mm256_blendv_ps(b_ps, v_f, gray_mask);

                        // 10) 反正規化到 [0..255]
                        r_ps = _mm256_mul_ps(r_ps, _255_ps);
                        g_ps = _mm256_mul_ps(g_ps, _255_ps);
                        b_ps = _mm256_mul_ps(b_ps, _255_ps);

                        // 11) Pack→store 回 8 個像素
                        __m256i ri = _mm256_cvttps_epi32(r_ps);
                        __m256i gi = _mm256_cvttps_epi32(g_ps);
                        __m256i bi = _mm256_cvttps_epi32(b_ps);
                        __m128i r16 = _mm_packus_epi32(_mm256_castsi256_si128(ri), _mm256_extracti128_si256(ri,1));
                        __m128i g16 = _mm_packus_epi32(_mm256_castsi256_si128(gi), _mm256_extracti128_si256(gi,1));
                        __m128i b16 = _mm_packus_epi32(_mm256_castsi256_si128(bi), _mm256_extracti128_si256(bi,1));
                        __m128i r8  = _mm_packus_epi16(r16, _zero_xmm);
                        __m128i g8  = _mm_packus_epi16(g16, _zero_xmm);
                        __m128i b8  = _mm_packus_epi16(b16, _zero_xmm);
                        unsigned char ru[8], gu[8], bu[8];
                        _mm_storel_epi64((__m128i*)ru, r8);
                        _mm_storel_epi64((__m128i*)gu, g8);
                        _mm_storel_epi64((__m128i*)bu, b8);
                        for (int k = 0; k < 8; ++k) {
                            outRow[j+k][0] = bu[k];  // B
                            outRow[j+k][1] = gu[k];  // G
                            outRow[j+k][2] = ru[k];  // R
                        }
                    }
    #endif
                    // 純量回落：處理尾端不足 8 像素
                    for (; j < jEnd; ++j) {
                        float H = inRow[j][0] * 2.0f;             // 0..360
                        float S = inRow[j][1] * inv255_scalar;    // 0..1
                        float V = inRow[j][2] * inv255_scalar;    // 0..1
                        float r, g, b;
                        if (S <= 1e-6f) {
                            // 灰階
                            r = g = b = V;
                        } else {
                            if (H >= 360.0f) H = 0.0f;
                            H *= inv60_scalar;         // H → [0..6)
                            int hi = int(floor(H));    // 區段
                            float f = H - hi;          // 小數
                            float p = V*(1 - S);
                            float q = V*(1 - S*f);
                            float t = V*(1 - S*(1 - f));
                            switch(hi) {
                                case 0: r=V; g=t; b=p; break;
                                case 1: r=q; g=V; b=p; break;
                                case 2: r=p; g=V; b=t; break;
                                case 3: r=p; g=q; b=V; break;
                                case 4: r=t; g=p; b=V; break;
                                default:r=V; g=p; b=q; break;
                            }
                        }
                        // 反正規化並 saturate
                        outRow[j][0] = saturate_cast<uchar>(b * 255.0f);
                        outRow[j][1] = saturate_cast<uchar>(g * 255.0f);
                        outRow[j][2] = saturate_cast<uchar>(r * 255.0f);
                    }
                }
            }
        }
    } // omp parallel end
}


string getexepath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}
