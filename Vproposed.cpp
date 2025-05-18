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
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
    }
    #pragma omp barrier
    
    for(int i = 0; i < numIter; ++i) {
        start_time = chrono::high_resolution_clock::now();
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
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
void rgbToHsvParOpt(const Mat& inputImage, Mat& outputImage, int blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const float inv255_scalar = 1.0f / 255.0f; 
    
    #pragma omp parallel
    {
        #ifdef __AVX2__
        const __m256 _zero_ps    = _mm256_setzero_ps();         
        const __m256 _one_ps     = _mm256_set1_ps(1.0f);          
        const __m256 _half_ps    = _mm256_set1_ps(0.5f);          
        const __m256 _2_ps       = _mm256_set1_ps(2.0f);          
        const __m256 _4_ps       = _mm256_set1_ps(4.0f);          
        const __m256 _60_ps      = _mm256_set1_ps(60.0f);         
        const __m256 _255_ps     = _mm256_set1_ps(255.0f);        
        const __m256 _360_ps     = _mm256_set1_ps(360.0f);        
        const __m256 _epsilon_ps = _mm256_set1_ps(1e-6f);         
        const __m256 _inv255_ps  = _mm256_set1_ps(1.0f / 255.0f); 
        const __m128i _zero_xmm  = _mm_setzero_si128();           
        #endif

        #pragma omp for collapse(2) schedule(guided)
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
                        
                        __m256 cmax_ps = _mm256_max_ps(r_ps_norm, _mm256_max_ps(g_ps_norm, b_ps_norm));
                        __m256 cmin_ps = _mm256_min_ps(r_ps_norm, _mm256_min_ps(g_ps_norm, b_ps_norm));
                        __m256 delta_ps = _mm256_sub_ps(cmax_ps, cmin_ps);

                        __m256 v_ps = cmax_ps; 

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
                        h_ps = _mm256_blendv_ps(h_ps, _zero_ps, delta_is_small_mask); 

                        h_ps = _mm256_mul_ps(h_ps, _60_ps); 
                        __m256 h_neg_mask = _mm256_cmp_ps(h_ps, _zero_ps, _CMP_LT_OQ); 
                        h_ps = _mm256_add_ps(h_ps, _mm256_and_ps(h_neg_mask, _360_ps)); 
                        
                        h_ps = _mm256_mul_ps(h_ps, _half_ps);   
                        s_ps = _mm256_mul_ps(s_ps, _255_ps);    
                        v_ps = _mm256_mul_ps(v_ps, _255_ps);    

                        __m256i h_epi32_final = _mm256_cvttps_epi32(h_ps); 
                        __m256i s_epi32_final = _mm256_cvttps_epi32(s_ps);
                        __m256i v_epi32_final = _mm256_cvttps_epi32(v_ps);

                        __m128i h_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(h_epi32_final), _mm256_extracti128_si256(h_epi32_final, 1));
                        __m128i s_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(s_epi32_final), _mm256_extracti128_si256(s_epi32_final, 1));
                        __m128i v_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(v_epi32_final), _mm256_extracti128_si256(v_epi32_final, 1));

                        __m128i h_u8_xmm_final = _mm_packus_epi16(h_u16_xmm, _zero_xmm); 
                        __m128i s_u8_xmm_final = _mm_packus_epi16(s_u16_xmm, _zero_xmm);
                        __m128i v_u8_xmm_final = _mm_packus_epi16(v_u16_xmm, _zero_xmm);

                        unsigned char h_u[8], s_u[8], v_u[8];
                        _mm_storel_epi64((__m128i*)h_u, h_u8_xmm_final); 
                        _mm_storel_epi64((__m128i*)s_u, s_u8_xmm_final);
                        _mm_storel_epi64((__m128i*)v_u, v_u8_xmm_final);

                        for (int k_store = 0; k_store < 8; ++k_store) {
                            outRow[j + k_store][0] = h_u[k_store];
                            outRow[j + k_store][1] = s_u[k_store];
                            outRow[j + k_store][2] = v_u[k_store];
                        }
                    }
#endif
                    for (; j < blockEndJ; ++j) { 
                        uchar blue_s = inRow[j][0];
                        uchar green_s = inRow[j][1];
                        uchar red_s = inRow[j][2];
                        
                        float r_norm_s = red_s * inv255_scalar;
                        float g_norm_s = green_s * inv255_scalar;
                        float b_norm_s = blue_s * inv255_scalar;
                        
                        float cmin_s = std::min({r_norm_s, g_norm_s, b_norm_s}); // C++11 initializer list for min/max
                        float cmax_s = std::max({r_norm_s, g_norm_s, b_norm_s});
                        float delta_s = cmax_s - cmin_s;
                        
                        float h_val_s = 0.0f, s_val_s = 0.0f, v_val_s = cmax_s;

                        if(delta_s < 1e-6f) { 
                            s_val_s = 0.0f;
                            h_val_s = 0.0f;
                        } else {
                            s_val_s = (cmax_s > 1e-6f) ? (delta_s / cmax_s) : 0.0f;
                            if(abs(cmax_s - r_norm_s) < 1e-6f)      h_val_s = (g_norm_s - b_norm_s) / delta_s;
                            else if(abs(cmax_s - g_norm_s) < 1e-6f) h_val_s = 2.0f + (b_norm_s - r_norm_s) / delta_s;
                            else                                   h_val_s = 4.0f + (r_norm_s - g_norm_s) / delta_s;
                            h_val_s *= 60.0f;
                            if(h_val_s < 0.0f) h_val_s += 360.0f;
                        }
                        outRow[j][0] = saturate_cast<uchar>(h_val_s * 0.5f);
                        outRow[j][1] = saturate_cast<uchar>(s_val_s * 255.0f);
                        outRow[j][2] = saturate_cast<uchar>(v_val_s * 255.0f);
                    }
                }
            }
        }
    }
}


#ifdef __AVX2__
static inline __m256 load_and_convert_uchar_to_float_avx2(const unsigned char* p) {
    __m128i u8_vec = _mm_loadl_epi64((const __m128i*)p); 
    __m256i i32_vec = _mm256_cvtepu8_epi32(u8_vec);      
    return _mm256_cvtepi32_ps(i32_vec);                  
}
#endif

// imageBlurParOpt 函數：優化後的圖像模糊函數
// 變更:
// - 水平遍歷SIMD循環: 從一次處理8個像素改為一次處理16個像素 (2組8像素)。
// - 垂直遍歷SIMD循環: 從一次處理8個像素改為一次處理16個像素 (2組8像素)。
// - 調整循環邊界條件以適應16像素處理。
// - 剩餘像素的標量循環將處理新的邊界。
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int halfSize = FILTERSIZE / 2; 
    
    alignas(64) const float GaussianKernel1D_F[FILTERSIZE] = {
        1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f
    }; 

    Mat tempImage(rows, cols, CV_32FC1);

    // --- 水平遍歷 ---
    Mat paddedInputHorizontal;
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

        // SIMD循環展開: 每次處理16個像素 (2組8像素)
        for (; j <= cols - 16; j += 16) { 
            // 第一組8個像素
            __m256 v_x0_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 0);
            __m256 v_x1_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 1);
            __m256 v_x2_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 2);
            __m256 v_x3_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 3);
            __m256 v_x4_0 = load_and_convert_uchar_to_float_avx2(srcRow + j + 4);
            
            __m256 sum_v_0 = _mm256_mul_ps(v_x0_0, gk0_v);
            sum_v_0 = _mm256_fmadd_ps(v_x1_0, gk1_v, sum_v_0);
            sum_v_0 = _mm256_fmadd_ps(v_x2_0, gk2_v, sum_v_0);
            sum_v_0 = _mm256_fmadd_ps(v_x3_0, gk3_v, sum_v_0);
            sum_v_0 = _mm256_fmadd_ps(v_x4_0, gk4_v, sum_v_0);
            _mm256_storeu_ps(tempRow + j, sum_v_0);

            // 第二組8個像素 (偏移量 +8)
            __m256 v_x0_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 0);
            __m256 v_x1_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 1);
            __m256 v_x2_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 2);
            __m256 v_x3_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 3);
            __m256 v_x4_1 = load_and_convert_uchar_to_float_avx2(srcRow + j + 8 + 4);

            __m256 sum_v_1 = _mm256_mul_ps(v_x0_1, gk0_v);
            sum_v_1 = _mm256_fmadd_ps(v_x1_1, gk1_v, sum_v_1);
            sum_v_1 = _mm256_fmadd_ps(v_x2_1, gk2_v, sum_v_1);
            sum_v_1 = _mm256_fmadd_ps(v_x3_1, gk3_v, sum_v_1);
            sum_v_1 = _mm256_fmadd_ps(v_x4_1, gk4_v, sum_v_1);
            _mm256_storeu_ps(tempRow + j + 8, sum_v_1);
        }
#endif
        // 標量循環處理剩餘的像素 (cols 可能不是16的倍數)
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
        const __m128i zero_xmm_pack = _mm_setzero_si128(); 

        // SIMD循環展開: 每次處理16個像素 (2組8像素)
        for (; j <= cols - 16; j += 16) {
            // 第一組8個像素
            __m256 in0_0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 0) + j);
            __m256 in1_0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 1) + j);
            __m256 in2_0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 2) + j);
            __m256 in3_0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 3) + j);
            __m256 in4_0 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 4) + j);

            __m256 sum_ps_0 = _mm256_mul_ps(in0_0, k0_v);
            sum_ps_0 = _mm256_fmadd_ps(in1_0, k1_v, sum_ps_0);
            sum_ps_0 = _mm256_fmadd_ps(in2_0, k2_v, sum_ps_0);
            sum_ps_0 = _mm256_fmadd_ps(in3_0, k3_v, sum_ps_0);
            sum_ps_0 = _mm256_fmadd_ps(in4_0, k4_v, sum_ps_0);
            
            __m256i sum_epi32_0 = _mm256_cvttps_epi32(sum_ps_0);
            __m128i sum_u16_packed_0 = _mm_packus_epi32(_mm256_castsi256_si128(sum_epi32_0), 
                                                      _mm256_extracti128_si256(sum_epi32_0, 1));
            __m128i sum_u8_packed_0 = _mm_packus_epi16(sum_u16_packed_0, zero_xmm_pack);
            _mm_storel_epi64((__m128i*)(outRow + j), sum_u8_packed_0);

            // 第二組8個像素 (偏移量 +8)
            __m256 in0_1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 0) + j + 8);
            __m256 in1_1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 1) + j + 8);
            __m256 in2_1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 2) + j + 8);
            __m256 in3_1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 3) + j + 8);
            __m256 in4_1 = _mm256_loadu_ps(paddedTempImage.ptr<float>(i + 4) + j + 8);

            __m256 sum_ps_1 = _mm256_mul_ps(in0_1, k0_v);
            sum_ps_1 = _mm256_fmadd_ps(in1_1, k1_v, sum_ps_1);
            sum_ps_1 = _mm256_fmadd_ps(in2_1, k2_v, sum_ps_1);
            sum_ps_1 = _mm256_fmadd_ps(in3_1, k3_v, sum_ps_1);
            sum_ps_1 = _mm256_fmadd_ps(in4_1, k4_v, sum_ps_1);

            __m256i sum_epi32_1 = _mm256_cvttps_epi32(sum_ps_1);
            __m128i sum_u16_packed_1 = _mm_packus_epi32(_mm256_castsi256_si128(sum_epi32_1),
                                                      _mm256_extracti128_si256(sum_epi32_1, 1));
            __m128i sum_u8_packed_1 = _mm_packus_epi16(sum_u16_packed_1, zero_xmm_pack);
            _mm_storel_epi64((__m128i*)(outRow + j + 8), sum_u8_packed_1);
        }
#endif
        // 標量循環處理剩餘的像素
        for (; j < cols; ++j) {
            float sum = 0.0f;
            sum += paddedTempImage.ptr<float>(i + 0)[j] * GaussianKernel1D_F[0];
            sum += paddedTempImage.ptr<float>(i + 1)[j] * GaussianKernel1D_F[1];
            sum += paddedTempImage.ptr<float>(i + 2)[j] * GaussianKernel1D_F[2];
            sum += paddedTempImage.ptr<float>(i + 3)[j] * GaussianKernel1D_F[3];
            sum += paddedTempImage.ptr<float>(i + 4)[j] * GaussianKernel1D_F[4];
            outRow[j] = saturate_cast<uchar>(sum); 
        }
    }
}

// subtractImageParOpt (Implementation as in the previous version)
void subtractImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage)
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;
    
    if(inputImage1.isContinuous() && inputImage2.isContinuous() && outputImage.isContinuous()) {
        const size_t totalPixels = static_cast<size_t>(rows) * cols; 
        const uchar* in1_ptr = inputImage1.ptr<uchar>(0);
        const uchar* in2_ptr = inputImage2.ptr<uchar>(0);
        uchar* out_ptr = outputImage.ptr<uchar>(0);
        
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < totalPixels / 32 * 32; i += 32) { 
            #ifdef __AVX2__
            __m256i a = _mm256_loadu_si256((__m256i*)(in1_ptr + i));
            __m256i b = _mm256_loadu_si256((__m256i*)(in2_ptr + i));
            __m256i result = _mm256_subs_epu8(a, b); 
            _mm256_storeu_si256((__m256i*)(out_ptr + i), result);
            #else
            for(size_t k = 0; k < 32; ++k) {
                int val = static_cast<int>(in1_ptr[i+k]) - static_cast<int>(in2_ptr[i+k]);
                out_ptr[i+k] = saturate_cast<uchar>(val);
            }
            #endif
        }
        for(size_t i = totalPixels / 32 * 32; i < totalPixels; ++i) {
            int val = static_cast<int>(in1_ptr[i]) - static_cast<int>(in2_ptr[i]);
            out_ptr[i] = saturate_cast<uchar>(val);
        }

    } else { 
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* in1Row = inputImage1.ptr<uchar>(i);
            const uchar* in2Row = inputImage2.ptr<uchar>(i);
            uchar* outRow = outputImage.ptr<uchar>(i);
            #pragma omp simd 
            for(int j = 0; j < cols; ++j) {
                int val = static_cast<int>(in1Row[j]) - static_cast<int>(in2Row[j]);
                outRow[j] = saturate_cast<uchar>(val);
            }
        }
    }
}

// sharpenImageParOpt (Implementation as in the previous version)
void sharpenImageParOpt(const Mat& inputImage1, const Mat& inputImage2, Mat& outputImage, double weight)
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;
    const float weight_f = static_cast<float>(weight); 

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < rows; ++i) {
        const uchar* in1Row = inputImage1.ptr<uchar>(i);
        const uchar* in2Row = inputImage2.ptr<uchar>(i);
        uchar* outRow = outputImage.ptr<uchar>(i);
        
        int j = 0;
#ifdef __AVX2__
        const __m256 weight_ps = _mm256_set1_ps(weight_f);       
        const __m256i zero_epi32 = _mm256_setzero_si256();       
        const __m256i c255_epi32 = _mm256_set1_epi32(255);       
        const __m128i zero_xmm_pack = _mm_setzero_si128();       

        for (; j <= cols - 8; j += 8) { 
            __m128i in1_u8_xmm = _mm_loadl_epi64((__m128i*)(in1Row + j));
            __m256i in1_epi32 = _mm256_cvtepu8_epi32(in1_u8_xmm);

            __m128i in2_u8_xmm = _mm_loadl_epi64((__m128i*)(in2Row + j));
            __m256i in2_epi32_temp = _mm256_cvtepu8_epi32(in2_u8_xmm);
            __m256 in2_ps = _mm256_cvtepi32_ps(in2_epi32_temp);

            __m256 product_ps = _mm256_mul_ps(in2_ps, weight_ps);

            __m256i product_epi32 = _mm256_cvttps_epi32(product_ps);
            __m256i sum_epi32 = _mm256_add_epi32(in1_epi32, product_epi32);

            sum_epi32 = _mm256_max_epi32(zero_epi32, sum_epi32); 
            sum_epi32 = _mm256_min_epi32(c255_epi32, sum_epi32); 

            __m128i sum_epu16 = _mm_packus_epi32(
                                _mm256_castsi256_si128(sum_epi32),      
                                _mm256_extracti128_si256(sum_epi32, 1) 
                                );
            __m128i sum_epu8 = _mm_packus_epi16(sum_epu16, zero_xmm_pack);
            _mm_storel_epi64((__m128i*)(outRow + j), sum_epu8);
        }
#endif
        for (; j < cols; ++j) {
            float val_f = static_cast<float>(in1Row[j]) + weight_f * static_cast<float>(in2Row[j]);
            outRow[j] = saturate_cast<uchar>(val_f);
        }
    }
}

// histogramCalcParOpt (Implementation as in the previous version)
void histogramCalcParOpt(const Mat& inputImage, unsigned int imHistogram[], int nThreads)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    
    std::vector<std::vector<unsigned int>> localHistData(nThreads, std::vector<unsigned int>(256, 0));
    
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            for(int j = 0; j < cols; ++j) {
                localHistData[tid][row[j]]++;
            }
        }
    }
    
    memset(imHistogram, 0, 256 * sizeof(unsigned int));
    for(int t = 0; t < nThreads; ++t) {
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHistData[t][j];
        }
    }
}

// histogramEqualParOpt (Implementation as in the previous version)
void histogramEqualParOpt(const Mat& inputImage, Mat& outputImage, const unsigned int imHistogram[])
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int totalPixels = rows * cols;
    
    if (totalPixels == 0) return; 

    int cdf[256] = {0};
    cdf[0] = imHistogram[0];
    for(int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + imHistogram[i];
    }
    
    uchar lut[256];
    float scale = 255.0f / totalPixels;
    
    #pragma omp parallel for simd schedule(static) 
    for(int i = 0; i < 256; ++i) {
        lut[i] = saturate_cast<uchar>(roundf(scale * cdf[i])); 
    }
    
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

// histogramCalcAndEqualParOpt (Implementation as in the previous version)
void histogramCalcAndEqualParOpt(const Mat& inputImage, Mat& outputImage, int nThreads)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int totalPixels = rows * cols;

    if (totalPixels == 0) { 
        if (outputImage.size() != inputImage.size() || outputImage.type() != inputImage.type()) {
            outputImage = Mat::zeros(inputImage.size(), inputImage.type());
        } else {
            outputImage.setTo(Scalar(0));
        }
        return;
    }
    
    std::vector<std::vector<unsigned int>> localHistData(nThreads, std::vector<unsigned int>(256, 0));
    
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            for(int j = 0; j < cols; ++j) {
                localHistData[tid][row[j]]++;
            }
        }
    }
    
    unsigned int imHistogram[256] = {0}; 
    for(int t = 0; t < nThreads; ++t) {
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHistData[t][j];
        }
    }
    
    int cdf[256] = {0};
    cdf[0] = imHistogram[0];
    for(int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + imHistogram[i];
    }
    
    uchar lut[256];
    float scale = 255.0f / totalPixels;
    
    #pragma omp parallel for simd schedule(static)
    for(int i = 0; i < 256; ++i) {
        lut[i] = saturate_cast<uchar>(roundf(scale * cdf[i]));
    }
    
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

// hsvToRgbParOpt (Implementation as in the previous version)
void hsvToRgbParOpt(const Mat& inputImage, Mat& outputImage, int blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const float inv255_scalar = 1.0f / 255.0f;
    const float inv60_scalar = 1.0f / 60.0f;
    
    #pragma omp parallel
    {
        #ifdef __AVX2__
        const __m256 _zero_ps    = _mm256_setzero_ps();
        const __m256 _one_ps     = _mm256_set1_ps(1.0f);
        const __m256 _255_ps     = _mm256_set1_ps(255.0f);
        const __m256 _inv255_ps  = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 _two_ps     = _mm256_set1_ps(2.0f);
        const __m256 _inv60_ps   = _mm256_set1_ps(1.0f / 60.0f);
        const __m256 _360_ps     = _mm256_set1_ps(360.0f); 
        const __m256 _epsilon_ps = _mm256_set1_ps(1e-6f); 
        const __m128i _zero_xmm  = _mm_setzero_si128();
        #endif

        #pragma omp for collapse(2) schedule(guided) 
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

                        __m256 h_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(h_epi32), _two_ps); 
                        __m256 s_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(s_epi32), _inv255_ps); 
                        __m256 v_ps = _mm256_mul_ps(_mm256_cvtepi32_ps(v_epi32), _inv255_ps); 

                        __m256 r_f_ps, g_f_ps, b_f_ps; 

                        __m256 s_is_small_mask = _mm256_cmp_ps(s_ps, _epsilon_ps, _CMP_LE_OQ);
                        
                        __m256 r_gray_ps = v_ps; 
                        __m256 g_gray_ps = v_ps; 
                        __m256 b_gray_ps = v_ps;

                        __m256 h_ge_360_mask = _mm256_cmp_ps(h_ps, _360_ps, _CMP_GE_OQ); 
                        h_ps = _mm256_blendv_ps(h_ps, _zero_ps, h_ge_360_mask); 
                        
                        h_ps = _mm256_mul_ps(h_ps, _inv60_ps); 
                        
                        __m256i hi_epi32 = _mm256_cvttps_epi32(h_ps); 
                        
                        __m256 f_ps = _mm256_sub_ps(h_ps, _mm256_cvtepi32_ps(hi_epi32)); 

                        __m256 p_val_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, s_ps));                     
                        __m256 q_val_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_ps, f_ps))); 
                        __m256 t_val_ps = _mm256_mul_ps(v_ps, _mm256_sub_ps(_one_ps, _mm256_mul_ps(s_ps, _mm256_sub_ps(_one_ps, f_ps)))); 

                        __m256 r_c0 = v_ps;     __m256 g_c0 = t_val_ps; __m256 b_c0 = p_val_ps;
                        __m256 r_c1 = q_val_ps; __m256 g_c1 = v_ps;     __m256 b_c1 = p_val_ps;
                        __m256 r_c2 = p_val_ps; __m256 g_c2 = v_ps;     __m256 b_c2 = t_val_ps;
                        __m256 r_c3 = p_val_ps; __m256 g_c3 = q_val_ps; __m256 b_c3 = v_ps;
                        __m256 r_c4 = t_val_ps; __m256 g_c4 = p_val_ps; __m256 b_c4 = v_ps;
                        __m256 r_c5 = v_ps;     __m256 g_c5 = p_val_ps; __m256 b_c5 = q_val_ps;

                        __m256i hi_eq_0_mask = _mm256_cmpeq_epi32(hi_epi32, _mm256_setzero_si256());
                        __m256i hi_eq_1_mask = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(1));
                        __m256i hi_eq_2_mask = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(2));
                        __m256i hi_eq_3_mask = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(3));
                        __m256i hi_eq_4_mask = _mm256_cmpeq_epi32(hi_epi32, _mm256_set1_epi32(4));
                        
                        r_f_ps = r_c5; 
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_c4, (__m256)hi_eq_4_mask);
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_c3, (__m256)hi_eq_3_mask);
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_c2, (__m256)hi_eq_2_mask);
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_c1, (__m256)hi_eq_1_mask);
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_c0, (__m256)hi_eq_0_mask);
                        
                        g_f_ps = g_c5;
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_c4, (__m256)hi_eq_4_mask);
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_c3, (__m256)hi_eq_3_mask);
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_c2, (__m256)hi_eq_2_mask);
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_c1, (__m256)hi_eq_1_mask);
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_c0, (__m256)hi_eq_0_mask);

                        b_f_ps = b_c5;
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_c4, (__m256)hi_eq_4_mask);
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_c3, (__m256)hi_eq_3_mask);
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_c2, (__m256)hi_eq_2_mask);
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_c1, (__m256)hi_eq_1_mask);
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_c0, (__m256)hi_eq_0_mask);
                        
                        r_f_ps = _mm256_blendv_ps(r_f_ps, r_gray_ps, s_is_small_mask);
                        g_f_ps = _mm256_blendv_ps(g_f_ps, g_gray_ps, s_is_small_mask);
                        b_f_ps = _mm256_blendv_ps(b_f_ps, b_gray_ps, s_is_small_mask);

                        r_f_ps = _mm256_mul_ps(r_f_ps, _255_ps);
                        g_f_ps = _mm256_mul_ps(g_f_ps, _255_ps);
                        b_f_ps = _mm256_mul_ps(b_f_ps, _255_ps);

                        __m256i r_epi32_final = _mm256_cvttps_epi32(r_f_ps);
                        __m256i g_epi32_final = _mm256_cvttps_epi32(g_f_ps);
                        __m256i b_epi32_final = _mm256_cvttps_epi32(b_f_ps);

                        __m128i r_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(r_epi32_final), _mm256_extracti128_si256(r_epi32_final, 1));
                        __m128i g_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(g_epi32_final), _mm256_extracti128_si256(g_epi32_final, 1));
                        __m128i b_u16_xmm = _mm_packus_epi32(_mm256_castsi256_si128(b_epi32_final), _mm256_extracti128_si256(b_epi32_final, 1));

                        __m128i r_u8_xmm_final = _mm_packus_epi16(r_u16_xmm, _zero_xmm);
                        __m128i g_u8_xmm_final = _mm_packus_epi16(g_u16_xmm, _zero_xmm);
                        __m128i b_u8_xmm_final = _mm_packus_epi16(b_u16_xmm, _zero_xmm);

                        unsigned char r_u[8], g_u[8], b_u[8];
                        _mm_storel_epi64((__m128i*)r_u, r_u8_xmm_final);
                        _mm_storel_epi64((__m128i*)g_u, g_u8_xmm_final);
                        _mm_storel_epi64((__m128i*)b_u, b_u8_xmm_final);

                        for (int k_store = 0; k_store < 8; ++k_store) {
                            outRow[j + k_store][0] = b_u[k_store]; 
                            outRow[j + k_store][1] = g_u[k_store]; 
                            outRow[j + k_store][2] = r_u[k_store]; 
                        }
                    }
#endif
                    for (; j < blockEndJ; ++j) {
                        float h_val_s = inRow[j][0] * 2.0f; 
                        float s_val_s = inRow[j][1] * inv255_scalar;
                        float v_val_s = inRow[j][2] * inv255_scalar;
                        
                        float r_f, g_f, b_f;
                        
                        if(s_val_s <= 1e-6f) { 
                            r_f = g_f = b_f = v_val_s;
                        } else {
                            if (h_val_s >= 360.0f) h_val_s = 0.0f; 
                            h_val_s *= inv60_scalar; 
                            int hi_s = static_cast<int>(floor(h_val_s));
                            float f_s = h_val_s - hi_s; 
                            
                            float p_s = v_val_s * (1.0f - s_val_s);
                            float q_s = v_val_s * (1.0f - s_val_s * f_s);
                            float t_s = v_val_s * (1.0f - s_val_s * (1.0f - f_s));
                            
                            switch(hi_s) {
                                case 0: r_f = v_val_s; g_f = t_s;   b_f = p_s;   break;
                                case 1: r_f = q_s;   g_f = v_val_s; b_f = p_s;   break;
                                case 2: r_f = p_s;   g_f = v_val_s; b_f = t_s;   break;
                                case 3: r_f = p_s;   g_f = q_s;   b_f = v_val_s; break;
                                case 4: r_f = t_s;   g_f = p_s;   b_f = v_val_s; break;
                                default:r_f = v_val_s; g_f = p_s;   b_f = q_s;   break; 
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

string getexepath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}
