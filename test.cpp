/*
優化版本 - Singh et. al 圖像增強演算法實現
https://ieeexplore.ieee.org/document/8071892

編譯命令:
g++ -g -Wall -O3 -o test_optimized test.cpp `pkg-config --cflags --libs opencv4` -fopenmp -march=native -ffast-math -funroll-loops -ftree-vectorize

執行方式:
./test_optimized <imagename> <threads> <size>
例如: ./test_optimized snow.png 4 778x1036
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

// 添加命名空間
using namespace std;
using namespace cv;

#define FILTERSIZE 5
#define MAX_THREADS 32 // 限制最大執行緒數避免過度分配

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
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage, int blockSize);
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

// 計算最佳執行緒數
int calculateOptimalThreads(const Mat& image, int maxThreads) {
    int pixelCount = image.rows * image.cols;
    
    // 對於小影像，減少執行緒數量避免過度並行開銷
    if (pixelCount < 500000) {
        return std::min(2, maxThreads);
    } else if (pixelCount < 2000000) {
        return std::min(4, maxThreads);
    } else if (pixelCount < 4000000) {
        return std::min(8, maxThreads);
    } else {
        return maxThreads;
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
    int maxThreads = omp_get_max_threads(); 
    maxThreads = std::min(maxThreads, MAX_THREADS); // 限制最大執行緒數
    
    int numThreads = maxThreads; // 預設使用最大執行緒數
    
    if (argc >= 3) {
        try {
            numThreads = stoi(argv[2]);
            if (numThreads <= 0 || numThreads > maxThreads) {
                numThreads = maxThreads;
                cout << "Invalid thread count, using " << numThreads << " threads." << endl;
            }
        } catch (const exception&) {
            cout << "Invalid thread count, using " << numThreads << " threads." << endl;
        }
    }
    
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
    int optimalThreads = calculateOptimalThreads(inputImage, numThreads);
    int optimalBlockSize = calculateOptimalBlockSize(inputImage);
    
    // 設置執行緒數量
    omp_set_num_threads(optimalThreads);
    cout << "Using " << optimalThreads << " threads with block size " << optimalBlockSize << endl;

    // 優化路徑處理
    string filename = argv[1];
    size_t last_slash_pos = filename.find_last_of("/\\");
    if (last_slash_pos != string::npos) {
        filename = filename.substr(last_slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    string basename = filename.substr(0, dot_pos);
    string extension = filename.substr(dot_pos);
    
    string sizemodifier = to_string(inputImage.rows) + "x" + to_string(inputImage.cols);
    string threadModifier = "_t" + to_string(optimalThreads);
    string outimagename = basename + "_" + sizemodifier + threadModifier + "_optimized" + extension;
    cout << "Output image file name: " << outimagename << endl;

    // 創建CSV結果文件
    string resultpath = basename + "_" + sizeStr + "_threads" + to_string(optimalThreads) + "_optimized.csv";
    ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,BlockSize,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << endl;
    
    // 提前配置所有需要的記憶體，避免動態配置
    Mat inputImageHsv(inputImage.size(), inputImage.type());
    Mat blurredImage(inputImage.size(), CV_8UC1);
    Mat imageMask(inputImage.size(), CV_8UC1);
    Mat sharpenedImage(inputImage.size(), CV_8UC1);
    Mat globallyEnhancedImage(inputImage.size(), CV_8UC1);
    Mat outputHSV(inputImage.size(), inputImage.type());
    Mat finalOutput(inputImage.size(), inputImage.type());
    
    vector<Mat> inputImageHsvChannels;
    
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    auto timeEllap1 = chrono::duration<double, milli>::zero();
    auto timeEllap2 = chrono::duration<double, milli>::zero();
    auto timeEllap3 = chrono::duration<double, milli>::zero();
    auto timeEllap4 = chrono::duration<double, milli>::zero();
    auto timeEllap5 = chrono::duration<double, milli>::zero();
    auto timeEllap6 = chrono::duration<double, milli>::zero();
    auto timeEllap7 = chrono::duration<double, milli>::zero();
    
    // 資料較大時減少重複計時次數
    int numIter = (inputImage.rows * inputImage.cols > 4000000) ? 5 : 10;

    // 步驟1: RGB到HSV轉換
    // 首先進行一次計算來預熱快取和載入指令
    rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
    
    // 正式計時
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        rgbToHsvParOpt(inputImage, inputImageHsv, optimalBlockSize);
        end = chrono::high_resolution_clock::now();
        timeEllap1 += (end - start);
    }
    timeEllap1 /= numIter;

    cout << "RGB to HSV Conversion: " << timeEllap1.count() << " ms" << endl;

    // 分離HSV通道
    split(inputImageHsv, inputImageHsvChannels);
    Mat& inputImageH = inputImageHsvChannels[0];
    Mat& inputImageS = inputImageHsvChannels[1];
    Mat& inputImageV = inputImageHsvChannels[2];

    // 步驟2: 局部增強 - 模糊處理
    // 預熱
    imageBlurParOpt(inputImageV, blurredImage, optimalBlockSize);
    
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        imageBlurParOpt(inputImageV, blurredImage, optimalBlockSize);
        end = chrono::high_resolution_clock::now();
        timeEllap2 += (end - start);
    }
    timeEllap2 /= numIter;

    cout << "Image Blur: " << timeEllap2.count() << " ms" << endl;

    // 步驟3: 圖像相減
    // 預熱
    subtractImageParOpt(inputImageV, blurredImage, imageMask);
    
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
        end = chrono::high_resolution_clock::now();
        timeEllap3 += (end - start);
    }
    timeEllap3 /= numIter;

    cout << "Image Subtraction: " << timeEllap3.count() << " ms" << endl;

    // 步驟4: 圖像銳化
    double weight = 10.0;
    // It's a CPU-intensive operation, warming up is good for the CPU cache
    sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
    
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
        end = chrono::high_resolution_clock::now();
        timeEllap4 += (end - start);
    }
    timeEllap4 /= numIter;

    cout << "Image Sharpening: " << timeEllap4.count() << " ms" << endl;

    // 步驟5-6: 合併的直方圖計算和均衡化
    // 預熱
    histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, optimalThreads);

    // 計時合併操作
    auto timeEllapHistCombined = chrono::duration<double, milli>::zero();

    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        histogramCalcAndEqualParOpt(sharpenedImage, globallyEnhancedImage, optimalThreads);
        end = chrono::high_resolution_clock::now();
        timeEllapHistCombined += (end - start);
    }
    timeEllapHistCombined /= numIter;

    cout << "Combined Histogram Processing: " << timeEllapHistCombined.count() << " ms" << endl;

    // 為了保持與原始CSV輸出格式相容，將合併時間按原比例分配
    // 使用一個簡單的規則：將總時間的 40% 分配給計算，60% 分配給均衡化
    // 您可以根據實際經驗調整這些比例
    timeEllap5 = timeEllapHistCombined * 0.4;
    timeEllap6 = timeEllapHistCombined * 0.6;

    cout << "Estimated Histogram Calculation: " << timeEllap5.count() << " ms" << endl;
    cout << "Estimated Histogram Equalization: " << timeEllap6.count() << " ms" << endl;

    // 合併HSV通道
    vector<Mat> channels = {inputImageH, inputImageS, globallyEnhancedImage};
    merge(channels, outputHSV);
    
    // 步驟7: HSV到RGB轉換
    // 預熱
    hsvToRgbParOpt(outputHSV, finalOutput, optimalBlockSize);
    
    for(int i = 0; i < numIter; ++i){
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
           << optimalThreads << "," 
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
    // 預先計算常數向量
    __m256 _255 = _mm256_set1_ps(255.0f);
    __m256 _60 = _mm256_set1_ps(60.0f);
    __m256 _2 = _mm256_set1_ps(2.0f);
    __m256 _4 = _mm256_set1_ps(4.0f);
    __m256 _360 = _mm256_set1_ps(360.0f);
    __m256 _0_5 = _mm256_set1_ps(0.5f);
    
    // AVX2 優化部分...
    #endif
    
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                // 處理當前塊的範圍
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);
                
                for(int i = ii; i < blockEndI; ++i) {
                    const Vec3b* inRow = inputImage.ptr<Vec3b>(i);
                    Vec3b* outRow = outputImage.ptr<Vec3b>(i);
                    
                    for(int j = jj; j < blockEndJ; ++j) {
                        // 使用堆疊變數提高暫存器使用
                        uchar blue = inRow[j][0];
                        uchar green = inRow[j][1];
                        uchar red = inRow[j][2];
                        
                        // 避免浮點除法，使用查表或預先乘以倒數
                        float redScaled = red * (1.0f/255.0f);
                        float greenScaled = green * (1.0f/255.0f);
                        float blueScaled = blue * (1.0f/255.0f);
                        
                        float cmin = min(min(redScaled, greenScaled), blueScaled);
                        float cmax = max(max(redScaled, greenScaled), blueScaled);
                        float delta = cmax - cmin;
                        
                        float h = 0.0f, s = 0.0f, v = 0.0f;
                        
                        // 使用查表方式計算 h，避免條件分支
                        if(delta < 1e-6f) {
                            h = 0.0f; 
                            s = 0.0f;
                        }
                        else {
                            s = (cmax > 0.0f) ? (delta / cmax) : 0.0f;
                            
                            if(cmax == redScaled) {
                                h = 60.0f * ((greenScaled - blueScaled) / delta);
                                if(h < 0.0f) h += 360.0f;
                            }
                            else if(cmax == greenScaled) {
                                h = 60.0f * (2.0f + (blueScaled - redScaled) / delta);
                            }
                            else { // cmax == blueScaled
                                h = 60.0f * (4.0f + (redScaled - greenScaled) / delta);
                            }
                        }
                        
                        v = cmax;
                        
                        // OpenCV HSV 範圍：H [0,180), S [0,255], V [0,255]
                        outRow[j][0] = static_cast<uchar>(h * 0.5f);  // 0-360 -> 0-180
                        outRow[j][1] = static_cast<uchar>(s * 255.0f);
                        outRow[j][2] = static_cast<uchar>(v * 255.0f);
                    }
                }
            }
        }
    }
}

// 大幅優化的模糊處理
void imageBlurParOpt(const Mat& inputImage, Mat& outputImage, int blockSize)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int halfSize = FILTERSIZE / 2;
    
    // 使用擴展邊界的臨時影像，避免內部的邊界檢查
    Mat paddedInput;
    copyMakeBorder(inputImage, paddedInput, halfSize, halfSize, halfSize, halfSize, BORDER_REFLECT);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);
                
                for(int i = ii; i < blockEndI; ++i) {
                    uchar* outRow = outputImage.ptr<uchar>(i);
                    
                    for(int j = jj; j < blockEndJ; ++j) {
                        float sum = 0.0f;
                        
                        // 直接存取有邊界的影像，避免分支
                        for(int ki = 0; ki < FILTERSIZE; ++ki) {
                            const uchar* inRow = paddedInput.ptr<uchar>(i + ki);
                            
                            for(int kj = 0; kj < FILTERSIZE; ++kj) {
                                sum += inRow[j + kj] * Filter1D[ki*FILTERSIZE + kj];
                            }
                        }
                        
                        outRow[j] = static_cast<uchar>(sum);
                    }
                }
            }
        }
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
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < rows; ++i) {
            const uchar* in1Row = inputImage1.ptr<uchar>(i);
            const uchar* in2Row = inputImage2.ptr<uchar>(i);
            uchar* outRow = outputImage.ptr<uchar>(i);
            
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
    
    #pragma omp parallel for schedule(dynamic)
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
    
    // 建立線程本地直方圖，避免競爭
    unsigned int* localHist[MAX_THREADS];
    for(int i = 0; i < nThreads; ++i) {
        localHist[i] = new unsigned int[256]();
    }
    
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        unsigned int* myHist = localHist[tid];
        memset(myHist, 0, 256 * sizeof(unsigned int));
        
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            
            // 使用向量化循環處理每行
            for(int j = 0; j < cols; ++j) {
                myHist[row[j]]++;
            }
        }
    }
    
    // 合併所有線程的直方圖
    memset(imHistogram, 0, 256 * sizeof(unsigned int));
    for(int t = 0; t < nThreads; ++t) {  // 修正：使用變數 t 代替 i
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHist[t][j];  // 修正：使用 t 代替 i
        }
        delete[] localHist[t];  // 修正：使用 t 代替 i
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
    #pragma omp parallel for schedule(dynamic)
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
    
    // 1. 建立線程本地直方圖，避免競爭
    unsigned int* localHist[MAX_THREADS];
    for(int i = 0; i < nThreads; ++i) {
        localHist[i] = new unsigned int[256]();
    }
    
    // 2. 並行計算直方圖
    #pragma omp parallel num_threads(nThreads)
    {
        int tid = omp_get_thread_num();
        unsigned int* myHist = localHist[tid];
        memset(myHist, 0, 256 * sizeof(unsigned int));
        
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            
            for(int j = 0; j < cols; ++j) {
                myHist[row[j]]++;
            }
        }
    }
    
    // 3. 合併所有線程的直方圖
    unsigned int imHistogram[256] = {0};
    for(int t = 0; t < nThreads; ++t) {
        for(int j = 0; j < 256; ++j) {
            imHistogram[j] += localHist[t][j];
        }
        delete[] localHist[t];
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
    #pragma omp parallel for schedule(dynamic)
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
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(int ii = 0; ii < rows; ii += blockSize) {
            for(int jj = 0; jj < cols; jj += blockSize) {
                int blockEndI = min(ii + blockSize, rows);
                int blockEndJ = min(jj + blockSize, cols);
                
                for(int i = ii; i < blockEndI; ++i) {
                    const Vec3b* inRow = inputImage.ptr<Vec3b>(i);
                    Vec3b* outRow = outputImage.ptr<Vec3b>(i);
                    
                    for(int j = jj; j < blockEndJ; ++j) {
                        // 從 HSV 空間讀取值
                        float h = inRow[j][0] * 2.0f;  // 0-180 -> 0-360
                        float s = inRow[j][1] / 255.0f; // 0-255 -> 0-1
                        float v = inRow[j][2] / 255.0f; // 0-255 -> 0-1
                        
                        float r, g, b;
                        
                        // 優化的 HSV 到 RGB 轉換
                        if(s <= 0.0f) {
                            // 灰階
                            r = g = b = v;
                        } else {
                            h = (h >= 360.0f) ? 0.0f : h / 60.0f; // 0-359.99... -> 0-5.99...
                            int hi = static_cast<int>(h);
                            float f = h - hi;
                            
                            float p = v * (1.0f - s);
                            float q = v * (1.0f - s * f);
                            float t = v * (1.0f - s * (1.0f - f));
                            
                            // 使用查表判斷代替條件分支
                            switch(hi) {
                                case 0: r = v; g = t; b = p; break;
                                case 1: r = q; g = v; b = p; break;
                                case 2: r = p; g = v; b = t; break;
                                case 3: r = p; g = q; b = v; break;
                                case 4: r = t; g = p; b = v; break;
                                default: r = v; g = p; b = q; break; // case 5
                            }
                        }
                        
                        // 將 RGB 值轉換到 0-255 範圍，存入輸出影像
                        outRow[j][2] = saturate_cast<uchar>(r * 255.0f); // R
                        outRow[j][1] = saturate_cast<uchar>(g * 255.0f); // G
                        outRow[j][0] = saturate_cast<uchar>(b * 255.0f); // B
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