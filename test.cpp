/*
OpenCV (C++) Implemantation of "Image enhancement with the application of local and global enhancement methods for dark images" by
Singh et. al https://ieeexplore.ieee.org/document/8071892
This implementation includes parallel version which uses OpenMP - Optimized version
Author: Batuhan HANGÜN (Original), Optimized version added

Compilation:
g++ -g -Wall -O3 -o test test.cpp `pkg-config --cflags --libs opencv4` -fopenmp -march=native

Execution:
./test <imagename> <threads> <size>
./test snow.png 4 778x1036
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
#define BLOCK_SIZE 64 // 定義處理區塊大小，優化記憶體階層存取

/* Blurring Mask */
const int Filter[5][5] = {
	{1,4,6,4,1},
	{4,16,24,16,4},
	{6,24,36,24,6},
	{4,16,24,16,4},
	{1,4,6,4,1},
};
const int sumOfElementsInFilter = 256;

// 優化後的HSV色彩空間轉換表
struct HSVtoRGBTable {
    double Rs[6][61]; // 每10度一個索引，總共360度
    double Gs[6][61];
    double Bs[6][61];
    
    HSVtoRGBTable() {
        // 預計算HSV到RGB的轉換值
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h_idx = 0; h_idx < 6; h_idx++) {
            for (int h_val = 0; h_val < 61; h_val++) {
                double hue = h_idx * 60.0 + h_val;
                double C = 1.0; // 假設S和V均為1.0
                double X = C * (1 - std::abs(std::fmod(hue / 60.0, 2) - 1));
                
                if (hue >= 0 && hue < 60) {
                    Rs[h_idx][h_val] = C;
                    Gs[h_idx][h_val] = X;
                    Bs[h_idx][h_val] = 0.0;
                } else if (hue >= 60 && hue < 120) {
                    Rs[h_idx][h_val] = X;
                    Gs[h_idx][h_val] = C;
                    Bs[h_idx][h_val] = 0.0;
                } else if (hue >= 120 && hue < 180) {
                    Rs[h_idx][h_val] = 0.0;
                    Gs[h_idx][h_val] = C;
                    Bs[h_idx][h_val] = X;
                } else if (hue >= 180 && hue < 240) {
                    Rs[h_idx][h_val] = 0.0;
                    Gs[h_idx][h_val] = X;
                    Bs[h_idx][h_val] = C;
                } else if (hue >= 240 && hue < 300) {
                    Rs[h_idx][h_val] = X;
                    Gs[h_idx][h_val] = 0.0;
                    Bs[h_idx][h_val] = C;
                } else { // 300-360
                    Rs[h_idx][h_val] = C;
                    Gs[h_idx][h_val] = 0.0;
                    Bs[h_idx][h_val] = X;
                }
            }
        }
    }
};

HSVtoRGBTable hsvToRgbTable; // 全局轉換表

// 優化後的並行實現函數原型
void rgbToHsvParOpt(Mat& inputImage, Mat& outputImage);
void imageBlurParOpt(Mat& inputImage, Mat& outputImage, const int Kernel[FILTERSIZE][FILTERSIZE]);
void subtractImageParOpt(Mat& inputImage1, Mat& inputImage2, Mat& outputImage);
void sharpenImageParOpt(Mat& inputImage1, Mat& inputImage2, Mat& outputImage, double weight);
void histogramCalcParOpt(const Mat& inputImage, unsigned int imHistogram[]);
void histogramEqualParOpt(const Mat& inputImage, Mat& outputImage, const unsigned int imHistogram[]);
void hsvToRgbParOpt(Mat& inputImage, Mat& outputImage);
string getexepath();

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
        return true;
    } catch (const exception&) {
        return false;
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        cout << "Usage: ./test <imagename> [threads] [size]" << endl;
        cout << "Examples:" << endl;
        cout << "  ./test snow.png                  - Run with default threads (all) and original size" << endl;
        cout << "  ./test snow.png 4                - Run with 4 threads and original size" << endl;
        cout << "  ./test snow.png 4 778x1036       - Run with 4 threads and resize to 778x1036" << endl;
        exit(EXIT_FAILURE);
    }

    // 解析執行緒數量參數
    int numThreads = omp_get_max_threads(); // 預設使用最大執行緒數
    if (argc >= 3) {
        try {
            numThreads = stoi(argv[2]);
            if (numThreads <= 0) {
                numThreads = omp_get_max_threads();
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

    // 設置執行緒數量
    omp_set_num_threads(numThreads);
    cout << "Using " << numThreads << " threads." << endl;

    // 優化路徑處理
    string filename = argv[1];
    size_t last_slash_pos = filename.find_last_of("/\\");
    if (last_slash_pos != string::npos) {
        filename = filename.substr(last_slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    string basename = filename.substr(0, dot_pos);
    string extension = filename.substr(dot_pos);
    
    string sizemodifier = to_string(inputImage.rows) + "_" + to_string(inputImage.cols);
    string threadModifier = "_t" + to_string(numThreads);
    string outimagename = basename + "_" + sizemodifier + threadModifier + "_new" + extension;
    cout << "Output image file name: " << outimagename << endl;

    // 創建CSV結果文件
    string resultpath = basename + "_" + sizeStr + "_threads" + to_string(numThreads) + "_new.csv";
    ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Calculation,Histogram Equalization,HSVtoRGB,Total Time" << endl;
    
    // 步驟1 - 色彩空間轉換
    // 預先分配所有需要的Mat，避免反覆重新分配記憶體
    Mat outputImage;
    outputImage.create(inputImage.size(), inputImage.type());
    
    Mat inputImageHsv;
    inputImageHsv.create(inputImage.size(), inputImage.type());

    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    auto timeEllap1 = chrono::duration<double, milli>::zero();
    auto timeEllap2 = chrono::duration<double, milli>::zero();
    auto timeEllap3 = chrono::duration<double, milli>::zero();
    auto timeEllap4 = chrono::duration<double, milli>::zero();
    auto timeEllap5 = chrono::duration<double, milli>::zero();
    auto timeEllap6 = chrono::duration<double, milli>::zero();
    auto timeEllap7 = chrono::duration<double, milli>::zero();
    int numIter = 10; // 減少迭代次數以加快大圖像的處理

    // 計時RGB到HSV轉換
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        rgbToHsvParOpt(inputImage, inputImageHsv);
        end = chrono::high_resolution_clock::now();
        timeEllap1 += (end - start);
    }
    timeEllap1 /= numIter;

    cout << "RGB to HSV Colorspace Conversion Processing Time for " 
             << inputImage.rows << " X " << inputImage.cols << " image by using " 
             << numThreads << " thread(s) is " << timeEllap1.count() << " ms..." << endl;

    // 分離HSV通道
    vector<Mat> inputImageHsvChannels;
    split(inputImageHsv, inputImageHsvChannels);

    Mat& inputImageH = inputImageHsvChannels[0];
    Mat& inputImageS = inputImageHsvChannels[1];
    Mat& inputImageV = inputImageHsvChannels[2];

    // 步驟2 - 局部增強
    Mat blurredImage;
    Mat imageMask;
    Mat sharpenedImage;
    
    blurredImage.create(inputImageV.size(), inputImageV.type());
    imageMask.create(inputImageV.size(), inputImageV.type());
    sharpenedImage.create(inputImageV.size(), inputImageV.type());

    // 計時圖像模糊處理
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        imageBlurParOpt(inputImageV, blurredImage, Filter);
        end = chrono::high_resolution_clock::now();
        timeEllap2 += (end - start);
    }
    timeEllap2 /= numIter;

    cout << "Image Blur Processing Time for " 
             << inputImage.rows << " X " << inputImage.cols << " image by using " 
             << numThreads << " thread(s) is " << timeEllap2.count() << " ms..." << endl;

    // 計時圖像相減
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        subtractImageParOpt(inputImageV, blurredImage, imageMask);
        end = chrono::high_resolution_clock::now();
        timeEllap3 += (end - start);
    }
    timeEllap3 /= numIter;

    cout << "Image Subtracting Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " thread(s) is " << timeEllap3.count() << " ms..." << endl;

    // 計時圖像銳化
    double weight = 10.0;
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        sharpenImageParOpt(inputImageV, imageMask, sharpenedImage, weight);
        end = chrono::high_resolution_clock::now();
        timeEllap4 += (end - start);
    }
    timeEllap4 /= numIter;

    cout << "Image Sharpening Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " thread(s) is " << timeEllap4.count() << " ms..." << endl;

    // 步驟3 - 全局增強
    unsigned int imHistogram[256] = {0};
    Mat globallyEnhancedImage;
    globallyEnhancedImage.create(sharpenedImage.size(), sharpenedImage.type());

    // 計時直方圖計算
    for(int i = 0; i < numIter; ++i){
        // 在每次迭代前重置直方圖
        fill(imHistogram, imHistogram + 256, 0);
        
        start = chrono::high_resolution_clock::now();
        histogramCalcParOpt(sharpenedImage, imHistogram);
        end = chrono::high_resolution_clock::now();
        timeEllap5 += (end - start);
    }
    timeEllap5 /= numIter;

    cout << "Histogram Calculation Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " threads is " << timeEllap5.count() << " ms..." << endl;

    // 計算一次真實的直方圖用於均衡化
    fill(imHistogram, imHistogram + 256, 0);
    histogramCalcParOpt(sharpenedImage, imHistogram);
    
    // 計時直方圖均衡化
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        histogramEqualParOpt(sharpenedImage, globallyEnhancedImage, imHistogram);
        end = chrono::high_resolution_clock::now();
        timeEllap6 += (end - start);
    }
    timeEllap6 /= numIter;

    cout << "Histogram Equalization Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " threads is " << timeEllap6.count() << " ms..." << endl;

    // 合併HSV通道
    vector<Mat> channels = {inputImageH, inputImageS, globallyEnhancedImage};
    merge(channels, outputImage);
    
    // 計時HSV到RGB轉換和保存結果
    Mat finalOutput;
    finalOutput.create(outputImage.size(), outputImage.type());
    
    for(int i = 0; i < numIter; ++i){
        start = chrono::high_resolution_clock::now();
        hsvToRgbParOpt(outputImage, finalOutput);
        end = chrono::high_resolution_clock::now();
        timeEllap7 += (end - start);
    }
    timeEllap7 /= numIter;

    cout << "HSV to RGB Colorspace Conversion Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " threads is " << timeEllap7.count() << " ms..." << endl;

    // 保存結果圖像
    imwrite(outimagename, finalOutput);
    
    // 計算總處理時間
    auto totalTime = timeEllap1 + timeEllap2 + timeEllap3 + timeEllap4 + timeEllap5 + timeEllap6 + timeEllap7;

    cout << "Total Processing Time for " 
             << inputImageV.rows << " X " << inputImageV.cols << " image by using " 
             << numThreads << " threads is " << totalTime.count() << " ms..." << endl;

    // 寫入結果到CSV文件
    outFile << basename << "," 
           << inputImage.cols << "x" << inputImage.rows << "," 
           << numThreads << "," 
           << timeEllap1.count() << "," 
           << timeEllap2.count() << "," 
           << timeEllap3.count() << "," 
           << timeEllap4.count() << "," 
           << timeEllap5.count() << "," 
           << timeEllap6.count() << "," 
           << timeEllap7.count() << "," 
           << totalTime.count() << endl;

    outFile.close();
    return 0;
}

// 優化版RGB到HSV轉換
void rgbToHsvParOpt(Mat& inputImage, Mat& outputImage)
{
    int nRows = inputImage.rows;
    int nCols = inputImage.cols;
    
    // 使用連續記憶體訪問並進行分塊處理
    bool isInputContinuous = inputImage.isContinuous();
    bool isOutputContinuous = outputImage.isContinuous();
    
    #pragma omp parallel
    {
        double redSc, greenSc, blueSc; // 局部私有變數
        double h, s, v;
        double cmin, cmax, delta;
        
        // 分塊處理以優化記憶體存取
        #pragma omp for schedule(dynamic, 16)
        for(int ii = 0; ii < nRows; ii += BLOCK_SIZE) {
            for(int jj = 0; jj < nCols; jj += BLOCK_SIZE) {
                // 處理當前塊
                for(int i = ii; i < min(ii + BLOCK_SIZE, nRows); ++i) {
                    // 使用指針直接訪問連續記憶體可以提高性能
                    Vec3b* inRow = inputImage.ptr<Vec3b>(i);
                    Vec3b* outRow = outputImage.ptr<Vec3b>(i);
                    
                    for(int j = jj; j < min(jj + BLOCK_SIZE, nCols); ++j) {
                        redSc = inRow[j][2] / 255.0;
                        greenSc = inRow[j][1] / 255.0;
                        blueSc = inRow[j][0] / 255.0;
                        
                        cmin = min(min(redSc, greenSc), blueSc);
                        cmax = max(max(redSc, greenSc), blueSc);
                        delta = cmax - cmin;
                        
                        // 計算HSV值
                        if(delta < 1e-6) {
                            h = 0.0;
                            s = 0.0;
                            v = cmax * 255.0;
                        } else {
                            // 計算色調
                            if(cmax == redSc)
                                h = 60.0 * ((greenSc - blueSc) / delta);
                            else if(cmax == greenSc)
                                h = 120.0 + 60.0 * ((blueSc - redSc) / delta);
                            else // cmax == blueSc
                                h = 240.0 + 60.0 * ((redSc - greenSc) / delta);
                            
                            if(h < 0.0)
                                h += 360.0;
                            
                            h = h / 2.0; // 縮放到0-180範圍
                            v = cmax * 255.0;
                            s = ((cmax < 1e-6) ? 0.0 : ((delta / cmax) * 255.0));
                        }
                        
                        // 賦值給輸出
                        outRow[j][0] = static_cast<uchar>(h);
                        outRow[j][1] = static_cast<uchar>(s);
                        outRow[j][2] = static_cast<uchar>(v);
                    }
                }
            }
        }
    }
}

// 優化版圖像模糊
void imageBlurParOpt(Mat& inputImage, Mat& outputImage, const int Kernel[FILTERSIZE][FILTERSIZE])
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    
    // 預計算邊界，避免在循環內部重複計算
    const int halfSize = FILTERSIZE / 2;
    
    // 使用分塊處理以提高快取命中率
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(int ii = 0; ii < rows; ii += BLOCK_SIZE) {
            for(int jj = 0; jj < cols; jj += BLOCK_SIZE) {
                for(int i = ii; i < min(ii + BLOCK_SIZE, rows); ++i) {
                    uchar* outRow = outputImage.ptr<uchar>(i);
                    
                    for(int j = jj; j < min(jj + BLOCK_SIZE, cols); ++j) {
                        int finalIntens = 0;
                        
                        // 使用本地變數進行累加，減少記憶體訪問
                        for(int k = -halfSize; k <= halfSize; ++k) {
                            int ri = i + k;
                            if(ri < 0 || ri >= rows)
                                continue;
                                
                            const uchar* inRow = inputImage.ptr<uchar>(ri);
                            
                            for(int l = -halfSize; l <= halfSize; ++l) {
                                int cj = j + l;
                                if(cj < 0 || cj >= cols)
                                    continue;
                                    
                                finalIntens += inRow[cj] * Kernel[k+halfSize][l+halfSize];
                            }
                        }
                        
                        outRow[j] = finalIntens / sumOfElementsInFilter;
                    }
                }
            }
        }
    }
}

// 優化版圖像相減
void subtractImageParOpt(Mat& inputImage1, Mat& inputImage2, Mat& outputImage)
{
    const int rows = inputImage1.rows;
    const int cols = inputImage1.cols;
    
    // 如果記憶體連續，可以直接使用指針遍歷
    if(inputImage1.isContinuous() && inputImage2.isContinuous() && outputImage.isContinuous()) {
        const size_t total = rows * cols;
        uchar* in1 = inputImage1.ptr<uchar>(0);
        uchar* in2 = inputImage2.ptr<uchar>(0);
        uchar* out = outputImage.ptr<uchar>(0);
        
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total; ++i) {
            int val = in1[i] - in2[i];
            out[i] = (val < 0) ? 0 : val;
        }
    } else {
        // 如果不連續，按行存取
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < rows; ++i) {
            uchar* in1Row = inputImage1.ptr<uchar>(i);
            uchar* in2Row = inputImage2.ptr<uchar>(i);
            uchar* outRow = outputImage.ptr<uchar>(i);
            
            for(int j = 0; j < cols; ++j) {
                int val = in1Row[j] - in2Row[j];
                outRow[j] = (val < 0) ? 0 : val;
            }
        }
    }
}

// 優化版圖像銳化
void sharpenImageParOpt(Mat& inputImage1, Mat& inputImage2, Mat& outputImage, double weight)
{
    int nchannels = inputImage1.channels();
    int rows = inputImage1.rows;
    int cols = inputImage1.cols * nchannels;
    
    // 檢查連續性，優化記憶體訪問
    if(inputImage1.isContinuous() && inputImage2.isContinuous() && outputImage.isContinuous()) {
        cols = rows * cols;
        rows = 1;
    }
    
    // 減少記憶體訪問次數，提高向量化機會
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < rows; ++i) {
        uchar* p = inputImage1.ptr<uchar>(i);
        uchar* q = inputImage2.ptr<uchar>(i);
        uchar* t = outputImage.ptr<uchar>(i);
        
        for(int j = 0; j < cols; ++j) {
            t[j] = saturate_cast<uchar>(p[j] + (weight * q[j]));
        }
    }
}

// 優化版直方圖計算
void histogramCalcParOpt(const Mat& inputImage, unsigned int imHistogram[])
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    
    // 使用本地直方圖，減少線程間衝突
    const int nThreads = omp_get_max_threads();
    vector<unsigned int*> localHist(nThreads);
    
    for(int i = 0; i < nThreads; ++i) {
        localHist[i] = new unsigned int[256]();
    }
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        unsigned int* myHist = localHist[tid];
        
        // 使用靜態分配以提高緩存友好性
        #pragma omp for schedule(static)
        for(int i = 0; i < rows; ++i) {
            const uchar* row = inputImage.ptr<uchar>(i);
            for(int j = 0; j < cols; ++j) {
                myHist[row[j]]++;
            }
        }
        
        // 合併本地直方圖
        #pragma omp critical
        {
            for(int i = 0; i < 256; ++i) {
                imHistogram[i] += myHist[i];
            }
        }
    }
    
    // 釋放記憶體
    for(int i = 0; i < nThreads; ++i) {
        delete[] localHist[i];
    }
}

// 優化版直方圖均衡化
void histogramEqualParOpt(const Mat& inputImage, Mat& outputImage, const unsigned int imHistogram[])
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int numTotalPixels = rows * cols;
    
    // 計算概率密度函數
    double cumDistFunc[256] = {0.0};
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < 256; ++i) {
        cumDistFunc[i] = static_cast<double>(imHistogram[i]) / static_cast<double>(numTotalPixels);
    }
    
    // 計算累積分佈函數（優化：一次遍歷）
    int transFunc[256] = {0};
    double sumProb = 0.0;
    for(int i = 0; i < 256; ++i) {
        sumProb += cumDistFunc[i];
        transFunc[i] = 255 * sumProb;
    }
    
    // 應用轉換函數
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < rows; ++i) {
        const uchar* inRow = inputImage.ptr<uchar>(i);
        uchar* outRow = outputImage.ptr<uchar>(i);
        
                for(int j = 0; j < cols; ++j) {
            outRow[j] = transFunc[inRow[j]];
        }
    }
}

// 優化版HSV到RGB轉換
void hsvToRgbParOpt(Mat& inputImage, Mat& outputImage)
{
    int channels = inputImage.channels();
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < rows; ++i) {
        Vec3b* inRow = inputImage.ptr<Vec3b>(i);
        Vec3b* outRow = outputImage.ptr<Vec3b>(i);
        
        for(int j = 0; j < cols; ++j) {
            double imHval = inRow[j][0] * 2.0;       // 0-255 -> 0-360
            double imSval = inRow[j][1] / 255.0;     // 0-255 -> 0-1
            double imVval = inRow[j][2] / 255.0;     // 0-255 -> 0-1
            
            // 使用查表來獲取基本RGB分量
            int hue_section = min(5, static_cast<int>(imHval / 60));
            int hue_mod = min(60, static_cast<int>(imHval) % 60);
            
            double Rs = hsvToRgbTable.Rs[hue_section][hue_mod];
            double Gs = hsvToRgbTable.Gs[hue_section][hue_mod];
            double Bs = hsvToRgbTable.Bs[hue_section][hue_mod];
            
            // 計算真實的RGB值（根據S和V進行縮放）
            double C = imSval * imVval;
            double m = imVval - C;
            
            // 對查表結果進行縮放
            Rs = Rs * C + m;
            Gs = Gs * C + m;
            Bs = Bs * C + m;
            
            // 轉換到0-255範圍並賦值
            outRow[j][2] = static_cast<uchar>(Rs * 255.0); // R
            outRow[j][1] = static_cast<uchar>(Gs * 255.0); // G
            outRow[j][0] = static_cast<uchar>(Bs * 255.0); // B
        }
    }
}

// 獲取執行路徑
string getexepath()
{
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return string(result, (count > 0) ? count : 0);
}