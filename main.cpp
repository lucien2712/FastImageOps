/*
OpenCV (C++) Implemantation of "Image enhancement with the application of local and global enhancement methods for dark images" by
Singh et. al https://ieeexplore.ieee.org/document/8071892
This implementation includes parallel version which uses OpenMP
Author: Batuhan HANGÜN

Compilation:
g++ -g -Wall -o main main.cpp `pkg-config --cflags --libs opencv4` -fopenmp

Execution:
./main <imagename> [threads] [size]
./main snow.png 4 778x1036
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
#include <sstream>

#define FILTERSIZE 5

/* Blurring Mask */
const int Filter[5][5] = {
	{1,4,6,4,1},
	{4,16,24,16,4},
	{6,24,36,24,6},
	{4,16,24,16,4},
	{1,4,6,4,1},
};
const int sumOfElementsInFilter = 256;


//Parallel Implementations(1st-OpenMP, 2nd-POSSIX Threads(tentative), 3rd-CUDA(tentative))
void rgbToHsvPar(cv::Mat inputImage, cv::Mat outputImage);
void imageBlurPar(cv::Mat inputImage, cv::Mat outputImage, const int Kernel[FILTERSIZE][FILTERSIZE]);
void subtractImagePar(cv::Mat inputImage1, cv::Mat inputImage2, cv::Mat outputImage);
void sharpenImagePar(cv::Mat inputImage1, cv::Mat inputImage2, cv::Mat outputImage);
void histogramCalcPar(const cv::Mat inputImage, unsigned int imHistogram[]);
void histogramEqualPar(const cv::Mat inputImage, cv::Mat outputImage, const unsigned int imHistogram[]);
void hsvToRgbPar(cv::Mat inputImage, cv::Mat outputImage);
std::string getexepath();
void histogramCalcAndEqualPar(const cv::Mat inputImage, cv::Mat outputImage);

// 解析尺寸字符串 (如 "778x1036")
bool parseSize(const std::string& sizeStr, int& width, int& height) {
    size_t xPos = sizeStr.find('x');
    if (xPos == std::string::npos) {
        return false;
    }
    
    try {
        width = std::stoi(sizeStr.substr(0, xPos));
        height = std::stoi(sizeStr.substr(xPos + 1));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// 檢查圖像是否有效
bool isImageValid(const cv::Mat& image) {
    return !image.empty() && image.rows > 0 && image.cols > 0;
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./main <imagename> [threads] [size]" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  ./main snow.png                  - Run with default threads (all) and original size" << std::endl;
        std::cout << "  ./main snow.png 4                - Run with 4 threads and original size" << std::endl;
        std::cout << "  ./main snow.png 4 778x1036       - Run with 4 threads and resize to 778x1036" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Current working directory: " << getexepath() << std::endl;

    // 解析執行緒數量參數
    int numthread = omp_get_max_threads(); // 預設使用最大執行緒數
    if (argc >= 3) {
        try {
            numthread = std::stoi(argv[2]);
            if (numthread <= 0) {
                numthread = omp_get_max_threads();
                std::cout << "Invalid thread count, using " << numthread << " threads." << std::endl;
            }
        } catch (const std::exception&) {
            std::cout << "Invalid thread count, using " << numthread << " threads." << std::endl;
        }
    }
    
    // 讀取原始圖像
    cv::Mat originalImage = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    
    // 檢查圖像是否正確讀取
    if (!isImageValid(originalImage)) {
        std::cerr << "Error: Could not read image " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // 解析大小參數並調整圖像大小（如果有指定）
    cv::Mat inputImage = originalImage;
    std::string sizeStr = "original";
    
    if (argc >= 4) {
        int width, height;
        if (parseSize(argv[3], width, height)) {
            sizeStr = argv[3];
            cv::resize(originalImage, inputImage, cv::Size(width, height));
            std::cout << "Resized input image to " << width << "x" << height << std::endl;
        } else {
            std::cout << "Invalid size format, using original size: " 
                      << originalImage.cols << "x" << originalImage.rows << std::endl;
        }
    } else {
        std::cout << "Using original size: " << originalImage.cols << "x" << originalImage.rows << std::endl;
    }

    // 設置執行緒數量
    omp_set_num_threads(numthread);
    std::cout << "Using " << numthread << " threads." << std::endl;

    // 優化檔案名處理
    std::string filename = argv[1];
    size_t last_slash_pos = filename.find_last_of("/\\");
    if (last_slash_pos != std::string::npos) {
        filename = filename.substr(last_slash_pos + 1);
    }
    
    size_t dot_pos = filename.find_last_of(".");
    std::string basename = filename.substr(0, dot_pos);
    std::string extension = filename.substr(dot_pos);

    std::string sizemodifier = std::to_string(inputImage.rows) + "_" + std::to_string(inputImage.cols);
    std::string threadModifier = "_t" + std::to_string(numthread);
    std::string outimagename = basename + "_" + sizemodifier + threadModifier + extension;
    std::cout << "Output image file name: " << outimagename << std::endl;

    // 創建CSV結果文件
    std::string resultpath = basename + "_" + sizeStr + "_threads" + std::to_string(numthread) + ".csv";
    std::ofstream outFile;
    outFile.open(resultpath);
    outFile << "Image,Size,Threads,RGBtoHSV,Image Blur,Image Subtraction,Image Sharpening,"
            << "Histogram Processing,HSVtoRGB,Total Time" << std::endl;

    // STEP 1 - COLORSPACE CONVERSION
    // Image was converted into HSV colorspace and V(Luminance component) channel of HSV image is used for local and global enhancement

    cv::Mat outputImage = inputImage.clone(); //Output image Parallel

    //Converting image from RGB to HSV colorspace
    cv::Mat inputImageHsv = inputImage.clone();

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto timeEllap1 = (end - start);
    auto timeEllap2 = (end - start);
    auto timeEllap3 = (end - start);
    auto timeEllap4 = (end - start);
    auto timeEllap5 = (end - start);
    auto timeEllap6 = (end - start);
    auto timeEllap7 = (end - start);
    int numIter = 10; // 減少迭代次數，特別是對大圖像

    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        rgbToHsvPar(inputImage, inputImageHsv);
        end = std::chrono::high_resolution_clock::now();
        timeEllap1 += (end - start);
    }
    timeEllap1 /= numIter;

    std::cout << "RGB to HSV Colorspace Conversion Processing Time for "  << inputImage.rows << " X " << inputImage.cols <<
    " image " << "by using " << numthread << " thread(s) " << "is " <<  std::chrono::duration <double, std::milli> (timeEllap1).count()
    << " ms..." << std::endl;

    //Splitting V channel for later use
    cv::Mat inputImageHsvChannels[3];
    cv::split(inputImageHsv, inputImageHsvChannels);

    cv::Mat inputImageH = inputImageHsvChannels[0];
    cv::Mat inputImageS = inputImageHsvChannels[1];
    cv::Mat inputImageV = inputImageHsvChannels[2];

    // STEP 2 - LOCAL ENHANCEMENT
    // 1.Blurring of the image
    // 2.Subtracting the blurred image from the original image to make the mask
    // 3.Adding mask to the original image(Sharpening the image)

    cv::Mat blurredImage = inputImageV.clone();
    cv::Mat imageMask = inputImageV.clone();

    cv::Mat sharpenedImage(inputImageV.rows, inputImageV.cols, CV_8UC1);

    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        imageBlurPar(inputImageV, blurredImage, Filter);
        end = std::chrono::high_resolution_clock::now();
        timeEllap2 += (end - start);
    }
    timeEllap2 /= numIter;

    std::cout << "Image Blur Processing Time for "  << inputImage.rows << " X " << inputImage.cols <<
    " image " << "by using " << numthread << " thread(s) " << "is " <<  std::chrono::duration <double, std::milli> (timeEllap2).count()
    << " ms..." << std::endl;

    //Subtracting the Blurred Image from the Original Image

    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        subtractImagePar(inputImageV, blurredImage, imageMask);
        end = std::chrono::high_resolution_clock::now();
        timeEllap3 += (end - start);
    }
    timeEllap3 /= numIter;

    std::cout << "Image Subtracting Processing Time for "  << inputImageV.rows << " X " << inputImageV.cols <<
    " image by using "  << numthread << " thread(s) " << "is " <<  std::chrono::duration <double, std::milli> (timeEllap3).count()  << " ms..." << std::endl;

    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        sharpenImagePar(inputImageV, imageMask, sharpenedImage);
        end = std::chrono::high_resolution_clock::now();
        timeEllap4 += (end - start);
    }
    timeEllap4 /= numIter;

    std::cout << "Image Sharpening Processing Time for "  << inputImageV.rows << " X " << inputImageV.cols <<
    " image by using "  << numthread << " thread(s) " << "is " <<  std::chrono::duration <double, std::milli> (timeEllap4).count()  << " ms..." << std::endl;

    // STEP 3 - GLOBAL ENHANCEMENT
    // Histogram Calculation and Equalization (Combined)
    cv::Mat locallyEnhancedImage = sharpenedImage.clone();
    cv::Mat locallyEnhancedImageTemp = sharpenedImage.clone();//Temp val for time eval

    cv::Mat globallyEnhancedImage = locallyEnhancedImage.clone();
    cv::Mat globallyEnhancedImageTemp = locallyEnhancedImage.clone();//Temp val for time eval

    // 執行處理（單次，無計時）
    histogramCalcAndEqualPar(locallyEnhancedImage, globallyEnhancedImage);

    // 時間評估 - 合併的直方圖計算和均衡化
    auto timeEllapHistogram = (end - start); // 初始化為 0
    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        histogramCalcAndEqualPar(locallyEnhancedImageTemp, globallyEnhancedImageTemp);
        end = std::chrono::high_resolution_clock::now();
        timeEllapHistogram += (end - start);
    }
    timeEllapHistogram /= numIter;

    std::cout << "Combined Histogram Calculation and Equalization Processing Time for "
             << inputImageV.rows << " X " << inputImageV.cols
             << " image by using " << numthread << " threads is "
             << std::chrono::duration<double, std::milli>(timeEllapHistogram).count()
             << " ms..." << std::endl;

    // 為了與原始程式碼保持相容性，我們需要保留原始的 timeEllap5 和 timeEllap6
    // 可以人為地將合併後的時間分配至這兩個變數
    // 分配比例可以是 50/50 或基於您的實驗結果設定其他比例

    // 假設將總時間的 50% 分配給 timeEllap5，另 50% 分配給 timeEllap6
    timeEllap5 = timeEllapHistogram / 2;
    timeEllap6 = timeEllapHistogram / 2;

    std::cout << "Estimated Histogram Calculation Time: " 
             << std::chrono::duration<double, std::milli>(timeEllap5).count() 
             << " ms" << std::endl;
    std::cout << "Estimated Histogram Equalization Time: " 
             << std::chrono::duration<double, std::milli>(timeEllap6).count() 
             << " ms" << std::endl;

    // Merging of enhanced H band and S, V bands
    cv::Mat channels[3] = {inputImageH, inputImageS, globallyEnhancedImage};
    cv::merge(channels, 3, outputImage);
    cv::Mat outputImageTemp = outputImage.clone();//Temp val for time eval

    //Creating results
    hsvToRgbPar(outputImage, outputImage);
    cv::imwrite(outimagename, outputImage);

    //Time Evaluation
    for(int i = 0; i < numIter; ++i){
        start = std::chrono::high_resolution_clock::now();
        hsvToRgbPar(outputImageTemp, outputImageTemp);
        end = std::chrono::high_resolution_clock::now();
        timeEllap7 += (end - start);
    }
    timeEllap7 /= numIter;

    std::cout << "HSV to RGB Colorspace Conversion Processing Time for "  << inputImageV.rows << " X " << inputImageV.cols <<
    " image by using " << numthread << " threads is "  <<  std::chrono::duration <double, std::milli> (timeEllap7).count()
    << " ms..." << std::endl;

    auto totalTime = timeEllap1 + timeEllap2 + timeEllap3 + timeEllap4 + timeEllapHistogram + timeEllap7;

    std::cout << "Total Processing Time for "  << inputImageV.rows << " X " << inputImageV.cols <<
    " image by using " << numthread << " threads is "  <<  std::chrono::duration <double, std::milli> (totalTime).count() << " ms..." << std::endl;

    auto t1 = std::chrono::duration <double, std::milli> (timeEllap1).count();
    auto t2 = std::chrono::duration <double, std::milli> (timeEllap2).count();
    auto t3 = std::chrono::duration <double, std::milli> (timeEllap3).count();
    auto t4 = std::chrono::duration <double, std::milli> (timeEllap4).count();
    auto t5 = std::chrono::duration <double, std::milli> (timeEllap5).count();
    auto t6 = std::chrono::duration <double, std::milli> (timeEllap6).count();
    auto t7 = std::chrono::duration <double, std::milli> (timeEllap7).count();
    auto t8 = std::chrono::duration <double, std::milli> (totalTime).count();

    // 寫入結果到CSV文件
    outFile << basename << "," 
           << inputImage.cols << "x" << inputImage.rows << "," 
           << numthread << "," 
           << t1 << "," 
           << t2 << "," 
           << t3 << "," 
           << t4 << "," 
           << std::chrono::duration <double, std::milli> (timeEllapHistogram).count() << "," 
           << t7 << "," 
           << t8 << std::endl;

    outFile.close();

    return 0;
}

// 保持原始實現函數不變
void rgbToHsvPar(cv::Mat inputImage, cv::Mat outputImage)
{
    // 原始代碼，未更改
	double redSc = 0, greenSc = 0, blueSc = 0; //Scaled R, G, B values of current pixel
	double h = 0, s = 0, v = 0; //R, G, B values of current pixel
	double cmin = 0, cmax = 0; //Min and max dummy variables
	double delta = 0; //Difference between min and max

	int nRows = inputImage.rows;
	int nCols = inputImage.cols;

	#pragma omp parallel for shared(inputImage, outputImage, nRows, nCols) private(h, s, v, redSc, greenSc, blueSc, cmin, cmax, delta)
	for(int i = 0; i < nRows; ++i){
		for(int j = 0; j < nCols; ++j){
			redSc = inputImage.at<cv::Vec3b>(i,j)[2] / 255.;
			greenSc = inputImage.at<cv::Vec3b>(i,j)[1] / 255.;
			blueSc = inputImage.at<cv::Vec3b>(i,j)[0] / 255.;

			cmin = std::min(std::min(redSc, greenSc), blueSc);
			cmax =  std::max(std::max(redSc, greenSc), blueSc);
			delta = cmax - cmin;
			if(!delta){
				h = 0.;
				s = 0.;
				v = cmax * 255.;
			}
			else{
				if(cmax == redSc)
				h = 60. * ((greenSc - blueSc)/delta);

				if(cmax == greenSc)
				h = 120 + (60. * (((blueSc - redSc)/delta)));

				if(cmax == blueSc)
				h = 240 + (60. * (((redSc - greenSc)/delta)));

				if(h < 0)
				h += 360;

				h = (h/2);
				v = cmax* 255.;
				s = ((cmax==0)?0:((delta/cmax)*255.));

				outputImage.at<cv::Vec3b>(i,j)[0] = h;
				outputImage.at<cv::Vec3b>(i,j)[1] = s;
				outputImage.at<cv::Vec3b>(i,j)[2] = v;
			}
		}
	}
}

void imageBlurPar(cv::Mat inputImage, cv::Mat outputImage, const int Kernel[FILTERSIZE][FILTERSIZE])
{
    // 原始代碼，未更改
	int curIntens = 0;
	int finalIntens = 0;

	#pragma omp parallel for shared(inputImage, outputImage, Filter) private (curIntens, finalIntens) 
	for(int i = 0; i < inputImage.rows; i++)
	{
		for(int j = 0; j < inputImage.cols; j++)
		{
			for(int k = -2; k <= 2; k++)
			{
				if(i+k < 0 || i+k> (inputImage.rows-1))
				continue;

				for(int l = -2; l <= 2; l++)
				{
					if(j+l < 0 || j+l > (inputImage.cols-1))
					continue;

					curIntens = inputImage.at<uchar>(i+k, j+l) * Filter[k+2][l+2];
					finalIntens += curIntens;
				}
			}
			outputImage.at<uchar>(i, j) = finalIntens / sumOfElementsInFilter;
			finalIntens = 0;
		}
	}
}

void subtractImagePar(cv::Mat inputImage1, cv::Mat inputImage2, cv::Mat outputImage)
{
    // 原始代碼，未更改
	int iVal = 0;
	#pragma omp parallel for shared(inputImage1, inputImage2, outputImage) private (iVal) 
	for(int i = 0; i < inputImage1.rows; ++i)
	{
		for(int j = 0; j < inputImage1.cols; ++j)
		{
			iVal = cv::saturate_cast<uchar>(inputImage1.at<uchar>(i,j) - inputImage2.at<uchar>(i,j));

			if(iVal < 0){
				outputImage.at<uchar>(i,j) = 0;
			}
			else{
				outputImage.at<uchar>(i,j) = iVal;
			}
		}
	}
}

void sharpenImagePar(cv::Mat inputImage1, cv::Mat inputImage2, cv::Mat outputImage)
{
    // 原始代碼，未更改
	int nchannels = inputImage1.channels();
	int nRows = inputImage1.rows;
	int nCols = inputImage1.cols*nchannels;
	double weight = 10.0;

	if (inputImage1.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	uchar* p;
	uchar* q;
	uchar* t;

	for(int i = 0; i < nRows; ++i){
		p = inputImage1.ptr<uchar>(i);
		q = inputImage2.ptr<uchar>(i);
		t = outputImage.ptr<uchar>(i);

		#pragma omp parallel for shared(inputImage1, inputImage2, outputImage, weight) 
		for(int j = 0; j < nCols; ++j){
			t[j] = cv::saturate_cast<uchar>(p[j] + (weight * q[j]));
		}
	}
}

void histogramCalcPar(const cv::Mat inputImage, unsigned int imHistogram[])
{
    // 原始代碼，未更改
	#pragma omp parallel for reduction(+: imHistogram[:256])
	for(int i=0; i<inputImage.rows; ++i){
		for(int j=0; j<inputImage.cols; ++j){
			imHistogram[inputImage.at<uchar>(i,j)] += 1;
		}
	}
}

void histogramEqualPar(const cv::Mat inputImage, cv::Mat outputImage, const unsigned int imHistogram[])
{
    // 原始代碼，未更改
	int numTotalPixels = inputImage.rows * inputImage.cols; 

	double cumDistFunc[256] = {.0};
	double sumProb  = 0.0;

	#pragma omp parallel for
	for(int i = 0; i < 256; ++i){
		cumDistFunc[i] = static_cast<double>(imHistogram[i])/static_cast<double>(numTotalPixels);
	}

	int transFunc[256] = {0}; //Transfer function to convert source histogram to target histogram

	#pragma omp parallel for reduction(+: sumProb)
	for(int i = 0; i < 256; i++){
		sumProb = 0.0;
		for(int j = 0; j <= i; j++){
			sumProb += cumDistFunc[j];
		}
		transFunc[i] = 255 * sumProb;
	}

	#pragma omp parallel for shared(outputImage)
	for(int i = 0; i < inputImage.rows; i++){
		for(int j = 0; j < inputImage.cols; j++){
			outputImage.at<uchar>(i,j) = transFunc[inputImage.at<uchar>(i,j)];
		}
	}
}

void hsvToRgbPar(cv::Mat inputImage, cv::Mat outputImage)
{
    // 原始代碼，未更改
	int channels = inputImage.channels();
	int nRows = inputImage.rows;
	int nCols = inputImage.cols*channels;
	uchar* p;
	uchar* q;

	if (inputImage.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	double imHval = 0.0, imSval = 0.0, imVval = 0.0;
	double C = 0.0, X = 0.0, m = 0.0, Rs = 0.0, Gs = 0.0, Bs = 0.0;
	int Rn = 0, Gn = 0, Bn = 0;
	
	for(int i = 0; i < nRows; ++i){
		p = inputImage.ptr<uchar>(i);
		q = outputImage.ptr<uchar>(i);
		#pragma omp parallel for shared(inputImage, outputImage, nRows, nCols) private(imHval, imSval, imVval, C, X, m ,Rs, Gs, Bs, Rn, Gn, Bn)
		for(int j = 0; j < nCols; j = j +3){
			imHval = p[j] * 2.0;          //  0 <= H <= 255 --> 0 <= H < 360
			imSval = p[j+1] / 255.0;       // 0 <= S <= 255 ---> 0 <= S <= 1
			imVval = p[j+2] / 255.0;      //  0 <= V <= 255 ---> 0 <= V <= 1

			C = imSval * imVval;
			X = C * (1 - abs(fmod(imHval / 60, 2)-1));
			m = imVval - C;

			if(imHval >= 0 && imHval < 60) {
				Rs = C;
				Gs = X;
				Bs = 0.0;
			}
			else if(imHval >= 60 && imHval < 120) {
				Rs = X;
				Gs = C;
				Bs = 0.0;
			}
			else if(imHval >= 120 && imHval< 180) {
				Rs = 0.0;
				Gs = C;
				Bs = X;
			}
			else if(imHval >= 180 && imHval < 240) {
				Rs = 0.0;
				Gs = X;
				Bs = C;
			}
			else if(imHval >= 240 && imHval < 300) {
				Rs = X;
				Gs = 0.0;
				Bs = C;
			}
			else if(imHval >= 300 && imHval < 360) {
				Rs = C;
				Gs = 0.0;
				Bs = X;
			}

			Rn = (Rs + m) * 255;
			Gn = (Gs + m) * 255;
			Bn = (Bs + m) * 255;

			Rn = static_cast<int>(Rn);
			Gn = static_cast<int>(Gn);
			Bn = static_cast<int>(Bn);

			q[j+2] = Rn; //R
			q[j+1] = Gn; //G
			q[j]   = Bn; //B
		}
	}
}

std::string getexepath()
{
	char result[ PATH_MAX ];
	ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
	return std::string( result, (count > 0) ? count : 0 );
}

// 新合併的直方圖計算和均衡化函數
void histogramCalcAndEqualPar(const cv::Mat inputImage, cv::Mat outputImage)
{
    int numTotalPixels = inputImage.rows * inputImage.cols; 
    unsigned int imHistogram[256] = {0};
    
    // 1. 計算直方圖
    #pragma omp parallel for reduction(+: imHistogram[:256])
    for(int i=0; i<inputImage.rows; ++i){
        for(int j=0; j<inputImage.cols; ++j){
            imHistogram[inputImage.at<uchar>(i,j)] += 1;
        }
    }
    
    // 2. 計算每個灰階值的機率密度
    double cumDistFunc[256] = {.0};
    #pragma omp parallel for
    for(int i = 0; i < 256; ++i){
        cumDistFunc[i] = static_cast<double>(imHistogram[i])/static_cast<double>(numTotalPixels);
    }
    
    // 3. 建立轉換函數
    int transFunc[256] = {0}; 
    double sumProb = 0.0;
    
    #pragma omp parallel for reduction(+: sumProb)
    for(int i = 0; i < 256; i++){
        sumProb = 0.0;
        for(int j = 0; j <= i; j++){
            sumProb += cumDistFunc[j];
        }
        transFunc[i] = 255 * sumProb;
    }
    
    // 4. 應用轉換函數到輸出影像
    #pragma omp parallel for shared(outputImage)
    for(int i = 0; i < inputImage.rows; i++){
        for(int j = 0; j < inputImage.cols; j++){
            outputImage.at<uchar>(i,j) = transFunc[inputImage.at<uchar>(i,j)];
        }
    }
}