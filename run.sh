#!/bin/bash

# 測試圖像
INPUT_IMAGE="chaewon.png"

# 所有要測試的解析度
RESOLUTIONS=("389x518" "778x1036" "1167x1554" "1556x2072" "1945x2590" "2334x3108" "2723x3626")

# 執行緒數量
THREADS=(1 2 4 8 16 32 48)

# 確保結果目錄存在
mkdir -p results

# 編譯所有程式
echo "編譯所有程式..."
g++ -g -Wall -o Vserial Vserial.cpp `pkg-config --cflags --libs opencv4` -std=c++17
g++ -g -Wall -o Vopenmp Vopenmp.cpp `pkg-config --cflags --libs opencv4` -fopenmp -std=c++17
g++ -g -Wall -O3 -o Vproposed Vproposed.cpp `pkg-config --cflags --libs opencv4` -fopenmp -march=native -ffast-math -funroll-loops -ftree-vectorize -std=c++17
g++ -o orig_opencv orig_opencv.cpp -std=c++17 `pkg-config --cflags --libs opencv4`

echo "檢查圖像文件是否存在..."
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "錯誤: 找不到輸入圖像 $INPUT_IMAGE"
    exit 1
fi

# 串行版本
echo "測試 Vserial 版本..."
for res in "${RESOLUTIONS[@]}"; do
    echo "處理解析度 $res"
    ./Vserial "$INPUT_IMAGE" "$res"
done

# OpenMP版本
echo "測試 Vopenmp 版本..."
for res in "${RESOLUTIONS[@]}"; do
    for thread in "${THREADS[@]}"; do
        echo "處理解析度 $res，執行緒數 $thread"
        ./Vopenmp "$INPUT_IMAGE" "$thread" "$res"
    done
done

# 優化版本
echo "測試 Vproposed 版本..."
for res in "${RESOLUTIONS[@]}"; do
    for thread in "${THREADS[@]}"; do
        echo "處理解析度 $res，執行緒數 $thread"
        ./Vproposed "$INPUT_IMAGE" "$thread" "$res"
    done
done

# OpenCV版本
echo "測試 orig_opencv 版本..."
for res in "${RESOLUTIONS[@]}"; do
    echo "處理解析度 $res"
    ./orig_opencv "$INPUT_IMAGE" "$res"
done

# 合併結果
echo "合併所有結果..."
chmod +x merge_results.sh
./merge_results.sh

echo "測試完成。所有結果已保存在 results 目錄中。"
echo "合併的結果可在 results/all_comparison_results.csv 中找到。" 