#!/bin/bash

# 結果CSV文件
OUTPUT_CSV="./all_comparison_results.csv"

# 創建CSV標頭
echo "Version,Resolution,Threads,RGBtoHSV,Image_Blur,Image_Subtraction,Image_Sharpening,Histogram_Processing,HSVtoRGB,Total_Time" > $OUTPUT_CSV

# 查找並合併Vserial產生的CSV
echo "處理Vserial結果..."
for csv in $(find results -name "*_Vserial.csv"); do
    if [ -f "$csv" ]; then
        echo "處理文件: $csv"
        resolution=$(echo $csv | grep -o "[0-9]\+x[0-9]\+" || echo "original")
        # 使用 -F, 將CSV正確分隔
        tail -n 1 "$csv" | awk -F, -v version="Serial" -v res="$resolution" '{print version "," res "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 "," $10}' >> $OUTPUT_CSV
    fi
done

# 查找並合併Vopenmp產生的CSV
echo "處理Vopenmp結果..."
for csv in $(find results -name "*_Vopenmp_threads*.csv"); do
    if [ -f "$csv" ]; then
        echo "處理文件: $csv"
        resolution=$(echo $csv | grep -o "[0-9]\+x[0-9]\+" || echo "original")
        threads=$(echo $csv | grep -o "threads[0-9]\+" | grep -o "[0-9]\+")
        # 使用 -F, 將CSV正確分隔
        tail -n 1 "$csv" | awk -F, -v version="OpenMP" -v res="$resolution" '{print version "," res "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 "," $10}' >> $OUTPUT_CSV
    fi
done

# 查找並合併Vproposed產生的CSV
echo "處理Vproposed結果..."
for csv in $(find results -name "*_Vproposed_threads*.csv"); do
    if [ -f "$csv" ]; then
        echo "處理文件: $csv"
        resolution=$(echo $csv | grep -o "[0-9]\+x[0-9]\+" || echo "original")
        threads=$(echo $csv | grep -o "threads[0-9]\+" | grep -o "[0-9]\+")
        # 使用 -F, 將CSV正確分隔
        tail -n 1 "$csv" | awk -F, -v version="Proposed" -v res="$resolution" '{print version "," res "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 "," $10}' >> $OUTPUT_CSV
    fi
done

# 查找並合併orig_opencv產生的CSV
echo "處理orig_opencv結果..."
for csv in $(find results -name "*_opencv.csv"); do
    if [ -f "$csv" ]; then
        echo "處理文件: $csv"
        resolution=$(echo $csv | grep -o "[0-9]\+x[0-9]\+" || echo "original")
        # 使用 -F, 將CSV正確分隔
        tail -n 1 "$csv" | awk -F, -v version="OpenCV" -v res="$resolution" '{print version "," res "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 "," $10}' >> $OUTPUT_CSV
    fi
done

echo "合併完成。所有結果已整合到 $OUTPUT_CSV" 