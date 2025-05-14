# FastImageOps - 高效能影像處理加速工具

![範例影像](snow.png)  
*專案範例輸出 (snow.png)*

## 🚀 專案亮點
- **多線程加速**：支援自定義線程數量，充分發揮多核CPU效能
- **動態尺寸調整**：可即時調整影像處理尺寸，平衡效能與品質
- **模組化設計**：核心演算法與介面分離，便於功能擴展
- **跨平台支援**：基於標準C++與OpenCV，可在Linux/Windows/macOS運行

## 📦 專案結構
| 檔案 | 說明 |
|------|------|
| `serial.cpp` | 序列化運算基礎實現 (對照組) |
| `main.cpp` | **主程式** (修改自 [batuhanhangun/MSc-Thesis](https://github.com/batuhanhangun/MSc-Thesis)) |
| `test.cpp` | 效能測試與基準分析 |
| `run.sh` | 一鍵執行腳本 |
| `snow.png` | 測試用範例影像 |
