# FastImageOps - 高效能影像處理加速工具

## 🚀 專案亮點
- **多線程加速**：支援自定義線程數量，充分發揮多核CPU效能
- **動態尺寸調整**：可即時調整影像處理尺寸，平衡效能與品質

## 📦 專案結構
| 檔案 | 說明 |
|------|------|
| `Vserial.cpp` | 序列化運算基礎實現 (對照組) |
| `Vopenmp.cpp` | **主程式** (修改自 [batuhanhangun/MSc-Thesis](https://github.com/batuhanhangun/MSc-Thesis)) |
| `Vproposed.cpp` | 效能測試與基準分析 |
| `run.sh` | 一鍵執行腳本 |
| `snow.png` | 測試用範例影像 |
