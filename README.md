# YOLOv8 GitHub Pages 物件識別網頁

這是一個可部署在 GitHub Pages 的純前端 YOLOv8 ONNX 物件識別網頁，支援：

1. 上傳圖片後偵測物件
2. 啟用手機後鏡頭即時偵測
3. 在畫面上框出物件、標示類別與信心值
4. 統計各類別數量與總數量

> GitHub Pages 無法直接執行 `.pt` 模型，請先將 `best.pt` 匯出為 `best.onnx`。

## 一、Colab 匯出 ONNX

在 Colab 放入 `best.pt` 後執行：

```python
!pip install -U ultralytics onnx onnxruntime
from ultralytics import YOLO

model = YOLO('/content/best.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
```

匯出後通常會得到：

```text
/content/best.onnx
```

請把 `best.onnx` 放到本專案的：

```text
models/best.onnx
```

## 二、修改類別名稱

開啟 `labels.json`，依照 Roboflow 資料集的類別順序填入名稱，例如：

```json
[
  "百合甲蟲"
]
```

多類別範例：

```json
[
  "Lychee stink bug",
  "scale insect",
  "scarab beetle"
]
```

## 三、GitHub Pages 部署

專案檔案結構：

```text
index.html
style.css
app.js
labels.json
models/best.onnx
README.md
```

上傳到 GitHub repository 後：

1. 進入 repository 的 Settings
2. 點 Pages
3. Source 選 Deploy from a branch
4. Branch 選 main / root
5. 儲存後等待網址產生

## 四、注意事項

- 手機開後鏡頭需要 HTTPS，GitHub Pages 符合此條件。
- 第一次載入模型會比較慢，屬正常現象。
- 若手機跑不動，建議重新匯出較小模型，例如 YOLOv8n，或降低 `imgsz=416`。
- 若框的位置不準，請確認匯出時的 `imgsz` 與 `app.js` 裡的 `MODEL_INPUT_SIZE` 一致。
