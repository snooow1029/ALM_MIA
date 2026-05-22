# Audio Flamingo 3 Membership Inference Attack

這個專案實現了針對 Audio Flamingo 3 多模態音頻-文本模型的 membership inference attack (MIA)。

## 文件說明

### 1. `audio_mia.py` - 核心MIA功能
包含所有MIA相關的函數：
- `raw_values_audio()`: 提取音頻-文本對的loss值
- `audio_perplexity()`: 計算perplexity
- `audio_k_min_probs()`: 計算k-最小機率
- `audio_zlib_ratio()`: 計算zlib壓縮比
- `aggregate_audio_metrics()`: 聚合所有metrics

### 2. `extract_logits.py` - 提取輸出和logits
直接回答您原本的問題：如何提取Audio Flamingo 3的output和logits
- `extract_audio_flamingo_logits()`: 提取完整的logits、機率、top-k predictions
- `analyze_audio_logits()`: 分析logits的統計資訊

### 3. `run_audio_mia.py` - 完整MIA分析腳本
提供命令行介面來運行完整的MIA分析

## 使用方法

### 快速開始 - 提取logits和輸出

```python
from extract_logits import extract_audio_flamingo_logits, load_audio_flamingo_model

# 載入模型
model, tokenizer = load_audio_flamingo_model("nvidia/audio-flamingo-3")

# 提取logits和輸出
results = extract_audio_flamingo_logits(
    model, tokenizer, 
    "path/to/audio.wav", 
    "Please describe the audio"
)

print(f"Generated: {results['generated_text']}")
print(f"Has logits: {results['has_logits']}")

# logits被保存在 audio_flamingo_outputs/ 目錄下
# - logits.npy: 原始logits [seq_len, vocab_size]
# - probabilities.npy: 軟最大機率
# - top_predictions.json: 每個位置的top-k預測
```

### 運行完整MIA分析

```bash
# 準備數據集 (JSON格式)
# {"audio_path": "path/to/audio.wav", "text": "prompt text"}

python run_audio_mia.py \
    --model-path nvidia/audio-flamingo-3 \
    --dataset-path your_dataset.json \
    --output-dir ./mia_results \
    --batch-size 4 \
    --metrics ppl k_min_probs k_max_probs zlib_ratio
```

### 程序化使用

```python
from audio_mia import aggregate_audio_metrics, load_audio_flamingo_model

# 載入模型
model, tokenizer = load_audio_flamingo_model("nvidia/audio-flamingo-3")

# 準備數據集
dataset = {
    "audio_path": ["audio1.wav", "audio2.wav"],
    "text": ["Describe this", "What do you hear?"]
}

# 計算MIA metrics
metrics = aggregate_audio_metrics(
    model, tokenizer, dataset, 
    ["ppl", "k_min_probs", "zlib_ratio"],
    args=None, batch_size=4
)

print(f"Average perplexity: {sum(metrics['ppl'])/len(metrics['ppl'])}")
```

## 支援的Metrics

1. **Perplexity (ppl)**: 模型對文本生成的困惑度
2. **K-min/max probabilities**: k分位數的最小/最大機率
3. **Zlib ratio**: 平均loss與zlib壓縮的比值
4. **Perturbation analysis**: 對文本/音頻擾動的響應
5. **Audio feature similarity**: 音頻特徵相似度

## 針對Audio Flamingo 3的特別考慮

1. **音頻處理**: 使用模型內建的音頻編碼器處理.wav文件
2. **多模態embeddings**: 同時考慮音頻和文本的嵌入向量
3. **注意力機制**: 分析音頻到文本的注意力權重
4. **跨模態相似度**: 計算不同音頻間的特徵相似度

## 數據集格式

### JSON格式
```json
[
  {
    "audio_path": "path/to/audio1.wav",
    "text": "Please describe what you hear"
  },
  {
    "audio_path": "path/to/audio2.wav", 
    "text": "Transcribe this audio"
  }
]
```

### CSV格式
```csv
audio_path,text
path/to/audio1.wav,Please describe what you hear
path/to/audio2.wav,Transcribe this audio
```

## 輸出格式

結果保存為JSON文件，包含：
```json
{
  "ppl": [2.34, 1.87, 3.45],
  "k_min_probs_0.05": [0.12, 0.15, 0.09],
  "zlib_ratio": [0.034, 0.045, 0.028],
  "audio_similarity": [0.76, 0.82, 0.69]
}
```

## 錯誤排除

1. **CUDA錯誤**: 確保有足夠的GPU記憶體
2. **音頻格式錯誤**: 確保音頻是.wav格式
3. **模型載入失敗**: 檢查模型路徑和網路連接
4. **依賴缺失**: 安裝所需套件

```bash
pip install torch transformers librosa pydantic
```

## 注意事項

1. 這個實現基於原始文本LLM MIA代碼改編
2. 針對Audio Flamingo 3的特殊架構進行了優化
3. 支援批次處理以提高效率
4. 包含詳細的錯誤處理和日誌記錄
