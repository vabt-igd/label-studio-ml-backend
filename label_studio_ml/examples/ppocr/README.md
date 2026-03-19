<!--
---
title: PP-OCR Text Detection and Recognition
type: guide
tier: all
order: 41
hide_menu: true
hide_frontmatter_title: true
meta_title: PP-OCR model connection for text detection and recognition
meta_description: The PP-OCR model connection integrates PaddleX's OCR pipeline with Label Studio for high-accuracy text detection and recognition supporting 100+ languages.
categories:
    - Computer Vision
    - Optical Character Recognition
    - PaddleOCR
image: "/guide/ml_tutorials/ppocr.png"
---
-->

# PP-OCR Model Connection

The [PP-OCR](https://github.com/PaddlePaddle/PaddleOCR) model connection integrates PaddleX's state-of-the-art OCR pipeline with Label Studio. It provides high-accuracy text detection and recognition with support for 100+ languages.

## Features

- **Multiple Versions**: This backend has been tested with PP-OCRv4 and PP-OCRv5, and is expected to work with future versions as long as the Python API remains compatible (configurable via `PPOCR_VERSION`)
- **High Accuracy**: Achieves state-of-the-art performance on various OCR benchmarks
- **Multi-language Support**: Supports 12 language families covering 100+ languages
- **Two Model Variants**:
  - `mobile`: Fast inference, suitable for real-time applications
  - `server`: Higher accuracy, suitable for batch processing
- **Document Preprocessing**: Optional orientation detection and image unwarping (disabled by default)
- **Flexible Output**: Polygon or rectangle bounding boxes

## Supported Languages

| Code | Language | Notes |
|------|----------|-------|
| `ch` | Chinese (Simplified) | Server and mobile variants (includes Traditional Chinese, Japanese) |
| `en` | English | Server and mobile variants (uses base PP-OCR model) |
| `arabic` | Arabic | Dedicated mobile model |
| `cyrillic` | Cyrillic (Russian, etc.) | Dedicated mobile model |
| `devanagari` | Devanagari (Hindi, etc.) | Dedicated mobile model |
| `el` | Greek | Dedicated mobile model |
| `eslav` | East Slavic | Dedicated mobile model |
| `korean` | Korean | Dedicated mobile model |
| `latin` | Latin-based | Dedicated mobile model |
| `ta` | Tamil | Dedicated mobile model |
| `te` | Telugu | Dedicated mobile model |
| `th` | Thai | Dedicated mobile model |

## Before You Begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the `ppocr` example.

## Labeling Configuration

The PP-OCR model connection works with the default OCR labeling configuration in Label Studio.

### Polygon + TextArea (Recommended)

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
    <Label value="Handwriting" background="blue"/>
  </Labels>

  <Polygon name="poly" toName="image" strokeWidth="3"/>

  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            required="false"
            maxSubmissions="1"
            rows="3"
            placeholder="Recognized text will appear here"
            displayMode="region-list"/>
</View>
```

### Rectangle + TextArea

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
  </Labels>

  <Rectangle name="bbox" toName="image" strokeWidth="3"/>

  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            displayMode="region-list"/>
</View>
```

## Running with Docker (Recommended)

### CPU Version (Default)

1. Start the ML backend:

```bash
docker-compose up -d
```

2. Verify the backend is running:

```bash
curl http://localhost:9090/health
# Expected: {"status":"UP"}
```

### GPU Version (CUDA 11.8)

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### GPU Version (CUDA 12.6)

```bash
CUDA_VERSION=cuda12.6 docker-compose -f docker-compose.gpu.yml up -d
```

### Using a Different PP-OCR Version

```bash
# Use PP-OCRv4
PPOCR_VERSION=v4 docker-compose up -d

# Use PP-OCRv5 (default)
PPOCR_VERSION=v5 docker-compose up -d
```

## Configuration Options

### Environment Variables

Set these in `docker-compose.yml` or via environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `PPOCR_VERSION` | `v5` | PP-OCR version: `v4`, `v5` (tested; future versions expected to work) |
| `MODEL_VARIANT` | `server` | `mobile` (fast) or `server` (accurate) |
| `OCR_LANG` | `ch` | Language code (see table above) |
| `DEVICE` | `cpu` | `cpu`, `gpu:0`, `gpu:1`, etc. |
| `SCORE_THRESHOLD` | `0.5` | Recognition confidence threshold (0.0-1.0) |
| `DET_SCORE_THRESHOLD` | `0.3` | Detection confidence threshold (0.0-1.0) |
| `OUTPUT_TYPE` | `polygon` | `polygon` or `rectangle` |
| `INCLUDE_TRANSCRIPTION` | `true` | Include OCR text in output |
| `USE_DOC_ORIENTATION` | `false` | Enable document orientation detection |
| `USE_DOC_UNWARPING` | `false` | Enable document image correction |
| `USE_TEXTLINE_ORIENTATION` | `false` | Enable text line orientation detection |

### Label Studio Connection (Required)

To access images from Label Studio, you must configure these variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LABEL_STUDIO_URL` | `http://host.docker.internal:8080` | Label Studio server URL |
| `LABEL_STUDIO_API_KEY` | (none) | API key from Label Studio (**Account & Settings** → **Access Token**) |

Edit `docker-compose.yml` and replace `LABEL_STUDIO_API_KEY` with your token:

```yaml
- LABEL_STUDIO_API_KEY=your_token_here
```

## Examples

### Chinese OCR (Default)

```bash
MODEL_VARIANT=server OCR_LANG=ch docker-compose up -d
```

### English OCR

```bash
MODEL_VARIANT=mobile OCR_LANG=en docker-compose up -d
```

### Korean OCR with CPU

```bash
OCR_LANG=korean docker-compose up -d
```

### High-Accuracy Mode

```bash
MODEL_VARIANT=server SCORE_THRESHOLD=0.7 docker-compose up -d
```

### Using PP-OCRv4 with GPU

```bash
PPOCR_VERSION=v4 docker-compose -f docker-compose.gpu.yml up -d
```

## Building from Source

```bash
# CPU
docker-compose build

# GPU
docker-compose -f docker-compose.gpu.yml build
```

## Running Without Docker

1. Install PaddlePaddle following the [official guide](https://www.paddleocr.ai/latest/version3.x/installation.html):

```bash
# CPU version
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU version (CUDA 11.8, requires driver >= 450.80.02)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU version (CUDA 12.6, requires driver >= 550.54.14)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

2. Install PaddleOCR:

```bash
python -m pip install paddleocr
```

3. Install dependencies:

```bash
pip install -r requirements-base.txt
pip install -r requirements.txt
```

4. Start the server:

```bash
python _wsgi.py
```

Or using label-studio-ml:

```bash
label-studio-ml start ./ppocr
```

## Connecting to Label Studio

1. Open Label Studio and go to **Settings** > **Machine Learning**
2. Click **Add Model**
3. Enter:
   - **Title**: `PP-OCR`
   - **URL**: `http://localhost:9090` (or `http://host.docker.internal:9090` if Label Studio is in Docker)
4. Click **Validate and Save**
5. Optionally enable **Use for interactive preannotations**

## Testing

### Direct API Test

```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "id": 1,
      "data": {"image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"}
    }],
    "label_config": "<View><Image name=\"image\" value=\"$image\"/><Polygon name=\"poly\" toName=\"image\"/><TextArea name=\"transcription\" toName=\"image\" perRegion=\"true\"/><Labels name=\"label\" toName=\"image\"><Label value=\"Text\"/></Labels></View>"
  }'
```

### Running Unit Tests

```bash
pip install -r requirements-test.txt
pytest
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Verify backend is running: `docker-compose ps` |
| "Can't resolve url" error | Set `LABEL_STUDIO_URL` environment variable (see Label Studio Connection section) |
| No predictions returned | Check logs: `docker-compose logs ppocr` |
| GPU not detected | Ensure NVIDIA drivers and nvidia-docker are installed |
| Slow first prediction | Model downloads on first run; subsequent calls are faster |
| Label Studio can't connect to ML backend | Use `http://host.docker.internal:9090` instead of `localhost` |
| ML backend can't access Label Studio images | Set both `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` |
| Server variant unavailable | Server variant only supports Chinese (`ch`) and English (`en`); use mobile for other languages |
| PIR/MKL-DNN error on Windows CPU | The model automatically uses `paddle_fp32` run mode to avoid this issue |

## Performance

| Configuration | Typical Inference Time |
|---------------|----------------------|
| GPU (server) | 1-3 seconds |
| GPU (mobile) | 0.5-1 second |
| CPU (server) | 5-10 seconds |
| CPU (mobile) | 2-5 seconds |

## Customization

The ML backend can be customized by modifying files in this directory:

- `model.py`: Main prediction logic
- `_wsgi.py`: Server configuration
- `docker-compose.yml`: Container settings (CPU)
- `docker-compose.gpu.yml`: Container settings (GPU)

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleX Documentation](https://paddlepaddle.github.io/PaddleX/)
- [PaddleOCR Installation Guide](https://www.paddleocr.ai/latest/version3.x/installation.html)
- [Label Studio ML Backend](https://github.com/HumanSignal/label-studio-ml-backend)
