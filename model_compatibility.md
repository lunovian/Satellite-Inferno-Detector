# YOLO Model Compatibility Guide

This document explains compatibility issues between different YOLO model versions and the Ultralytics library.

## YOLOv9, YOLOv10, YOLOv11, and YOLOv12 Models

These newer YOLO models require **Ultralytics version 8.1.0 or later**. Using these models with older versions of Ultralytics may result in various errors, including:
```
'Conv' object has no attribute 'bn'
```

### Solution:
```bash
pip install ultralytics>=8.1.0
```

## YOLOv8 Models

YOLOv8 models are compatible with most Ultralytics versions, but for best compatibility use Ultralytics 8.0.0 or later.

## YOLOv5 Models

YOLOv5 models are generally backward compatible with newer Ultralytics versions, but for best results:
- For pure YOLOv5: Use ultralytics 7.x versions
- For mixed YOLOv5/v8 usage: Use ultralytics 8.0.x versions

## General Tips

1. **Check the model filename** - Often the filename contains the YOLO version (v5, v8, v10, etc.)
2. **Match library version to model version** - Use the appropriate Ultralytics version for your models
3. **Mixed model ensembles** - When using models from different YOLO versions, use the latest compatible Ultralytics version

## Troubleshooting

If you encounter model loading errors:

1. Check if your model is YOLOv10 and update Ultralytics if needed
2. Try running with a specific Ultralytics version:
   ```bash
   # For YOLOv10:
   pip install ultralytics>=8.1.0
   
   # For YOLOv8:
   pip install ultralytics==8.0.145
   
   # For YOLOv5:
   pip install ultralytics==7.0.0
   ```
3. Check model file integrity - corrupted model files can cause loading issues

