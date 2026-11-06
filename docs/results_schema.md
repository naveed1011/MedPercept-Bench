# Predictions schema (`results/predictions/`)

## Aggregated file: `results/predictions/<tag>.json`

A JSON object keyed by `image_id`. Each value:

```json
{
  "image_id": "string",
  "task": "detection | referring | point | count_box | count_point | segmentation | matting | ocr | ocr_grounding | visual_prompt | freeform",
  "query": "string or list[str]",
  "boxes_xyxy": [[x1, y1, x2, y2], ...],          // PIXEL space, float
  "labels": ["string", ...],
  "scores": [1.0, ...],                            // model emits no scores; all 1.0
  "masks_b64_png": ["base64-encoded PNG", ...],    // segmentation only; binary {0,255}
  "count": 0,                                      // number of parsed instances
  "rejected": false,                               // true if no instances parsed
  "raw_response": "raw model text with <ref>/<box>/<OBJ> tokens",
  "latency_s": 0.0
}
```

## Per-image files: `results/predictions/per_image/<tag>_<image_id>.json`

Same schema, one file per image.

## Notes

- For `segmentation`/`matting` tasks, `boxes_xyxy` and `masks_b64_png` come from `InferenceResult.boxes` (normalized [0,1] → pixel) and `InferenceResult.masks` (thresholded at 0.5).
- For all other tasks, boxes/labels/count are obtained by `parse_predictions(response, width, height)` on the raw response string.
- `rejected == true` (empty dict from parser) is treated as "model returned None" — used for rejection-rate metrics.
- Count for `count_box` = number of parsed boxes (the prompt asks the model to "count them with boxes").
