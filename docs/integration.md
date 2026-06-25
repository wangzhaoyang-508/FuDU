# Detector Integration Notes

FuDU only needs image-level features and compact detector uncertainty summaries.
This makes it easy to attach to different detection frameworks.

## Feature Extraction

Use a fixed image-level vector for each image. Common choices:

- Global average pooled backbone feature.
- Global average pooled neck feature.
- The detector's image token or pooled encoder feature.

Write one row per image:

```csv
image_id,image_path,f0,f1,f2
img_0001,/data/images/img_0001.jpg,0.01,0.02,-0.03
```

For prototype building, add `label`:

```csv
image_id,label,f0,f1,f2
img_0001,normal,0.01,0.02,-0.03
img_0002,defect,0.80,0.76,0.91
```

## Detection Summaries

FuDU accepts two levels of detail.

### Preferred

Provide class probabilities and localization-bin probabilities:

```json
{"image_id":"img_0001","boxes":[{"class_probs":[0.8,0.1,0.1],"loc_probs":[[0.7,0.2,0.1],[0.6,0.2,0.2],[0.8,0.1,0.1],[0.7,0.2,0.1]]}]}
```

### Framework-Agnostic Fallback

If your detector does not expose localization bins, precompute normalized
entropy values and pass them directly:

```json
{"image_id":"img_0001","boxes":[{"confidence":0.62,"class_entropy_norm":0.72,"loc_entropy_norm":0.55}]}
```

## YOLO-Style Retraining

After scoring, create a selected train list:

```bash
fudu make-yolo-list \
  --scores runs/round_1/scores.csv \
  --output runs/round_1/selected_train.txt
```

Then point your detector training config to `selected_train.txt`. The exact
training command is intentionally left to the user's detector framework.

## Fuzzy Parameter Presets

The default code path uses the nuclear fuel rod parameters from the
supplementary material. The ELES photovoltaic-cell parameters are listed in
[fuzzy_parameters.md](fuzzy_parameters.md). If your dataset has a different
uncertainty distribution, recalibrate the trapezoidal parameters by:

1. collecting `Ug` and `Ud` on the cold-start stream,
2. using quantiles such as P10/P30/P60/P80 as initial anchors,
3. checking false-negative and false-positive coverage, and
4. validating the final sampling surface with domain experts.
