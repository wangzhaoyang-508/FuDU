# Data Format

FuDU is detector-agnostic. It consumes features and prediction summaries instead
of private images or detector weights.

## Initial Feature CSV

Used by `fudu build-prototypes`.

Required columns:

- `image_id`: unique image identifier.
- `label`: normal/defect label. By default, `normal` is normal and every other
  value is treated as defect.
- `f0`, `f1`, ...: image-level feature vector.

Example:

```csv
image_id,label,f0,f1
seed_normal_001,normal,0.01,-0.04
seed_defect_001,defect,1.05,0.95
```

## Stream Feature CSV

Used by `fudu score`.

Required columns:

- `image_id`
- `f0`, `f1`, ...

Optional columns:

- `image_path`: copied to `scores.csv`; useful for `make-yolo-list`.

Example:

```csv
image_id,image_path,f0,f1
stream_001,/data/images/stream_001.jpg,0.02,0.01
```

## Detection JSONL

Used by `fudu score`.

Each line is one image record:

```json
{"image_id":"stream_001","boxes":[{"class_probs":[0.96,0.03,0.01],"loc_probs":[[0.9,0.05,0.03,0.02],[0.9,0.05,0.03,0.02],[0.9,0.05,0.03,0.02],[0.9,0.05,0.03,0.02]]}]}
```

Box fields:

- `class_probs`: class probability vector from the detector head.
- `confidence` or `conf`: optional. If absent, FuDU uses `max(class_probs)`.
- `loc_probs`: optional localization-bin distributions, shaped as `(4, K)` for
  left/top/right/bottom. If your detector does not expose this, you may provide
  `loc_entropy_norm` directly or omit localization entropy.
- `class_entropy_norm`: optional normalized classification entropy.
- `loc_entropy_norm` or `h_loc`: optional normalized localization entropy.

If an image has no predicted boxes, use an empty list:

```json
{"image_id":"stream_002","boxes":[]}
```

In that case `Ud=0`; potential false negatives are mainly captured by `Ug`.

## Output Scores CSV

`fudu score` writes:

- `image_id`
- `image_path`
- `global_uncertainty`: PGUQ image-level uncertainty `Ug`.
- `defect_uncertainty`: DeUE image-level uncertainty `Ud`.
- `action`: one of `DNS`, `LS`, `HS`, `MS`.
- `sampling_probability`: one of `0.0`, `0.1`, `0.5`, `1.0`.
- `selected`: `1` if sampled for annotation, otherwise `0`.

