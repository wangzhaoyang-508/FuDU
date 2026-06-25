"""Command line interface for the FuDU workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fuzzy import PRESETS, FuzzySampler
from .io import read_detection_jsonl, read_feature_csv, read_scores_csv, write_scores_csv
from .prototypes import PrototypeLibrary, build_prototype_library
from .stream import score_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fudu", description="FuDU streaming active-learning utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-prototypes", help="build PGUQ prototypes from labeled features")
    p_build.add_argument("--features", required=True, help="CSV with image_id,label,f0,f1,...")
    p_build.add_argument("--output", required=True, help="output .npz prototype library")
    p_build.add_argument("--normal-label", default="normal", help="label value used for normal samples")
    p_build.add_argument("--n-normal", type=int, default=50, help="number of normal prototypes")
    p_build.add_argument("--n-defect", type=int, default=50, help="number of defect prototypes")
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--normalize", action="store_true", help="L2-normalize features before clustering/scoring")
    p_build.add_argument("--no-calibrate", action="store_true", help="keep alpha=1,beta=0 instead of percentile calibration")
    p_build.add_argument("--alpha", type=float, default=None, help="override PGUQ sigmoid alpha after prototype building")
    p_build.add_argument("--beta", type=float, default=None, help="override PGUQ sigmoid beta after prototype building")
    p_build.set_defaults(func=_cmd_build)

    p_score = subparsers.add_parser("score", help="score stream images and sample with FuDU")
    p_score.add_argument("--features", required=True, help="CSV with image_id,f0,f1,... and optional image_path")
    p_score.add_argument("--detections", required=True, help="JSONL detector outputs")
    p_score.add_argument("--prototypes", required=True, help=".npz prototype library")
    p_score.add_argument("--output", required=True, help="output scores CSV")
    p_score.add_argument("--seed", type=int, default=0)
    p_score.add_argument("--deterministic", action="store_true", help="select by probability threshold instead of random sampling")
    p_score.add_argument("--deterministic-threshold", type=float, default=0.5)
    p_score.add_argument("--w-cls", type=float, default=1.0, help="classification entropy weight w1")
    p_score.add_argument("--w-loc", type=float, default=2.0, help="localization entropy weight w2")
    p_score.add_argument("--divisor", default="paper", choices=["paper", "sum_weights"], help="DeUE normalization")
    p_score.add_argument("--fuzzy-preset", default="nuclear_fuel_rod", choices=sorted(PRESETS), help="membership parameters from the paper supplement")
    p_score.set_defaults(func=_cmd_score)

    p_yolo = subparsers.add_parser("make-yolo-list", help="write selected image paths for detector retraining")
    p_yolo.add_argument("--scores", required=True, help="scores CSV produced by fudu score")
    p_yolo.add_argument("--output", required=True, help="output train list")
    p_yolo.add_argument("--image-root", default=None, help="used when scores CSV has no image_path column")
    p_yolo.add_argument("--suffix", default="", help="optional suffix such as .jpg when using image_id")
    p_yolo.set_defaults(func=_cmd_make_yolo_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_build(args: argparse.Namespace) -> int:
    table = read_feature_csv(args.features, require_labels=True)
    library = build_prototype_library(
        table.features,
        table.labels or [],
        n_normal=args.n_normal,
        n_defect=args.n_defect,
        normal_label=args.normal_label,
        seed=args.seed,
        normalize=args.normalize,
        calibrate=not args.no_calibrate,
    )
    if args.alpha is not None:
        library.alpha = float(args.alpha)
    if args.beta is not None:
        library.beta = float(args.beta)
    library.save(args.output)
    print(
        f"saved prototypes to {args.output} "
        f"(normal={len(library.normal)}, defect={len(library.defect)}, dim={library.feature_dim}, "
        f"alpha={library.alpha:.4f}, beta={library.beta:.4f})"
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    table = read_feature_csv(args.features)
    detections = read_detection_jsonl(args.detections)
    library = PrototypeLibrary.load(args.prototypes)
    records = score_records(
        features=table,
        detections=detections,
        prototypes=library,
        sampler=FuzzySampler.from_preset(args.fuzzy_preset),
        seed=args.seed,
        stochastic=not args.deterministic,
        deterministic_threshold=args.deterministic_threshold,
        deue_kwargs={"w_cls": args.w_cls, "w_loc": args.w_loc, "divisor": args.divisor},
    )
    rows = [record.to_row() for record in records]
    write_scores_csv(args.output, rows)
    selected = sum(int(row["selected"]) for row in rows)
    print(f"wrote {len(rows)} scores to {args.output}; selected={selected}")
    return 0


def _cmd_make_yolo_list(args: argparse.Namespace) -> int:
    rows = read_scores_csv(args.scores)
    selected_rows = [row for row in rows if str(row.get("selected", "")).strip() in {"1", "true", "True"}]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for row in selected_rows:
        path = row.get("image_path") or ""
        if not path:
            if not args.image_root:
                raise ValueError("--image-root is required when selected rows have no image_path")
            path = str(Path(args.image_root) / f"{row['image_id']}{args.suffix}")
        lines.append(path)

    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} selected image paths to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
