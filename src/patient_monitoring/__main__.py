"""Standard-library command line interface for the M0 environment check."""

import argparse
import platform
from pathlib import Path
import sys


def doctor() -> None:
    """Print observations from this interpreter, without probing hardware."""
    source_root = Path(__file__).resolve().parents[2]
    project_root = (
        str(source_root)
        if (source_root / "pyproject.toml").is_file()
        else "unknown (use an editable installation from the project workspace)"
    )
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"OS: {platform.platform()}")
    print(f"Virtualenv active: {sys.prefix != sys.base_prefix}")
    print(f"Project root: {project_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient monitoring: M0 environment scaffold."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report the observed Python environment.")
    inspect = commands.add_parser("inspect-video", help="Audit a local video's timeline.")
    inspect.add_argument("path", type=Path, help="Path to a newly recorded local video.")
    inspect.add_argument("--output-dir", type=Path, required=True,
                         help="Directory for new metadata.json and timeline.csv files.")
    preview = commands.add_parser("preview-video", help="Export timestamp-selected video frames.")
    preview.add_argument("path", type=Path)
    preview.add_argument("--output-dir", type=Path, required=True,
                         help="New directory for preview images and timing summary.")
    preview.add_argument("--samples", type=int, default=8, choices=range(2, 13))
    pose = commands.add_parser("pose-preview", help="Run the CPU YOLO11n pose smoke test.")
    pose.add_argument("input_dir", type=Path)
    pose.add_argument("--output-dir", type=Path, required=True)
    pose.add_argument("--keypoint-confidence", type=float, default=0.5,
                      help="Display-only confidence threshold (default: 0.5).")
    extract = commands.add_parser("extract-poses", help="Extract poses for every decoded video frame.")
    extract.add_argument("path", type=Path)
    extract.add_argument("--output-dir", type=Path, required=True,
                         help="New directory for poses.jsonl, summary.json and run_info.json.")
    review = commands.add_parser("review-poses", help="Review saved pose detections without inference.")
    review.add_argument("path", type=Path)
    review.add_argument("--poses", type=Path, required=True)
    review.add_argument("--summary", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    target = commands.add_parser("select-target", help="Experimental causal single-target association from saved poses.")
    target.add_argument("path", type=Path, help="Input poses.jsonl")
    target.add_argument("--output-dir", type=Path, required=True)
    target.add_argument("--seed-frame", type=int, required=True)
    target.add_argument("--seed-detection", type=int, required=True)
    target.add_argument("--min-iou", type=float, default=0.1)
    target.add_argument("--ambiguity-margin", type=float, default=0.05)
    target.add_argument("--max-gap", type=float, default=0.5, help="Maximum elapsed seconds since last accepted box")
    target.add_argument("--video", type=Path, help="Optional original video for review images only")
    target.add_argument("--review-frames", type=int, nargs="*", default=[], help="Review-only indices; never used by association")
    benchmark = commands.add_parser("benchmark-pose", help="Short CPU predict-call benchmark using local weights.")
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--weights", type=Path, required=True)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--frames", type=int, default=30)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser('compare-pose-backends', help='Small PT/NCNN consistency check, not accuracy.')
    compare.add_argument('path', type=Path)
    compare.add_argument('--pt-weights', type=Path, required=True)
    compare.add_argument('--ncnn-model', type=Path, required=True)
    compare.add_argument('--output-dir', type=Path, required=True)
    compare.add_argument('--imgsz', type=int, default=320)
    compare.add_argument('--frames', type=int, default=3)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == 'compare-pose-backends':
        try:
            from .pose_comparison import compare_backends
            result = compare_backends(args.path, args.pt_weights, args.ncnn_model, args.output_dir, args.imgsz, args.frames)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f'Comparison failed: {exc}\n')
        print(f"Comparison: {result['status']}; outputs: {args.output_dir.resolve()}")
        if not result['completed']:
            parser.exit(1, f"Comparison incomplete: {result['errors']}\n")
    elif args.command == "benchmark-pose":
        try:
            from .pose_benchmark import benchmark_pose
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Benchmark dependency unavailable: {exc}\n")
        try:
            result = benchmark_pose(args.path, args.weights, args.output_dir, args.imgsz, args.warmup, args.frames)
        except (InspectionError, OSError, ValueError) as exc:
            parser.exit(1, f"Benchmark failed: {exc}\n")
        print(f"Benchmark: {result['status']}; measured={result['statistics']['measured_completed']}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if not result["completed"]:
            parser.exit(1, f"Benchmark incomplete: {result['errors']}\n")
    elif args.command == "select-target":
        try:
            from .target_selection import select_target
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Selection dependency unavailable: {exc}\n")
        if args.review_frames and args.video is None:
            parser.error("--review-frames requires --video")
        try:
            result = select_target(args.path, args.output_dir, args.seed_frame, args.seed_detection,
                                   args.min_iou, args.ambiguity_margin, args.max_gap)
            if result["completed"] and args.video is not None:
                from .target_review import target_review
                review_result = target_review(args.video, args.path, args.output_dir, args.review_frames)
                if review_result["status"] != "complete":
                    parser.exit(1, f"Selection complete; review incomplete: {review_result['errors']}\n")
        except (InspectionError, OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"Target selection failed: {exc}\n")
        print(f"Selection: {result['status']}; statuses: {result['status_counts']}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if not result["completed"]:
            parser.exit(1, f"Selection incomplete: {result['errors']}\n")
    elif args.command == "review-poses":
        try:
            from .pose_review import review_poses
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Review dependency unavailable: {exc}\n")
        try:
            result = review_poses(args.path, args.poses, args.summary, args.output_dir)
        except (InspectionError, OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"Pose review failed: {exc}\n")
        print(f"Review export: {result['export_status']}; human verdict: pending")
        print(f"Multiple-detection frames: {result['actual_multiple_count']}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if result["export_status"] != "complete":
            parser.exit(1, f"Review export incomplete: {result['errors']}\n")
    elif args.command == "extract-poses":
        try:
            from .pose_extraction import extract_poses
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Pose dependency unavailable: {exc}\n")
        try:
            summary = extract_poses(args.path, args.output_dir)
        except (InspectionError, OSError, ValueError) as exc:
            parser.exit(1, f"Pose extraction failed: {exc}\n")
        print(f"Extraction status: {summary['status']}")
        print(f"Decoded/processed: {summary['decoded_frame_count']}/{summary['processed_frame_count']}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if not summary["completed"]:
            parser.exit(1, f"Extraction incomplete: {summary['errors']}\n")
    elif args.command == "pose-preview":
        try:
            from .pose_preview import pose_preview
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Pose dependency unavailable: {exc}\n")
        try:
            run = pose_preview(args.input_dir, args.output_dir, args.keypoint_confidence)
        except (InspectionError, OSError, ValueError) as exc:
            parser.exit(1, f"Pose preview failed: {exc}\n")
        print(f"Pose status: {run['status']}")
        print(f"Images processed: {run['processed_image_count']} / {run['expected_image_count']}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if run["status"] != "complete":
            parser.exit(1, f"Pose inference incomplete: {run['error']}\n")
    elif args.command == "preview-video":
        try:
            from .video_preview import preview_video
            from .video_inspection import InspectionError
        except ImportError as exc:
            parser.exit(1, f"Preview dependency unavailable: {exc}. Install project dependencies from wheels.\n")
        try:
            summary = preview_video(args.path, args.output_dir, args.samples)
        except (InspectionError, OSError, ValueError) as exc:
            parser.exit(1, f"Video preview failed: {exc}\n")
        print(f"Preview status: {summary['preview_status']}")
        print(f"Distinct frames: {summary['distinct_sample_count']} / {args.samples}")
        print(f"Outputs: {args.output_dir.resolve()}")
        if summary["preview_status"] != "complete":
            parser.exit(1, f"Video preview incomplete: {summary['error']}\n")
    elif args.command == "inspect-video":
        # Keep M0 commands independent of the video dependency.
        try:
            from .video_inspection import InspectionError, inspect_video
        except ImportError as exc:
            parser.exit(1, f"Video dependency unavailable: {exc}. Install av==16.1.0 from a binary wheel.\n")
        try:
            metadata = inspect_video(args.path, args.output_dir)
        except (InspectionError, OSError, ValueError) as exc:
            parser.exit(1, f"Video inspection failed: {exc}\n")
        print(f"Decode status: {metadata['decode_status']}")
        print(f"Decoded frames: {metadata['actual_decoded_count']}")
        print(f"Reports: {args.output_dir.resolve()}")
        if not metadata["decode_completed"]:
            parser.exit(1, f"Inspection incomplete: {metadata['error'] or 'see metadata.json warnings'}\n")


if __name__ == "__main__":
    main()
