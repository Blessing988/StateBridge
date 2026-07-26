"""Command line entrypoints for the SynWTS scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_caption_predictions, evaluate_vqa_predictions, write_eval_report
from .exporters import export_llamafactory, load_records, write_llamafactory_dataset_info
from .index import build_index, write_index
from .io import write_json
from .validators import validate_caption_submission, validate_vqa_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="SynWTS tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Build dataset index JSONL")
    p_index.add_argument("--dataset-root", required=True)
    p_index.add_argument("--output", required=True)
    p_index.add_argument("--splits", default="train,val")
    p_index.add_argument("--relative-paths", action="store_true")

    p_wts_index = subparsers.add_parser("index-wts-test", help="Build index for WTS public test layout")
    p_wts_index.add_argument("--test-root", required=True)
    p_wts_index.add_argument("--output", required=True)
    p_wts_index.add_argument("--relative-paths", action="store_true")

    p_wts_vqa = subparsers.add_parser("export-wts-test-vqa", help="Export WTS public VQA test inference dataset")
    p_wts_vqa.add_argument("--test-root", required=True)
    p_wts_vqa.add_argument("--vqa-json", required=True)
    p_wts_vqa.add_argument("--output", required=True)
    p_wts_vqa.add_argument("--dataset-info-output")
    p_wts_vqa.add_argument("--dataset-name", default="wts_public_test_vqa")
    p_wts_vqa.add_argument("--clip-output-root")
    p_wts_vqa.add_argument("--make-clips", action="store_true")
    p_wts_vqa.add_argument("--overwrite", action="store_true")
    p_wts_vqa.add_argument("--ffmpeg-bin", default="ffmpeg")
    p_wts_vqa.add_argument("--clip-mode", choices=("h264", "mpeg4", "copy"), default="mpeg4")
    p_wts_vqa.add_argument("--bbox-mode", choices=("none", "summary"), default="none")
    p_wts_vqa.add_argument("--frame-width", type=int, default=1920)
    p_wts_vqa.add_argument("--frame-height", type=int, default=1080)
    p_wts_vqa.add_argument("--max-videos-per-row", type=int, default=0)
    p_wts_vqa.add_argument("--env-clip-duration", type=float)
    p_wts_vqa.add_argument("--visual-variants", default="")
    p_wts_vqa.add_argument("--visual-output-root")
    p_wts_vqa.add_argument("--visual-max-clips-per-row", type=int, default=0)
    p_wts_vqa.add_argument("--visual-crop-padding", type=float, default=0.35)
    p_wts_vqa.add_argument("--visual-crop-size", type=int, default=768)
    p_wts_vqa.add_argument("--visual-max-tracks-per-role", type=int, default=2)

    p_export = subparsers.add_parser("export-llamafactory", help="Export LLaMA-Factory SFT JSON")
    p_export.add_argument("--index", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--tasks", default="caption,vqa")
    p_export.add_argument("--media-policy", choices=("all", "first", "none"), default="all")
    p_export.add_argument("--include-missing-media", action="store_true")
    p_export.add_argument("--bbox-mode", choices=("none", "summary"), default="none")
    p_export.add_argument("--frame-width", type=int, default=1920)
    p_export.add_argument("--frame-height", type=int, default=1080)
    p_export.add_argument("--phase-clip-manifest")
    p_export.add_argument("--dataset-info-output")
    p_export.add_argument("--dataset-name", default="synwts_train")

    p_frame_export = subparsers.add_parser(
        "export-frame-dataset",
        help="Convert a video LLaMA-Factory dataset into an image-frame dataset",
    )
    p_frame_export.add_argument("--dataset", required=True)
    p_frame_export.add_argument("--output", required=True)
    p_frame_export.add_argument("--frame-root", required=True)
    p_frame_export.add_argument("--dataset-info-output")
    p_frame_export.add_argument("--dataset-name")
    p_frame_export.add_argument("--ffmpeg-bin", default="ffmpeg")
    p_frame_export.add_argument("--ffprobe-bin", default="ffprobe")
    p_frame_export.add_argument("--frame-time", default="middle")
    p_frame_export.add_argument("--max-frames-per-row", type=int, default=0)
    p_frame_export.add_argument("--overwrite", action="store_true")

    p_pref = subparsers.add_parser("export-vqa-preference", help="Export VQA chosen/rejected pairs for DPO")
    p_pref.add_argument("--index", required=True)
    p_pref.add_argument("--output", required=True)
    p_pref.add_argument("--media-policy", choices=("all", "first", "none"), default="all")
    p_pref.add_argument("--include-missing-media", action="store_true")
    p_pref.add_argument("--bbox-mode", choices=("none", "summary"), default="none")
    p_pref.add_argument("--frame-width", type=int, default=1920)
    p_pref.add_argument("--frame-height", type=int, default=1080)
    p_pref.add_argument("--phase-clip-manifest")
    p_pref.add_argument("--dataset-info-output")
    p_pref.add_argument("--dataset-name", default="synwts_vqa_preference")
    p_pref.add_argument("--negative-policy", choices=("all", "first", "random"), default="all")
    p_pref.add_argument("--max-rejected-per-question", type=int, default=0)
    p_pref.add_argument("--response-mode", choices=("letter", "letter_text"), default="letter")
    p_pref.add_argument("--seed", type=int, default=13)

    p_hard_candidates = subparsers.add_parser(
        "export-vqa-hard-negative-candidates",
        help="Export one multimodal option-scoring row per VQA question",
    )
    p_hard_candidates.add_argument("--index", required=True)
    p_hard_candidates.add_argument("--output", required=True)
    p_hard_candidates.add_argument("--media-policy", choices=("all", "first", "none"), default="all")
    p_hard_candidates.add_argument("--include-missing-media", action="store_true")
    p_hard_candidates.add_argument("--bbox-mode", choices=("none", "summary"), default="none")
    p_hard_candidates.add_argument("--frame-width", type=int, default=1920)
    p_hard_candidates.add_argument("--frame-height", type=int, default=1080)
    p_hard_candidates.add_argument("--phase-clip-manifest")

    p_score_options = subparsers.add_parser(
        "score-vqa-options",
        help="Score every VQA answer option with Qwen3-VL and an optional SFT adapter",
    )
    p_score_options.add_argument("--candidates", required=True)
    p_score_options.add_argument("--output", required=True)
    p_score_options.add_argument("--model-name-or-path", required=True)
    p_score_options.add_argument("--adapter-name-or-path")
    p_score_options.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p_score_options.add_argument("--attn-implementation", default="sdpa")
    p_score_options.add_argument("--device-map", default="auto")
    p_score_options.add_argument("--video-max-pixels", type=int, default=65536)
    p_score_options.add_argument("--fps", type=float, default=2.0)
    p_score_options.add_argument("--response-mode", choices=("letter", "letter_text"), default="letter")
    p_score_options.add_argument("--include-eos", action="store_true")
    p_score_options.add_argument("--no-resume", action="store_true")
    p_score_options.add_argument("--max-rows", type=int, default=0)
    p_score_options.add_argument("--num-shards", type=int, default=1)
    p_score_options.add_argument("--shard-index", type=int, default=0)
    p_score_options.add_argument("--trust-remote-code", action="store_true")

    p_export_public_options = subparsers.add_parser(
        "export-vqa-option-candidates-from-inference",
        help="Export option-scoring candidates from a VQA inference dataset",
    )
    p_export_public_options.add_argument("--dataset", required=True)
    p_export_public_options.add_argument("--output", required=True)
    p_export_public_options.add_argument(
        "--prompt-variant",
        choices=("base", "direct", "evidence", "anti-prior"),
        default="base",
    )

    p_assemble_option_scores = subparsers.add_parser(
        "assemble-vqa-option-scores",
        help="Assemble official VQA predictions from option-score JSONL files",
    )
    p_assemble_option_scores.add_argument("--candidates", required=True)
    p_assemble_option_scores.add_argument("--scores", nargs="+", required=True)
    p_assemble_option_scores.add_argument("--output", required=True)
    p_assemble_option_scores.add_argument("--weights")
    p_assemble_option_scores.add_argument(
        "--normalization",
        choices=("none", "center", "zscore"),
        default="center",
    )
    p_assemble_option_scores.add_argument("--fallback")
    p_assemble_option_scores.add_argument("--report-output")

    p_hard_build = subparsers.add_parser(
        "build-vqa-hard-negative-preference",
        help="Select model-confusable wrong answers and export a DPO dataset",
    )
    p_hard_build.add_argument("--candidates", required=True)
    p_hard_build.add_argument("--scores", nargs="+", required=True)
    p_hard_build.add_argument("--output", required=True)
    p_hard_build.add_argument("--report-output")
    p_hard_build.add_argument("--dataset-info-output")
    p_hard_build.add_argument("--dataset-name", default="synwts_vqa_hard_negative")
    p_hard_build.add_argument(
        "--selection",
        choices=("all", "errors", "margin", "errors_and_margin"),
        default="errors_and_margin",
    )
    p_hard_build.add_argument("--max-gold-margin", type=float, default=2.0)
    p_hard_build.add_argument(
        "--balance-fields",
        default="question_type,scope,phase,correct",
    )
    p_hard_build.add_argument("--max-per-group", type=int, default=0)
    p_hard_build.add_argument("--max-rows", type=int, default=0)
    p_hard_build.add_argument("--response-mode", choices=("letter", "letter_text"), default="letter")
    p_hard_build.add_argument(
        "--remap-option-letters",
        action="store_true",
        help="Rewrite option labels so chosen/rejected response letters are balanced.",
    )
    p_hard_build.add_argument("--allow-missing-scores", action="store_true")

    p_caption_bank = subparsers.add_parser(
        "build-caption-style-bank",
        help="Build a bank of synthetic caption exemplars for retrieval-conditioned captioning",
    )
    p_caption_bank.add_argument("--index", required=True)
    p_caption_bank.add_argument("--output", required=True)
    p_caption_bank.add_argument("--splits", default="train")
    p_caption_bank.add_argument("--bbox-mode", choices=("none", "summary"), default="summary")
    p_caption_bank.add_argument("--frame-width", type=int, default=1920)
    p_caption_bank.add_argument("--frame-height", type=int, default=1080)

    p_caption_retrieval = subparsers.add_parser(
        "augment-caption-retrieval",
        help="Inject nearest synthetic caption examples into a LLaMA-Factory caption dataset",
    )
    p_caption_retrieval.add_argument("--dataset", required=True)
    p_caption_retrieval.add_argument("--style-bank", required=True)
    p_caption_retrieval.add_argument("--output", required=True)
    p_caption_retrieval.add_argument("--k", type=int, default=3)
    p_caption_retrieval.add_argument("--allow-same-scenario", action="store_true")
    p_caption_retrieval.add_argument("--max-caption-chars", type=int, default=600)
    p_caption_retrieval.add_argument("--dataset-info-output")
    p_caption_retrieval.add_argument("--dataset-name")

    p_cider_bank = subparsers.add_parser(
        "build-caption-cider-bank",
        help="Build synthetic train n-gram bank for CIDEr-style public caption reranking",
    )
    p_cider_bank.add_argument("--index", required=True)
    p_cider_bank.add_argument("--output", required=True)
    p_cider_bank.add_argument("--splits", default="train")
    p_cider_bank.add_argument("--max-ngram", type=int, default=4)

    p_cider_rerank = subparsers.add_parser(
        "rerank-caption-cider-bank",
        help="Rerank caption submissions using synthetic train CIDEr-style phrase bank",
    )
    p_cider_rerank.add_argument("--phrase-bank", required=True)
    p_cider_rerank.add_argument("--base-caption", required=True)
    p_cider_rerank.add_argument("--candidate", action="append", required=True, help="Named caption as name=path")
    p_cider_rerank.add_argument("--output", required=True)
    p_cider_rerank.add_argument("--report-output")
    p_cider_rerank.add_argument("--fallback-name", default="base")
    p_cider_rerank.add_argument("--max-changed-rows", type=int, default=10)
    p_cider_rerank.add_argument("--min-margin", type=float, default=0.06)
    p_cider_rerank.add_argument("--min-context-delta", type=int, default=0)
    p_cider_rerank.add_argument("--min-attention-delta", type=int, default=0)
    p_cider_rerank.add_argument("--min-source-overlap", type=float, default=0.72)

    p_caption_facts = subparsers.add_parser(
        "augment-caption-facts",
        help="Inject gold or predicted VQA facts into a LLaMA-Factory caption dataset",
    )
    p_caption_facts.add_argument("--dataset", required=True)
    p_caption_facts.add_argument("--output", required=True)
    p_caption_facts.add_argument("--index")
    p_caption_facts.add_argument("--wts-vqa-json")
    p_caption_facts.add_argument("--vqa-submission")
    p_caption_facts.add_argument("--dataset-info-output")
    p_caption_facts.add_argument("--dataset-name")
    p_caption_facts.add_argument("--max-global-facts", type=int, default=12)
    p_caption_facts.add_argument("--max-phase-facts", type=int, default=16)

    p_facts = subparsers.add_parser("export-facts", help="Export canonical VQA-derived fact JSONL")
    p_facts.add_argument("--index", required=True)
    p_facts.add_argument("--output", required=True)

    p_clips = subparsers.add_parser("make-phase-clips", help="Create phase clip manifest and optional MP4 clips")
    p_clips.add_argument("--index", required=True)
    p_clips.add_argument("--output-manifest", required=True)
    p_clips.add_argument("--output-root", required=True)
    p_clips.add_argument("--ffmpeg-bin", default="ffmpeg")
    p_clips.add_argument("--make-clips", action="store_true")
    p_clips.add_argument("--overwrite", action="store_true")
    p_clips.add_argument("--relative-paths", action="store_true")
    p_clips.add_argument("--clip-mode", choices=("h264", "mpeg4", "copy"), default="h264")

    p_visual_clips = subparsers.add_parser(
        "make-visual-clips",
        help="Create bbox-overlay and interaction-crop phase clip manifest",
    )
    p_visual_clips.add_argument("--index", required=True)
    p_visual_clips.add_argument("--phase-clip-manifest", required=True)
    p_visual_clips.add_argument("--output-manifest", required=True)
    p_visual_clips.add_argument("--output-root", required=True)
    p_visual_clips.add_argument("--variants", default="overlay,interaction_crop")
    p_visual_clips.add_argument("--include-base", action="store_true")
    p_visual_clips.add_argument("--max-clips-per-key", type=int, default=0)
    p_visual_clips.add_argument("--make-clips", action="store_true")
    p_visual_clips.add_argument("--overwrite", action="store_true")
    p_visual_clips.add_argument("--ffmpeg-bin", default="ffmpeg")
    p_visual_clips.add_argument("--clip-mode", choices=("h264", "mpeg4"), default="mpeg4")
    p_visual_clips.add_argument("--frame-width", type=int, default=1920)
    p_visual_clips.add_argument("--frame-height", type=int, default=1080)
    p_visual_clips.add_argument("--crop-padding", type=float, default=0.35)
    p_visual_clips.add_argument("--crop-size", type=int, default=768)
    p_visual_clips.add_argument("--max-tracks-per-role", type=int, default=2)
    p_visual_clips.add_argument("--relative-paths", action="store_true")

    p_validate_clips = subparsers.add_parser("validate-phase-clips", help="Validate generated phase clips")
    p_validate_clips.add_argument("--manifest", required=True)
    p_validate_clips.add_argument("--ffprobe-bin", default="ffprobe")
    p_validate_clips.add_argument("--max-errors", type=int, default=50)
    p_validate_clips.add_argument("--output")

    p_filter_clips = subparsers.add_parser("filter-phase-clips", help="Remove invalid clips from a manifest")
    p_filter_clips.add_argument("--manifest", required=True)
    p_filter_clips.add_argument("--validation-report", required=True)
    p_filter_clips.add_argument("--output-manifest", required=True)

    p_eval_cap = subparsers.add_parser("eval-caption", help="Evaluate caption predictions locally")
    p_eval_cap.add_argument("--dataset-root", required=True)
    p_eval_cap.add_argument("--predictions", required=True)
    p_eval_cap.add_argument("--split", default="val")
    p_eval_cap.add_argument("--output")

    p_eval_vqa = subparsers.add_parser("eval-vqa", help="Evaluate VQA predictions locally")
    p_eval_vqa.add_argument("--dataset-root", required=True)
    p_eval_vqa.add_argument("--predictions", required=True)
    p_eval_vqa.add_argument("--split", default="val")
    p_eval_vqa.add_argument("--output")

    p_val_cap = subparsers.add_parser("validate-caption", help="Validate caption submission JSON")
    p_val_cap.add_argument("--predictions", required=True)
    p_val_cap.add_argument("--output")

    p_val_vqa = subparsers.add_parser("validate-vqa", help="Validate VQA submission JSON")
    p_val_vqa.add_argument("--predictions", required=True)
    p_val_vqa.add_argument("--output")

    p_assemble_cap = subparsers.add_parser("assemble-caption", help="Assemble official caption submission")
    p_assemble_cap.add_argument("--inference-dataset", required=True)
    p_assemble_cap.add_argument("--predictions", required=True)
    p_assemble_cap.add_argument("--output", required=True)

    p_assemble_vqa = subparsers.add_parser("assemble-vqa", help="Assemble official VQA submission")
    p_assemble_vqa.add_argument("--inference-dataset", required=True)
    p_assemble_vqa.add_argument("--predictions", required=True)
    p_assemble_vqa.add_argument("--output", required=True)

    p_ensemble_vqa = subparsers.add_parser("ensemble-vqa", help="Ensemble VQA submission JSON files")
    p_ensemble_vqa.add_argument("--inputs", nargs="+", required=True)
    p_ensemble_vqa.add_argument("--output", required=True)
    p_ensemble_vqa.add_argument("--fallback")
    p_ensemble_vqa.add_argument("--weights")

    p_vqa_types = subparsers.add_parser("summarize-vqa-types", help="Summarize public-test VQA question types")
    p_vqa_types.add_argument("--vqa-json", required=True)
    p_vqa_types.add_argument("--output")

    p_fuse_vqa_types = subparsers.add_parser("fuse-vqa-by-type", help="Question-type-aware VQA fusion")
    p_fuse_vqa_types.add_argument("--vqa-json", required=True)
    p_fuse_vqa_types.add_argument("--submission", action="append", required=True, help="Named submission as name=path")
    p_fuse_vqa_types.add_argument("--rules", required=True)
    p_fuse_vqa_types.add_argument("--output", required=True)
    p_fuse_vqa_types.add_argument("--report-output")

    p_repair_vqa = subparsers.add_parser(
        "repair-vqa-consistency",
        help="Conservatively repair repeated stable VQA facts within each scenario",
    )
    p_repair_vqa.add_argument("--candidates", required=True)
    p_repair_vqa.add_argument("--submission", required=True)
    p_repair_vqa.add_argument("--output", required=True)
    p_repair_vqa.add_argument("--report-output")
    p_repair_vqa.add_argument("--stable-types")
    p_repair_vqa.add_argument("--min-group-size", type=int, default=2)
    p_repair_vqa.add_argument("--min-top-count", type=int, default=2)
    p_repair_vqa.add_argument("--min-top-share", type=float, default=0.75)
    p_repair_vqa.add_argument("--max-changes", type=int, default=0)

    p_fuse_caption = subparsers.add_parser(
        "fuse-caption-candidates",
        help="Fact-aware fusion/reranking for multiple caption submission JSON files",
    )
    p_fuse_caption.add_argument("--caption", action="append", required=True, help="Named caption as name=path")
    p_fuse_caption.add_argument("--fallback-name", required=True)
    p_fuse_caption.add_argument("--output", required=True)
    p_fuse_caption.add_argument("--wts-vqa-json")
    p_fuse_caption.add_argument("--vqa-submission")
    p_fuse_caption.add_argument("--report-output")
    p_fuse_caption.add_argument("--target-words", type=int, default=250)
    p_fuse_caption.add_argument("--min-switch-margin", type=float, default=2.0)

    p_realizer_train = subparsers.add_parser(
        "export-caption-realizer-train",
        help="Export text-only SFT rows for learned WTS caption realization",
    )
    p_realizer_train.add_argument("--index", required=True)
    p_realizer_train.add_argument("--output", required=True)
    p_realizer_train.add_argument("--dataset-info-output")
    p_realizer_train.add_argument("--dataset-name")
    p_realizer_train.add_argument("--splits", default="train")
    p_realizer_train.add_argument("--max-rows", type=int, default=0)

    p_realizer_test = subparsers.add_parser(
        "export-caption-realizer-test",
        help="Export text-only inference rows from caption candidates for learned realization",
    )
    p_realizer_test.add_argument("--caption", action="append", required=True, help="Named caption as name=path")
    p_realizer_test.add_argument("--fallback-name", required=True)
    p_realizer_test.add_argument("--output", required=True)
    p_realizer_test.add_argument("--wts-vqa-json")
    p_realizer_test.add_argument("--vqa-submission")
    p_realizer_test.add_argument("--dataset-info-output")
    p_realizer_test.add_argument("--dataset-name")

    p_realizer_assemble = subparsers.add_parser(
        "assemble-caption-realizer",
        help="Assemble caption realizer predictions into official caption JSON",
    )
    p_realizer_assemble.add_argument("--inference-dataset", required=True)
    p_realizer_assemble.add_argument("--predictions", required=True)
    p_realizer_assemble.add_argument("--fallback-caption", required=True)
    p_realizer_assemble.add_argument("--output", required=True)
    p_realizer_assemble.add_argument("--report-output")

    p_segment_train = subparsers.add_parser(
        "export-caption-segment-train",
        help="Export divide-and-conquer segment caption SFT rows",
    )
    p_segment_train.add_argument("--dataset", required=True)
    p_segment_train.add_argument("--output", required=True)
    p_segment_train.add_argument("--dataset-info-output")
    p_segment_train.add_argument("--dataset-name")
    p_segment_train.add_argument("--max-rows", type=int, default=0)
    p_segment_train.add_argument("--extraction", choices=("rules", "remap"), default="rules")
    p_segment_train.add_argument("--role")
    p_segment_train.add_argument("--segment")

    p_segment_test = subparsers.add_parser(
        "export-caption-segment-test",
        help="Export divide-and-conquer segment caption inference rows",
    )
    p_segment_test.add_argument("--dataset", required=True)
    p_segment_test.add_argument("--output", required=True)
    p_segment_test.add_argument("--dataset-info-output")
    p_segment_test.add_argument("--dataset-name")
    p_segment_test.add_argument("--max-rows", type=int, default=0)
    p_segment_test.add_argument("--role")
    p_segment_test.add_argument("--segment")

    p_segment_assemble = subparsers.add_parser(
        "assemble-caption-segments",
        help="Assemble segment caption predictions into official caption JSON",
    )
    p_segment_assemble.add_argument("--inference-dataset", required=True)
    p_segment_assemble.add_argument("--predictions", required=True)
    p_segment_assemble.add_argument("--output", required=True)
    p_segment_assemble.add_argument("--fallback-caption")
    p_segment_assemble.add_argument("--report-output")

    p_rewrite_train = subparsers.add_parser(
        "export-caption-rewrite-train",
        help="Export rewrite-module SFT rows from segmented caption notes to original captions",
    )
    p_rewrite_train.add_argument("--dataset", required=True)
    p_rewrite_train.add_argument("--output", required=True)
    p_rewrite_train.add_argument("--dataset-info-output")
    p_rewrite_train.add_argument("--dataset-name")
    p_rewrite_train.add_argument("--extraction", choices=("rules", "remap"), default="remap")
    p_rewrite_train.add_argument("--max-rows", type=int, default=0)
    p_rewrite_train.add_argument("--lock-target-caption", action="store_true")
    p_rewrite_train.add_argument("--index")

    p_rewrite_test = subparsers.add_parser(
        "export-caption-rewrite-test",
        help="Export rewrite-module inference rows from assembled segment captions",
    )
    p_rewrite_test.add_argument("--segment-submission", required=True)
    p_rewrite_test.add_argument("--output", required=True)
    p_rewrite_test.add_argument("--dataset-info-output")
    p_rewrite_test.add_argument("--dataset-name")
    p_rewrite_test.add_argument("--fallback-caption")
    p_rewrite_test.add_argument("--max-rows", type=int, default=0)
    p_rewrite_test.add_argument("--trusted-caption")
    p_rewrite_test.add_argument("--wts-vqa-json")
    p_rewrite_test.add_argument("--vqa-submission")

    p_rewrite_assemble = subparsers.add_parser(
        "assemble-caption-rewrite",
        help="Assemble rewrite-module predictions into official caption JSON",
    )
    p_rewrite_assemble.add_argument("--inference-dataset", required=True)
    p_rewrite_assemble.add_argument("--predictions", required=True)
    p_rewrite_assemble.add_argument("--output", required=True)
    p_rewrite_assemble.add_argument("--fallback-caption")
    p_rewrite_assemble.add_argument("--report-output")

    p_selector_train = subparsers.add_parser(
        "export-caption-selector-train",
        help="Export text-only SFT rows for learned caption reward selection",
    )
    p_selector_train.add_argument("--index", required=True)
    p_selector_train.add_argument("--output", required=True)
    p_selector_train.add_argument("--dataset-info-output")
    p_selector_train.add_argument("--dataset-name")
    p_selector_train.add_argument("--splits", default="train")
    p_selector_train.add_argument("--max-rows", type=int, default=0)

    p_selector_test = subparsers.add_parser(
        "export-caption-selector-test",
        help="Export candidate rows for learned caption selector inference",
    )
    p_selector_test.add_argument("--caption", action="append", required=True, help="Named caption as name=path")
    p_selector_test.add_argument("--fallback-name", required=True)
    p_selector_test.add_argument("--output", required=True)
    p_selector_test.add_argument("--wts-vqa-json")
    p_selector_test.add_argument("--vqa-submission")
    p_selector_test.add_argument("--dataset-info-output")
    p_selector_test.add_argument("--dataset-name")

    p_selector_assemble = subparsers.add_parser(
        "assemble-caption-selector",
        help="Assemble learned caption selector predictions into official caption JSON",
    )
    p_selector_assemble.add_argument("--inference-dataset", required=True)
    p_selector_assemble.add_argument("--predictions", required=True)
    p_selector_assemble.add_argument("--caption", action="append", required=True, help="Named caption as name=path")
    p_selector_assemble.add_argument("--fallback-name", required=True)
    p_selector_assemble.add_argument("--output", required=True)
    p_selector_assemble.add_argument("--wts-vqa-json")
    p_selector_assemble.add_argument("--vqa-submission")
    p_selector_assemble.add_argument("--report-output")
    p_selector_assemble.add_argument("--min-good-margin", type=float, default=0.0)
    p_selector_assemble.add_argument("--min-source-overlap", type=float, default=0.70)
    p_selector_assemble.add_argument("--max-changed-rows", type=int, default=40)

    p_caption_consistency = subparsers.add_parser(
        "repair-caption-consistency",
        help="Apply attribute-lock and phase-transition consistency gates to captions",
    )
    p_caption_consistency.add_argument("--caption", required=True)
    p_caption_consistency.add_argument("--fallback-caption", required=True)
    p_caption_consistency.add_argument("--output", required=True)
    p_caption_consistency.add_argument("--wts-vqa-json", required=True)
    p_caption_consistency.add_argument("--vqa-submission", required=True)
    p_caption_consistency.add_argument("--report-output")

    p_caption_audit = subparsers.add_parser(
        "audit-caption-completeness",
        help="Find truncated or incomplete caption fields",
    )
    p_caption_audit.add_argument("--caption", required=True)
    p_caption_audit.add_argument("--output")
    p_caption_audit.add_argument("--max-examples", type=int, default=50)

    p_caption_repair_complete = subparsers.add_parser(
        "repair-caption-completeness",
        help="Replace truncated caption fields with fallback caption fields",
    )
    p_caption_repair_complete.add_argument("--caption", required=True)
    p_caption_repair_complete.add_argument("--fallback-caption", required=True)
    p_caption_repair_complete.add_argument("--output", required=True)
    p_caption_repair_complete.add_argument("--report-output")
    p_caption_repair_complete.add_argument("--max-examples", type=int, default=50)

    p_rewrite_caption_slots = subparsers.add_parser(
        "rewrite-caption-slots",
        help="Rewrite caption submission into phase-aware location/attention/behavior/context slots",
    )
    p_rewrite_caption_slots.add_argument("--caption", required=True)
    p_rewrite_caption_slots.add_argument("--output", required=True)
    p_rewrite_caption_slots.add_argument("--wts-vqa-json", required=True)
    p_rewrite_caption_slots.add_argument("--vqa-submission", required=True)
    p_rewrite_caption_slots.add_argument("--report-output")
    p_rewrite_caption_slots.add_argument(
        "--mode",
        choices=("reorder", "balanced", "fill", "template"),
        default="balanced",
    )
    p_rewrite_caption_slots.add_argument("--max-added-sentences", type=int, default=2)

    p_rerank_caption_slots = subparsers.add_parser(
        "rerank-caption-slots",
        help="Selectively apply slot-rewritten caption rows only when proxy score improves",
    )
    p_rerank_caption_slots.add_argument("--base-caption", required=True)
    p_rerank_caption_slots.add_argument("--candidate", action="append", required=True, help="Named candidate as name=path")
    p_rerank_caption_slots.add_argument("--output", required=True)
    p_rerank_caption_slots.add_argument("--wts-vqa-json", required=True)
    p_rerank_caption_slots.add_argument("--vqa-submission", required=True)
    p_rerank_caption_slots.add_argument("--report-output")
    p_rerank_caption_slots.add_argument("--min-switch-margin", type=float, default=1.0)
    p_rerank_caption_slots.add_argument("--min-source-overlap", type=float, default=0.72)
    p_rerank_caption_slots.add_argument("--max-changed-rows", type=int, default=0)
    p_rerank_caption_slots.add_argument("--reward-slot-order", action="store_true")

    p_oracle_cap = subparsers.add_parser("make-oracle-caption", help="Create reference captions as predictions")
    p_oracle_cap.add_argument("--dataset-root", required=True)
    p_oracle_cap.add_argument("--split", default="val")
    p_oracle_cap.add_argument("--output", required=True)

    p_oracle_vqa = subparsers.add_parser("make-oracle-vqa", help="Create reference VQA answers as predictions")
    p_oracle_vqa.add_argument("--dataset-root", required=True)
    p_oracle_vqa.add_argument("--split", default="val")
    p_oracle_vqa.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "index":
        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
        records = build_index(
            args.dataset_root,
            splits=splits,
            absolute_paths=not args.relative_paths,
        )
        write_index(records, args.output)
        print(f"Wrote {len(records)} records to {args.output}")
        return

    if args.command == "index-wts-test":
        from .wts_test import build_wts_public_test_index

        records = build_wts_public_test_index(
            args.test_root,
            args.output,
            absolute_paths=not args.relative_paths,
        )
        print(f"Wrote {len(records)} WTS public test records to {args.output}")
        return

    if args.command == "export-wts-test-vqa":
        from .wts_test import export_wts_public_vqa

        rows = export_wts_public_vqa(
            test_root=args.test_root,
            vqa_json=args.vqa_json,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            clip_output_root=args.clip_output_root,
            make_clips=args.make_clips,
            overwrite=args.overwrite,
            ffmpeg_bin=args.ffmpeg_bin,
            clip_mode=args.clip_mode,
            bbox_mode=args.bbox_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            max_videos_per_row=args.max_videos_per_row or None,
            env_clip_duration=args.env_clip_duration,
            visual_variants={variant.strip() for variant in args.visual_variants.split(",") if variant.strip()} or None,
            visual_output_root=args.visual_output_root,
            visual_max_clips_per_row=args.visual_max_clips_per_row or None,
            visual_crop_padding=args.visual_crop_padding,
            visual_crop_size=args.visual_crop_size,
            visual_max_tracks_per_role=args.visual_max_tracks_per_role,
        )
        print(f"Wrote {len(rows)} WTS public VQA inference rows to {args.output}")
        return

    if args.command == "export-llamafactory":
        records = load_records(args.index)
        tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
        rows = export_llamafactory(
            records,
            args.output,
            tasks=tasks,
            media_policy=args.media_policy,
            include_missing_media=args.include_missing_media,
            bbox_mode=args.bbox_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            phase_clip_manifest=args.phase_clip_manifest,
        )
        if args.dataset_info_output:
            write_llamafactory_dataset_info(
                args.dataset_info_output,
                dataset_name=args.dataset_name,
                file_name=Path(args.output).name,
            )
        print(f"Wrote {len(rows)} LLaMA-Factory rows to {args.output}")
        return

    if args.command == "export-frame-dataset":
        from .frame_export import export_frame_dataset

        rows = export_frame_dataset(
            dataset=args.dataset,
            output=args.output,
            frame_root=args.frame_root,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            frame_time=args.frame_time,
            max_frames_per_row=args.max_frames_per_row,
            overwrite=args.overwrite,
        )
        print(f"Wrote {len(rows)} image-frame rows to {args.output}")
        return

    if args.command == "export-vqa-preference":
        from .preferences import export_vqa_preference_llamafactory, write_vqa_preference_dataset_info

        records = load_records(args.index)
        rows = export_vqa_preference_llamafactory(
            records,
            args.output,
            media_policy=args.media_policy,
            include_missing_media=args.include_missing_media,
            bbox_mode=args.bbox_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            phase_clip_manifest=args.phase_clip_manifest,
            negative_policy=args.negative_policy,
            max_rejected_per_question=args.max_rejected_per_question or None,
            response_mode=args.response_mode,
            seed=args.seed,
        )
        if args.dataset_info_output:
            write_vqa_preference_dataset_info(
                args.dataset_info_output,
                dataset_name=args.dataset_name,
                file_name=Path(args.output).name,
            )
        print(f"Wrote {len(rows)} VQA preference rows to {args.output}")
        return

    if args.command == "export-vqa-hard-negative-candidates":
        from .preferences import export_vqa_option_candidates

        records = load_records(args.index)
        rows = export_vqa_option_candidates(
            records,
            args.output,
            media_policy=args.media_policy,
            include_missing_media=args.include_missing_media,
            bbox_mode=args.bbox_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            phase_clip_manifest=args.phase_clip_manifest,
        )
        print(f"Wrote {len(rows)} VQA hard-negative candidates to {args.output}")
        return

    if args.command == "score-vqa-options":
        from .option_scoring import score_qwen3vl_options

        report = score_qwen3vl_options(
            candidates_path=args.candidates,
            output=args.output,
            model_name_or_path=args.model_name_or_path,
            adapter_name_or_path=args.adapter_name_or_path,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation or None,
            device_map=args.device_map,
            video_max_pixels=args.video_max_pixels or None,
            fps=args.fps or None,
            response_mode=args.response_mode,
            include_eos=args.include_eos,
            resume=not args.no_resume,
            max_rows=args.max_rows or None,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            trust_remote_code=args.trust_remote_code,
        )
        _emit_report(report, None)
        return

    if args.command == "export-vqa-option-candidates-from-inference":
        from .option_submission import export_vqa_option_candidates_from_inference_dataset

        rows = export_vqa_option_candidates_from_inference_dataset(
            dataset=args.dataset,
            output=args.output,
            prompt_variant=args.prompt_variant,
        )
        print(f"Wrote {len(rows)} VQA option-scoring candidates to {args.output}")
        return

    if args.command == "assemble-vqa-option-scores":
        from .option_submission import assemble_vqa_submission_from_option_scores

        weights = None
        if args.weights:
            weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
        rows, report = assemble_vqa_submission_from_option_scores(
            candidates_path=args.candidates,
            scores_path=args.scores,
            output=args.output,
            weights=weights,
            normalization=args.normalization,
            fallback=args.fallback,
            report_output=args.report_output,
        )
        print(f"Wrote {len(rows)} VQA predictions from option scores to {args.output}")
        _emit_report(report, None)
        return

    if args.command == "build-vqa-hard-negative-preference":
        from .hard_negatives import build_hard_negative_preference_dataset

        fields = tuple(
            field.strip()
            for field in args.balance_fields.split(",")
            if field.strip()
        )
        rows, report = build_hard_negative_preference_dataset(
            candidates_path=args.candidates,
            scores_path=args.scores,
            output=args.output,
            report_output=args.report_output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            selection=args.selection,
            max_gold_margin=args.max_gold_margin,
            balance_fields=fields,
            max_per_group=args.max_per_group or None,
            max_rows=args.max_rows or None,
            response_mode=args.response_mode,
            remap_option_letters=args.remap_option_letters,
            allow_missing_scores=args.allow_missing_scores,
        )
        print(
            f"Wrote {len(rows)} hard-negative VQA preference rows to {args.output}; "
            f"scorer accuracy={report['model_accuracy']:.4f}"
        )
        return

    if args.command == "build-caption-style-bank":
        from .caption_retrieval import build_caption_style_bank

        splits = {split.strip() for split in args.splits.split(",") if split.strip()}
        rows = build_caption_style_bank(
            args.index,
            args.output,
            splits=splits,
            bbox_mode=args.bbox_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
        )
        print(f"Wrote {len(rows)} caption style exemplars to {args.output}")
        return

    if args.command == "augment-caption-retrieval":
        from .caption_retrieval import augment_caption_dataset_with_retrieval

        rows = augment_caption_dataset_with_retrieval(
            args.dataset,
            args.style_bank,
            args.output,
            k=args.k,
            exclude_same_scenario=not args.allow_same_scenario,
            max_caption_chars=args.max_caption_chars,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
        )
        print(f"Wrote {len(rows)} retrieval-augmented caption rows to {args.output}")
        return

    if args.command == "build-caption-cider-bank":
        from .caption_cider_reranker import build_caption_phrase_bank

        bank = build_caption_phrase_bank(
            index=args.index,
            output=args.output,
            splits={split.strip() for split in args.splits.split(",") if split.strip()},
            max_ngram=args.max_ngram,
        )
        print(f"Wrote CIDEr phrase bank with {len(bank.get('banks', {}))} banks to {args.output}")
        return

    if args.command == "rerank-caption-cider-bank":
        from .caption_cider_reranker import rerank_caption_cider_bank

        rows, report = rerank_caption_cider_bank(
            phrase_bank=args.phrase_bank,
            base_caption=args.base_caption,
            candidates=_parse_named_paths(args.candidate),
            output=args.output,
            report_output=args.report_output,
            fallback_name=args.fallback_name,
            max_changed_rows=args.max_changed_rows,
            min_margin=args.min_margin,
            min_context_delta=args.min_context_delta,
            min_attention_delta=args.min_attention_delta,
            min_source_overlap=args.min_source_overlap,
        )
        print(
            f"Wrote CIDEr-reranked caption submission for {len(rows)} scenarios to {args.output}; "
            f"changed={report['changed_rows']}"
        )
        return

    if args.command == "augment-caption-facts":
        from .caption_facts import augment_caption_dataset_with_facts

        rows = augment_caption_dataset_with_facts(
            args.dataset,
            args.output,
            index_path=args.index,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            max_global_facts=args.max_global_facts,
            max_phase_facts=args.max_phase_facts,
        )
        print(f"Wrote {len(rows)} fact-conditioned caption rows to {args.output}")
        return

    if args.command == "export-facts":
        from .facts import export_fact_jsonl

        rows = export_fact_jsonl(args.index, args.output)
        print(f"Wrote {len(rows)} fact rows to {args.output}")
        return

    if args.command == "make-phase-clips":
        from .clips import build_phase_clip_manifest

        rows = build_phase_clip_manifest(
            args.index,
            args.output_manifest,
            output_root=args.output_root,
            ffmpeg_bin=args.ffmpeg_bin,
            make_clips=args.make_clips,
            overwrite=args.overwrite,
            absolute_paths=not args.relative_paths,
            clip_mode=args.clip_mode,
        )
        action = "clips and manifest" if args.make_clips else "manifest"
        print(f"Wrote {len(rows)} phase clip rows ({action}) to {args.output_manifest}")
        return

    if args.command == "make-visual-clips":
        from .visual_clips import build_visual_phase_clip_manifest

        variants = {variant.strip() for variant in args.variants.split(",") if variant.strip()}
        rows = build_visual_phase_clip_manifest(
            index_path=args.index,
            phase_clip_manifest=args.phase_clip_manifest,
            output_manifest=args.output_manifest,
            output_root=args.output_root,
            variants=variants,
            include_base=args.include_base,
            max_clips_per_key=args.max_clips_per_key or None,
            make_clips=args.make_clips,
            overwrite=args.overwrite,
            ffmpeg_bin=args.ffmpeg_bin,
            clip_mode=args.clip_mode,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            crop_padding=args.crop_padding,
            crop_size=args.crop_size,
            max_tracks_per_role=args.max_tracks_per_role,
            absolute_paths=not args.relative_paths,
        )
        action = "clips and manifest" if args.make_clips else "manifest"
        print(f"Wrote {len(rows)} visual phase clip rows ({action}) to {args.output_manifest}")
        return

    if args.command == "validate-phase-clips":
        from .clips import validate_phase_clips

        report = validate_phase_clips(
            args.manifest,
            ffprobe_bin=args.ffprobe_bin,
            max_errors=args.max_errors,
        )
        _emit_report(report, args.output)
        return

    if args.command == "filter-phase-clips":
        from .clips import filter_phase_clip_manifest

        report = filter_phase_clip_manifest(
            args.manifest,
            args.output_manifest,
            validation_report=args.validation_report,
        )
        _emit_report(report, None)
        return

    if args.command == "eval-caption":
        report = evaluate_caption_predictions(args.dataset_root, args.predictions, split=args.split)
        _emit_report(report, args.output)
        return

    if args.command == "eval-vqa":
        report = evaluate_vqa_predictions(args.dataset_root, args.predictions, split=args.split)
        _emit_report(report, args.output)
        return

    if args.command == "validate-caption":
        report = validate_caption_submission(args.predictions)
        _emit_report(report, args.output)
        return

    if args.command == "validate-vqa":
        report = validate_vqa_submission(args.predictions)
        _emit_report(report, args.output)
        return

    if args.command == "assemble-caption":
        from .submission import assemble_caption_submission

        submission = assemble_caption_submission(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            output=args.output,
        )
        print(f"Wrote caption submission for {len(submission)} scenarios to {args.output}")
        return

    if args.command == "assemble-vqa":
        from .submission import assemble_vqa_submission

        submission = assemble_vqa_submission(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            output=args.output,
        )
        print(f"Wrote {len(submission)} VQA predictions to {args.output}")
        return

    if args.command == "ensemble-vqa":
        from .submission import ensemble_vqa_submissions

        weights = (
            [float(value.strip()) for value in args.weights.split(",") if value.strip()]
            if args.weights
            else None
        )
        submission = ensemble_vqa_submissions(
            inputs=args.inputs,
            output=args.output,
            fallback=args.fallback,
            weights=weights,
        )
        print(f"Wrote {len(submission)} ensembled VQA predictions to {args.output}")
        return

    if args.command == "summarize-vqa-types":
        from .vqa_fusion import summarize_vqa_question_types

        report = summarize_vqa_question_types(vqa_json=args.vqa_json, output=args.output)
        if not args.output:
            _emit_report(report, None)
        else:
            print(f"Wrote VQA type summary to {args.output}")
        return

    if args.command == "fuse-vqa-by-type":
        from .vqa_fusion import fuse_vqa_by_question_type

        submissions = _parse_named_paths(args.submission)
        rows = fuse_vqa_by_question_type(
            vqa_json=args.vqa_json,
            submissions=submissions,
            rules_path=args.rules,
            output=args.output,
            report_output=args.report_output,
        )
        print(f"Wrote {len(rows)} type-fused VQA predictions to {args.output}")
        return

    if args.command == "repair-vqa-consistency":
        from .vqa_consistency import repair_vqa_scenario_consistency

        stable_types = None
        if args.stable_types:
            stable_types = tuple(
                value.strip()
                for value in args.stable_types.split(",")
                if value.strip()
            )
        rows, report = repair_vqa_scenario_consistency(
            candidates_path=args.candidates,
            submission=args.submission,
            output=args.output,
            report_output=args.report_output,
            stable_types=stable_types,
            min_group_size=args.min_group_size,
            min_top_count=args.min_top_count,
            min_top_share=args.min_top_share,
            max_changes=args.max_changes or None,
        )
        print(
            f"Wrote {len(rows)} consistency-repaired VQA predictions to {args.output}; "
            f"changed={report['changed']}"
        )
        return

    if args.command == "fuse-caption-candidates":
        from .caption_fusion import fuse_caption_submissions

        captions = _parse_named_paths(args.caption)
        submission, report = fuse_caption_submissions(
            submissions=captions,
            output=args.output,
            fallback_name=args.fallback_name,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=args.report_output,
            target_words=args.target_words,
            min_switch_margin=args.min_switch_margin,
        )
        print(
            f"Wrote caption fusion for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_from_fallback']}/{report['total_rows']}"
        )
        return

    if args.command == "export-caption-realizer-train":
        from .caption_realizer import export_caption_realizer_train

        splits = {value.strip() for value in args.splits.split(",") if value.strip()}
        rows = export_caption_realizer_train(
            index=args.index,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            splits=splits,
            max_rows=args.max_rows,
        )
        print(f"Wrote {len(rows)} caption realizer train rows to {args.output}")
        return

    if args.command == "export-caption-realizer-test":
        from .caption_realizer import export_caption_realizer_test

        captions = _parse_named_paths(args.caption)
        rows = export_caption_realizer_test(
            captions=captions,
            fallback_name=args.fallback_name,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
        )
        print(f"Wrote {len(rows)} caption realizer inference rows to {args.output}")
        return

    if args.command == "assemble-caption-realizer":
        from .caption_realizer import assemble_caption_realizer_submission

        submission, report = assemble_caption_realizer_submission(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            fallback_caption=args.fallback_caption,
            output=args.output,
            report_output=args.report_output,
        )
        print(
            f"Wrote caption realizer submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_from_fallback']} parse_failed={report['parse_failed']}"
        )
        return

    if args.command == "export-caption-segment-train":
        from .caption_segments import export_caption_segment_train

        rows = export_caption_segment_train(
            dataset=args.dataset,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            max_rows=args.max_rows,
            extraction=args.extraction,
            role_filter=args.role,
            segment_filter=args.segment,
        )
        print(f"Wrote {len(rows)} caption segment train rows to {args.output}")
        return

    if args.command == "export-caption-segment-test":
        from .caption_segments import export_caption_segment_test

        rows = export_caption_segment_test(
            dataset=args.dataset,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            max_rows=args.max_rows,
            role_filter=args.role,
            segment_filter=args.segment,
        )
        print(f"Wrote {len(rows)} caption segment inference rows to {args.output}")
        return

    if args.command == "assemble-caption-segments":
        from .caption_segments import assemble_caption_segments

        submission, report = assemble_caption_segments(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            output=args.output,
            fallback_caption=args.fallback_caption,
            report_output=args.report_output,
        )
        print(
            f"Wrote segment-composed caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_from_fallback']} parse_failed={report['parse_failed']}"
        )
        return

    if args.command == "export-caption-rewrite-train":
        from .caption_segments import export_caption_rewrite_train

        rows = export_caption_rewrite_train(
            dataset=args.dataset,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            extraction=args.extraction,
            max_rows=args.max_rows,
            lock_target_caption=args.lock_target_caption,
            index_path=args.index,
        )
        print(f"Wrote {len(rows)} caption rewrite train rows to {args.output}")
        return

    if args.command == "export-caption-rewrite-test":
        from .caption_segments import export_caption_rewrite_test

        rows = export_caption_rewrite_test(
            segment_submission=args.segment_submission,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            fallback_caption=args.fallback_caption,
            max_rows=args.max_rows,
            trusted_caption=args.trusted_caption,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
        )
        print(f"Wrote {len(rows)} caption rewrite inference rows to {args.output}")
        return

    if args.command == "assemble-caption-rewrite":
        from .caption_segments import assemble_caption_rewrite

        submission, report = assemble_caption_rewrite(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            output=args.output,
            fallback_caption=args.fallback_caption,
            report_output=args.report_output,
        )
        print(
            f"Wrote rewrite caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_from_fallback']} parse_failed={report['parse_failed']}"
        )
        return

    if args.command == "export-caption-selector-train":
        from .caption_selector import export_caption_selector_train

        splits = {value.strip() for value in args.splits.split(",") if value.strip()}
        rows = export_caption_selector_train(
            index=args.index,
            output=args.output,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
            splits=splits,
            max_rows=args.max_rows,
        )
        print(f"Wrote {len(rows)} caption selector train rows to {args.output}")
        return

    if args.command == "export-caption-selector-test":
        from .caption_selector import export_caption_selector_test

        captions = _parse_named_paths(args.caption)
        rows = export_caption_selector_test(
            captions=captions,
            fallback_name=args.fallback_name,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            dataset_info_output=args.dataset_info_output,
            dataset_name=args.dataset_name,
        )
        print(f"Wrote {len(rows)} caption selector inference rows to {args.output}")
        return

    if args.command == "assemble-caption-selector":
        from .caption_selector import assemble_caption_selector_submission

        captions = _parse_named_paths(args.caption)
        submission, report = assemble_caption_selector_submission(
            inference_dataset=args.inference_dataset,
            predictions=args.predictions,
            captions=captions,
            fallback_name=args.fallback_name,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=args.report_output,
            min_good_margin=args.min_good_margin,
            min_source_overlap=args.min_source_overlap,
            max_changed_rows=args.max_changed_rows,
        )
        print(
            f"Wrote caption selector submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_rows']}"
        )
        return

    if args.command == "repair-caption-consistency":
        from .caption_selector import repair_caption_consistency

        submission, report = repair_caption_consistency(
            caption=args.caption,
            fallback_caption=args.fallback_caption,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=args.report_output,
        )
        print(
            f"Wrote consistency-repaired caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_rows']}"
        )
        return

    if args.command == "audit-caption-completeness":
        from .caption_quality import audit_caption_completeness

        report = audit_caption_completeness(
            caption=args.caption,
            output=args.output,
            max_examples=args.max_examples,
        )
        print(f"Caption completeness ok={report['ok']} errors={report['num_errors']}")
        return

    if args.command == "repair-caption-completeness":
        from .caption_quality import repair_caption_completeness

        submission, report = repair_caption_completeness(
            caption=args.caption,
            fallback_caption=args.fallback_caption,
            output=args.output,
            report_output=args.report_output,
            max_examples=args.max_examples,
        )
        print(
            f"Wrote completeness-repaired caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed={report['changed_rows']} remaining={report['remaining_errors']}"
        )
        return

    if args.command == "rewrite-caption-slots":
        from .caption_slots import rewrite_caption_submission_slots

        submission, report = rewrite_caption_submission_slots(
            caption=args.caption,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=args.report_output,
            mode=args.mode,
            max_added_sentences=args.max_added_sentences,
        )
        print(
            f"Wrote slot-rewritten caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed_rows={report['changed_rows']}/{report['total_rows']}"
        )
        return

    if args.command == "rerank-caption-slots":
        from .caption_slots import rerank_caption_slot_variants

        candidates = _parse_named_paths(args.candidate)
        submission, report = rerank_caption_slot_variants(
            base_caption=args.base_caption,
            candidates=candidates,
            output=args.output,
            wts_vqa_json=args.wts_vqa_json,
            vqa_submission=args.vqa_submission,
            report_output=args.report_output,
            min_switch_margin=args.min_switch_margin,
            min_source_overlap=args.min_source_overlap,
            max_changed_rows=args.max_changed_rows or None,
            preserve_base_order=not args.reward_slot_order,
        )
        print(
            f"Wrote slot-reranked caption submission for {len(submission)} scenarios to {args.output}; "
            f"changed_rows={report['changed_rows']}/{report['total_rows']}"
        )
        return

    if args.command == "make-oracle-caption":
        from .oracles import make_oracle_caption_predictions

        write_json(args.output, make_oracle_caption_predictions(args.dataset_root, split=args.split))
        print(f"Wrote oracle caption predictions to {args.output}")
        return

    if args.command == "make-oracle-vqa":
        from .oracles import make_oracle_vqa_predictions

        write_json(args.output, make_oracle_vqa_predictions(args.dataset_root, split=args.split))
        print(f"Wrote oracle VQA predictions to {args.output}")
        return


def _emit_report(report: dict, output: str | None) -> None:
    if output:
        write_eval_report(report, output)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"Expected name=path, got: {value}")
        parsed[name.strip()] = path.strip()
    return parsed


if __name__ == "__main__":
    main()
