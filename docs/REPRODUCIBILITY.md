# Reproducibility Checklist

Use this checklist before publishing or rerunning the paper pipeline.

1. Download SynWTS train/validation data through the official challenge access
   channel.
2. Keep all generated clips, adapters, prediction files, and submission zips
   outside git.
3. Build the synthetic index with `python -m synwts.cli index`.
4. Build synthetic phase clips with `python -m synwts.cli make-phase-clips`.
5. Export LLaMA-Factory SFT data with `python -m synwts.cli export-llamafactory`
   and `--bbox-mode summary`.
6. Train public VLM adapters with configs copied from `configs/release/`.
7. Export public-test caption and VQA inference data only after training.
8. Assemble and validate `caption_submission.json` and `vqa_submission.json`.
9. Report only results whose adapters were trained on synthetic SynWTS data.

The public-test videos are inference inputs. They should not be used for manual
annotation, checkpoint selection, prompt tuning against hidden labels, or
train/validation supervision.
