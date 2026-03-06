#!/usr/bin/env python3
"""Generate OpenCLIP golden JSONL data using Hugging Face Transformers.

Example:
  python3 tools/openclip_generate_golden.py \
    --cases-jsonl /tmp/openclip_cases_v1.jsonl \
    --output-jsonl /tmp/openclip_endpoint_golden/v1/openclip_vit_b_32_laion2b_s34b_b79k_prefix64_v1.jsonl \
    --metadata-path /tmp/openclip_endpoint_golden/v1/metadata.json \
    --prefix-length 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_MODEL_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
DEFAULT_SEQUENCE_LENGTH = 77
DEFAULT_LOGIT_SCALE = 1.0 / 0.07


@dataclass
class GoldenCase:
    case_id: str
    text: str
    image: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic OpenCLIP golden rows from "
            "id/text/image-recipe input cases."
        )
    )
    parser.add_argument(
        "--cases-jsonl",
        type=Path,
        required=True,
        help="Input JSONL with rows {id,text,image}.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help=(
            "Output JSONL rows "
            "{id,text,image,text_prefix,image_prefix,logits_row}."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Output metadata.json path (default: alongside --output-jsonl).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="Pinned model revision/commit.",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=64,
        help="Number of leading embedding dimensions to include per row.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Tokenizer truncation/padding length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local inference.",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=DEFAULT_LOGIT_SCALE,
        help="Scale used for CLIP similarity logits.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing Hugging Face token.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"line {line_number}: expected object row, got {type(payload)}"
                )

            case_id = str(payload.get("id", "")).strip()
            text = str(payload.get("text", "")).strip()
            image = payload.get("image")
            if not case_id:
                raise ValueError(f"line {line_number}: missing non-empty id")
            if not text:
                raise ValueError(f"line {line_number}: missing non-empty text")
            if not isinstance(image, dict):
                raise ValueError(
                    f"line {line_number}: image must be an object, got {type(image)}"
                )

            validate_image_recipe(image, line_number)
            cases.append(GoldenCase(case_id=case_id, text=text, image=image))
    return cases


def validate_image_recipe(recipe: dict[str, object], line_number: int) -> None:
    kind = str(recipe.get("kind", "solid")).strip().lower()
    width = int(recipe.get("width", 0))
    height = int(recipe.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"line {line_number}: image width/height must be > 0, got {width}x{height}"
        )

    if kind == "solid":
        parse_rgb(recipe.get("color"), "color", line_number)
        return
    if kind == "checkerboard":
        parse_rgb(recipe.get("color_a"), "color_a", line_number)
        parse_rgb(recipe.get("color_b"), "color_b", line_number)
        block_size = int(recipe.get("block_size", 0))
        if block_size <= 0:
            raise ValueError(
                f"line {line_number}: checkerboard block_size must be > 0, got {block_size}"
            )
        return

    raise ValueError(f"line {line_number}: unsupported image.kind={kind!r}")


def parse_rgb(value: object, label: str, line_number: int) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"line {line_number}: {label} must be [r,g,b] with 3 integers"
        )
    channels: list[int] = []
    for i, raw in enumerate(value):
        channel = int(raw)
        if channel < 0 or channel > 255:
            raise ValueError(
                f"line {line_number}: {label}[{i}] must be between 0 and 255, got {channel}"
            )
        channels.append(channel)
    return channels[0], channels[1], channels[2]


def render_image(recipe: dict[str, object]) -> np.ndarray:
    kind = str(recipe.get("kind", "solid")).strip().lower() or "solid"
    width = int(recipe["width"])
    height = int(recipe["height"])

    if kind == "solid":
        r, g, b = parse_rgb(recipe["color"], "color", 0)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b
        return image

    if kind == "checkerboard":
        r1, g1, b1 = parse_rgb(recipe["color_a"], "color_a", 0)
        r2, g2, b2 = parse_rgb(recipe["color_b"], "color_b", 0)
        block_size = int(recipe["block_size"])

        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                use_a = ((x // block_size) + (y // block_size)) % 2 == 0
                if use_a:
                    image[y, x] = (r1, g1, b1)
                else:
                    image[y, x] = (r2, g2, b2)
        return image

    raise ValueError(f"unsupported image.kind={kind!r}")


T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    norms = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors / norms


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.prefix_length <= 0:
        print("error: --prefix-length must be > 0", file=sys.stderr)
        return 2
    if args.sequence_length <= 0:
        print("error: --sequence-length must be > 0", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be > 0", file=sys.stderr)
        return 2
    if args.logit_scale <= 0:
        print("error: --logit-scale must be > 0", file=sys.stderr)
        return 2

    try:
        cases = parse_cases(args.cases_jsonl)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not cases:
        print("error: --cases-jsonl has no rows", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    hf_token = os.getenv(args.hf_token_env, "").strip() or None

    print(
        f"Loading OpenCLIP model/tokenizer/processor: {args.model_name}@{args.revision}",
        file=sys.stderr,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        token=hf_token,
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_name,
        revision=args.revision,
        token=hf_token,
    )
    model = CLIPModel.from_pretrained(
        args.model_name,
        revision=args.revision,
        token=hf_token,
    ).to(device)
    model.eval()

    all_text_embeddings: list[torch.Tensor] = []
    all_image_embeddings: list[torch.Tensor] = []

    for batch_index, case_batch in enumerate(batched(cases, args.batch_size), start=1):
        print(f"Encoding batch {batch_index} ({len(case_batch)} rows)", file=sys.stderr)
        texts = [case.text for case in case_batch]
        images = [render_image(case.image) for case in case_batch]

        text_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.sequence_length,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

        image_inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)

        with torch.inference_mode():
            text_features = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            image_features = model.get_image_features(pixel_values=pixel_values)
            text_features = normalize_rows(text_features)
            image_features = normalize_rows(image_features)

        all_text_embeddings.append(text_features.cpu())
        all_image_embeddings.append(image_features.cpu())

    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    if text_embeddings.shape != image_embeddings.shape:
        print(
            "error: text/image embedding shape mismatch after encoding: "
            f"{tuple(text_embeddings.shape)} vs {tuple(image_embeddings.shape)}",
            file=sys.stderr,
        )
        return 1

    embedding_dim = int(text_embeddings.shape[1])
    prefix_length = min(args.prefix_length, embedding_dim)
    logits = torch.matmul(image_embeddings, text_embeddings.T) * float(args.logit_scale)

    text_np = text_embeddings.numpy()
    image_np = image_embeddings.numpy()
    logits_np = logits.numpy()

    rows: list[dict[str, object]] = []
    for i, case in enumerate(cases):
        rows.append(
            {
                "id": case.case_id,
                "text": case.text,
                "image": case.image,
                "text_prefix": text_np[i, :prefix_length].astype(float).tolist(),
                "image_prefix": image_np[i, :prefix_length].astype(float).tolist(),
                "logits_row": logits_np[i, :].astype(float).tolist(),
            }
        )

    write_jsonl(args.output_jsonl, rows)
    digest = sha256_file(args.output_jsonl)

    metadata_path = args.metadata_path
    if metadata_path is None:
        metadata_path = args.output_jsonl.parent / "metadata.json"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "local:tools/openclip_generate_golden.py",
        "source_type": "local_transformers_clip",
        "model_repo": args.model_name,
        "model_revision": args.revision,
        "row_count": len(rows),
        "dataset_digest_sha256": digest,
        "settings": {
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "prefix_length": prefix_length,
            "logit_scale": float(args.logit_scale),
            "device": device,
        },
        "row_schema": {
            "id": "string",
            "text": "string",
            "image": "{kind,width,height,...}",
            "text_prefix": "[]float32",
            "image_prefix": "[]float32",
            "logits_row": "[]float32",
        },
    }
    write_metadata(metadata_path, metadata)

    print(f"Wrote JSONL: {args.output_jsonl}", file=sys.stderr)
    print(f"Wrote metadata: {metadata_path}", file=sys.stderr)
    print(f"Rows: {len(rows)}", file=sys.stderr)
    print(f"Embedding dimension: {embedding_dim}", file=sys.stderr)
    print(f"Prefix length: {prefix_length}", file=sys.stderr)
    print(f"Digest (SHA-256): {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
