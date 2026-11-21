---
pretty_name: VN100 Graph Editing
language:
- en
size_categories:
- n<1K
tags:
- scene-graph
- image-editing
- vietnam
- evaluation
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev.jsonl
---

# VN100 Graph Editing

VN100 is a development evaluation set for instruction-guided image editing in Vietnamese street and travel scenes. It contains 200 English editing requests on 100 source photographs. Each source has two independent requests. Both edits start from the same original image.

This metadata edition distributes prompts, revision-pinned upstream image references, outcome labels and graph-based evaluation scores. Source and edited image pixels are omitted. The project selected the images, prepared instructions with AI assistance, ran the editor and graph extractor, and evaluated the resulting edits.

## Contents

| Source | Distinct photographs | Requests |
|---|---:|---:|
| Vietnamese Sidewalk Violation VQA | 50 | 100 |
| VietTravelVQA-v2 | 50 | 100 |

The requests comprise 106 additions, 66 removals, 18 replacements and 10 color changes. The project used Qwen-Image-2.1 with seed 42 and 40 steps. REACT++ extracted scene graphs for 100 source images and 200 edited images.

## Reading the table

Each line in `data/dev.jsonl` describes one request.

| Fields | Meaning |
|---|---|
| `task_id`, `image_id` | Unique request and shared source photograph identifiers. |
| `source_dataset`, `source_url`, `source_revision`, `source_sha256` | Source collection, revision-pinned upstream file page, upstream revision and digest of the original image bytes. |
| `source_dataset_license` | License declared by the upstream collection. |
| `instruction`, `operation` | English prompt supplied to the editor and the intended edit category. |
| `edit_fulfillment`, `preservation`, `overall_pass` | Reviewed fulfillment and preservation labels. Overall pass requires both. |
| `defaulted_label_fields` | Names of originally blank review fields interpreted as pass under the review protocol. Explicit labels are preserved. |
| `edit_score`, `preservation_score`, `graph_consistency` | Graph-based edit score E, preservation score P and their product G when evaluable. Missing scores are JSON null. |
| `preservation_evaluable`, `reason_codes` | Whether P can be used for evaluation, and reasons for abstention. An ambiguous alignment can leave a diagnostic P while G is null. |

For example, request `travel-a4fe5a686b0396-q200-2` asks for a dog standing on foreground rocks beside a natural pool. Both reviewed criteria pass. The graph scorer gives E = 1 and G ≈ 0.829.

## Evaluation and limitations

There is one development split, `dev`. The images were selected purposefully for context and editability. For future partitions, group by `image_id` to keep both requests together. This cohort was used for development and qualitative analysis.

The review protocol treats 287 originally blank fields across 144 requests as passes. Of the 200 requests, 175 pass both reviewed criteria. Inter-annotator agreement was not measured. The graph scorer yields G for 91 requests. Its evidence depends on detected categories, boxes and predicted relations. REACT++ does not encode color in this evaluation. A null score denotes insufficient evidence.

## Source attribution and image rights

The sidewalk photographs come from a collection published under CC BY-NC 4.0. [Vietnamese Sidewalk Violation VQA](https://huggingface.co/datasets/manhdungcr7/dataset_viahe_vqa)

VietTravelVQA-v2 is published under CC BY 4.0 at the collection level. The original authors identify Wikimedia Commons as the source of the travel photographs. Original attribution and image-specific terms should be checked at the source. [VietTravelVQA-v2](https://huggingface.co/datasets/phunguyendinhkh/VietTravelVQA-v2) · [VietTravelVQA v2 author list and DOI](https://doi.org/10.17632/fvxtpnh8mh.2)

The project records the source images as local-evaluation assets with per-image redistribution rights unverified. The license column describes the upstream collection only.
