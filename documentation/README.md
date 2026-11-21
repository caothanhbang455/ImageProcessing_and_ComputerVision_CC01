# Documentation

Start with a guide based on what needs to be done. The reference files contain the exact dataset and metric rules; the archive preserves earlier experiments and implementation notes.

## Guides

| Topic | English | Tiếng Việt |
|---|---|---|
| Use the repository, add a method and inspect results | [Implementation guide](implementation_guide.md) | [Hướng dẫn sử dụng repository](implementation_guide-vi.md) |
| Understand the different approaches to document unwarping | [Methods overview](methods_overview.md) | [Tìm hiểu các phương pháp](methods_overview-vi.md) |
| Design experiments and interpret evaluation results | [Experiments and evaluation](experiments_and_evaluation.md) | [Thực nghiệm và đánh giá](experiments_and_evaluation-vi.md) |

## Technical reference

- [Dataset protocols](reference/dataset_protocols.md): sample identities, variants, pairing and split rules.
- [Geometry conventions](reference/geometry_conventions.md): coordinates, map direction, interpolation and composition.
- [Evaluation protocols](reference/evaluation_protocols.md): supported benchmark procedures, dependencies and metric limitations.

## Archive

The `archive/implementation/` folder preserves older implementation and environment-fix notes. The `archive/reports/` folder holds dataset investigations, baseline pilot results and early DocUNet inspection files. `archive/project_context_20261118.md` is a historical planning snapshot.

Archived files may describe earlier repository paths or an earlier verification status. Use the guides and technical references above for current instructions, and use saved runs for actual experimental evidence. Nothing in `archive/` is required when writing a new method.
