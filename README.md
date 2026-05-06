# ReportQA: QA-Based Radiology Report Evaluation

## Getting Started

### 1. QA construction

``` bash
# generate qas from free-form reports
bash scripts/generate_qas/generate_qas.sh

# self-filter & report-based filter
bash scripts/filter/filter_ctrg_brain_zh.sh
```

### 2. QA-based evaluation
``` bash
# model inference (zero-shot or SFT)
# install `ms-swift` first
bash scripts/infer/internvl/infer_ctrg_brain_zh.sh

# evaluation & scoring
bash scripts/eval/eval_ctrg_brain_zh.sh
```
