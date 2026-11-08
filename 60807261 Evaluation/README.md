# COMP8410 Assignment 1 – Evaluation code and data (60807261)

Four head-to-head evaluations of AI models (eight models in total) for a fictional online homewares retailer, Harbour & Home.

## How to reproduce

**Requirements:**
- An NVIDIA GPU with at least 8 GB of memory (tested on an RTX 4060 Laptop, 8 GB), Windows 11.
- Python 3.12 and Jupdyter Notebook.
- About 30 GB of free disk space, for the models and data.

Run the notebooks in this order, from this folder:

| Notebook | What it does | Approx. time |
|---|---|---|
| `0_Environment_Setup_and_Dataset_Preparation.ipynb` | Installs the pinned packages (restart the kernel afterwards), records hardware in `results/env.json`, downloads both datasets into `data/`, records model versions in `data/model_versions.json` | 30–60 min (downloads) |
| `01_pair1_image_editing.ipynb` | Pair 1: ControlNet-Canny + SD 1.5 vs InstructPix2Pix | ~30 min |
| `02_pair2_personalisation.ipynb` | Pair 2: DreamBooth-LoRA vs IP-Adapter | ~1 h |
| `03_pair3_fraud_detection.ipynb` | Pair 3: GraphSAGE vs FAGCN (MLP and GCN as baselines) | ~30 min |
| `04_pair4_customer_agent.ipynb` | Pair 4: Qwen2.5-1.5B-Instruct vs Phi-3.5-mini-instruct | ~1 h |
| `05_final_summary.ipynb` | Merges the blind self-ratings and builds the overview table | 1 min |

**Outputs:**
- Tables and figures used in the report: `results/summary/`.
- Raw per-item outputs: `results/pair1` … `results/pair4`.
- Blind rating sheets and keys: `results/ratings/`.

**Data and models are not included.** Notebook 00 downloads them from their official sources:
- DreamBooth dataset: github.com/google/dreambooth.
- Amazon fraud graph: CARE-GNN GitHub repository.
- Model weights: Hugging Face.

Seeds are fixed: Pair 1 uses 0–1, Pair 2 uses 0–3, Pair 3 uses 0–4, and Pair 4 uses greedy decoding. GPU fp16 arithmetic can cause small differences in generated images between runs.

## Implementation statement for each pair

**Pair 1 – catalogue photo variants.**
- Uses pretrained diffusers pipelines (`StableDiffusionControlNetPipeline` with `lllyasviel/sd-controlnet-canny`, and `StableDiffusionInstructPix2PixPipeline`). No training.
- My own parts: the 24 test cases (8 products × colour/material/scene), the metrics (CLIP-T, CLIP direction, DINO structure distance, edge F1), the control-strength ablation and the blind rating sheet.

**Pair 2 – same-product campaign images.**
- I trained four DreamBooth-LoRA adapters myself: rank 4, 800 steps, learning rate 1e-4, prior preservation with 16 class images, with checkpoints at 400 and 800 steps.
- The first run used the course-notebook settings (400 steps, lr 5e-5), which under-fitted the products.
- IP-Adapter uses the pretrained `h94/IP-Adapter` (`ip-adapter_sd15.bin`), with no training.
- Metrics follow the DreamBooth paper (DINO-I and CLIP-I against a held-out photo, CLIP-T).

**Pair 3 – fake-review account detection.**
- All four models are trained from scratch with PyTorch Geometric layers (`SAGEConv`, `FAConv`, `GCNConv`).
- Experiments:
  - 5 seeds, 40/20/40 stratified splits;
  - each relation of the graph separately;
  - a camouflage-edge attack;
  - AUPRC by user degree.

**Pair 4 – customer-service agent.**
- Pretrained instruction-tuned models, both loaded in 4-bit NF4.
- My own parts: the simulated shop, the 40 tasks, the JSON agent loop, the programmatic scorer and the guard-on/off ablation.
- After the first run, the refusal keywords were extended to accept escalation to a human, which the policy allows.

## Notes
- **Self-ratings:** the ratings in `results/ratings/` are my own blind ratings, from a single rater. They are not an independent user study.