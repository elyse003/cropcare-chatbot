# 🌾 CropCare — Agricultural Disease Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)

**A domain-specific agricultural assistant fine-tuned with LoRA (PEFT) on crop disease QA data.**

[ Model](https://huggingface.co/UyiringiyeMarie/cropcare-tinyllama) • 
[ Experiments](https://huggingface.co/UyiringiyeMarie/cropcare-tinyllama-experiments)  • [ Demo Video](#)

</div>

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Fine-Tuning Methodology](#-fine-tuning-methodology)
- [Experiments & Results](#-experiments--results)
- [Performance Metrics](#-performance-metrics)
- [Gradio UI](#-gradio-ui)
- [How to Run](#-how-to-run)
- [Example Conversations](#-example-conversations)
- [Project Structure](#-project-structure)

---

##  Project Overview

**CropCare** is a domain-specific conversational AI assistant designed to help farmers and agricultural professionals diagnose crop diseases and receive expert treatment advice. It was built by fine-tuning **TinyLlama-1.1B-Chat** using **LoRA (Low-Rank Adaptation)** via the `peft` library, enabling efficient training on limited GPU resources.

### Why CropCare?
Smallholder farmers often lack access to agricultural experts. CropCare bridges this gap by providing instant, accurate advice on:
- Crop disease identification and diagnosis
- Treatment and management recommendations
- Pest control strategies
- Soil and fertilizer guidance
- Prevention best practices

The model is specifically designed to **decline out-of-domain questions**, ensuring responses are always relevant and trustworthy within the agricultural domain.

---

##  Dataset

| Property | Details |
|---|---|
| **Source** | [YuvrajSingh9886/Agriculture-Plan-Diseases-QA-Pairs-Dataset](https://huggingface.co/datasets/YuvrajSingh9886/Agriculture-Plan-Diseases-QA-Pairs-Dataset) |
| **Domain** | Agricultural Crop Disease Diagnosis |
| **Format** | Question-Answer pairs |
| **Total Samples** | ~2,570 QA pairs |
| **Train Split** | 80% (~2,054 samples) |
| **Validation Split** | 5% (~128 samples) |
| **Test Split** | 15% (~388 samples) |

### Preprocessing Steps
1. **Cleaning** — Removed extra whitespace, special characters, and malformed entries
2. **Filtering** — Removed samples with question or answer length < 10 characters
3. **Instruction Formatting** — Converted raw QA pairs into structured instruction-response templates:
```
### Instruction:
As an agricultural expert, please diagnose and advise on this crop issue: {question}

### Response:
{answer}
```
4. **Tokenization** — Used TinyLlama's tokenizer with `max_length=512`, `padding="max_length"`, and `truncation=True`
5. **Label Assignment** — Labels set as a copy of `input_ids` for causal language modeling (next-token prediction)

---

##  Model Architecture

| Component | Details |
|---|---|
| **Base Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Parameters** | 1.1 Billion |
| **Precision** | float16 |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) via PEFT |
| **Trainable Parameters** | 4,505,600 (0.41% of total) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | q_proj, v_proj, k_proj, o_proj |
| **LoRA Dropout** | 0.05 |

Using LoRA means only **0.41%** of parameters are trained, making fine-tuning feasible on a single GPU in Google Colab.

---

##  Fine-Tuning Methodology

The model was fine-tuned using **3 experiments** with progressively optimized hyperparameters:

### Training Setup
- **Optimizer**: AdamW (default)
- **Scheduler**: Cosine Annealing
- **Gradient Accumulation**: 4 steps (effective batch size = 16)
- **Mixed Precision**: fp16
- **Early Stopping**: Based on `eval_loss`
- **Evaluation Strategy**: Every 50 steps

---

##  Experiments & Results

| Hyperparameter | Experiment 1 (Baseline) | Experiment 2 (Optimized) | Experiment 3 (Max Convergence) |
|---|---|---|---|
| **Epochs** | 3 | 5 | 10 |
| **Learning Rate** | 2e-4 | 5e-5 | 8e-5 |
| **Batch Size** | 4 | 4 | 4 |
| **Warmup Steps** | 50 | 150 | 250 |
| **Weight Decay** | 0.01 | 0.05 | 0.1 |
| **Max Grad Norm** | 1.0 (default) | 0.3 | 0.3 |
| **Gradient Accumulation** | 4 | 4 | 4 |

### Results Comparison

| Metric | Exp 1 (Baseline) | Exp 2 (Optimized) | Exp 3 (Max Conv) |
|---|---|---|---|
| **Training Loss** | 1.2224| 1.3949 | 1.1069 |
| **Test Loss** | 1.0509 | 1.1410 | 1.0178 |
| **ROUGE-1** | 0.4729 | 0.4370 | 0.4883 |
| **ROUGE-2** | 0.2445 | 0.2174 | 0.2610 |
| **ROUGE-L** | 0.4567 | 0.4200 | 0.4735 |
| **BLEU** | 6.6765 | 5.7589 | 5.0784 |



### Key Observations
- **Experiment 1** established the baseline with 3 epochs and a high learning rate
- **Experiment 2** refined domain knowledge using a lower LR and stronger regularization
- **Experiment 3** pushed convergence further with 10 epochs and aggressive weight decay (0.1) to prevent overfitting

---

## Performance Metrics

The fine-tuned model was evaluated using:

| Metric | Description | Why Used |
|---|---|---|
| **ROUGE-1** | Unigram overlap between prediction and reference | Measures basic vocabulary coverage |
| **ROUGE-2** | Bigram overlap | Measures phrase-level accuracy |
| **ROUGE-L** | Longest common subsequence | Measures expert reasoning quality |
| **BLEU** | N-gram precision score | Standard MT/generation evaluation |
| **Test Loss** | Cross-entropy loss on held-out test set | Measures generalization |

### Overfitting Analysis
The gap between training loss and test loss was monitored across all experiments:
- Gap < 0.1 →  Excellent generalization
- Gap 0.1–0.3 →  Good balance
- Gap > 0.3 →  Potential overfitting

---

##  Gradio UI

CropCare features a polished Gradio web interface with:
-  Real-time chat with the fine-tuned model
-  Out-of-domain detection (declines non-agricultural questions)
-  Example questions for easy testing
-  Model info panel showing metrics
-  Agricultural tips sidebar

### Out-of-Domain Handling
The chatbot automatically detects and declines non-agricultural questions:
```
User: What is the capital of France?

CropCare: 🌱 I'm CropCare, an agricultural assistant specialized in crop
diseases and plant health. I can only answer questions related to
agriculture, plant diseases, pest management, and farming practices.
```

---

##  How to Run

### Option 1 — Google Colab (Recommended)
1. Click the **Open in Colab** badge above
2. Select `Runtime → Run All`
3. Wait for training to complete (~45-60 mins on T4 GPU)
4. The Gradio UI will launch automatically with a shareable link

### Option 2 — Local Setup
```bash
# Clone the repository
git clone https://github.com/elyse003/cropcare-chatbot.git
cd cropcare-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Final_summative-mltech1 (1).ipynb
```

### Requirements
```
transformers==4.44.0
peft==0.12.0
trl==0.11.0
accelerate==0.34.0
datasets
evaluate
sacrebleu
rouge-score
gradio
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Example Conversations

###  In-Domain — Disease Diagnosis
```
User: What causes powdery mildew on cucumber leaves?

CropCare: 🌾 Powdery mildew on cucumber is caused by the fungus
Podosphaera xanthii. Treatment includes applying sulfur-based
fungicide every 7-10 days, removing infected leaves, and improving
air circulation. Neem oil is an effective organic alternative.
Avoid overhead watering and choose resistant varieties for prevention.
```

###  In-Domain — Pest Management
```
User: How do I identify and treat root rot in vegetable crops?

CropCare: 🌾 Root rot is caused by waterlogged soil and pathogens
like Pythium and Phytophthora. Key symptoms include yellowing leaves,
wilting despite adequate water, and brown/black mushy roots.
Treatment: improve drainage immediately, apply Trichoderma-based
biocontrol agents, and treat soil with copper oxychloride.
```

###  Out-of-Domain — Declined
```
User: How do I code a website in Python?

CropCare: 🌱 I'm CropCare, an agricultural assistant specialized
in crop diseases and plant health. I can only answer questions
related to agriculture, plant diseases, pest management, and
farming practices.
```

---

## Project Structure

```
cropcare-chatbot/
│
├── Final_summative-mltech1.ipynb     # Main notebook — complete pipeline
├── README.md                   # This file
├── requirements.txt            # Package dependencies
│
└── cropcare_trained_model/     # Trained model artifacts (auto-created)
```

---

##  Author

**Marie Elyse Uyiringiye** — African Leadership University (ALU)
- GitHub: [@elyse003](https://github.com/elyse003)
- Hugging Face: [UyiringiyeMarie](https://huggingface.co/UyiringiyeMarie)

---

##  License

This project is submitted as part of the ALU MLTech Summative Assessment.

---

<div align="center">
<i>Built with ❤️ for African farmers 🌍</i>
</div>
```
