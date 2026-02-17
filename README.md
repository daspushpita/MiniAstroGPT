# MiniAstroLM (AstroGPT) – v1  
### Controlled Distillation of Mechanism-Focused Scientific Explanations

MiniAstroLM v1 is a research-oriented teacher–judge–student distillation pipeline for generating structured, mechanism-focused explanations of astrophysics abstracts.

This project investigates a focused modeling question:

> Can a small causal language model reliably reproduce a constrained scientific explanation style when supervision is generated and filtered systematically?

Rather than relying on prompt engineering alone, MiniAstroLM constructs supervision explicitly and distills it into GPT-2 under controlled training conditions.

---

## Research Motivation

Astrophysics abstracts compress physical mechanisms, observational constraints, and inference chains into dense prose.

Large language models can unpack this effectively — but:

- Prompt-only approaches are unstable.
- Style drift is common.
- Faithfulness is difficult to enforce.
- Inference cost is high.

MiniAstroLM treats explanation generation as a **controlled distillation problem under constraints**, not as a generic summarization task.

---

## v1 Task Definition

Given an astrophysics abstract, generate an explanation that:

- Is 180–220 words.
- Rewrites the abstract from scratch (no structural copying).
- Focuses on physical mechanism and inference.
- Avoids academic reporting phrases.
- Maintains structured paragraph flow.

This creates a tightly constrained output distribution suitable for small-model distillation experiments.

---

## System Overview
```
Raw arXiv abstracts
    -> SQLite ingestion + batching
    -> Teacher-generated explanations
    -> Filtering + validation
    -> Curated JSONL dataset
    -> Student fine-tuning: GPT-2
    -> Clean, readable explanations
```

---

## Architecture

```mermaid
flowchart TB

    subgraph Supervision
        A[Astro abstracts]
        B[Teacher LLM]
        C[Structured JSON output]
        D[Judge evaluation]
        E[Filtered dataset]
        A --> B --> C --> D --> E
    end

    subgraph Distillation
        E --> F[Causal LM collator<br/>(prefix masked)]
        F --> G[GPT-2 fine-tuning]
        G --> H[Student checkpoint]
    end

    subgraph Inference
        I[New abstract]
        H --> J[Greedy decoding]
        I --> J
        J --> K[Mechanism-focused explanation]
    end
```
## Technical Highlights

### 1. Explicit Supervision Construction

Teacher outputs follow a strict JSON schema:
	•	id
	•	title
	•	abstract
	•	target_explanation
	•	judge_feedback
	•	accepted

Only samples meeting scoring thresholds are used for student training.

⸻

### 2. Masked Causal LM Training

Training format:
``` 
    [prompt tokens]  → masked (-100)
    [target tokens]  → supervised
```

The student learns conditional generation of explanations without being penalized for the prefix.

⸻

### 3. Overfitting Validation

Before scaling experiments, the pipeline is validated through:
	•	Single-sample overfit verification
	•	20-sample memorization confirmation
	•	EOS supervision debugging
	•	Repetition collapse mitigation
	•	Controlled greedy decoding

This validates:
	•	Dataset formatting
	•	Label masking correctness
	•	Optimization stability
	•	Stop-condition learning

⸻

### 4. Small-Model Realism

GPT-2 small (124M parameters) is used intentionally:
	•	Runs on consumer hardware (MPS-compatible)
	•	Exposes training pathologies clearly
	•	Forces careful supervision design
	•	Emphasizes signal quality over model scale

⸻

Experimental Observations (v1)
	•	Small curated datasets (≤20 samples) are fully memorized.
	•	Repetition collapse occurs without explicit EOS supervision.
	•	Generation stability depends strongly on decoding configuration.
	•	Structured style constraints are learnable via distillation.

These observations provide insight into small-model conditional generation under tight stylistic control.

⸻

## Repository Structure
```
MiniAstroLM/
├── configs/                  # Training + generation configs
├── prompts/                  # Teacher prompt definitions
├── data/                     # Curated datasets + checkpoints
├── src/miniastrolm/
│   ├── llm/                  # Teacher & judge modules
│   ├── data_scripts/         # Dataset construction
│   ├── student/
│   │   ├── data.py
│   │   ├── collate.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── model.py
│   └── eval/                 # Evaluation utilities
└── README.md
```
## Running the Student

### Train

```python -m miniastrolm.student.train \
    --config configs/student_train.yaml
```

### Inference
```
python -m miniastrolm.student.infer \
    --config configs/generation.yaml \
    --model_dir data/student/checkpoint \
    --abstract "Abstract text"
```
## Research Directions

Planned next steps:
	•	Scale curated dataset to 5k–10k samples.
	•	Quantitative teacher–student alignment metrics.
	•	Faithfulness verification via entity anchoring.
	•	LoRA vs full fine-tuning comparison.
	•	Robust decoding under style constraints.
	•	Cross-domain generalization experiments.

⸻

## Positioning

MiniAstroLM is not a summarization demo.

It is a controlled experiment in:
	•	Structured supervision design
	•	Small-model distillation
	•	Style-constrained scientific generation
	•	Training dynamics under explicit output rules

⸻

Author

Pushpita Das
Computational Astrophysicist → Generative AI Systems Research