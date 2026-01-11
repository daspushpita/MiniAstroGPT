# AstroGPT
### A Domain-Specific GenAI System for Explaining Astrophysics Research

AstroGPT is a **domain-adapted generative AI system** designed to translate technical astrophysics research abstracts into **clear, accurate, and public-friendly explanations**.

The system is built around a **teacher–student training paradigm**, where a high-capability language model first generates high-quality explanations from raw arXiv abstracts, and a smaller **GPT-2–based student model** is then fine-tuned to reproduce this explanatory style efficiently.

AstroGPT demonstrates how **structured prompting, automated dataset construction, and classical language-model fine-tuning** can be combined into a robust, end-to-end GenAI pipeline.

---

## Why AstroGPT?

Scientific knowledge—especially in fields like astrophysics—is often locked behind dense, technical language. AstroGPT addresses this gap by:

- Preserving scientific correctness  
- Eliminating unnecessary jargon  
- Producing explanations readable by non-experts  
- Enabling scalable, automated science communication  

The project also serves as a **research-grade example** of how to build and evaluate domain-specific generative models from scratch.

---

## Core Capabilities

- Automated arXiv ingestion for astro-ph research abstracts  
- Teacher-prompt framework with strict style and content constraints  
- SQLite-backed data pipeline for reproducible dataset generation  
- End-to-end GPT-2 fine-tuning workflow (no black-box tooling)  
- Evaluation rules to enforce readability and stylistic correctness  
- CLI-based inference for generating explanations on demand  
- Modular design supporting model scaling and alignment extensions  

---

## System Overview

```Raw arXiv abstracts
		↓
Structured SQLite database
		↓
Teacher prompt generation
		↓
Teacher explanations (JSON)
		↓
Filtered training dataset (JSONL)
		↓
Fine-tuned GPT-2 student model
		↓
Clear, layperson-friendly explanations
```

⸻

This architecture mirrors real-world GenAI workflows:
**data ingestion → supervision → filtering → training → evaluation → inference**.

---

## Repository Structure

```AstroGPT/
├── data/
│   ├── raw/                         # Downloaded arXiv metadata
│   └── processed/
│       ├── mini_astrolm.db          # SQLite database
│       ├── batches/                 # Abstract batches
│       ├── teacher_outputs/         # Teacher explanations
│       └── train.jsonl              # Final training dataset
│
├── src/
│   └── miniastrolm/
│       ├── data/
│       │   ├── arxiv_fetcher.py     # arXiv ingestion logic
│       │   ├── db.py                # DB schema & helpers
│       │   └── dataset_builder.py   # Batch & JSONL construction
│       │
│       ├── llm/
│       │   ├── teacher.py           # Teacher model interface
│       │   ├── student.py           # Student inference
│       │   └── prompts.py           # Prompt templates & style rules
│       │
│       ├── training/
│       │   ├── finetune_gpt2.py     # Fine-tuning entry point
│       │   └── collate.py           # Tokenization & batching
│       │
│       └── eval/
│           ├── eval_rules.py        # Structural & style checks
│           └── sample_compare.py    # Before/after comparisons
│
├── scripts/
│   ├── build_dataset.py             # Dataset generation
│   ├── train.py                     # Training launcher
│   └── explain.py                   # CLI inference tool
│
├── notebooks/
│   └── debug_*.ipynb                # Experiments & debugging
│
└── README.md
```
---

## Design Philosophy

- **Explicit over implicit** — every transformation step is inspectable  
- **Reproducible by construction** — datasets are derived, not manual  
- **Small-model realism** — optimized for efficient training and inference  
- **Production-inspired layout** — mirrors real ML system organization  

No monolithic scripts. No hidden magic.

---

## Installation

*Coming soon.*

(Planned support via Hugging Face `transformers` with minimal dependencies.)

---

## Usage

*Coming soon.*

(The CLI will support explaining raw abstracts or arXiv IDs.)

---

## Planned Extensions

- Scaling to instruction-tuned foundation models  
- Preference-based alignment and quality ranking  
- Automated readability and faithfulness metrics  
- Web interface for daily arXiv summaries  
- Cross-domain generalization beyond astrophysics  

---

## Author

**Pushpita Das**  
Astrophysicist · Machine Learning Researcher · GenAI Systems Developer