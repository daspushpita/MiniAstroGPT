# AstroGPT

A GenAI System for Simplifying Astrophysics Research

AstroGPT is a domain-specific generative AI system fine-tuned from GPT-2 to translate technical astrophysics research abstracts—sourced directly from arXiv—into clear, accessible explanations for non-experts.

The project implements an end-to-end teacher–student pipeline, including automated data ingestion, structured prompt generation, dataset construction, and language-model fine-tuning.

⸻

## Key Features

	•	Automated arXiv ingestion for astro-ph research abstracts
	•	Teacher-prompt framework for generating high-quality, human-readable explanations
	•	End-to-end GPT-2 fine-tuning pipeline, built entirely in Python
	•	SQLite-backed dataset management for scalable preprocessing
	•	Modular, class-based architecture (data → prompts → models → evaluation)
	•	CLI-based inference for generating public-friendly explanations
	•	Designed to be easily extensible to larger models, instruction tuning, or RLHF

⸻

## High-Level Pipeline

arXiv abstracts
      ↓
SQLite database
      ↓
Teacher prompt generation
      ↓
Teacher explanations (JSON)
      ↓
Training dataset (JSONL)
      ↓
Fine-tuned GPT-2 student model
      ↓
Layperson-friendly explanations

⸻

## Project Structure

AstroGPT/
├── data/
│   ├── raw/                     # Downloaded arXiv metadata
│   └── processed/
│       ├── mini_astrolm.db      # SQLite database
│       ├── batches/             # Abstract batches
│       ├── teacher_outputs/     # Teacher model outputs
│       └── train.jsonl          # Final training dataset
│
├── src/
│   └── miniastrolm/
│       ├── data/
│       │   ├── arxiv_fetcher.py     # arXiv ingestion
│       │   ├── db.py                # SQLite schema & helpers
│       │   └── dataset_builder.py   # Batch + JSONL creation
│       │
│       ├── llm/
│       │   ├── teacher.py           # Teacher model loading & generation
│       │   ├── student.py           # Student model inference
│       │   └── prompts.py           # Prompt templates & style rules
│       │
│       ├── training/
│       │   ├── finetune_gpt2.py     # Fine-tuning entry point
│       │   └── collate.py           # Tokenization & formatting
│       │
│       └── eval/
│           ├── eval_rules.py        # Structural & style checks
│           └── sample_compare.py    # Before/after comparisons
│
├── scripts/
│   ├── build_dataset.py             # Dataset construction
│   ├── train.py                     # Model training
│   └── explain.py                   # CLI inference tool
│
├── notebooks/
│   └── debug_*.ipynb                # Experiments & debugging
│
└── README.md

## Installation

Coming soon.
(Planned support for local inference via Hugging Face transformers.)

⸻

## Usage

Coming soon.
(The CLI will support generating explanations from raw arXiv abstracts or IDs.)

## Author

Pushpita Das
Astrophysicist · Machine Learning Researcher · GenAI Developer

⸻
