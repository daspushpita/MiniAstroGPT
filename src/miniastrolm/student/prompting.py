from __future__ import annotations


def build_prompt(tokenizer, paper_id: str, abstract: str) -> str:
    """
    Prompt used during inference.
    Must match the prefix of build_full_text exactly.
    """
    bos = tokenizer.bos_token or ""
    return (
        f"{bos}"
        "### Task:\n"
        "Explain the abstract in simple, non-technical language.\n"
        "Stay strictly on-topic.\n\n"
        "Rules:\n"
        "- 180-220 words\n"
        "- Exactly 4 paragraphs separated by \\n\\n\n"
        "- Rewrite from scratch\n"
        "- No 'we present' or 'this paper'\n"
        "- Stay strictly anchored to the abstract\n\n"
        f"### Abstract:\n{abstract}\n\n"
        "### Output:\n"
    )


def build_full_text(tokenizer, paper_id: str, abstract: str, explanation: str) -> str:
    """
    Full training string (prompt + explanation target).
    """
    eos = tokenizer.eos_token or ""
    return build_prompt(tokenizer, paper_id, abstract) + explanation + eos
