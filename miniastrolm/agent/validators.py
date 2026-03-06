from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

@dataclass
class ValidationResult:
    passed: bool
    failures: List[str]
    word_count: int
    paragraph_count: int
    forbidden_hits: List[str]

class Validator:
    def __init__(self, draft: str,
                min_words: int = 170,
                max_words: int = 220,
                required_paragraphs: int = 4):
        
        self.draft = draft
        self.FORBIDDEN_PHRASES_DEFAULT = ("this paper", "we present", "in this study", "in this work", "we propose")
        self.min_words = min_words
        self.max_words = max_words
        self.required_paragraphs = required_paragraphs
    
    def _normalize_text(self, text: str) -> str:
        """Light normalization to make counting consistent."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return text
        
    def split_paragraphs(self) -> List[str]:
        normalized_draft = self._normalize_text(self.draft)
        if not normalized_draft:
            return []
        # Split on blank-line separators
        paragraphs = re.split(r"\n\s*\n+", normalized_draft)
        # Remove empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def count_paragraphs(self) -> int:
        return len(self.split_paragraphs())
    
    def count_words(self) -> int:
        normalized_draft = self._normalize_text(self.draft)
        if not normalized_draft:
            return 0
        words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", normalized_draft)
        return len(words)
    
    def find_forbidden_phrases(self) -> list[str]:
        normalized_draft = self._normalize_text(self.draft).lower()
        hits = []
        for phrase in self.FORBIDDEN_PHRASES_DEFAULT:
            if phrase in normalized_draft:
                hits.append(phrase)
        return hits
    
    def validate(self) -> ValidationResult:
        
        failures: List[str] = []
        
        norm = self._normalize_text(self.draft)
        if not norm:
            return ValidationResult(
                passed=False,
                failures=["empty"],
                word_count=0,
                paragraph_count=0,
                forbidden_hits=[],
            )
        word_count = self.count_words()
        paragraph_count = self.count_paragraphs()
        forbidden_hits = self.find_forbidden_phrases()
        
        
        if word_count < self.min_words or word_count > self.max_words:
            failures.append("word_count")
        if paragraph_count != self.required_paragraphs:
            failures.append("paragraph_count")
        if forbidden_hits:
            failures.append("forbidden_phrases")
        
        return ValidationResult(
            passed=(len(failures) == 0),
            failures=failures,
            word_count=word_count,
            paragraph_count=paragraph_count,
            forbidden_hits=forbidden_hits)
    
    
