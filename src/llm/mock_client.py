from src.llm.base import LLMClient, LLMResult, LLMUsage


class MockLLMClient(LLMClient):
    """Deterministic local client used for pipeline smoke testing."""

    def generate(
        self,
        stage: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, object]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        if prompt is None and messages is None:
            raise ValueError("Either messages or prompt must be provided.")

        _ = prompt
        _ = messages
        _ = system_prompt
        _ = temperature
        _ = max_new_tokens
        _ = json_mode

        responses = {
            "plan": "Paragraph 1: Context\nParagraph 2: Method\nParagraph 3: Findings\nParagraph 4: Implications",
            "write": (
                "This is a placeholder draft.\n\n"
                "It represents the writer stage output.\n\n"
                "Replace with model-generated text later.\n\n"
                "Keep exactly four paragraphs in final mode."
            ),
            "critic": '{"passed": true, "failures": [], "fix_instructions": []}',
            "glossary": '{"redshift": "Change in wavelength due to relative motion or expansion."}',
            "revise": "This is a placeholder revised draft, which should be an improved version of the initial draft based on the critique and glossary."
        }
        text = responses.get(stage, "Unsupported stage.")
        return LLMResult(text=text, model="mock", usage=LLMUsage())
