import json

class Prompts:
    def __init__(self, mode: str):
        self.mode = mode
        
    def build_planner_prompt(self, abstract: str) -> str:
        """Build a prompt for the planner stage."""

        prompt = f"""You are an astronomer tasked with generating magazine style explanations of astronomy abstracts.
        TASK:
        Create a short internal plan describing what each of the four paragraphs will cover.

        FINAL TARGET (for the rewrite stage):
        1. Exactly 4 short paragraphs.
        2. Total length 180-250 words.
        3. Engaging but clear language.
        4. Suitable for scientifically curious general readers.

        PARAGRAPH STRUCTURE:
        Paragraph 1 — Introduce the central problem in an engaging but simple way.
        Paragraph 2 — Explain what the researchers did, avoiding heavy jargon.
        Paragraph 3 — Present the key findings clearly and accurately.
        Paragraph 4 — State implications only if explicitly supported by abstract; otherwise state limitations/uncertainty from abstract.

        PLAN RULES:
        1. Stay strictly anchored to the abstract.
        2. Do NOT add new scientific claims.
        3. Keep each paragraph description to one concise sentence.
        4. Do NOT write the final explanation.

        OUTPUT FORMAT:
        Write four lines, each starting with:
        "Paragraph 1:", "Paragraph 2:", etc.

        ABSTRACT:
        {abstract}
        """
        return prompt


    def build_writer_prompt(self, abstract: str, plan: str, glossary: str = "") -> str:
        prompt = f"""You are writing a short magazine-style explanation of the following astronomy abstract:
        {abstract}

        You have generated the following plan for how to structure the explanation:
        {plan}
        """
        if glossary.strip():
            prompt += f"""
        You also have a small glossary of simple term explanations (definitions only):
        {glossary}

        Important rule about the glossary:
        - Use it ONLY to explain terms that appear in the abstract.
        - Do NOT use it to add new results, numbers, or claims beyond the abstract.
        """

        prompt += """
        Your task is to write a concise magazine-style explanation targeted at readers with general science literacy.

        Requirements:
        1. The explanation must follow the structure and order of the plan.
        2. The explanation must contain exactly 4 short paragraphs separated by a blank line.
        3. The total length must be between 180 and 250 words.
        4. Use clear, engaging language, but avoid hype or exaggeration.
        5. Prefer short sentences and common words.
        6. Do NOT introduce facts or claims that are not supported by the abstract.
        7. Do NOT use phrases such as “this paper”, “we present”, or “in this study”.

        Output only the final 4-paragraph explanation. Do not include titles, bullet points, or extra commentary.
        """
        return prompt

    def build_critic_prompt(self, abstract: str, draft: str) -> str:
        """Build a prompt for the critic stage."""
        prompt = f"""You are a strict editorial reviewer evaluating a magazine-style explanation of an astronomy abstract.
        ABSTRACT (source of truth):
        {abstract}

        DRAFT TO REVIEW:
        {draft}

        Evaluate semantic quality only:
        - Faithfulness to the abstract (no invented facts or implications)
        - Logical structure (problem → method → findings → implications)
        - Clarity for general science readers

        Do NOT evaluate formatting constraints like paragraph count, word count, or forbidden phrases; those are handled elsewhere.

        SCORING (integers 0-5):
        - scores.hallucination: 0 = fully supported by abstract, 5 = many unsupported claims
        - scores.structure: 0 = excellent problem→method→findings→implications flow, 5 = poor or jumbled flow
        - scores.clarity: 0 = very clear and accessible, 5 = confusing or too technical

        If the draft contains any non-trivial statement not clearly supported by the abstract, list it in hallucinated_claims and increase scores.hallucination.

        OUTPUT:
        Return ONLY valid JSON. No markdown. No code fences. No extra keys.
        If there are no hallucinated claims, return an empty list [].
        If no fixes are needed, return an empty list [].

        JSON SCHEMA (must match exactly):
        {{
        "scores": {{
            "hallucination": 0,
            "structure": 0,
            "clarity": 0
        }},
        "fix_instructions": [],
        "hallucinated_claims": []
        }}
        """
        return prompt
    
    def build_reviser_prompt(self, abstract: str, plan: str, draft: str, critique_json: str, glossary: str = "") -> str:
        """Build a prompt for the revise stage."""

        prompt = f"""You are revising a short magazine-style explanation of an astronomy abstract.
        ABSTRACT (source of truth):
        {abstract}

        PLAN (intended structure; follow the same order):
        {plan}

        DRAFT (to revise):
        {draft}

        CRITIQUE (JSON; must address all fix_instructions):
        {critique_json}
        """
        if glossary.strip():
            prompt += f"""
        GLOSSARY (definitions only; use only to explain terms already in the abstract):
        {glossary}

        Important rule about the glossary:
        - Use it ONLY to explain terms that appear in the abstract.
        - Do NOT use it to add new results, numbers, or claims beyond the abstract.
        """

        prompt += """
        Your task is to produce a revised final version that fixes all issues in the critique.

        HARD CONSTRAINTS:
        1. Exactly 4 paragraphs separated by a blank line.
        2. Total length between 180 and 250 words.
        3. Stay strictly anchored to the abstract. Remove or soften any unsupported claims.
        4. Follow the plan's order: problem → method → findings → implications.
        5. Use clear, engaging language for general science readers (not academic, not hype).
        6. Do NOT use phrases such as “this paper”, “we present”, or “in this study”.
        7. Do NOT include titles, bullet points, or extra commentary.
        8. Never introduce any claim that is not directly supported by the abstract.
        9. If CRITIQUE contains "hallucinated_claims", explicitly remove those claims from the revised draft.

        OUTPUT:
        Return only the revised 4-paragraph explanation (no JSON, no notes).
        """
        return prompt
        
    def build_glossary_prompt(self, abstract: str) -> str:
        """Build a prompt for generating a glossary with optional web search."""

        prompt = f"""You are creating a glossary of simple explanations to support a short magazine-style rewrite.

        ABSTRACT:
        {abstract}

        TASK:
        1. Identify 3-5 domain-specific noun phrases that are required to understand the abstract.
        2. A term must name a thing/concept/method/instrument/physical quantity (noun or noun phrase).
        3. Do not include: adverbs, adjectives, or generic academic phrasing.

        If needed, you may use the web_search tool to find reliable definitions.
        Use trustworthy sources (e.g., NASA, ESA, university pages, Wikipedia, arXiv).

        GLOSSARY RULES:
        1. Include only terms that appear verbatim in the abstract.
        2. Provide concise definitions (1-2 sentences).
        3. Definitions must be written in plain language.
        4. Do NOT include scientific results or new claims.
        5. If a term is already widely known (e.g., galaxy, black hole), DO NOT include it.
        6. If no technical terms require explanation, return an empty JSON object {{}}.
        7. Each key must be a noun phrase (e.g., “Lyman continuum”, “integral-field spectroscopy”). Never include adverbs/adjectives like “phenomenologically”, “kinematically”, “robust”, “consistent”.
        8. For each candidate term, silently apply this test: If removing the term would not block a general reader from following the story, exclude it.
        9. Output must contain 0 keys or 3-5 keys. Never never >5.

        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {{
        "term1": "simple definition",
        "term2": "simple definition"
        }}
        """
        return prompt
