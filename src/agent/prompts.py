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
        Paragraph 4 — Explain why the results matter or what they change.

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

        Your task is to check whether the draft satisfies all constraints and stays fully supported by the abstract.

        Check the following:

        1. Exactly 4 paragraphs separated by a blank line.
        2. Total length between 180 and 250 words.
        3. No invented facts, numbers, claims, or implications not present in the abstract.
        4. Follows logical structure: problem → method → findings → implications.
        5. Clear, accessible language for general science readers.
        6. Does NOT use phrases such as "this paper", "we present", or "in this study".

        If the draft contains information (other than basic background information) not clearly supported by the abstract, mark it as a hallucination.

        OUTPUT FORMAT:
        Return ONLY valid JSON with this exact structure:

        {{
        "passed": true or false,
        "failures": ["word_count", "paragraph_count", "hallucination", "structure", "clarity", "forbidden_phrases"],
        "hallucinated_claims": ["claim 1", "claim 2"],
        "fix_instructions": [
            "specific instruction 1",
            "specific instruction 2"
        ]
        }}

        Do NOT rewrite the draft.
        Do NOT include any text outside the JSON.
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
        Identify technical terms in the abstract that may be unfamiliar to scientifically curious general readers.

        If needed, you may use the web_search tool to find reliable definitions.
        Use trustworthy sources (e.g., NASA, ESA, university pages, Wikipedia, arXiv).

        GLOSSARY RULES:
        1. Include only terms that appear verbatim in the abstract.
        2. Provide concise definitions (1-2 sentences).
        3. Definitions must be written in plain language.
        4. Do NOT include scientific results or new claims.
        5. If a term is already widely known (e.g., galaxy, black hole), do NOT include it.
        6. If no technical terms require explanation, return an empty JSON object {{}}.

        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {{
        "term1": "simple definition",
        "term2": "simple definition"
        }}
        """
        return prompt
