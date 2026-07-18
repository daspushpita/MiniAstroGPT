import json
import random
import html
from pathlib import Path
import gradio as gr

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "cache" / "papers.jsonl"


def read_papers(path: Path):
    with open(path, "r", encoding="utf-8") as f_in:
        return [json.loads(line) for line in f_in if line.strip()]


def empty_outputs():
    return (
        "",
        "## No papers found",
        "",
        "No paper data is available yet.",
        "<em>No glossary available for this paper.</em>",
        "### Plan\n\n_No plan available._",
        "### Draft\n\n_No draft available._",
        "### Critic\n\n_No critic available._",
    )
    
def format_glossary_markdown(glossary_data) -> str:
    if not glossary_data:
        return "<em>No glossary available for this paper.</em>"

    glossary_dict = json.loads(glossary_data)
    parts = ["<h3>Glossary</h3>"]

    for term, definition in glossary_dict.items():
        safe_term = html.escape(str(term))
        safe_definition = html.escape(str(definition))
        parts.append(
            f"<div class='glossary-item'>"
            f"<strong>{safe_term}</strong><br>{safe_definition}</div>"
        )

    return "\n".join(parts)

def format_paper(paper):
    paper_id = str(paper.get("id", ""))
    raw_title = str(paper.get("title", ""))

    arxiv_url = paper_id
    short_id = paper_id.split("/")[-1] if paper_id else "unknown"

    title = f"""## {raw_title}
<span style="color:#93c5fd; font-size: 14px;">
arXiv: <a href="{arxiv_url}" target="_blank" style="color:#93c5fd; text-decoration:none;">{short_id}</a>
</span>
"""

    abstract = str(paper.get("abstract", ""))
    explanation = str(paper.get("final_explanation", ""))

    glossary = format_glossary_markdown(paper.get("glossary", ""))

    plan = f"### Plan\n\n{str(paper.get('plan', ''))}"
    draft = f"### Draft\n\n{str(paper.get('draft', ''))}"

    critic_raw = str(paper.get("critic", ""))
    try:
        critic_payload = json.loads(critic_raw)
        critic_pretty = json.dumps(critic_payload, indent=2)
    except Exception:
        critic_payload = None
        critic_pretty = critic_raw

    critic_summary = ""
    if isinstance(critic_payload, dict):
        scores = critic_payload.get("scores", {})
        if isinstance(scores, dict):
            hallucination = scores.get("hallucination", "n/a")
            structure = scores.get("structure", "n/a")
            clarity = scores.get("clarity", "n/a")
            critic_summary = (
                "| Metric | Value |\n"
                "|---|---:|\n"
                f"| Hallucination risk (lower is better) | {hallucination}/5 |\n"
                f"| Structure quality (higher is better) | {structure}/5 |\n"
                f"| Clarity quality (higher is better) | {clarity}/5 |\n\n"
            )

    critic = f"### Critic Scores\n\n{critic_summary}```json\n{critic_pretty}\n```"

    return paper_id, title, abstract, explanation, glossary, plan, draft, critic


def initial_paper():
    data = read_papers(DATA_PATH)
    if not data:
        return "", "No papers found", "", "", "", "", "", ""
    initial_paper = data[19]
    return format_paper(initial_paper)

def load_random():
    data = read_papers(DATA_PATH)

    if not data:
        return "", "No papers found", "", "", "", "", "", ""

    paper = random.choice(data)
    return format_paper(paper)

startup_paper_id, startup_title, startup_abstract, startup_explanation, startup_glossary, startup_plan, startup_draft, startup_critic = initial_paper()

css = """
:root {
    color-scheme: dark;
    --body-text-color: #f8fafc;
    --body-text-color-subdued: #dbeafe;
    --background-fill-primary: transparent;
    --background-fill-secondary: rgba(15, 23, 42, 0.92);
    --block-background-fill: transparent;
    --block-border-color: rgba(148, 163, 184, 0.18);
    --button-primary-background-fill: #52525b;
    --button-primary-background-fill-hover: #71717a;
    --button-primary-text-color: #ffffff;
    --link-text-color: #60a5fa;
    --link-text-color-hover: #93c5fd;
}

html, body {
    margin: 0;
    min-height: 100vh;
    background: #020617;
    color: #f8fafc;
}

html {
    overflow-y: scroll;
    scrollbar-gutter: stable;
}

* {
    box-sizing: border-box;
}

::selection {
    background: rgba(96, 165, 250, 0.55);
    color: #ffffff;
}

body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    background-image: url("https://cdn.esawebb.org/archives/images/wallpaper5/weic2425c.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

body::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: rgba(2, 6, 23, 0.40);
}

.gradio-container {
    position: relative;
    z-index: 1;
    background: transparent !important;
    color: #f8fafc !important;
    max-width: 1100px;
    margin: 24px auto !important;
    padding: 20px !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container > div,
.gradio-container .main,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background: transparent !important;
    color: #f8fafc !important;
}

#hero_box {
    max-width: 900px;
    margin: 0 auto 16px auto;
    padding: 10px 6px;
}

#hero_box h1 {
    margin-bottom: 0.2rem;
    color: #f8fafc !important;
    font-weight: 700;
}

#hero_box h3 {
    margin-top: 0;
    color: #dbeafe !important;
    font-weight: 500;
}

#hero_box button,
#hero_box .gr-button {
    background: #52525b !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

#hero_box button:hover,
#hero_box .gr-button:hover {
    background: #71717a !important;
}

#hero_box button *,
#hero_box .gr-button * {
    color: #ffffff !important;
}

#paper_card {
    background: rgba(2, 6, 23, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    max-width: 900px;
    margin: 20px auto;
    color: #f8fafc !important;
}

#paper_card,
#paper_card *,
#paper_card .markdown,
#paper_card .prose,
#paper_card .md,
#paper_card .output-html {
    color: #f8fafc !important;
}

#paper_card p {
    line-height: 1.7;
    font-size: 16px;
    color: #f8fafc !important;
}

#paper_card h2 {
    margin-bottom: 0.4rem;
    color: #f8fafc !important;
    font-weight: 700;
}

#paper_card h3 {
    margin-top: 1.2rem;
    color: #f8fafc !important;
    font-weight: 700;
}

#paper_card a,
#paper_card a:visited {
    color: #60a5fa !important;
    text-decoration-color: rgba(96, 165, 250, 0.75) !important;
}

#paper_card a:hover {
    color: #93c5fd !important;
}

textarea, input {
    background: rgba(15, 23, 42, 0.75) !important;
    color: #f8fafc !important;
}

footer {
    display: none !important;
}

#paper_card .glossary-item {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    line-height: 1.6;
    color: #dbeafe !important;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

#paper_card .glossary-item strong {
    color: #93c5fd !important;
}

#paper_card .glossary-item:hover {
    transform: translateY(-2px);
    border-color: #60a5fa;
    box-shadow: 0 8px 22px rgba(96, 165, 250, 0.12);
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=css) as demo:
    with gr.Column(elem_id="hero_box"):
        gr.Markdown("""
# AstroGPT
### Daily Astronomy ArXiv
""")
        with gr.Row():
            btn_random = gr.Button("Discover another paper")

    state_paper_id = gr.State(startup_paper_id)

    with gr.Column(elem_id="paper_card"):
        title_box = gr.Markdown(value=startup_title)
        out_box = gr.Markdown(value=startup_explanation)
        glossary_box = gr.HTML(value=startup_glossary, label="Glossary")

        with gr.Accordion("Abstract", open=False):
            abstract_box = gr.Markdown(value=startup_abstract)

        with gr.Accordion("Generation Trace", open=False):
            plan_box = gr.Markdown(value=startup_plan)
            draft_box = gr.Markdown(value=startup_draft)
            critic_box = gr.Markdown(value=startup_critic)

    btn_random.click(
        fn=load_random,
        inputs=[],
        outputs=[state_paper_id, title_box, abstract_box, out_box, 
                glossary_box, plan_box, draft_box, critic_box,],
        )

    demo.load(
        fn=initial_paper,
        inputs=[],
        outputs=[state_paper_id, title_box, abstract_box, out_box, 
                glossary_box, plan_box, draft_box, critic_box,],
    )
if __name__ == "__main__":
    demo.launch(show_error=True)
