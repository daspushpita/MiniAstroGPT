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
        critic_pretty = json.dumps(json.loads(critic_raw), indent=2)
    except Exception:
        critic_pretty = critic_raw

    critic = f"### Critic\n\n```json\n{critic_pretty}\n```"

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
html, body {
    margin: 0;
    min-height: 100%;
    background: #020617;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image: url("https://cdn.esawebb.org/archives/images/wallpaper5/weic2425c.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    transform: translateZ(0);
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
    max-width: 1100px;
    margin: 24px auto !important;
    padding: 20px !important;
}

.gradio-container > div,
.gradio-container .main,
.gradio-container .block {
    background: transparent !important;
}

#hero_box {
    max-width: 900px;
    margin: 0 auto 16px auto;
    padding: 10px 6px;
}

#hero_box h1 {
    margin-bottom: 0.2rem;
}

#hero_box h3 {
    margin-top: 0;
    color: #dbeafe;
    font-weight: 500;
}

#paper_card {
    background: rgba(2, 6, 23, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    max-width: 900px;
    margin: 20px auto;
}

#paper_card p {
    line-height: 1.7;
    font-size: 16px;
}

#paper_card h2 {
    margin-bottom: 0.4rem;
}

#paper_card h3 {
    margin-top: 1.2rem;
}

textarea, input {
    background: rgba(15, 23, 42, 0.75) !important;
}

footer {
    display: none !important;
}

.glossary-item {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    line-height: 1.6;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.glossary-item strong {
    color: #93c5fd;
}

.glossary-item:hover {
    transform: translateY(-2px);
    border-color: #60a5fa;
    box-shadow: 0 8px 22px rgba(96, 165, 250, 0.12);
}
"""

with gr.Blocks(css=css) as demo:
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
