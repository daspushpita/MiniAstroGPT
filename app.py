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


def load_random():
    data = read_papers(DATA_PATH)

    if not data:
        return "", "No papers found", "", "", "", "", "", ""

    paper = random.choice(data)

    # paper_id = str(paper.get("id", ""))
    # title = f"## {str(paper.get('title', ''))}"
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


css = f"""
html, body {{
    margin: 0;
    min-height: 100%;
}}

#app_bg {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image: url("https://cdn.esawebb.org/archives/images/wallpaper5/weic2425c.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

#app_bg::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(2, 6, 23, 0.28);
}}

.gradio-container {{
    position: relative;
    z-index: 1;
    background: transparent !important;
    max-width: 1100px;
    margin: 24px auto !important;
    padding: 20px !important;
}}

.gradio-container > div,
.gradio-container .main,
.gradio-container .block {{
    background: transparent !important;
}}

#paper_card {{
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    max-width: 900px;
    margin: 20px auto;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}}

textarea, input {{
    background: rgba(15, 23, 42, 0.75) !important;
}}

footer {{
    display: none !important;
}}
.glossary-item {{
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    line-height: 1.6;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}

.glossary-item strong {{
    color: #93c5fd;
}}

.glossary-item:hover {{
    transform: translateY(-2px);
    border-color: #60a5fa;
    box-shadow: 0 8px 22px rgba(96, 165, 250, 0.12);
}}

"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<div id='app_bg'></div>")

    gr.Markdown("""
    # AstroGPT
    ### Daily astronomy arXiv
    """)    
    state_paper_id = gr.State("")

    with gr.Row():
        btn_random = gr.Button("Discover another paper")

    with gr.Column(elem_id="paper_card"):
        title_box = gr.Markdown()
        out_box = gr.Markdown()
        glossary_box = gr.HTML(label="Glossary")

        with gr.Accordion("Abstract", open=False):
            abstract_box = gr.Markdown()

        with gr.Accordion("Generation Trace", open=False):
            plan_box = gr.Markdown()
            draft_box = gr.Markdown()
            critic_box = gr.Markdown()

    btn_random.click(
        fn=load_random,
        inputs=[],
        outputs=[state_paper_id, title_box, abstract_box, out_box, 
                glossary_box, plan_box, draft_box, critic_box,],
        )

if __name__ == "__main__":
    demo.launch(show_error=True)