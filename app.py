"""Customer Support Ticket Router: Gradio app.

Run: python app.py   (then open http://127.0.0.1:7860)

The routing decision comes from the quantized DistilBERT student (INT8 ONNX).
The other three models of the ladder are shown side by side on the same message.
"""

from src.models import load_models  # imported first: sets HF_HOME

import gradio as gr

from src.evaluation import OOS
from src.preprocess import load_known_words
from src.router import HUMAN_AGENT, Router, load_intent_to_domain

LADDER = ("baseline", "ffn", "bert", "student_int8")
PRIMARY = "student_int8"
EXAMPLES = [
    "Bulaşık makinem su boşaltmıyor, model DW-6534",
    "my dishwaser is leaking water from the door",
    "where can I buy a new door seal for my WM4410",
    "the oven was installed yesterday but it won't turn on",
    "I lost my credit card",
    "Hello there",
    "Thanks for fixing my order",
    "who won the football match last night",
    "can you recommend a good novel to read",
]
COLUMNS = ["Model", "Intent", "Confidence", "Team", "Latency (ms)"]


def team_label(team):
    if team is None:
        return "small talk reply"
    if team == HUMAN_AGENT:
        return "Human agent"
    return team.replace("_", " ").capitalize()


def decision_markdown(result):
    lines = []
    if result.reply and result.team is None:
        lines += ["### Reply", f"> {result.reply}"]
    elif result.team == HUMAN_AGENT:
        top = next(p.top_intent for p in result.models if p.key == PRIMARY)
        reason = (f"Predicted out of scope (confidence {result.confidence:.2f})." if top == OOS else
                  f"Not confident enough: top intent `{top}` at {result.confidence:.2f} is below the threshold.")
        lines += [f"### Routed to: **{team_label(result.team)}**", reason]
    else:
        lines += [f"### Routed to: **{team_label(result.team)}**",
                  f"Intent `{result.intent}`, confidence {result.confidence:.2f}"]
    lines += ["", "**Preprocessing**", f"- Normalized text: `{result.normalized or '(empty)'}`",
              f"- Model codes: {', '.join(result.model_codes) or 'none'}",
              "- Spelling corrections: " + (", ".join(f"{a} → {b}" for a, b in result.corrections) or "none")]
    return "\n".join(lines)


def comparison_rows(result):
    rows = []
    for p in result.models:
        name = p.name + (" (decides)" if p.key == PRIMARY else "")
        intent = p.intent if p.intent == p.top_intent else f"oos (top: {p.top_intent})"
        rows.append([name, intent, round(p.confidence, 3), team_label(p.team), round(p.latency_ms, 2)])
    return rows


def build_app(router):
    def handle(message):
        result = router.route(message or "")
        return decision_markdown(result), comparison_rows(result)

    with gr.Blocks(title="Ticket Router") as demo:
        gr.Markdown("# Customer Support Ticket Router\n"
                    "Type a support message in English or Turkish. The quantized DistilBERT student picks the "
                    "team; low-confidence and out-of-scope messages go to a human agent, and small talk gets a "
                    "rule-based reply.")
        message = gr.Textbox(label="Support message", lines=1,placeholder="e.g. my dishwasher won't drain, model DW-6534")
        submit = gr.Button("Route message", variant="primary")
        gr.Examples(EXAMPLES, inputs=message)
        decision = gr.Markdown()
        table = gr.Dataframe(headers=COLUMNS, label="All four models on the same message", interactive=False, wrap=True)
        submit.click(handle, inputs=message, outputs=[decision, table])
        message.submit(handle, inputs=message, outputs=[decision, table])
    return demo


def main():
    print("loading models...", flush=True)
    router = Router(load_models(LADDER), load_intent_to_domain(), load_known_words(), primary=PRIMARY)
    for _ in range(3):  # warm-up, so the latencies shown for the first real message are representative
        router.route("my oven is broken")
    build_app(router).launch()


if __name__ == "__main__":
    main()
