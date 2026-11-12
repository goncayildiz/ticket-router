import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from pipeline import Router

router = Router()

EXAMPLES = [
    "My dishwaser is not draining, model DW-6534",
    "hello",
    "thanks for your help",
    "i want to pay my bill",
    "book a flight to istanbul",
    "what is the status of flight AB123",
    "my washer is leaking water everywhere",
    "bulaşık makinem su boşaltmıyor",
    "asdf qwerty",
]


def _clean(s):
    return s.replace("`", "'")


def run(message):
    res = router.analyze(message)
    prep, d = res["prep"], res["decision"]

    head = f"## Routed to: **{d['team']}**\n\nDecision: {d['source']}"
    if d["reply"]:
        head += f"\n\nAutomatic reply: *{d['reply']}*"

    corr = ", ".join(f"`{a}` → `{b}`" for a, b in prep["corrections"]) or "none"
    code = f"`{prep['model_code']}`" if prep["model_code"] else "none"
    prep_md = (
        "### Preprocessing\n\n"
        f"- Raw message: `{_clean(prep['raw'])}`\n"
        f"- Model code extracted: {code}\n"
        f"- Spelling corrections: {corr}\n"
        f"- Cleaned text sent to the models: `{_clean(prep['clean'])}`"
    )

    if not res["rows"]:
        return head, prep_md, ""

    lines = [
        "### All four models side by side\n",
        "| Model | Top intent | Confidence | τ | Routed to | Time (ms) |",
        "|---|---|---|---|---|---|",
    ]
    last = len(res["rows"]) - 1
    for k, r in enumerate(res["rows"]):
        name = f"**{r['model']} ← decides**" if k == last else r["model"]
        lines.append(
            f"| {name} | `{r['intent']}` | {r['conf']:.2f} | {r['tau']:.2f} | {r['team']} | {r['ms']:.1f} |"
        )
    top3 = ", ".join(f"`{n}` {p:.2f}" for n, p in res["top3"])
    lines.append(f"\nTop-3 intents of the int8 student: {top3}")
    return head, prep_md, "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Support Ticket Router\n"
        "Type a customer message. It is normalized, classified by four models and routed to a team. "
        "Low-confidence or out-of-scope messages go to a human agent; small talk gets a rule-based reply."
    )
    msg = gr.Textbox(label="Customer message", lines=2,
                     placeholder="e.g. my dishwaser is not draining, model DW-6534")
    btn = gr.Button("Route message")
    decision = gr.Markdown()
    prep_out = gr.Markdown()
    table_out = gr.Markdown()
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=msg)
    btn.click(run, inputs=msg, outputs=[decision, prep_out, table_out])
    msg.submit(run, inputs=msg, outputs=[decision, prep_out, table_out])

if __name__ == "__main__":
    demo.launch()