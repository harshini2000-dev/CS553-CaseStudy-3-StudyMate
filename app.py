import os
import re
import time

import gradio as gr
import torch
from huggingface_hub import InferenceClient
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from transformers import pipeline

# ============================================================
# CONFIGURATION
# ============================================================
# MODE: "local" (local model only), "api" (remote API only),
#       "both" (single container offering both products)
MODE = os.getenv("MODE", "both").lower()
if MODE not in {"local", "api", "both"}:
    raise ValueError(f"MODE must be local, api, or both (got {MODE!r})")

LOCAL_MODEL = "Qwen/Qwen3-0.6B"
REMOTE_MODEL = "Qwen/Qwen3-32B"
REMOTE_CHOICE = "🌐 Remote API (Qwen3-32B)"
LOCAL_CHOICE = "💻 Local (Qwen3-0.6B)"

SUBJECTS = [
    "Machine Learning",
    "Data Science",
    "Deep Learning",
    "Statistics",
    "MLOps",
    "Computer Science",
    "General",
]

# ============================================================
# PROMETHEUS METRICS (served on port 8000 inside the container)
# ============================================================
REQUESTS = Counter(
    "studymate_requests_total", "Chat requests handled", ["mode", "status"]
)
LATENCY = Histogram(
    "studymate_request_latency_seconds",
    "End-to-end request latency",
    ["mode"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160),
)
TTFT = Histogram(
    "studymate_time_to_first_token_seconds",
    "Time until the first streamed token (remote API only)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
)
RESPONSE_CHARS = Histogram(
    "studymate_response_chars",
    "Length of generated responses in characters",
    ["mode"],
    buckets=(50, 200, 500, 1000, 2000, 4000, 8000),
)
INFLIGHT = Gauge(
    "studymate_inflight_requests", "Requests currently being processed", ["mode"]
)
ERRORS = Counter(
    "studymate_errors_total", "Failed requests", ["mode", "error_type"]
)
EMPTY = Counter(
    "studymate_empty_responses_total", "Empty model responses", ["mode"]
)
SUBJECT_REQUESTS = Counter(
    "studymate_subject_requests_total", "Requests per subject", ["subject"]
)
MODEL_LOAD = Gauge(
    "studymate_local_model_load_seconds", "Seconds taken to load the local model"
)

ACTIVE_MODES = {"local": ["local"], "api": ["api"], "both": ["local", "api"]}[MODE]
for _m in ACTIVE_MODES:  # make series appear at 0 before the first request
    INFLIGHT.labels(_m).set(0)
    EMPTY.labels(_m).inc(0)
    for _s in ("success", "error", "empty"):
        REQUESTS.labels(_m, _s).inc(0)

# Start the metrics server first so it is up while the model loads.
start_http_server(int(os.getenv("METRICS_PORT", "8000")))

# ============================================================
# MODEL LOADING
# ============================================================
pipe = None
if MODE in ("local", "both"):
    _t0 = time.time()
    pipe = pipeline("text-generation", model=LOCAL_MODEL)
    MODEL_LOAD.set(time.time() - _t0)


def clean_token(raw):
    """Strip whitespace and stray quote characters from a token."""
    if not raw:
        return None
    return raw.strip().strip('"').strip("'").strip()


_env_token = clean_token(os.environ.get("HF_TOKEN"))
if _env_token:
    print(
        f"HF_TOKEN found: length={len(_env_token)}, "
        f"starts_with_hf_={_env_token.startswith('hf_')}"
    )
else:
    print("HF_TOKEN not set - remote model will rely on the login button.")

# ============================================================
# PROMPT BUILDING
# ============================================================
def build_system_prompt(subject):
    return f"""You are StudyMate AI, a helpful academic study assistant.
The student's current subject is: {subject}

Your responsibilities:
- Explain concepts clearly and accurately.
- Use simple language when appropriate.
- Give examples when they help understanding.
- Help students learn rather than simply providing unexplained answers.
- Organize longer answers using headings or bullet points.
- Avoid unnecessary repetition.
- Do not mention that you are an AI unless it is relevant.
"""


def content_to_text(content):
    """Chatbot history content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def build_messages(message, history, subject, use_local):
    system_prompt = build_system_prompt(subject)
    if use_local:
        # Qwen3 soft switch: skip the <think> block for the small model.
        system_prompt += "\n/no_think"
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = content_to_text(msg.get("content"))
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": message})
    return messages


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ============================================================
# MODEL CALLS
# ============================================================
def local_generate(messages, max_tokens, temperature, top_p):
    """Generate a response with the local model (CPU, or CUDA if present)."""
    if torch.cuda.is_available():
        pipe.model.to("cuda")
        pipe.device = torch.device("cuda")
    outputs = pipe(
        messages,
        max_new_tokens=int(max_tokens),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return strip_thinking(outputs[0]["generated_text"][-1]["content"])


def remote_stream(messages, max_tokens, temperature, top_p, token):
    client = InferenceClient(model=REMOTE_MODEL, provider="auto", token=token)
    response = ""
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ):
        choices = chunk.choices
        if len(choices) and choices[0].delta.content:
            response += choices[0].delta.content
            yield strip_thinking(response)


def resolve_token(hf_token):
    token = clean_token(os.environ.get("HF_TOKEN"))
    if not token and hf_token is not None:
        token = clean_token(hf_token.token)
    return token


# ============================================================
# CHAT HANDLER (instrumented)
# ============================================================
def chat_and_update(
    message,
    history,
    subject,
    model_choice,
    response_length,
    temperature,
    top_p,
    hf_token: gr.OAuthToken | None,
):
    if not message or not message.strip():
        yield history, ""
        return

    history = history or []
    use_local = model_choice == LOCAL_CHOICE
    mode = "local" if use_local else "api"
    messages = build_messages(message, history, subject, use_local)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Thinking..."},
    ]
    yield history, ""

    def set_reply(text):
        history[-1] = {"role": "assistant", "content": text}
        return history, ""

    SUBJECT_REQUESTS.labels(subject).inc()
    INFLIGHT.labels(mode).inc()
    start = time.time()
    status = "success"
    reply_text = ""

    try:
        # ---------------- Local model ----------------
        if use_local:
            try:
                reply_text = local_generate(
                    messages, response_length, temperature, top_p
                )
                if not reply_text:
                    status = "empty"
                    EMPTY.labels(mode).inc()
                yield set_reply(
                    reply_text or "⚠️ The local model returned an empty response."
                )
            except Exception as e:
                status = "error"
                ERRORS.labels(mode, type(e).__name__).inc()
                yield set_reply(
                    f"❌ Local model failed:\n\n`{type(e).__name__}: {e}`"
                )
            return

        # ---------------- Remote model ----------------
        token = resolve_token(hf_token)
        if not token:
            status = "error"
            ERRORS.labels(mode, "MissingToken").inc()
            yield set_reply(
                "⚠️ No token found. Log in with the button above, "
                "or set the HF_TOKEN environment variable."
            )
            return
        if not token.startswith("hf_"):
            status = "error"
            ERRORS.labels(mode, "BadToken").inc()
            yield set_reply(
                "⚠️ The token doesn't start with `hf_`. Check for stray "
                "quotes or whitespace in your HF_TOKEN value."
            )
            return

        try:
            got_text = False
            for partial in remote_stream(
                messages, response_length, temperature, top_p, token
            ):
                if partial:
                    if not got_text:
                        TTFT.observe(time.time() - start)
                    got_text = True
                    reply_text = partial
                    yield set_reply(partial)
            if not got_text:
                status = "empty"
                EMPTY.labels(mode).inc()
                yield set_reply(
                    "Sorry, I couldn't generate a response. "
                    "Please try asking the question again."
                )
        except Exception as e:
            status = "error"
            ERRORS.labels(mode, type(e).__name__).inc()
            yield set_reply(f"❌ Remote model failed:\n\n`{type(e).__name__}: {e}`")
    finally:
        INFLIGHT.labels(mode).dec()
        LATENCY.labels(mode).observe(time.time() - start)
        RESPONSE_CHARS.labels(mode).observe(len(reply_text))
        REQUESTS.labels(mode, status).inc()


def clear_chat():
    return [], ""


def model_info(model_choice):
    if model_choice == LOCAL_CHOICE:
        return (
            "**Deployment:** Runs locally in this application container\n\n"
            f"**Model:** `{LOCAL_MODEL}`\n\n"
            "**Pros:** no external API call or API usage cost; can run offline "
            "after the model is cached\n\n"
            "**Cons:** smaller model; response speed depends on available CPU/GPU"
        )
    return (
        "**Deployment:** Remote API via Hugging Face Inference Providers\n\n"
        f"**Model:** `{REMOTE_MODEL}`\n\n"
        "**Pros:** much larger model, fast streaming when provider capacity is available\n\n"
        "**Cons:** requires a Hugging Face token and may be subject to "
        "rate limits and usage costs"
    )


# ============================================================
# GRADIO USER INTERFACE
# ============================================================
if MODE == "local":
    CHOICES, DEFAULT_CHOICE = [LOCAL_CHOICE], LOCAL_CHOICE
elif MODE == "api":
    CHOICES, DEFAULT_CHOICE = [REMOTE_CHOICE], REMOTE_CHOICE
else:
    CHOICES, DEFAULT_CHOICE = [REMOTE_CHOICE, LOCAL_CHOICE], LOCAL_CHOICE

with gr.Blocks(title="StudyMate AI") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown(
                """
# 🎓 StudyMate AI
### Your AI-Powered Study Assistant
Ask questions, understand difficult concepts, and learn with
a remotely hosted LLM or a model running in this application.
"""
            )
        with gr.Column(scale=1, min_width=160):
            if MODE != "local":
                gr.LoginButton()

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Conversation", height=500)
            message = gr.Textbox(
                label="Ask StudyMate",
                placeholder="Example: Explain overfitting in machine learning...",
                lines=3,
            )
            with gr.Row():
                send_button = gr.Button("🚀 Ask StudyMate", variant="primary")
                clear_button = gr.Button("🗑️ Clear Chat")

        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Study Settings")
            subject = gr.Dropdown(
                choices=SUBJECTS, value="Machine Learning", label="📚 Subject"
            )
            model_choice = gr.Radio(
                choices=CHOICES, value=DEFAULT_CHOICE, label="🤖 Model"
            )
            response_length = gr.Slider(
                minimum=256, maximum=2048, value=1024, step=256,
                label="📝 Response Length (max new tokens)",
            )
            with gr.Accordion("🔧 Advanced", open=False):
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top-p (nucleus sampling)",
                )
            gr.Markdown("## ℹ️ Model Information")
            info = gr.Markdown(model_info(DEFAULT_CHOICE))
            gr.Markdown(
                """
---
💡 **Tip:** Select a subject so StudyMate can tailor its
explanations. The local model is smaller, so keep questions short.
"""
            )

    chat_inputs = [
        message, chatbot, subject, model_choice,
        response_length, temperature, top_p,
    ]
    send_button.click(chat_and_update, inputs=chat_inputs, outputs=[chatbot, message])
    message.submit(chat_and_update, inputs=chat_inputs, outputs=[chatbot, message])
    clear_button.click(clear_chat, inputs=None, outputs=[chatbot, message])
    model_choice.change(model_info, inputs=model_choice, outputs=info)

PORT = int(os.getenv("PORT", "6550"))

if __name__ == "__main__":
    # Bind to all interfaces so Docker port publishing can reach Gradio.
    demo.launch(server_name="0.0.0.0", server_port=PORT)