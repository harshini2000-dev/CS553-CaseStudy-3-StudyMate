import os
import re

import gradio as gr
import torch
from huggingface_hub import InferenceClient
from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

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

# Load the local model when the application starts.
# The Hugging Face cache can be persisted with a Docker volume.
pipe = pipeline("text-generation", model=LOCAL_MODEL)


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
        # Qwen3 soft switch: skip the <think> block so the small model
        # spends its tokens on the answer instead of reasoning.
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
    """Generate a response using the local model on CPU or available CUDA."""
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
    generated_text = outputs[0]["generated_text"][-1]["content"]
    return strip_thinking(generated_text)


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
# CHAT HANDLER
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
    messages = build_messages(message, history, subject, use_local)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Thinking..."},
    ]
    yield history, ""

    def set_reply(text):
        history[-1] = {"role": "assistant", "content": text}
        return history, ""

    # ---------------- Local model ----------------
    if use_local:
        try:
            reply = local_generate(
                messages, response_length, temperature, top_p
            )
            yield set_reply(
                reply or "⚠️ The local model returned an empty response."
            )
        except Exception as e:
            yield set_reply(
                f"❌ Local model failed:\n\n`{type(e).__name__}: {e}`"
            )
        return

    # ---------------- Remote model ----------------
    token = resolve_token(hf_token)

    if not token:
        yield set_reply(
            "⚠️ No token found. Log in with the button above, "
            'or set the HF_TOKEN environment variable before running locally.'
        )
        return

    if not token.startswith("hf_"):
        yield set_reply(
            "⚠️ The token doesn't start with `hf_`, so the Hugging Face "
            "router may reject it. Check for stray quotes or whitespace "
            "in your HF_TOKEN value."
        )
        return

    try:
        got_text = False
        for partial in remote_stream(
            messages, response_length, temperature, top_p, token
        ):
            if partial:
                got_text = True
                yield set_reply(partial)

        if not got_text:
            yield set_reply(
                "Sorry, I couldn't generate a response. "
                "Please try asking the question again."
            )
    except Exception as e:
        yield set_reply(
            f"❌ Remote model failed:\n\n`{type(e).__name__}: {e}`"
        )


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

with gr.Blocks(title="StudyMate AI") as demo:

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 🎓 StudyMate AI
                ### Your AI-Powered Study Assistant
                Ask questions, understand difficult concepts, and learn with
                either a remotely hosted LLM or a model running in this application.
                """
            )
        with gr.Column(scale=1, min_width=160):
            gr.LoginButton()

    with gr.Row():

        # ---------------- Left: chat ----------------
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

        # ---------------- Right: settings ----------------
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Study Settings")

            subject = gr.Dropdown(
                choices=SUBJECTS,
                value="Machine Learning",
                label="📚 Subject",
            )

            # Local mode is selected by default for the first VM deployment.
            model_choice = gr.Radio(
                choices=[REMOTE_CHOICE, LOCAL_CHOICE],
                value=LOCAL_CHOICE,
                label="🤖 Model",
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
            info = gr.Markdown(model_info(LOCAL_CHOICE))

            gr.Markdown(
                """
                ---
                💡 **Tip:** Select a subject so StudyMate can tailor its
                explanations to your topic. The local model is smaller, so
                keep questions relatively short when using it.
                """
            )

    # ---------------- Events ----------------
    chat_inputs = [
        message, chatbot, subject, model_choice,
        response_length, temperature, top_p,
    ]

    send_button.click(
        chat_and_update,
        inputs=chat_inputs,
        outputs=[chatbot, message],
    )
    message.submit(
        chat_and_update,
        inputs=chat_inputs,
        outputs=[chatbot, message],
    )
    clear_button.click(
        clear_chat,
        inputs=None,
        outputs=[chatbot, message],
    )
    model_choice.change(model_info, inputs=model_choice, outputs=info)


PORT = int(os.getenv("PORT", "6550"))

if __name__ == "__main__":
    # Bind to all interfaces so Docker port publishing can reach Gradio.
    demo.launch(server_name="0.0.0.0", server_port=PORT)
