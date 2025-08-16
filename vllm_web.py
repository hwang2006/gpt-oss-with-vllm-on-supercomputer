# vllm_web.py
import os
import time
import json
import html
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
import gradio as gr

# --- Behavior knobs ---
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
ENV_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip() or None
FILTER_THINK = True  # strip <think>...</think> from outputs

def _strip_think(text: str) -> str:
    if not text:
        return text
    if FILTER_THINK:
        # remove <think>...</think> (single-line or multi-line)
        import re
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S | re.I)
    return text

class VLLMChat:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    # --- API helpers ---
    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self):
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=15)
            r.raise_for_status()
            payload = r.json()
            return [m["id"] for m in payload.get("data", [])]
        except Exception as e:
            print(f"[list_models] error: {e}")
            return []

    def chat_stream(self, messages, model: str, temperature: float, max_tokens: int):
        """
        Streams content from /v1/chat/completions using SSE-like lines.
        """
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        try:
            with self.session.post(url, json=body, stream=True, timeout=600) as resp:
                if resp.status_code != 200:
                    yield f"\n\n[Error {resp.status_code}] {resp.text}"
                    return
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield html.unescape(delta)
                    except Exception as e:
                        # Non-JSON line or parse hiccup; ignore but log
                        print(f"[stream-parse] {e}: {line[:200]}")
        except requests.RequestException as e:
            yield f"\n\n[Request error] {e}"

# --- wait for server/models on startup ---
def wait_for_server(chat: VLLMChat, timeout_s=120, poll=2):
    start = time.time()
    last_models = []
    while time.time() - start < timeout_s:
        ok = chat.health()
        if ok:
            models = chat.list_models()
            if models:
                return True, models
            last_models = models
        time.sleep(poll)
    # fallback: return whatever we saw (likely empty)
    return False, last_models

def health_text(ok: bool) -> str:
    return "Health: ✅ **OK**" if ok else "Health: ❌ **DOWN**"

def create_interface(initial_base_url: str):
    chat = VLLMChat(initial_base_url)
    ok, models = wait_for_server(chat, timeout_s=180, poll=2)
    # choose default model safely
    if models:
        if ENV_DEFAULT_MODEL and ENV_DEFAULT_MODEL in models:
            default_model = ENV_DEFAULT_MODEL
        else:
            default_model = models[0]
    else:
        default_model = None

    # Optional: quick warmup to reduce first-token latency
    try:
        if default_model:
            list(chat.chat_stream(
                messages=[{"role": "user", "content": "ping"}],
                model=default_model,
                temperature=0.0,
                max_tokens=1,
            ))
    except Exception:
        pass

    with gr.Blocks(title="vLLM Chat Interface") as demo:
        gr.HTML("""
        <style>
        .gradio-container { max-width: 1500px !important; }
        #chat_col .gr-chatbot { max-width: 100% !important; }
        #side_col { min-width: 280px; max-width: 320px; }
        </style>
        """)

        gr.Markdown("# vLLM Chat Interface")

        with gr.Row():
            with gr.Column(scale=11, elem_id="chat_col"):
                chatbot = gr.Chatbot(height=520, type="messages")
                message = gr.Textbox(label="Message", placeholder="Type your message here...")
                send = gr.Button("Send", variant="primary")

            with gr.Column(scale=3, elem_id="side_col"):
                server_url = gr.Textbox(label="Server", value=chat.base_url)
                health_md = gr.Markdown(health_text(ok))
                model_dd = gr.Dropdown(
                    choices=models if models else ["No models available"],
                    value=default_model,
                    label="Select Model",
                    interactive=bool(models),
                )
                refresh_btn = gr.Button("🔄 Refresh Models")

                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(16, 8192, value=1536, step=1, label="Max Tokens")

        # ------------- actions -------------
        def on_send(msg, hist, server, model, temp, max_toks):
            # guardrails: rebind server base & ensure valid model
            chat.base_url = server.strip().rstrip("/")
            available = chat.list_models()
            if not available:
                yield "", hist + [{"role": "assistant", "content": "Server has no models yet."}]
                return
            if model not in available:
                model = available[0]

            if not msg.strip():
                yield "", hist
                return

            new_hist = hist + [{"role": "user", "content": msg}]
            acc = ""
            for chunk in chat.chat_stream(
                messages=new_hist,
                model=model,
                temperature=temp,
                max_tokens=int(max_toks),
            ):
                acc += chunk
                # optimistic filtering of <think>
                safe = _strip_think(acc)
                yield "", new_hist + [{"role": "assistant", "content": safe}]

        def on_refresh(current_model, server):
            try:
                chat.base_url = server.strip().rstrip("/")
                ok_now = chat.health()
                models_now = chat.list_models()
                if models_now:
                    new_value = current_model if current_model in models_now else (
                        ENV_DEFAULT_MODEL if ENV_DEFAULT_MODEL in models_now else models_now[0]
                    )
                    dd = gr.update(choices=models_now, value=new_value, interactive=True)
                else:
                    dd = gr.update(choices=["No models available"], value=None, interactive=False)
                return health_text(ok_now), dd
            except Exception as e:
                return f"Health: ❌ Error: {e}", gr.update()

        def on_clear():
            return [], ""

        send.click(
            on_send,
            inputs=[message, chatbot, server_url, model_dd, temperature, max_tokens],
            outputs=[message, chatbot],
        )
        refresh_btn.click(
            on_refresh,
            inputs=[model_dd, server_url],
            outputs=[health_md, model_dd],
        )
        message.submit(
            on_send,
            inputs=[message, chatbot, server_url, model_dd, temperature, max_tokens],
            outputs=[message, chatbot],
        )
        gr.Button("Clear Chat").click(on_clear, None, [chatbot, message], queue=False)

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    ui = create_interface(args.base_url)
    ui.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)

