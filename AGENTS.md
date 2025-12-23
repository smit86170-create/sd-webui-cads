## Review guidelines
- This repo is an Automatic1111 WebUI extension script. Keep compatibility with common A1111 versions.
- Do not add heavy dependencies. Use only stdlib + gradio already used by WebUI.
- Keep changes minimal and well-scoped per PR.
- Avoid breaking infotext/paste functionality.
- For SDXL conditioning dicts, keep behavior for keys crossattn and vector.
- Add clear UI labels/tooltips and keep defaults sane.
- If you change behavior, update README or UI help text accordingly.
