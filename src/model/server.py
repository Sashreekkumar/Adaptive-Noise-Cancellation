"""
server.py

Drag-and-drop web UI (Gradio). Model is loaded once at startup (see
model.get_model) and reused for every upload, so per-file latency
after the first request is just inference time, not model-load time.
"""

import tempfile
from pathlib import Path

from .cli import _kwargs_from_args
from .model import get_model
from .pipeline import clean_audio_file


def serve(args):
    import gradio as gr

    print("Loading DeepFilterNet3 model...")
    get_model(args.device)
    kwargs = _kwargs_from_args(args)

    def _run(file_path):
        if file_path is None:
            return None
        out_path = Path(tempfile.mkdtemp()) / (Path(file_path).stem + "_clean.mp3")
        clean_audio_file(file_path, out_path, **kwargs)
        return str(out_path)

    demo = gr.Interface(
        fn=_run,
        inputs=gr.Audio(sources=["upload"], type="filepath", label="Drop a noisy mp3/wav here"),
        outputs=gr.Audio(type="filepath", label="Cleaned mp3"),
        title="Audio Denoiser (DeepFilterNet3)",
        description="Drag and drop a noisy recording to get a cleaned mp3 back.",
    )
    demo.launch(server_port=args.port)
