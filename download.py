"""Modal download entry for paddle.

Run:
  modal run download.py::download

Self-contained. This plugin installs required models/deps in the image build,
so no separate download step is needed.
"""

from __future__ import annotations

import modal

app = modal.App("paddle-download")


@app.local_entrypoint()
def download() -> None:
    print("No download step required for paddle.")
