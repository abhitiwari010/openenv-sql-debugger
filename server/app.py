from pathlib import Path

from openenv.core.env_server import create_fastapi_app
from fastapi.responses import HTMLResponse
import os
import uvicorn

from models import SqlAction, SqlObservation
from server.environment import SqlEnvironment

app = create_fastapi_app(SqlEnvironment, SqlAction, SqlObservation)

_PLAYGROUND_HTML = Path(__file__).resolve().parent / "playground.html"


@app.get("/")
def root():
    html = _PLAYGROUND_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {"status": "healthy"}


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
