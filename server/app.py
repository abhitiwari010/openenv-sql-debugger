from openenv.core.env_server import create_fastapi_app
from fastapi.responses import HTMLResponse
import os
import uvicorn

from models import SqlAction, SqlObservation
from server.environment import SqlEnvironment

app = create_fastapi_app(SqlEnvironment, SqlAction, SqlObservation)


@app.get("/")
def root():
    return HTMLResponse(
        """
        <html>
          <head><title>SQL OpenEnv Space</title></head>
          <body style="font-family: sans-serif; margin: 2rem;">
            <h2>SQL Data Analyst OpenEnv Environment</h2>
            <p>Environment is running.</p>
            <ul>
              <li>Health check: <a href="/health">/health</a></li>
              <li>OpenAPI docs: <a href="/docs">/docs</a></li>
            </ul>
            <p>Use the OpenEnv client in <code>inference.py</code> to run baseline episodes.</p>
          </body>
        </html>
        """
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
