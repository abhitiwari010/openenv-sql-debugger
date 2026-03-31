from openenv.core.env_server import create_fastapi_app

from models import SqlAction, SqlObservation
from server.environment import SqlEnvironment

app = create_fastapi_app(SqlEnvironment, SqlAction, SqlObservation)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}
