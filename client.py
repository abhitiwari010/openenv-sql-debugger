from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import SqlAction, SqlObservation, SqlState

class SqlEnvClient(EnvClient[SqlAction, SqlObservation, SqlState]):
    def _step_payload(self, action: SqlAction) -> dict:
        return {"query": action.query}

    def _parse_result(self, payload: dict) -> StepResult[SqlObservation]:
        obs = SqlObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SqlState:
        return SqlState(**payload)
