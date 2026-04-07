# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, State]
):
    """
    Client for the Email Triage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with EmailTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.email_text)
        ...
        ...     result = client.step(EmailTriageAction(category="spam"))
        ...     print(result.observation.feedback)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = EmailTriageEnv.from_docker_image("email_triage-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(EmailTriageAction(category="urgent"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """
        Convert EmailTriageAction to JSON payload for step message.

        Args:
            action: EmailTriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "category": action.category,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        """
        Parse server response into StepResult[EmailTriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with EmailTriageObservation
        """
        obs_data = payload.get("observation", {})
        observation = EmailTriageObservation(
            email_text=obs_data.get("email_text", ""),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
