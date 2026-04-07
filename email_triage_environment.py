# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageAction, EmailTriageObservation
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = EmailTriageEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Email Triage environment ready!"
        >>>
        >>> obs = env.step(EmailTriageAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the email_triage environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._emails = [
            {"text": "CONGRATULATIONS! You won a free iPhone. Click here to claim your prize.", "category": "spam", "ideal_action": "delete", "ideal_resolution": "none"},
            {"text": "Hey, when are we meeting for coffee tomorrow?", "category": "standard", "ideal_action": "reply", "ideal_resolution": "none"},
            {"text": "URGENT: Production server is down! Needs immediate attention.", "category": "urgent", "ideal_action": "forward", "ideal_resolution": "it"},
            {"text": "Hot singles in your area want to meet you tonight!", "category": "spam", "ideal_action": "delete", "ideal_resolution": "none"},
            {"text": "Weekly team sync notes are attached.", "category": "standard", "ideal_action": "categorize", "ideal_resolution": "none"},
            {"text": "Your account has been compromised. Send Bitcoin to unlock.", "category": "spam", "ideal_action": "delete", "ideal_resolution": "none"},
            {"text": "Project deadline shifted to tomorrow morning! Please Review ASAP.", "category": "urgent", "ideal_action": "reply", "ideal_resolution": "none"},
            {"text": "Are we still on for lunch?", "category": "standard", "ideal_action": "reply", "ideal_resolution": "none"},
            {"text": "Get rich quick! Buy this new crypto coin now before it moons.", "category": "spam", "ideal_action": "delete", "ideal_resolution": "none"},
            {"text": "CRITICAL FIREWALL ALERT - MULTIPLE INTRUSIONS DETECTED", "category": "urgent", "ideal_action": "forward", "ideal_resolution": "it"},
        ]
        self._current_email = {}

    def reset(self) -> EmailTriageObservation:
        """
        Reset the environment.

        Returns:
            EmailTriageObservation with the email to classify.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        self._current_email = random.choice(self._emails)

        return EmailTriageObservation(
            email_text=self._current_email.get("text", ""),
            feedback="Please classify this email and determine the appropriate action and resolution.",
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:  # type: ignore[override]
        """
        Execute a step in the environment by classifying the email.

        Args:
            action: EmailTriageAction containing the classification

        Returns:
            EmailTriageObservation with feedback and reward
        """
        self._state.step_count += 1

        # Reward Function logic: Multi-factor scoring
        ideal_category = self._current_email.get("category", "")
        ideal_action = self._current_email.get("ideal_action", "")
        ideal_resolution = self._current_email.get("ideal_resolution", "")

        reward = 0.0
        feedback_parts = []

        if action.category.lower() == ideal_category:
            reward += 1.0
            feedback_parts.append(f"Correct category ('{ideal_category}').")
        else:
            reward -= 0.5
            feedback_parts.append(f"Incorrect category (Expected '{ideal_category}', got '{action.category}').")

        if action.action_type.lower() == ideal_action:
            reward += 1.0
            feedback_parts.append(f"Correct action ('{ideal_action}').")
        else:
            reward -= 0.5
            feedback_parts.append(f"Incorrect action (Expected '{ideal_action}', got '{action.action_type}').")

        if action.resolution.lower() == ideal_resolution:
            reward += 1.0
            feedback_parts.append(f"Correct resolution ('{ideal_resolution}').")
        else:
            reward -= 0.5
            feedback_parts.append(f"Incorrect resolution (Expected '{ideal_resolution}', got '{action.resolution}').")

        feedback = " ".join(feedback_parts)

        return EmailTriageObservation(
            email_text=self._current_email.get("text", ""),
            feedback=feedback,
            done=True,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
