# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage Environment.

The email_triage environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EmailTriageAction(Action):
    """Action for the Email Triage environment - classifying an email."""

    category: str = Field(..., description="The classification of the email: 'spam', 'urgent', or 'standard'")
    action_type: str = Field(default="categorize", description="The action to take: 'delete', 'reply', 'forward', or 'categorize'")
    resolution: str = Field(default="none", description="The specific team or next step: 'sales', 'support', 'it', or 'none'")


class EmailTriageObservation(Observation):
    """Observation from the Email Triage environment - the email content."""

    email_text: str = Field(default="", description="The content of the email to triage")
    feedback: str = Field(default="", description="Feedback on the classification")
