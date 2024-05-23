import logging
import pyautogui
from typing import Iterator, Optional

from forge.agent import BaseAgentSettings
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.agent.protocols import CommandProvider, DirectiveProvider

logger = logging.getLogger(__name__)


class GuiComponent(DirectiveProvider, CommandProvider):
    """
    Adds commands to work with gui of system.
    """

    def __init__(self, state: BaseAgentSettings, file_storage: FileStorage):
        self.state = state

        if not state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.files = file_storage.clone_with_subroot(f"agents/{state.agent_id}/")
        self.workspace = file_storage.clone_with_subroot(
            f"agents/{state.agent_id}/workspace"
        )
        self._file_storage = file_storage

    def get_resources(self) -> Iterator[str]:
        yield "The ability to intract with GUI"

    def get_commands(self) -> Iterator[Command]:
        yield self.click
        yield self.type

    @command(
        ["click"],
        "Left click in your system",
        {
            "x": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="x",
                required=True,
            ),
            "y": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="y",
                required=True,
            ),
        },
    )
    def click(self, x: float, y: float):
        """Read a file and return the contents

        Args:
            x (float): x
            y (float): y
        """
        pyautogui.moveTo(x, y)
        pyautogui.click()

    @command(
        ["type"],
        "type on the system",
        {
            "word": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The word should be type by keyboard",
                required=True,
            ),
        },
    )
    def type(self, word: str):
        """Type on the system

        Args:
            word (str): The word will be typed
        """
        pyautogui.write(word, interval=0.25)
