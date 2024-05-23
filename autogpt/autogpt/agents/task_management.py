"""Commands to search the web with"""

from __future__ import annotations
from asyncio import create_task
from datetime import datetime
from typing import Iterator, Optional
import uuid
import traceback


from forge.models.json_schema import JSONSchema
from autogpt.agents.agent_member import *
from forge.agent.protocols import CommandProvider
from forge.command import Command, command

COMMAND_CATEGORY = "create_task"
COMMAND_CATEGORY_TITLE = "Create task"


class TaskManagementComponent(CommandProvider):
    """Component for manage tasks."""

    def __init__(self, agent: AgentMember) -> None:
        self.agent = agent

    def get_commands(self) -> Iterator[Command]:
        yield self.create_task

    @command(
        ["create_task"],
        "Create new task for yourself or one of your members. Show the assignee "
        "by agent_id of yourself or your members. the task description should be matched "
        "with the assignee description. If you can't find someone for this create or hire a new agent for this one.",
        {
            "task": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description for task that will be created",
                required=True,
            ),
            "agent_id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The id of agent will be the owner of this task",
                required=True,
            ),
            "parent_task_id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The task id that is cause of this task",
                required=False,
            ),
        },
    )
    async def create_task(
        self,
        task: str,
        agent_id: str,
        parent_task_id: Optional[str] = None,
    ):
        """Create new agent for some one

        Args:
            task (str): The description of that task
            agent_id (str): The id of the agent will be the owner of this task.
            parent_task_id (str): The task id that is the cause of this task.
        """
        try:
            owner_of_task = self.agent.group.members[agent_id]
            await owner_of_task.create_task(TaskRequestBody(input=task), parent_task_id=parent_task_id)
            return f"{task} created"
        except Exception as ex:
            traceback.print_exc()
            return f"can't create {task}"
