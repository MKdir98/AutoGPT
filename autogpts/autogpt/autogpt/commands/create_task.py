"""Commands to search the web with"""

from __future__ import annotations
from typing import Optional

from autogpt.agents.agent_member import AgentMember
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from forge.sdk.model import AgentTaskStatus

COMMAND_CATEGORY = "create_task"
COMMAND_CATEGORY_TITLE = "Create task"

@command(
    "create_task",
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
            description="The id of agent will be the ownwer of this task",
            required=True,
        ),
        "father_task_id": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The task id that is cause of this task",
            required=False,
        ),
    },
)
async def create_task(task: str, agent_id:str, father_task_id: Optional[str], agent: AgentMember):
    """Create new agent for some one

    Args:
        task (str): The description of that task
        agent_id (str): The id of the agent will be the owner of this task.
        father_task_id (str): The task id that is the cause of this task.
    """
    agentTask = agent.db.create_agent_task(input=task, father_task_id=father_task_id, status=AgentTaskStatus.INITIAL)
    owner_of_task = agent.group.members[agent_id]
    owner_of_task.tasks.append(agentTask)

