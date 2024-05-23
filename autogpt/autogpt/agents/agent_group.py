from datetime import datetime
import logging
from typing import Optional
from autogpt.config.config import ConfigBuilder
from forge.sdk.model import TaskRequestBody
from autogpt.file_storage.base import FileStorage
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.agent_manager.agent_manager import AgentManager
from autogpt.agent_factory.profile_generator import AgentProfileGenerator
from autogpt.core.resource.model_providers.schema import ChatModelProvider
from autogpt.agents.agent_member import (
    AgentMember,
    AgentMemberSettings,
    AgentTask,
    AgentTaskSettings,
)

logger = logging.getLogger(__name__)


class AgentGroup:
    leader: AgentMember
    members: dict[str, AgentMember]

    def assign_group_to_members(self):
        self.leader.recursive_assign_group(self)

    def reload_members(self):
        members = self.leader.get_list_of_all_your_team_members()
        members_dict = {}
        for agent_member in members:
            members_dict[agent_member.state.agent_id] = agent_member
        self.members = members_dict

    def __init__(self, leader: AgentMember):
        self.leader = leader
        self.assign_group_to_members()
        self.reload_members()

    def print_state(self):
        logger.info("======== Status of group =======")
        logger.info("agents:")
        self.leader.print_state()

    async def create_task(self, task: TaskRequestBody):
        await self.leader.create_task(task)

    @staticmethod
    def configure_agent_group_with_state(
        state: AgentMemberSettings,
        file_storage: FileStorage,
        llm_provider: ChatModelProvider,
    ) -> "AgentGroup":
        leader: AgentMember = AgentMember(
            settings=state,
            llm_provider=llm_provider,
        )
        agents, tasks = AgentGroup.create_agents_and_tasks_dict_from_state(
            leader, file_storage, llm_provider
        )
        members = AgentGroup.get_agents_tree_from_state(agents, tasks, leader)
        if state.recruiter_id:
            leader.recruiter = agents[state.recruiter_id]
        leader.members = members
        return AgentGroup(leader=leader)

    @staticmethod
    def get_agents_tree_from_state(
        agents: dict[str, AgentMember], tasks: dict[str, AgentTask], agent: AgentMember
    ):
        members = []
        for member_id in agent.state.members:
            members_of_member = AgentGroup.get_agents_tree_from_state(
                agents, tasks, agents[member_id]
            )
            agents[member_id].members = members_of_member
            agent_tasks = []
            for task in agent.state.tasks:
                if tasks[task.task_id].parent_task_id:
                    tasks[task.task_id].parent_task = tasks[
                        tasks[task.task_id].parent_task_id
                    ]
                agent_tasks.append(tasks[task.task_id])
            agents[member_id].tasks = agent_tasks
            members.append(agents[member_id])
        return members

    @staticmethod
    def create_agents_and_tasks_dict_from_state(
        agent: "AgentMember", file_storage: FileStorage, llm_provider: ChatModelProvider
    ) -> tuple[dict[str, AgentMember], dict[str, AgentTask]]:
        agent_manager = AgentManager(file_storage)
        agents = {}
        agents[agent.state.agent_id] = agent
        sub_tasks = {}
        for member_id in agent.state.members:
            member_state = agent_manager.load_agent_member_state(member_id)
            agent_member = AgentMember(
                settings=member_state,
                llm_provider=llm_provider,
            )
            agents[agent_member.state.agent_id] = agent_member
            members, sub_tasks = AgentGroup.create_agents_and_tasks_dict_from_state(
                agent_member, file_storage, llm_provider
            )
            for agent_id, member in members.items():
                agents[agent_id] = member
            for sub_task in agent.state.tasks:
                sub_tasks[sub_task.task_id] = AgentTask(
                    task_id=sub_task.task_id,
                    input=sub_task.input,
                    parent_task_id=sub_task.parent_task_id,
                    status=sub_task.status,
                    created_at=datetime.now(),
                    modified_at=datetime.now(),
                    sub_tasks=[],
                )
        return agents, sub_tasks
