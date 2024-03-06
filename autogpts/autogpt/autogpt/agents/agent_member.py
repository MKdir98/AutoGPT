import os
import copy
import sys
from uuid import uuid4
import asyncio
import logging
from typing import Optional
from pydantic import Field
from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.core.resource.model_providers.schema import AssistantChatMessage
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import Action
from forge.sdk.db import AgentDB
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from forge.sdk.model import (TaskRequestBody, AgentTask, AgentTaskStatus )
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config.config import ConfigBuilder
from autogpt.agents.base import BaseAgentSettings
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.models.command_registry import CommandRegistry
from autogpt.agents.prompt_strategies.divide_and_conquer import Command, DivideAndConquerAgentPromptConfiguration, DivideAndConquerAgentPromptStrategy
from autogpt.core.prompting.schema import (
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
info = "-v" in sys.argv
debug = "-vv" in sys.argv
granular = "--granular" in sys.argv

logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO if info else logging.WARNING
)
logger = logging.getLogger(__name__)

class AgentMemberSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: DivideAndConquerAgentPromptConfiguration = Field(
        default_factory=(
            lambda: DivideAndConquerAgentPromptStrategy.default_configuration.copy(deep=True)
        )
    )

class AgentMember(Agent):
    
    id: str
    role: str
    initial_prompt: str
    boss: Optional['AgentMember']
    recruiter: Optional['AgentMember']
    tasks: list[AgentTask]
    members: list['AgentMember']
    create_agent: bool
    db: AgentDB
    group: 'AgentGroup'

    def assign_group(self, group: 'AgentGroup'):
        self.group = group
        for members in self.members:
            members.assign_group(group)

    def get_list_of_all_your_team_members(self) -> list['AgentMember']:
        members = []
        members.append(self)
        for member in self.members:
            members.extend(member.get_list_of_all_your_team_members())
        return members

    def __init__(
        self,
        role: str,
        initial_prompt: str,
        settings: AgentMemberSettings,
        boss: Optional['AgentMember'] = None,
        recruiter: Optional['AgentMember'] = None,
        create_agent: bool = False,
    ):
        config = ConfigBuilder.build_config_from_env()
        config.logging.plain_console_output = True

        config.continuous_mode = False
        config.continuous_limit = 20
        config.noninteractive_mode = True
        config.memory_backend = "no_memory"

        commands = copy.deepcopy(COMMAND_CATEGORIES)
        commands.remove("autogpt.commands.system")
        if create_agent:
            commands.append("autogpt.commands.create_agent")
        else:
            commands.append("autogpt.commands.hire_agent")
        commands.append("autogpt.commands.create_task")

        command_registry = CommandRegistry.with_command_modules(commands, config)

        hugging_chat_settings = OpenAIProvider.default_settings.copy(deep=True)
        hugging_chat_settings.credentials = config.openai_credentials

        llm_provider = OpenAIProvider(
            settings=hugging_chat_settings,
            logger=logger.getChild(f"Role-{role}-OpenAIProvider"),
        )

        super().__init__(settings, llm_provider, command_registry, config)

        self.id = str(uuid4())
        self.role = role
        self.initial_prompt = initial_prompt
        self.boss = boss
        self.recruiter = recruiter
        self.tasks = []
        self.members = []
        self.create_agent = create_agent
        database = AgentDB(
            database_string=os.getenv("AP_SERVER_DB_URL", "sqlite:///agetn_group.db"),
            debug_enabled=debug,
        )
        self.db = database
        self.prompt_strategy = DivideAndConquerAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )

    def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        tasks: list['AgentTask'],
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """
        if not extra_commands:
            extra_commands = []
        if not extra_messages:
            extra_messages = []

        # Apply additions from plugins
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(scratchpad)
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

        prompt = self.prompt_strategy.build_prompt(
            include_os_info=True,
            tasks=tasks,
            agent_member=self,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt


    async def process_tasks(self, tasks: list['AgentTask']):
        try:
            self._prompt_scratchpad = PromptScratchpad()
            logger.info(f"tasks: {str(tasks)}")
            prompt = self.build_prompt(scratchpad=self._prompt_scratchpad, tasks=tasks)
            result = await self.llm_provider.create_chat_completion(
                prompt.messages,
                model_name=self.config.smart_llm,
                functions=prompt.functions,
                completion_parser=lambda r: self.parse_and_process_response(
                    r,
                    prompt,
                    scratchpad=self._prompt_scratchpad,
                ),
            )
            commands:list[Command] = result.parsed_result
            # self.log_cycle_handler.log_cycle(
            #     self.ai_profile.ai_name,
            #     self.created_at,
            #     self.config.cycle_count,
            #     assistant_reply_dict,
            #     NEXT_ACTION_FILE_NAME,
            # )

            for command in commands:
                self.event_history.register_action(
                    Action(
                        name=command.command,
                        args=command.args,
                        reasoning="",
                    )
                )
                result = await self.execute(
                    command_name=command.command,
                    command_args=command.args,
                )
        except Exception as e:
            print(e)
            logger.error(f"Error occurred while creating task: {e}")

    async def create_task(self, task_request: TaskRequestBody):
        try:
            task = await self.db.create_agent_task(
                input=task_request.input,
                additional_input=task_request.additional_input,
                status="INITIAL"
            )
            self.tasks.append(task)
        except Exception as e:
            logger.error(f"Error occurred while creating task: {e}")
        
    async def run_tasks(self):
        while True:
            try:
                current_tasks = []
                for task in self.tasks:
                    if task.status == AgentTaskStatus.REJECTED:
                        task.status = AgentTaskStatus.INITIAL

                    elif task.status == AgentTaskStatus.DOING:
                        sub_tasks_done = all(sub_task.status == AgentTaskStatus.DONE for sub_task in task.sub_tasks)
                        sub_tasks_checking = any(sub_task.status == AgentTaskStatus.CHECKING for sub_task in task.sub_tasks)

                        if sub_tasks_done:
                            task.status = AgentTaskStatus.CHECKING
                        elif sub_tasks_checking:
                            current_tasks.append(task)

                    elif task.status == AgentTaskStatus.INITIAL:
                        current_tasks.append(task)
                        task.status = AgentTaskStatus.DOING

                if current_tasks:
                    await self.process_tasks(current_tasks)

                await asyncio.sleep(3)

            except Exception as e:
                logger.error(f"Error occurred while running tasks: {e}")


    def parse_and_process_response(
        self, llm_response: AssistantChatMessage, *args, **kwargs
    ) -> list[Command]:
        result = self.prompt_strategy.parse_response_content(llm_response)
        return result
