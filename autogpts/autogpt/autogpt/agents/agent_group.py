import sys
import logging
from typing import Optional
from autogpt.agent_factory.profile_generator import AgentProfileGenerator
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.agents.agent_member import AgentMember, AgentMemberSettings
from autogpt.config.config import ConfigBuilder
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from forge.sdk.model import ( Task, TaskRequestBody )

info = "-v" in sys.argv
debug = "-vv" in sys.argv
granular = "--granular" in sys.argv

logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO if info else logging.WARNING
)
logger = logging.getLogger(__name__)


class AgentGroup:
    
    leader: AgentMember
    members: dict[str, AgentMember]

    def assign_group_to_all_of_member(self):
        self.leader.assign_group(self)

    def create_list_of_members(self):
        members = self.leader.get_list_of_all_your_team_members()
        self.members = {}
        for agent_member in members:
            self.members[agent_member.id] = agent_member
        
    def __init__(
        self,
        leader: AgentMember
    ):
        self.leader = leader
        self.assign_group_to_all_of_member()
        self.create_list_of_members()

    async def create_task(self, task: TaskRequestBody):
        await self.leader.create_task(task)

async def create_agent_member(
    role: str,
    initial_prompt:str,
    boss: Optional['AgentMember'] = None,
    recruiter: Optional['AgentMember'] = None,
    create_agent: bool = False,
) -> Optional[AgentMember]:
    config = ConfigBuilder.build_config_from_env()
    config.logging.plain_console_output = True

    config.continuous_mode = False
    config.continuous_limit = 20
    config.noninteractive_mode = True
    config.memory_backend = "no_memory"
    settings = await generate_agent_profile_for_task(
            task=initial_prompt,
            app_config=config,
        )

    agentMember = AgentMember(
        role=role,
        initial_prompt=initial_prompt,
        settings=settings,
        boss=boss,
        recruiter=recruiter,
        create_agent=create_agent
    )

    if boss:
        boss.members.append(agentMember)

    return agentMember

async def generate_agent_profile_for_task(
    task: str,
    app_config
) -> AgentMemberSettings:
    openai_settings = OpenAIProvider.default_settings.copy(deep=True)
    openai_settings.credentials = app_config.openai_credentials
    llm_provider = OpenAIProvider(
        settings=openai_settings,
        logger=logger.getChild(f"OpenAIProvider"),
    )
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    prompt = agent_profile_generator.build_prompt(task)
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = app_config.openai_functions

    return AgentMemberSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=ai_directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
            plugins=app_config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )
