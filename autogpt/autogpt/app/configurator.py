"""Configurator module."""
from __future__ import annotations

import logging
from typing import Literal, Optional

import click
from colorama import Back, Style
from forge.config.config import GPT_3_MODEL, GPT_4_MODEL, Config
from forge.llm.providers import ModelName, MultiProvider

logger = logging.getLogger(__name__)


async def apply_overrides_to_config(
    config: Config,
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    skip_reprompt: bool = False,
    gpt3only: bool = False,
    gpt4only: bool = False,
    browser_name: Optional[str] = None,
    allow_downloads: bool = False,
    skip_news: bool = False,
) -> None:
    """Updates the config object with the given arguments.

    Args:
        config (Config): The config object to update.
        continuous (bool): Whether to run in continuous mode.
        continuous_limit (int): The number of times to run in continuous mode.
        skip_reprompt (bool): Whether to skip the re-prompting messages on start.
        speak (bool): Whether to enable speak mode.
        debug (bool): Whether to enable debug mode.
        log_level (int): The global log level for the application.
        log_format (str): The format for the log(s).
        log_file_format (str): Override the format for the log file.
        gpt3only (bool): Whether to enable GPT3.5 only mode.
        gpt4only (bool): Whether to enable GPT4 only mode.
        browser_name (str): The name of the browser to use for scraping the web.
        allow_downloads (bool): Whether to allow AutoGPT to download files natively.
        skips_news (bool): Whether to suppress the output of latest news on startup.
    """
    config.continuous_mode = False

    if continuous:
        logger.warning(
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        config.continuous_mode = True

        if continuous_limit:
            config.continuous_limit = continuous_limit

    # Check if continuous limit is used without continuous mode
    if continuous_limit and not continuous:
        raise click.UsageError("--continuous-limit can only be used with --continuous")

    # # Set the default LLM models
    # if gpt3only:
    #     # --gpt3only should always use gpt-3.5-turbo, despite user's FAST_LLM config
    #     config.fast_llm = GPT_3_MODEL
    #     config.smart_llm = GPT_3_MODEL
    # elif (
    #     gpt4only
    #     and check_model(
    #         GPT_4_MODEL,
    #         model_type="smart_llm",
    #         api_credentials=config.openai_credentials,
    #     )
    #     == GPT_4_MODEL
    # ):
    #     # --gpt4only should always use gpt-4, despite user's SMART_LLM config
    #     config.fast_llm = GPT_4_MODEL
    #     config.smart_llm = GPT_4_MODEL
    # else:
    #     config.fast_llm = check_model(
    #         config.fast_llm, "fast_llm", api_credentials=config.openai_credentials
    #     )
    #     config.smart_llm = check_model(
    #         config.smart_llm, "smart_llm", api_credentials=config.openai_credentials
    #     )

    if skip_reprompt:
        config.skip_reprompt = True

    if browser_name:
        config.selenium_web_browser = browser_name

    if allow_downloads:
        logger.warning(
            msg=f"{Back.LIGHTYELLOW_EX}"
            "AutoGPT will now be able to download and save files to your machine."
            f"{Back.RESET}"
            " It is recommended that you monitor any files it downloads carefully.",
        )
        logger.warning(
            msg=f"{Back.RED + Style.BRIGHT}"
            "NEVER OPEN FILES YOU AREN'T SURE OF!"
            f"{Style.RESET_ALL}",
        )
        config.allow_downloads = True

    if skip_news:
        config.skip_news = True


async def check_model(
    model_name: ModelName, model_type: Literal["smart_llm", "fast_llm"]
) -> ModelName:
    """Check if model is available for use. If not, return gpt-3.5-turbo."""
    multi_provider = MultiProvider()
    models = await multi_provider.get_available_models()

    if any(model_name == m.name for m in models):
        return model_name

    logger.warning(
        f"You don't have access to {model_name}. Setting {model_type} to {GPT_3_MODEL}."
    )
    return GPT_3_MODEL
