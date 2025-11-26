import os
import time

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, SecretStr
from typing import Iterator, List, Literal, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from openai import APIConnectionError

from dexter.prompts import DEFAULT_SYSTEM_PROMPT

DEFAULT_MODEL_TYPE = "standard"  # strong or standard
DEFAULT_MODEL_PROVIDER = "openai"  # openai, anthropic, gemini
MODEL_PROVIDER = Literal["openai", "anthropic", "gemini"]


def get_chat_model(
    model: MODEL_PROVIDER = DEFAULT_MODEL_PROVIDER,
    model_type: str = DEFAULT_MODEL_TYPE,
    temperature: float = 0,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Factory function to get the appropriate chat model based on the llm type.
    """
    api_key = os.getenv(f"LLM_API_{model.upper()}_KEY")
    if not api_key:
        raise ValueError(
            f"LLM_API_{model.upper()}_KEY not found in environment variables"
        )

    model_name = os.getenv(
        f"LLM_API_{model.upper()}_{model_type.upper() + '_' if model_type.upper() == 'STRONG' else ''}MODEL",
        "gpt-5-mini",
    )

    if model == "claude":
        # Anthropic models
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            api_key=SecretStr(api_key),
            streaming=streaming,
            timeout=300,
            stop=["\nHuman:", "\nAssistant:"],
        )

    elif model == "gemini":
        # Google Gemini models
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            streaming=streaming,
            convert_system_message_to_human=True,
        )

    else:
        # Default to OpenAI (gpt-* or others)
        # OpenAI client handles reading OPENAI_API_KEY from env automatically if not passed,
        # but we pass it explicitly to match existing pattern if set.
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=SecretStr(api_key),
            streaming=streaming,
        )


# Initialize the OpenAI client
# Make sure your OPENAI_API_KEY is set in your .env
def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    model_type: Literal["standard", "strong"] = "standard",
    model: MODEL_PROVIDER = "openai",
) -> AIMessage:  # type: ignore
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", final_system_prompt), ("user", "{prompt}")]
    )

    # Initialize the LLM.
    llm = get_chat_model(
        model=model,
        model_type=model_type,
        temperature=0,
        streaming=False,
    )

    # Add structured output or tools to the LLM.
    runnable = llm
    if output_schema:
        runnable = llm.with_structured_output(output_schema, method="function_calling")
        if output_schema.__name__ == "OptimizedToolArgs":
            # For tool argument optimization, we don't need the full message
            runnable = llm.with_structured_output(output_schema, method="json_mode")
    elif tools:
        runnable = llm.bind_tools(tools)

    chain = prompt_template | runnable

    # Retry logic for transient connection errors
    for attempt in range(3):
        try:
            return chain.invoke({"prompt": prompt})  # type: ignore
        except KeyboardInterrupt:
            # Don't retry on user interrupt, propagate immediately
            raise
        except APIConnectionError as e:
            if attempt == 2:  # Last attempt
                raise
            time.sleep(0.5 * (2**attempt))  # 0.5s, 1s backoff


def call_llm_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_type: str = "strong",
    model: MODEL_PROVIDER = "openai",
) -> Iterator[str]:
    """
    Stream LLM responses as text chunks.

    Note: Streaming does not support structured output or tools.
    Use this when you want to display text incrementally.
    """
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", final_system_prompt), ("user", "{prompt}")]
    )

    # Initialize the LLM with streaming enabled
    llm = get_chat_model(
        model=model,
        model_type=model_type,
        temperature=0,
        streaming=True,
    )

    chain = prompt_template | llm

    # Retry logic for transient connection errors
    for attempt in range(3):
        try:
            for chunk in chain.stream({"prompt": prompt}):
                # LangChain streams AIMessage chunks, extract content
                if hasattr(chunk, "content"):
                    content = chunk.content
                    if content:  # Only yield non-empty content
                        yield content  # type: ignore
            break
        except KeyboardInterrupt:
            # Don't retry on user interrupt, propagate immediately
            raise
        except APIConnectionError as e:
            if attempt == 2:  # Last attempt
                raise
            time.sleep(0.5 * (2**attempt))  # 0.5s, 1s backoff
