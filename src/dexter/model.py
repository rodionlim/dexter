import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, SecretStr
from typing import Type, List, Literal, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from openai import APIConnectionError

from dexter.prompts import DEFAULT_SYSTEM_PROMPT


# Initialize the OpenAI client
# Make sure your OPENAI_API_KEY is set in your .env
def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    model_type: Literal["standard", "strong"] = "standard",
) -> AIMessage:
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", final_system_prompt), ("user", "{prompt}")]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Initialize the LLM.
    llm = ChatOpenAI(
        model=os.getenv(
            f"OPENAI_API_{'STRONG_' if model_type.upper() == 'STRONG' else ''}MODEL",
            "gpt-5-nano",
        ),
        temperature=0,
        api_key=SecretStr(api_key),
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
            return chain.invoke({"prompt": prompt})
        except APIConnectionError as e:
            if attempt == 2:  # Last attempt
                raise
            time.sleep(0.5 * (2**attempt))  # 0.5s, 1s backoff
