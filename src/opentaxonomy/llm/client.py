from typing import TypeVar, Type

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class TaxonomyLLM:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def call_structured(
        self,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        max_tokens: int = 8096,
    ) -> dict:
        """Call Claude with forced tool use to get structured JSON output."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise RuntimeError(f"LLM did not return expected tool '{tool_name}'")

    def complete(
        self,
        schema_class: Type[T],
        system: str,
        user: str,
        max_tokens: int = 8096,
    ) -> T:
        """Call Claude and parse the response into a Pydantic model via tool use."""
        schema = schema_class.model_json_schema()
        # Remove the Pydantic-generated title — cleaner for tool calling
        schema.pop("title", None)
        result = self.call_structured(
            system=system,
            user=user,
            tool_name="output",
            tool_description=f"Return a {schema_class.__name__}",
            input_schema=schema,
            max_tokens=max_tokens,
        )
        return schema_class.model_validate(result)
