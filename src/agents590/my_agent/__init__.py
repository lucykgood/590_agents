"""The course starter agent, independent of Harbor."""

import asyncio
import json
from typing import cast

from openai.types.responses import ResponseFunctionToolCall, ResponseInputParam

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class StarterAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        # Use context.model to select the model and load credentials.
        settings = self.provider_settings(context)

        # Initialize the agent's memory (conversation history).
        history: ResponseInputParam = [{"role": "user", "content": instruction}]

        # Open an async client.
        async with settings.create_async_client() as client:
            # Allow at most 10 model requests to avoid infinite loops.
            for _ in range(10):
                # Send the conversation and describe the shell tool the model can request.
                response = await client.responses.create(
                    model=settings.model, # Use the selected model.
                    input=history,
                    tools=[
                        {
                            "type": "function",
                            "name": "shell",
                            "description": "Run a shell command in the task's working directory.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        }
                    ],
                    store=False,
                    background=False,
                )

                # Gather any tool requests from the model's response.
                tool_calls: list[ResponseFunctionToolCall] = []
                for item in response.output:
                    if item.type == "function_call":
                        tool_calls.append(item)

                # If none, return the model's final output.
                if not tool_calls:
                    return response.output_text

                # Save the model's response in memory. 
                history.extend(
                    cast(
                        ResponseInputParam,
                        [item.model_dump(exclude_none=True) for item in response.output],
                    )
                )

                # Execute each tool call.
                for tool_call in tool_calls:
                    arguments = json.loads(tool_call.arguments)
                    result = await self.run_shell(arguments["command"], context)

                    history.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": result,
                        }
                    )
            raise RuntimeError("Agent reached its turn limit.")

    async def run_shell(self, command: str, context: RunContext) -> str:
        """Run a shell command in the run's working directory and return its output."""

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=context.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        text = output.decode(errors="replace")
        return f"Exit code: {process.returncode}\n{text}"
