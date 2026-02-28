from dataclasses import dataclass
import json
from .prompts import Prompts
from src.llm.base import LLMClient

@dataclass
class AgentRun:
    mode: str
    plan: str
    draft: str
    glossary: str
    critic: str
    revised_draft: str
    
GOOGLE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Search the web to define hard words for a glossary.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The term to search for"},
            },
            "required": ["query"],
        }
    }
}

class AstroAgent:
    def __init__(self, llm_client: LLMClient, max_turns: int) -> None:
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.TOOL_MAPPING = {
            "google_search": self._google_search_tool,
        }

    @staticmethod
    def _google_search_tool(query: str) -> str:
        from src.tools.google_search import google_search
        return google_search(query)

    def _execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call based on the tool name and arguments.
        """
        if tool_name not in self.TOOL_MAPPING:
            return "Tool error: unsupported tool."
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "Tool error: missing 'query' argument."

        tool_func = self.TOOL_MAPPING.get(tool_name)
        if tool_func is None:
            return "Tool error: unsupported tool."

        try:
            return tool_func(query)
        except Exception as e:
            return f"Tool error: {e}"
        
    def glossary_agent(self, abstract: str, max_tool_turns: int) -> str:
        # Generate the glossary
        glossary_prompt = Prompts(mode="glossary").build_glossary_prompt(abstract)
        glossary_system_prompt = f"""You are an expert astronomer and science communicator. Your task is to identify any technical terms in the given abstract that might be difficult for a general audience to understand, and provide simple definitions for those terms."""

        messages = [{"role": "system", "content": glossary_system_prompt},
                    {"role": "user", "content": glossary_prompt}]
        
        for _ in range(max_tool_turns):
            glossary_result = self.llm_client.generate(
                messages=messages,
                stage = "glossary",
                temperature = 0.2,
                max_new_tokens = 512,
            )
            tool_calls = glossary_result.tool_calls or []

            if not tool_calls:
                return glossary_result.text.strip()
            
            messages.append(
                {"role": "assistant",
                "content": glossary_result.text or "",
                "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                function_block = call.get("function", {})
                tool_name = function_block.get("name", "")
                arguments_json = function_block.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_json or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                result = self._execute_tool_call(tool_name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": result,
                    }
                )

        return "Glossary generation incomplete after max tool turns."

    def run(self, abstract: str) -> AgentRun:
        """Run the whole series of agents with the given abstract.

        Args:
            abstract (str): The abstract to process.

        Returns:
            AgentRun: The result of running the agent.
        """
        #Defining the tools the agent can use
    
        # Generate the plan
        plan_prompt = Prompts(mode="plan").build_planner_prompt(abstract)
        system_plan_prompt = f"""You are an expert astronomer and science communicator. Your task is to create a clear, concise plan for how to rewrite the given abstract into a magazine-style explanation."""
        plan_result = self.llm_client.generate(
            prompt = plan_prompt,
            stage = "plan",
            system_prompt = system_plan_prompt,
            temperature = 0.2,
            max_new_tokens = 512,
        )
        plan = plan_result.text.strip()
        
        # Generate the draft explanation
        writer_prompt = Prompts(mode="write").build_writer_prompt(abstract, plan)
        system_writer_prompt = f"""You are an expert astronomer and science communicator. Your task is to write a clear, concise magazine-style explanation of the given abstract, following the provided plan."""
        writer_result = self.llm_client.generate(prompt = writer_prompt,
                                                stage = "write",
                                                system_prompt=system_writer_prompt,
                                                temperature=0.2,
                                                max_new_tokens=800,
                                                json_mode=False)
        draft = writer_result.text.strip()
        
        # Generate the critique
        critic_prompt = Prompts(mode="critic").build_critic_prompt(abstract, draft)
        system_critic_prompt = f"""You are a strict editorial reviewer evaluating a magazine-style explanation of an astronomy abstract. Your task is to critique the provided magazine-style explanation of the given abstract, identifying any factual inaccuracies, missing key points, or areas where the explanation could be clearer."""
        critic_result = self.llm_client.generate(prompt = critic_prompt,
                                                stage = "critic",
                                                system_prompt=system_critic_prompt,
                                                temperature=0.2,
                                                max_new_tokens=512,
                                                json_mode=False)
        critic = critic_result.text.strip()
        
        #Generate the glossary
        glossary = self.glossary_agent(abstract=abstract, max_tool_turns=self.max_turns)
        
        #Revise the draft based on the critique and glossary
        revision_prompt = Prompts(mode="revise").build_reviser_prompt(abstract=abstract, plan = plan, draft = draft, critique_json = critic, glossary=glossary)
        system_revision_prompt = f"""You are an expert astronomer and science communicator. Your task is to revise the magazine-style explanation draft based on the critique and glossary, ensuring that the final explanation is clear, accurate, and accessible to a general audience."""
        revision_result = self.llm_client.generate(prompt=revision_prompt,
                                                stage = "revise",
                                                system_prompt=system_revision_prompt,
                                                temperature=0.2,
                                                max_new_tokens=800,
                                                json_mode=False)
        
        revised_draft = revision_result.text.strip()
        return AgentRun(
            mode="final",
            plan=plan,
            draft=draft,
            glossary=glossary,
            critic=critic,
            revised_draft=revised_draft
        )
