import argparse
from src.agent.astro_agent import AstroAgent
import os


def build_client(provider: str, llama_model_path: str | None):
    if provider == "mock":
        from src.llm.mock_client import MockLLMClient
        return MockLLMClient()

    if provider == "openai":
        from src.llm.providers.openai_client import OpenAIClient
        return OpenAIClient()

    if provider == "hf":
        from src.llm.providers.hf_client import HFLLMClient
        return HFLLMClient()

    if provider == "llama":
        from src.llm.providers.llama_cpp_client import LlamaCppClient
        model_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            raise ValueError("For --provider llama, pass --llama-model-path or set LLAMA_MODEL_PATH.")
        return LlamaCppClient(model_path=model_path)

    raise ValueError(f"Unsupported provider: {provider}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "openai", "hf", "llama"], default="mock")
    parser.add_argument("--abstract", type=str, default=None)
    parser.add_argument("--abstract-file", type=str, default=None)
    parser.add_argument("--llama-model-path", type=str, default=None)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    abstract = args.abstract
    if abstract is None and args.abstract_file is not None:
        with open(args.abstract_file, "r", encoding="utf-8") as f:
            abstract = f.read().strip()
    client = build_client(provider=args.provider, llama_model_path=args.llama_model_path)
    agent = AstroAgent(llm_client=client, max_turns=3)
    run_result = agent.run(abstract=abstract or "")
    
    print(run_result.plan)
    print(run_result.draft)
    print(run_result.glossary)
    print(run_result.critic)
    print(run_result.revised_draft)

if __name__ == "__main__":
    main()
