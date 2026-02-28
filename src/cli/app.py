import argparse
from src.agent.astro_agent import AstroAgent
from src.llm.mock_client import MockLLMClient

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simple", "magazine"], default="simple")
    parser.add_argument("--abstract", type=str, default=None)
    parser.add_argument("--abstract-file", type=str, default=None)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    # TODO: load abstract from args
    # TODO: run agent
    # TODO: print plan/output/critic
    raise NotImplementedError
