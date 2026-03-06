from __future__ import annotations
import os
from pathlib import Path
import time


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from miniastrolm.utils import (
    AgentConfig,
    get_daily_paths_and_date,
    run_fetch_and_build,
    get_random_papers,
    orchestration,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default

# --- CONFIG: set these once ---
CFG = AgentConfig(
    provider=os.getenv("ASTRO_PROVIDER", "openai"),
    max_turns=3,
    max_revision_attempts=1,
    threshold_hallucination=2,
    threshold_clarity=2,
    threshold_structure=2,
    llama_model_path=os.getenv("LLAMA_MODEL_PATH"),
    fast_mode=_env_flag("ASTRO_FAST_MODE", "0"),
    fast_max_new_tokens=_env_int("ASTRO_FAST_MAX_NEW_TOKENS", 700),
)

def explain(paper_id: str, abstract: str, debug: bool):
    if not str(abstract).strip():
        return "Provide or load an abstract first."
    try:
        run_result = orchestration(
            config=CFG,
            mode="single",
            abstract=abstract,
            debug=debug,
        )
        final_text = getattr(run_result, "revised_draft", None) or str(run_result)
        return final_text
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"



paper_id = "GRMHD Simulations of Accreting Neutron Stars with Non-Dipole Fields"
title = "https://arxiv.org/abs/2204.00249"
abstract = "NASA's NICER telescope has recently provided evidence for non-dipolar magnetic field structures in rotation-powered millisecond pulsars. These stars are assumed to have gone through a prolonged accretion spin-up phase, begging the question of what accretion flows onto stars with complex magnetic fields would look like. We present results from a suite of GRMHD simulations of accreting neutron stars for dipole, quadrupole, and quadrudipolar stellar field geometries. This is a first step towards simulating realistic hotspot shapes in a general relativistic framework to understand hotspot variability in accreting millisecond pulsars. We find that the location and size of the accretion columns resulting in hotspots changes significantly depending on initial stellar field strength and geometry. We also find that the strongest contributions to the stellar torque are from disk-connected fieldlines and the pulsar wind, leading to spin-down in almost all of the parameter regime explored here. We further analyze angular momentum transport in the accretion disk due to large scale magnetic stresses, turbulent stresses, wind- and compressible effects which we identify with convective motions. The disk collimates the initial open stellar flux forming jets. For dipoles, the disk-magnetosphere interaction can either enhance or reduce jet power compared to the isolated case. However for quadrupoles, the disk always leads to an enhanced net open flux making the jet power comparable to the dipolar case. We discuss our results in the context of observed neutron star jets and provide a viable mechanism to explain radio power both in the low- and high-magnetic field case."

start_time = time.time()
final_text = explain(paper_id, abstract, debug=True)
print(final_text)


end_time = time.time()
execution_time = end_time - start_time

print(f"Executed in: {execution_time:.4f} seconds")
