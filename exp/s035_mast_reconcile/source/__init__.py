"""
Pre-configured YRSN Applications.

This module provides ready-to-use YRSN apps with all tools pre-registered
and configured for common use cases.

Available Apps:
    - quality_app: Quality analysis and collapse detection
    - temperature_app: Temperature control and inference optimization
    - retrieval_app: Context retrieval with YRSN ranking
    - crewai_app: CrewAI-ready tools and agents
    - full_app: All YRSN capabilities
    - sudoku_app: Sudoku puzzle analysis demo
    - stress_mast_app: MAST failure taxonomy stress testing

Note: The 9 pre-built tools (rag_optimizer, prompt_clinic, context_trimmer,
search_reranker, chat_memory, doc_classifier, hallucination_detector,
multi_source_fusion, adaptive_router) have moved to yrsn-tools package.

Example:
    # Quick start - serve quality tools as REST API
    from yrsn.framework.apps import quality_app
    quality_app.serve(port=8000)

    # Get tools for CrewAI
    from yrsn.framework.apps import crewai_app
    tools = crewai_app.to_crewai()

    # Export MCP manifest
    from yrsn.framework.apps import full_app
    manifest = full_app.to_mcp()

    # Run MAST stress tests
    from yrsn.framework.apps import stress_mast_app
    report = stress_mast_app.run("run_mast_stress_suite")
"""

# Core apps
from yrsn.framework.apps.quality import quality_app
from yrsn.framework.apps.temperature import temperature_app
from yrsn.framework.apps.retrieval import retrieval_app
from yrsn.framework.apps.crewai_ready import crewai_app
from yrsn.framework.apps.full import full_app
from yrsn.framework.apps.sudoku import sudoku_app
from yrsn.framework.apps.stress_mast import stress_mast_app

__all__ = [
    "quality_app",
    "temperature_app",
    "retrieval_app",
    "crewai_app",
    "full_app",
    "sudoku_app",
    "stress_mast_app",
]
