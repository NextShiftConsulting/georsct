"""Constants for R5 harness evolution."""

# S3 paths
BUCKET = "swarm-floodrsct-data"
RESULTS_PREFIX = "results/s035"
R5_PREFIX = "results/r5_harness_evolution"

# Scenarios
SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida",
]

# VLM models available for agent role
AGENT_VLMS = [
    "gpt4o", "gemini_flash", "gemini_pro", "jina", "nova", "qwen",
]

# Models available for evolver role
EVOLVER_MODELS = [
    "claude", "gpt4o", "gemini_pro", "qwen", "deepseek",
]

# Pilot defaults
PILOT_AGENT = "gemini_flash"
PILOT_EVOLVER = "claude"
PILOT_TRAIN_SCENARIO = "houston"
PILOT_TEST_SCENARIO = "nyc"
PILOT_STEPS = 3
PILOT_BATCH_SIZE = 50
