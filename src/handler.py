import os
import sys
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Add token validation before engine initialization
def validate_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No Hugging Face token found in environment variables")
        return False

    print(f"HF Token present (first 5 chars): {token[:5]}...")
    return True

# Validate token before initializing engine
if not validate_token():
    print("ERROR: Failed to validate Hugging Face token. Please check your token and model access permissions.")
    sys.exit(1)

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_input = JobInput(job["input"])
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)