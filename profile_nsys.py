import torch
from nanovllm import LLM, SamplingParams

# Nsight Systems shows a timeline of CPU and GPU activity, letting you see when kernels run, 
# where there are idle gaps, and how much time is spent 
# on memory transfers vs actual computation. It helps you identify bottlenecks like 
# "GPU is waiting for CPU" or "too much time in memory copies."

# I'm not very well-acquainted with the nvidia ecosystem yet but this is a useful tool.

# Initialize model.
llm = LLM('/workspace/huggingface/Qwen3-0.6B/', enforce_eager=True)

# Warmup (important - do this BEFORE profiling).
llm.generate(["warmup"], SamplingParams(max_tokens=10))
torch.cuda.synchronize()

# Mark the region we want to profile.
torch.cuda.nvtx.range_push("inference")

# Run inference.
prompt = "Explain what machine learning is in simple terms."
outputs = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=100))

torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()

print(f"Generated: {outputs[0]['text'][:100]}...")
