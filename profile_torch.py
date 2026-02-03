import torch
from torch.profiler import profile, ProfilerActivity, schedule
from nanovllm import LLM, SamplingParams

# Torch profiler is more high-level than Nvidia Nsight. It shows PyTorch operations like linear, attention,
# and softmax, and maps them to underlying kernels. You can see stuff like which layers are slow.

# Initialize model
llm = LLM('/workspace/huggingface/Qwen3-0.6B/', enforce_eager=True)

# Warmup
llm.generate(["warmup"], SamplingParams(max_tokens=10))
torch.cuda.synchronize()

# Profile inference
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    outputs = llm.generate(
        ["Explain what machine learning is in simple terms."],
        SamplingParams(temperature=0.6, max_tokens=50)
    )

torch.cuda.synchronize()

# Print table sorted by CUDA time
print("\n=== Top operations by CUDA time ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Print table sorted by CPU time
print("\n=== Top operations by CPU time ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# Print memory usage
print("\n=== Top operations by memory ===")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# Export chrome trace (can view in chrome://tracing or perfetto)
prof.export_chrome_trace("torch_trace.json")
print("\nExported trace to torch_trace.json")
print("View it at: https://ui.perfetto.dev/ (drag and drop the file)")
