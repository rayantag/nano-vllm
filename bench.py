import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
# open sesame


def main():
    seed(0)
    num_seqs = 256

    # Truncate input and output lengths.
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # Batch size: technically there is not a fixed batch size due to continuous batching, but here there are 256 sequences
    # Sequence length: random length between 100 and 1024 tokens
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # Warmup call -- not timed.
    # This is to get rid of all the inference overhead, including torch JIT compilation, CUDA kernel
    # compilation, and memory allocations.
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()

    # Actual benchmark - timed.
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens} tokens, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
