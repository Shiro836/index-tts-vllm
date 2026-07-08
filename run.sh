export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CUDA 13.x nvcc cannot parse gcc-16 libstdc++ headers; force JIT compiles
# (flashinfer, BigVGAN kernel) to use gcc-15 as host compiler.
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-15'

# flashinfer's get_cuda_path() only checks $CUDA_HOME then /usr/local/cuda,
# but on Arch CUDA lives in /opt/cuda.
export CUDA_HOME=/opt/cuda
export PATH=/opt/cuda/bin:$PATH

# reduce caching-allocator fragmentation in the multi-model main process
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --kv_cache_memory_bytes: explicit KV budget (~20k tokens, ~11 max-len
#   sentences in flight) instead of a fraction of total VRAM, so boot no
#   longer depends on what the other GPU services have already allocated.
# --qwen_emo_mode lazy: the Qwen emotion engine (~3GB) is only built if an
#   emo_text request ever arrives (prod has never sent one).
# reference models (w2v-bert, campplus) stay on the GPU: +2.5GB VRAM, but a
#   new voice costs ~0.3s instead of ~4.5s of CPU inference that also blocks
#   the event loop. Pass --ref_device cpu to trade back if VRAM gets tight.
# --mel_workers 4: concurrent requests (universal bursts) overlap their
#   s2mel/vocoder tails instead of queueing behind one thread; activation
#   VRAM grows per busy worker — drop back to 1 if the card gets tight.
LD_PRELOAD=/home/forsen/repos/index-tts-vllm-fast/fake_dns.so \
    /home/forsen/miniconda3/envs/index-tts-vllm-fast/bin/python api_server_v2.py \
    --host 0.0.0.0 \
    --port 5113 \
    --gpu_memory_utilization 0.10 \
    --kv_cache_memory_bytes 2500000000 \
    --qwen_emo_mode lazy \
    --ref_cache_size 32 \
    --mel_workers 4
