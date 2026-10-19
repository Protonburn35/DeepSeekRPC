"""Construct the pinned vLLM configuration without starting its engine/loader."""
from pathlib import Path
import os

from .runtime_policy import configure_exact_exl3


def make_vllm_config(model_root: Path):
    # Offline operation also applies to Transformers config discovery. The
    # caller verifies this local directory's metadata before entering here.
    configure_exact_exl3()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    import torch
    import vllm_exl3
    vllm_exl3.register()
    from vllm.config import (
        CacheConfig, CompilationConfig, DeviceConfig, EngramConfig, KernelConfig,
        LoadConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig,
    )
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm_exl3.exl3 import Exl3Config

    model = ModelConfig(model=str(model_root.resolve(strict=True)),
                        skip_tokenizer_init=True, trust_remote_code=False,
                        dtype=torch.bfloat16, quantization="exl3", enforce_eager=True,
                        max_model_len=1048576, enable_sleep_mode=False,
                        enable_cumem_allocator=False, mm_encoder_attn_backend="TORCH_SDPA")
    # No stock loader, cache manager, whole Engram table, speculative worker,
    # scheduler or executor is instantiated. Actual decoder modules consume
    # these validated config objects through the explicit owned adapters.
    return VllmConfig(
        model_config=model,
        quant_config=Exl3Config.from_config(model.hf_config.quantization_config),
        cache_config=CacheConfig(cache_dtype="bfloat16", enable_prefix_caching=False,
                                 swa_bounded_replay=False),
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1,
                                       data_parallel_size=1, prefill_context_parallel_size=1,
                                       decode_context_parallel_size=1, enable_expert_parallel=False,
                                       enable_dbo=False, disable_custom_all_reduce=True,
                                       distributed_executor_backend="external_launcher"),
        scheduler_config=SchedulerConfig(max_model_len=1048576, is_encoder_decoder=False, max_num_seqs=1,
                                         max_num_batched_tokens=64, enable_chunked_prefill=True,
                                         is_multimodal_model=True),
        device_config=DeviceConfig(device="cuda"), load_config=LoadConfig(load_format="safetensors"),
        engram_config=EngramConfig(cpu_offload=False, embedding_across_dp=False,
                                  dp_shared_memory=False, use_thp=False),
        kernel_config=KernelConfig(enable_jit_warmup=False, enable_cutedsl_warmup=False),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                                             cudagraph_mode=CUDAGraphMode.NONE),
    )
