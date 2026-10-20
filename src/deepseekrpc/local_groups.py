"""The actual in-process rendezvous for a single owned stage rank."""
from datetime import timedelta

from .jsonutil import ValidationError


def initialize_local_process_group():
    """Initialize only the TP=PP=1 local Gloo group, without a filesystem path."""
    import torch.distributed as distributed

    if distributed.is_initialized():
        raise ValidationError("local stage groups require a fresh worker")
    if not distributed.is_gloo_available():
        raise RuntimeError("the Windows Torch build must provide the actual Gloo backend")
    # There is exactly one rank in this process. HashStore is the real Torch
    # rendezvous store; it needs neither a port nor FileStore's narrow Windows
    # filename conversion. Cross-PC traffic uses the separately owned TLS mesh.
    distributed.init_process_group(backend="gloo", store=distributed.HashStore(),
                                   rank=0, world_size=1, timeout=timedelta(seconds=30))
