"""Actual CUDA/DXGI budgets and physical Windows host admission counters."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .jsonutil import ValidationError, integer
from .native_windows import physical_host_memory


@dataclass(frozen=True)
class DeviceBudget:
    cuda_capacity: int
    local_budget: int
    local_usage: int
    reservation_available: int
    reservation_current: int
    nonlocal_budget: int
    nonlocal_usage: int
    dedicated_capacity: int
    shared_capacity: int
    cuda_free: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            integer(value, f"device memory {name}", 0, (1 << 63) - 1)
        if not self.cuda_capacity or not self.dedicated_capacity or self.cuda_free > self.cuda_capacity:
            raise ValidationError("invalid actual CUDA/WDDM device memory counters")

    @property
    def available(self):
        return min(self.cuda_free, max(0, self.local_budget - self.local_usage))


def query_device_budget(device=0) -> DeviceBudget:
    from .cuda_ops import extension
    import torch
    values = extension().windows_memory_budget(device)
    if len(values) != 9:
        raise ValidationError("Windows memory query ABI returned unexpected counters")
    free, _ = torch.cuda.mem_get_info(device)
    return DeviceBudget(*values, free)


class RuntimeBudgetGuard:
    """Pause work on budget pressure; this is not a claim of pinned VRAM.

    Retain the pre-allocation baseline; query after allocation/warmup and before
    and after model windows. Nonlocal usage is compared to that baseline plus
    known transfer staging. WDDM telemetry and successful eager allocations do
    not establish that Windows can never evict a GPU allocation.
    """
    def __init__(self, memory_plan, *, baseline: DeviceBudget, device=0):
        if not isinstance(baseline, DeviceBudget):
            raise ValidationError("runtime budget guard requires its actual pre-allocation device snapshot")
        self.device, self.plan = device, memory_plan
        self.baseline = baseline
        self.check()

    def check(self):
        device, host = query_device_budget(self.device), physical_host_memory()
        if device.local_usage + self.plan.gpu["device_budget_headroom"] > device.local_budget:
            raise ValidationError("WDDM local budget pressure: model execution is not admitted")
        allowed_nonlocal = (self.baseline.nonlocal_usage + self.plan.host["bounded_transfer_reservation"] +
                            self.plan.host["pinned_and_boot_weight_staging"])
        if device.nonlocal_usage > allowed_nonlocal:
            raise ValidationError("WDDM nonlocal usage exceeds the known transfer allowance; model execution is not admitted")
        if host["available_bytes"] < self.plan.host["physical_host_headroom"]:
            raise ValidationError("physical Windows host headroom is exhausted; model execution is not admitted")
        return {"device": asdict(device), "host": host,
                "scope": "actual budget pressure counters; not an absolute GPU residency guarantee"}
