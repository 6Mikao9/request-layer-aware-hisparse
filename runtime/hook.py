"""Explicit process-local hook. Importing this module never enables the overlay."""
import hashlib
import importlib
import json
from pathlib import Path
import weakref
from .policy import StaticPolicy

_state = None
_ever_bound = False
_bound = weakref.WeakSet()

def verify_sources(root):
    manifest = json.loads(Path(__file__).with_name('PROVENANCE.json').read_text())
    for rel, expected in {**manifest['source_sha256'], **manifest['runtime_dependency_sha256']}.items():
        path = Path(root) / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Pinned source mismatch: {rel}; expected {expected}, got {actual}')
    for name, expected in manifest['generated_sha256'].items():
        path = Path(__file__).with_name('generated') / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Generated overlay mismatch: {name}')

def install(policy: StaticPolicy, *, enabled=False, split_config=None):
    """Register explicitly in each fresh worker BEFORE coordinator construction/capture.

    At coordinator construction, rank agreement is checked using its TP group.
    Global world must equal TP; no DP/PP. Registration starts no process group,
    model, or GPU service.
    """
    global _state
    if not enabled:
        if _state is not None:
            raise RuntimeError('Already enabled; disabled install cannot switch a live policy')
        return False
    if not isinstance(policy, StaticPolicy):
        raise TypeError('policy must be StaticPolicy')
    from .split_copy import SplitCopyConfig, execution_identity
    if split_config is not None and not isinstance(split_config, SplitCopyConfig):
        raise TypeError('split_config must be an immutable SplitCopyConfig')
    identity = execution_identity(policy, split_config)
    if _state is not None:
        if _state['identity'] != identity:
            raise RuntimeError('Policy and execution mode are frozen for this worker')
        return True
    import torch
    import torch.distributed as dist
    import sglang
    root = Path(sglang.__file__).resolve().parent
    verify_sources(root)
    module = importlib.import_module('sglang.srt.managers.hisparse_coordinator')
    cls = module.HiSparseCoordinator
    from .generated.coordinator_method import swap_in_selected_blocks
    original_init, original_swap = cls.__init__, cls.swap_in_selected_blocks
    def init(self, *args, **kwargs):
        global _ever_bound
        original_init(self, *args, **kwargs)
        if self.is_m3_hisparse:
            from sglang.srt.runtime_context import get_parallel
            if (not dist.is_initialized() or self.tp_world_size != policy.expected_tp_size
                    or dist.get_world_size() != policy.expected_tp_size or get_parallel().pp_size != 1):
                raise RuntimeError('Requires TP4/TP8 with global world = TP and PP1; DP/PP unsupported')
            if 'A100' not in torch.cuda.get_device_name():
                raise RuntimeError('Prototype target is A100')
            hashes = [None] * self.tp_world_size
            dist.all_gather_object(hashes, identity, group=self.tp_group)
            if hashes != [identity] * self.tp_world_size:
                raise RuntimeError('TP rank policy or execution mode mismatch')
            if self.device_buffer_size != policy.hmax:
                raise ValueError('Coordinator physical Hmax differs from frozen policy')
            if not 0 <= self.mem_pool_device.start_layer < self.mem_pool_device.end_layer <= 60:
                raise ValueError('Unexpected absolute layer range')
            if (self.padded_buffer_size != policy.hmax + self.mem_pool_device.page_size
                    or self.req_device_buffer_tokens.shape != self.req_device_buffer_token_locs.shape
                    or self.req_device_buffer_tokens.size(2) != self.padded_buffer_size
                    or self.lru_slots.size(2) != policy.hmax):
                raise ValueError('Coordinator physical metadata padding/LRU shape changed')
            self._request_layer_hisparse_policy = policy
            if split_config is not None:
                from .split_copy import bind
                bind(self, policy, split_config)
            _bound.add(self)
            _ever_bound = True
    def swap(self, *args, **kwargs):
        if not self.is_m3_hisparse:
            return original_swap(self, *args, **kwargs)
        if self not in _bound:
            raise RuntimeError('Install must precede coordinator construction; live retrofit forbidden')
        if split_config is not None:
            from .split_copy import swap_in_selected_blocks as split_swap
            return split_swap(self, *args, **kwargs)
        return swap_in_selected_blocks(self, *args, **kwargs)
    cls.__init__, cls.swap_in_selected_blocks = init, swap
    _state = dict(policy=policy, identity=identity, cls=cls, init=original_init, swap=original_swap,
                  hooked_init=init, hooked_swap=swap)
    return True

def uninstall():
    """Restore methods only if this process has NEVER bound an MHA coordinator.

    A graph can retain tensors without retaining its coordinator. WeakSet
    emptiness therefore cannot establish safe teardown; use a new process.
    """
    global _state
    if _state is None:
        return False
    if _ever_bound:
        raise RuntimeError('An MHA coordinator has been bound in this process; start a new process to restore stock behavior')
    cls = _state['cls']
    if cls.__init__ is not _state['hooked_init'] or cls.swap_in_selected_blocks is not _state['hooked_swap']:
        raise RuntimeError('Another hook replaced these methods; refusing to overwrite it')
    cls.__init__, cls.swap_in_selected_blocks = _state['init'], _state['swap']
    _state = None
    return True
