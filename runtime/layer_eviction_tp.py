"""Rank-local demotion with ordered TP votes on a dedicated control group."""

from dataclasses import dataclass
from threading import Lock, local, get_ident
from contextlib import contextmanager

import torch.distributed as dist


class DemotionProtocolError(RuntimeError):
    """The worker must exit and its TP peers must be stopped by supervision."""


@dataclass(frozen=True)
class DemotionResult:
    committed: bool
    reclaimed_bytes_by_rank: tuple[int, ...]


class GlooDemotionTransport:
    """Group-relative rank order. Configure a finite group timeout at creation."""

    def __init__(self, group):
        if group is None or dist.get_backend(group) != "gloo":
            raise ValueError("Pass a dedicated Gloo process group")
        self._group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Worker is not a member of the control group")

    def all_gather(self, value):
        # The trusted TP workers exchange small control records over CPU Gloo.
        records = [None] * self.world_size
        dist.all_gather_object(records, value, group=self._group)
        return records


class LayerEvictionTP:
    """One coordinator per local KV worker.

    The scheduler must quiesce forwards, serialize worker access and invoke
    demote exactly once per epoch on every TP rank. Apply the decode gate before
    reusing this worker. Supervise TP as a failure unit, including process
    crashes or asymmetric transport errors following collective completion.
    """

    def __init__(self, kv, transport, *, config_digest):
        if type(config_digest) is not str or not config_digest:
            raise ValueError("Expected a shared nonempty configuration digest")
        self.kv = kv
        self._transport = transport
        self._config = (
            kv.layout.digest, config_digest, kv._max_context, kv._host_budget,
            kv._kv._logical_slots, kv._kv._head_dim, kv._kv._device.type,
        )
        self._last_epoch = -1
        self._failure = None
        self._lock = Lock()
        self._control_local = local()
        self._forward_ticket = None
        self._forward_thread = None
        self._forward_running = False

    @property
    def poisoned(self):
        return self._failure is not None or self.kv._kv._poisoned_at is not None

    def require_decode_ready(self, *, forward_ticket=None):
        if self.poisoned:
            raise DemotionProtocolError("TP eviction worker is poisoned")
        if self._lock.locked():
            if getattr(self._control_local, "depth", 0):
                return
            if (self._forward_running and forward_ticket is not None
                    and forward_ticket is self._forward_ticket
                    and get_ident() == self._forward_thread):
                return
            raise DemotionProtocolError("TP demotion is still in progress")

    @contextmanager
    def _control_access(self):
        """Internal short bridge call only; never wrap a user model callback."""
        depth = getattr(self._control_local, "depth", 0)
        self._control_local.depth = depth + 1
        try:
            yield
        finally:
            self._control_local.depth = depth

    def _gather(self, phase, value):
        records = self._transport.all_gather((phase, self._transport.rank, value))
        if len(records) != self._transport.world_size or any(
            type(record) is not tuple or len(record) != 3 or record[:2] != (phase, rank)
            for rank, record in enumerate(records)
        ):
            raise DemotionProtocolError(f"Invalid rank or phase in {phase} vote")
        return tuple(record[2] for record in records)

    def demote(self, epoch, lease, layer, hot_capacity, *, local_ready=True):
        if not self._lock.acquire(blocking=False):
            self._failure = "concurrent demotion"
            raise DemotionProtocolError(self._failure)
        try:
            return self._demote(epoch, lease, layer, hot_capacity, local_ready)
        except BaseException as exc:
            self._failure = str(exc)
            if isinstance(exc, DemotionProtocolError):
                raise
            raise DemotionProtocolError("TP demotion failed; stop the worker group") from exc
        finally:
            self._lock.release()

    def _lease_is_active(self, row, lease):
        try:
            return self.kv._kv._pages.active_lease(row) is lease
        except ValueError:
            return False

    def _demote(self, epoch, lease, layer, hot_capacity, local_ready):
        row = getattr(lease, "row", None)
        generation = getattr(lease, "generation", None)
        identity = (epoch, row, generation, layer, hot_capacity, self._config)
        valid = (
            not self.poisoned
            and local_ready is True
            and all(type(value) is int for value in identity[:5])
            and 0 < row < self.kv.layout.max_requests
            and epoch > self._last_epoch
            and self._lease_is_active(row, lease)
        )
        proposals = self._gather("propose", (valid, identity))
        if not valid or any(proposal != (True, identity) for proposal in proposals):
            raise DemotionProtocolError("TP identity, configuration or epoch disagreement")
        self._last_epoch = epoch

        ticket = None
        try:
            ticket = self.kv.prepare_demote(lease, layer, hot_capacity)
        except (ValueError, MemoryError):
            vote = ("unsafe" if self.poisoned or self.kv._pending else "reject", 0)
        except BaseException:
            vote = ("unsafe", 0)
        else:
            vote = ("unsafe", 0) if self.poisoned else ("ready", ticket.length)
        votes = self._gather("prepare", vote)
        if any(status not in ("ready", "reject") for status, _ in votes):
            raise DemotionProtocolError("Unsafe TP demotion preparation failure")
        lengths = {length for status, length in votes if status == "ready"}
        if len(lengths) > 1 or any(status == "reject" for status, _ in votes):
            cancelled = True
            try:
                if ticket is not None:
                    self.kv.cancel_demote(ticket)
            except BaseException:
                cancelled = False
            if not all(status is True for status in self._gather("cancel", cancelled)):
                raise DemotionProtocolError("TP demotion cancellation failed")
            if len(lengths) > 1:
                raise DemotionProtocolError("TP ranks have different request lengths")
            return DemotionResult(False, (0,) * self._transport.world_size)

        try:
            reclaimed = self.kv.commit_demote(ticket)
            if (self.poisoned or type(reclaimed) is not int or reclaimed <= 0
                    or reclaimed != ticket.reclaimable_bytes):
                reclaimed = None
        except BaseException:
            reclaimed = None
        counts = self._gather("commit", reclaimed)
        if any(type(count) is not int or count <= 0 for count in counts):
            raise DemotionProtocolError("TP demotion commit failed after publication may have started")
        if self.poisoned:
            raise DemotionProtocolError("TP worker was poisoned during demotion")
        return DemotionResult(True, counts)
