"""Collective admission of eager forwards, sharing the demotion control domain.

This deliberately synchronous CPU-Gloo correctness path is not a performance
claim. The service must provide finite data/control timeouts and fail-stop TP
supervision: a local exception cannot wake peers already stuck in model NCCL,
and asymmetric failures after the final collective need external supervision.
"""

import hashlib
import json
from threading import get_ident

from .layer_eviction_tp import DemotionProtocolError
from .layer_hybrid_pool import PageBudgetExceeded


class LayerForwardTP:
    def __init__(self, bridge, *, on_fatal=None):
        if bridge.tp is None:
            raise ValueError("Forward admission needs the worker's shared TP controller")
        if on_fatal is not None and not callable(on_fatal):
            raise TypeError("on_fatal must notify the worker-group supervisor")
        self.bridge = bridge
        self.tp = bridge.tp
        self.on_fatal = on_fatal
        self._ticket = None
        self._identity = None
        self._owns_lock = False

    def _fail(self, exc, *, release=True):
        self.tp._failure = "collective forward: " + str(exc)
        # A foreign thread must never clear another operation's ticket/lock.
        if release and (not self._owns_lock or self.tp._forward_thread == get_ident()):
            self._release()
        if self.on_fatal is not None:
            try:
                self.on_fatal(exc)
            except BaseException:
                # The local gate is already closed. Preserve the protocol
                # error even if the supervisor notification itself fails.
                pass
        raise DemotionProtocolError("Collective forward failed; stop the worker group") from exc

    def _release(self):
        self._ticket = self._identity = None
        if self._owns_lock:
            self.tp._forward_ticket = None
            self.tp._forward_thread = None
            self.tp._forward_running = False
            self._owns_lock = False
            self.tp._lock.release()

    def _snapshot(self, epoch, leases, chunks, mode):
        if type(epoch) is not int or epoch <= self.tp._last_epoch or self.tp.poisoned:
            raise ValueError("Invalid epoch or poisoned worker")
        with self.tp._control_access():
            self.bridge._idle()
        if mode not in ("prefill", "decode") or not leases or len(leases) != len(chunks):
            raise ValueError("Invalid forward geometry")
        requests = []
        for lease, chunk in zip(leases, chunks):
            self.bridge._lease(lease)
            rid = self.bridge._request_ids[lease.row]
            if type(rid) is not str or not rid:
                raise ValueError("Collective forward requires a stable scheduler request ID")
            layers = []
            for layer in range(self.bridge.adapter.layout.layer_count):
                state = self.bridge.adapter._state(lease, layer)
                layers.append((len(state.logical), state.logical_digest.hexdigest(),
                               None if state.hot is None else state.hot.capacity))
            requests.append((rid, lease.row, lease.generation, chunk, layers))
        # Physical addresses are rank local. Hash only stable logical identity,
        # completed history and residency geometry; never hash() or mutable refs.
        digest = hashlib.sha256(json.dumps(
            (self.tp._config, mode, requests), separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode()).hexdigest()
        return ("forward", epoch, digest)

    def begin(self, epoch, leases, logical_chunks, *, mode):
        """Return prepared ticket, or None after a collectively cancelled OOM.

        Every rank must call in the same scheduler order. Native logical slots
        remain caller owned: retain the prepared batch for retry, or perform a
        complete native rollback before releasing them. Do not repeat prepare.
        """
        if self._owns_lock or not self.tp._lock.acquire(blocking=False):
            return self._fail(RuntimeError("Overlapping TP operation"))
        self._owns_lock = True
        self.tp._forward_thread = get_ident()
        try:
            try:
                leases = tuple(leases)
                chunks = tuple(tuple(c) for c in logical_chunks)
                identity = self._snapshot(epoch, leases, chunks, mode)
            except Exception:
                identity = None
            proposals = self.tp._gather("forward_propose", identity)
            if identity is None or any(p != identity for p in proposals):
                raise ValueError("Forward epoch, request, history or configuration disagreement")
            self._identity = identity
            self.tp._last_epoch = epoch
            try:
                with self.tp._control_access():
                    ticket = self.bridge.begin_forward(leases, chunks, mode=mode)
            except (PageBudgetExceeded, MemoryError):
                # A healthy, entirely unsubmitted reservation failure is the
                # only recoverable rejection. All other cases require exit.
                status = "reject" if (not self.tp.poisoned and self.bridge._active is None) else "unsafe"
            except BaseException:
                status = "unsafe"
            else:
                self._ticket = ticket
                self.tp._forward_ticket = ticket
                status = "ready"
            votes = self.tp._gather("forward_prepare", (identity, status))
            if any(p != identity or s not in ("ready", "reject") for p, s in votes):
                raise ValueError("Unsafe forward preparation failure")
            if any(s == "reject" for _, s in votes):
                self._cancel_local_and_vote("forward_prepare_cancel")
                self._require_healthy()
                self._release()
                return None
            self._require_healthy()
            return self._ticket
        except BaseException as exc:
            return self._fail(exc)

    def collect_admission(self, epoch, leases, logical_chunks, *, mode, proposals):
        """Collect current allocator snapshots without reserving or running work.

        A local invalid native table/lease must still cast a failed vote so peers
        do not hang. This consumes a control epoch and releases the lock before
        returning. The serialized scheduler must revalidate freshness before
        demoting and use a later epoch for the actual collective forward.
        """
        from .layer_admission import capture_admission

        if self._owns_lock or not self.tp._lock.acquire(blocking=False):
            return self._fail(RuntimeError("Overlapping TP operation"))
        self._owns_lock = True
        self.tp._forward_thread = get_ident()
        try:
            try:
                leases = tuple(leases)
                chunks = tuple(tuple(c) for c in logical_chunks)
                identity = self._snapshot(epoch, leases, chunks, mode)
                with self.tp._control_access():
                    snapshot = capture_admission(self.bridge, self.tp._transport.rank,
                        leases, chunks, mode=mode, proposals=proposals)
            except BaseException:
                identity, snapshot = None, None
            votes = self.tp._gather("admission_inspection", (identity, snapshot))
            if identity is None or any(i != identity or s is None for i, s in votes):
                raise ValueError("A rank failed admission inspection or disagreed on its forward")
            snapshots = tuple(s for _, s in votes)
            fields = ("world_size", "configuration", "forward_identity",
                      "logical_identity", "proposal_identity", "groups")
            if any(s.rank != rank or s.world_size != len(snapshots)
                   for rank, s in enumerate(snapshots)) or any(
                getattr(s, f) != getattr(snapshot, f) for s in snapshots for f in fields
            ):
                raise ValueError("Admission membership, request state or proposal disagreement")
            self.tp._last_epoch = epoch
            self._require_healthy()
            self._release()
            return snapshots
        except BaseException as exc:
            return self._fail(exc)

    def collect_catalog(self, epoch, leases, logical_chunks, *, mode,
                        proposal_rounds, capture_id, workload_id, runtime_sha256):
        """Opt-in, ordered multi-rank diagnostic capture; no KV migration.

        All workers call once at the same idle/prepared scheduler boundary.
        Success consumes an epoch: old capacity snapshots/budget passes must
        refresh before reuse. The prepared native handle remains untouched.
        This deliberately synchronous diagnostic must be disabled in performance
        runs. Failure closes the group; it is not a recoverable missing sample.
        """
        handoff = getattr(self.bridge, "_native_handoff", None)
        owner_thread = getattr(handoff, "_thread", None)
        if type(owner_thread) is int and get_ident() != owner_thread:
            return self._fail(RuntimeError("Only the native scheduler owner may capture"), release=False)
        if self._owns_lock or not self.tp._lock.acquire(blocking=False):
            return self._fail(RuntimeError("Overlapping catalog control operation"), release=False)
        self._owns_lock = True
        self.tp._forward_thread = get_ident()
        try:
            identity, local_error = None, None
            try:
                # Import/tuple/metadata failures also vote, rather than strand peers.
                from analysis.runtime_admission_catalog import (
                    _capture_rank_catalog_locked, capture_digest, _name, _sha,
                )
                from .layer_admission import DemotionProposal
                from .layer_native_batch import NativeBatchHandoff
                if (not isinstance(handoff, NativeBatchHandoff) or handoff.controller is not self
                        or not callable(self.on_fatal)):
                    raise ValueError("Native handoff must own the diagnostic boundary")
                handoff._owner()
                leases = tuple(leases)
                chunks = tuple(tuple(c) for c in logical_chunks)
                forward = self._snapshot(epoch, leases, chunks, mode)
                if not _name(capture_id) or not _name(workload_id) or not _sha(runtime_sha256):
                    raise ValueError("Capture scope requires IDs and a pinned runtime SHA")
                if type(proposal_rounds) is not tuple or not proposal_rounds:
                    raise ValueError("Frozen nonempty proposal rounds required")
                common_rounds = []
                for proposals in proposal_rounds:
                    if type(proposals) is not tuple or any(type(p) is not DemotionProposal for p in proposals):
                        raise ValueError("Expected frozen typed proposals")
                    common_rounds.append(tuple(sorted((p.key.row, p.key.generation,
                        p.key.layer, p.hot_capacity, p.resident_since, p.decode_eligible)
                        for p in proposals)))
                # Costs differ across ranks. Only common selection/clock identity votes here.
                identity = ("catalog", epoch, capture_id, workload_id, runtime_sha256,
                            forward, tuple(common_rounds))
            except BaseException as exc:
                local_error = exc
            proposals = self.tp._gather("catalog_propose", identity)
            if identity is None or any(p != identity for p in proposals):
                raise ValueError("Catalog capture phase, forward or proposal disagreement") from local_error
            self.tp._last_epoch = epoch
            local, local_error = None, None
            try:
                self._require_healthy()
                with self.tp._control_access():
                    local = _capture_rank_catalog_locked(self.bridge, self.tp._transport.rank,
                        leases, chunks, mode=mode, proposal_rounds=proposal_rounds,
                        capture_id=capture_id, workload_id=workload_id, runtime_sha256=runtime_sha256)
            except BaseException as exc:
                local_error = exc
            captured = self.tp._gather("catalog_capture", (identity, local))
            digest, captures = None, None
            try:
                self._require_healthy()
                if local is None or any(i != identity or c is None for i,c in captured):
                    raise ValueError("A rank could not capture its runtime state") from local_error
                captures = tuple(c for _,c in captured)
                if any(c.rounds[0].rank != rank for rank,c in enumerate(captures)):
                    raise ValueError("Catalog payload rank order differs from transport membership")
                digest = capture_digest(captures)
            except BaseException as exc:
                local_error = exc
            # Even local validation/serialization errors participate in the final vote.
            digests = self.tp._gather("catalog_digest", (identity, digest))
            if digest is None or any(v != (identity, digest) for v in digests):
                raise ValueError("Catalog state validation or final digest agreement failed") from local_error
            self._require_healthy()
            self._release()
            return captures
        except BaseException as exc:
            return self._fail(exc)

    def _validate_collective(self, phase, ticket, *, running):
        # A foreign thread/controller must never inject another collective into
        # an already active control stream. Fail locally and notify supervision.
        if not self._owns_lock or get_ident() != self.tp._forward_thread:
            raise ValueError("Only the owning control thread may enter this collective")
        valid = (self._owns_lock and self._ticket is not None and ticket is self._ticket
                 and get_ident() == self.tp._forward_thread
                 and self.tp._forward_running is running and not self.tp.poisoned)
        votes = self.tp._gather(phase, (self._identity, valid))
        if not valid or any(p != self._identity or ok is not True for p, ok in votes):
            raise ValueError("Forward ticket or phase disagreement")

    def start(self, ticket):
        """Arm all ranks before the caller submits *any* model work."""
        try:
            self._validate_collective("forward_start", ticket, running=False)
            try:
                with self.tp._control_access():
                    self.bridge.begin_model(ticket)
            except BaseException:
                armed = False
            else:
                armed = True
            votes = self.tp._gather("forward_armed", (self._identity, armed))
            if any(p != self._identity or ok is not True for p, ok in votes):
                raise ValueError("A rank failed to arm model execution")
            self.tp._forward_running = True
            self._require_healthy()
        except BaseException as exc:
            return self._fail(exc)

    def complete(self, ticket):
        """Only after model.forward submitted all work, including attention."""
        try:
            self._validate_collective("forward_end", ticket, running=True)
            try:
                with self.tp._control_access():
                    self.bridge.end_forward(ticket)
            except BaseException:
                complete = False
            else:
                complete = True
            votes = self.tp._gather("forward_completed", (self._identity, complete))
            if any(p != self._identity or ok is not True for p, ok in votes):
                raise ValueError("A rank failed layer completion or its device fence")
            self._require_healthy()
            self._release()
        except BaseException as exc:
            return self._fail(exc)

    def _cancel_local_and_vote(self, phase):
        try:
            if self._ticket is not None:
                with self.tp._control_access():
                    self.bridge.abort_forward(self._ticket)
        except BaseException:
            cancelled = False
        else:
            cancelled = True
        votes = self.tp._gather(phase, (self._identity, cancelled))
        if any(p != self._identity or ok is not True for p, ok in votes):
            raise ValueError("Forward reservation cancellation failed")

    def cancel(self, ticket):
        try:
            self._validate_collective("forward_cancel", ticket, running=False)
            self._cancel_local_and_vote("forward_cancelled")
            self._require_healthy()
            self._release()
        except BaseException as exc:
            return self._fail(exc)

    def fail(self, ticket, exc):
        """Local model failure: notify supervision, never wait for a finish vote.

        A peer can be blocked in a model collective and unable to join Gloo.
        This method must be wired to the engine's worker-group error handling.
        """
        return self._fail(exc)

    def _require_healthy(self):
        if self.tp.poisoned:
            raise ValueError("Worker was poisoned during a collective operation")

    def run(self, ticket, model_forward):
        """Preferred integration: catches exceptions from native model work."""
        self.start(ticket)
        try:
            result = model_forward()
        except BaseException as exc:
            return self.fail(ticket, exc)
        self.complete(ticket)
        return result
