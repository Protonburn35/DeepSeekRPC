"""Host-owned, all-stage state save/restore with bounded parallel streaming."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace
import uuid

from .coordinator import ClusterCoordinator
from .jsonutil import ValidationError
from .planner import Role
from .snapshot_archive import SnapshotStore, file_chunks
from .snapshot_state import state_inventory
from .snapshot_wire import receive_archive, send_archive, template
from .stage_service import COMMAND_KINDS


class SnapshotCoordinator:
    def __init__(self, coordinator, store):
        if (not isinstance(coordinator, ClusterCoordinator) or not isinstance(store, SnapshotStore) or
                coordinator.host.execution.stage.identity != store.identity):
            raise ValidationError("snapshot administration requires the actual cluster and matching host-owned store")
        self.coordinator, self.store = coordinator, store
        self.cancelled = threading.Event()
        self._observer = self.cancelled.is_set

    def cancel(self):
        self.cancelled.set()
        coordinator = self.coordinator
        with coordinator._cancel_lock:
            if coordinator._external_cancelled is not self._observer or not coordinator._request_lock.locked():
                return False
            coordinator._cancelled.set()
            # Administrative cancellation must also interrupt an ended/empty
            # session, whose ordinary request cancel is intentionally ignored.
            coordinator.host.actor.invalidate()
            coordinator.host.data.close()
            return True

    def _failed(self, future):
        if future.cancelled() or future.exception() is not None:
            self.cancel()

    @contextmanager
    def _ownership(self, identifier):
        coordinator = self.coordinator
        with coordinator._cancel_lock:
            if not coordinator._request_lock.acquire(blocking=False):
                raise RuntimeError("snapshot administration requires an idle cluster request owner")
            coordinator._cancelled.clear()
            coordinator._external_cancelled = self._observer
            coordinator._cancel_sent = False
        previous_timeout = coordinator.timeout
        coordinator.timeout = 3600
        ledger = coordinator.host.execution.ledger
        zero = "0" * 32
        coordinator._window = None
        coordinator._spec = SimpleNamespace(session=ledger.session or zero, request=ledger.request or zero,
            epoch=max(0, ledger.epoch), transaction=identifier, tokens=max(1, ledger.encoder_tokens))
        try:
            coordinator._check_cancel()
            yield
        except BaseException:
            self.cancel()
            raise
        finally:
            with coordinator._cancel_lock:
                coordinator._spec = coordinator._window = coordinator._external_cancelled = None
                coordinator.timeout = previous_timeout
                coordinator._request_lock.release()

    def _commands(self, commands):
        coordinator, host = self.coordinator, self.coordinator.host
        coordinator._check_cancel()
        futures = {role: host.controls[role].submit(coordinator._frame(role, body["action"]), body)
                   for role, body in commands.items() if role != Role.HOST}
        local = host.submit_local(commands[Role.HOST])
        for future in [*futures.values(), local]:
            future.add_done_callback(self._failed)
        result = {role: host.controls[role].wait(future, timeout=3600) for role, future in futures.items()}
        result[Role.HOST] = local.result(timeout=3600)
        coordinator._check_cancel()
        return result

    def save(self):
        coordinator, host = self.coordinator, self.coordinator.host
        if host.execution.ledger.phase != "decode" or not host.execution.ledger.quiescent:
            raise ValidationError("save requires an idle committed session; pause or finish generation first")
        identifier = uuid.uuid4().hex
        with self._ownership(identifier):
            coordinator.admit()
            offers = coordinator._all({"action": "snapshot_prepare", "id": identifier})
            reference = offers[Role.HOST]
            for role, offer in offers.items():
                if (set(offer) != {"id", "bytes", "header_sha256", "timeline", "settings_digest"} or
                        offer["id"] != identifier or offer["timeline"] != reference["timeline"] or
                        offer["settings_digest"] != reference["settings_digest"]):
                    raise ValidationError("snapshot stages disagree on their complete committed timeline/settings")
            expected = sum(offer["bytes"] for offer in offers.values())
            with self.store.prepare(timeline=reference["timeline"], expected_bytes=expected, identifier=identifier) as transaction:
                exports = {role: host.controls[role].submit(coordinator._frame(role, "snapshot_export"),
                    {"action": "snapshot_export", "id": identifier}) for role in (Role.MIDDLE, Role.TERMINAL)}
                local = host.export_local_snapshot(transaction.write_stage)
                for future in [*exports.values(), local]:
                    future.add_done_callback(self._failed)

                def receive(role):
                    connection, spec = host.data.connections[role], coordinator._spec
                    frame = template(connection, identifier, sender=role, session=spec.session, request=spec.request, epoch=spec.epoch)
                    chunks = receive_archive(connection, frame, total=offers[role]["bytes"], cancelled=self._observer)
                    return transaction.write_stage("ABC"[int(role)], chunks)

                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="deepseekrpc-state-read") as receivers:
                    reads = {role: receivers.submit(receive, role) for role in exports}
                    for future in reads.values():
                        future.add_done_callback(self._failed)
                    for role, future in reads.items():
                        stored = future.result(timeout=3600)
                        exported = host.controls[role].wait(exports[role], timeout=3600)
                        if (stored["bytes"], stored["sha256"], stored["header_sha256"]) != (
                                exported["bytes"], exported["sha256"], offers[role]["header_sha256"]):
                            raise ValidationError("host snapshot bytes differ from the actual stage export receipt")
                local.result(timeout=3600)
                coordinator._all({"action": "snapshot_seal", "id": identifier, "files": transaction.files})
                coordinator._check_cancel()
                manifest = transaction.commit()
                return {"id": identifier, "bytes": expected, "timeline": manifest["timeline"]}

    def _restore_pass(self, identifier, streams, manifest, *, apply=None):
        coordinator, host = self.coordinator, self.coordinator.host
        action = "restore_validate" if apply is None else "restore_apply"
        body = {"action": action, "id": identifier, **({} if apply is None else apply)}
        remote = {role: host.controls[role].submit(coordinator._frame(role, action), body)
                  for role in (Role.MIDDLE, Role.TERMINAL)}
        for stream in streams.values():
            stream.seek(0)
        local = host.restore_local(file_chunks(streams["A"], manifest["files"]["A"]["bytes"]), apply=apply)
        for future in [*remote.values(), local]:
            future.add_done_callback(self._failed)

        def send(role):
            connection, spec, stage = host.data.connections[role], coordinator._spec, "ABC"[int(role)]
            frame = template(connection, identifier, sender=Role.HOST, session=spec.session, request=spec.request, epoch=spec.epoch)
            record = manifest["files"][stage]
            result = send_archive(connection, frame, file_chunks(streams[stage], record["bytes"]), total=record["bytes"], cancelled=self._observer)
            if result["sha256"] != record["sha256"]:
                raise ValidationError("host snapshot file changed after complete validation")
            return result

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="deepseekrpc-state-write") as senders:
            writes = [senders.submit(send, role) for role in remote]
            for future in writes:
                future.add_done_callback(self._failed)
            for future in writes:
                future.result(timeout=3600)
        results = {role: host.controls[role].wait(future, timeout=3600) for role, future in remote.items()}
        results[Role.HOST] = local.result(timeout=3600)
        coordinator._check_cancel()
        return results

    def restore(self, identifier):
        coordinator, host = self.coordinator, self.coordinator.host
        # Hold deny-write/delete Windows handles from the complete checksum
        # pass through both remote passes and the all-stage readiness seal.
        with self.store.validated(identifier) as (manifest, headers, streams):
            for role in Role:
                metadata, records = headers["ABC"[int(role)]]
                expected, _, _ = state_inventory(host.execution.stage.plan, role, metadata,
                    identity=host.execution.stage.identity, run=host.data.run)
                if expected != records or metadata["timeline"] != manifest["timeline"]:
                    raise ValidationError("snapshot stage inventory/timeline differs before any live mutation")
            with self._ownership(identifier):
                peers = coordinator.admit()
                commands = {role: {"action": "restore_prepare", "id": identifier,
                    **{key: manifest["files"]["ABC"[int(role)]][key] for key in ("bytes", "sha256", "header_sha256")}} for role in Role}
                prepared = self._commands(commands)
                if any(result != {"prepared": True, "id": identifier} for result in prepared.values()):
                    raise ValidationError("not all stages froze the admitted restore inventory")
                validated = self._restore_pass(identifier, streams, manifest)
                for role, result in validated.items():
                    if result != {"validated": True, "id": identifier,
                            "sha256": manifest["files"]["ABC"[int(role)]]["sha256"], "timeline": manifest["timeline"]}:
                        raise ValidationError("not all stages validated their complete state before GPU mutation")
                apply = {"session": uuid.uuid4().hex, "request": uuid.uuid4().hex,
                         "epoch": max(manifest["timeline"]["epoch"] + 1, *(peer["minimum_next_epoch"] for peer in peers))}
                results = self._restore_pass(identifier, streams, manifest, apply=apply)
                sealed = coordinator._all({"action": "restore_seal", "id": identifier,
                                          "receipts": [results[role] for role in Role]})
                expected = {"restored": True, "id": identifier, "history": manifest["timeline"]["history"], "epoch": apply["epoch"]}
                if any(result != expected for result in sealed.values()):
                    raise ValidationError("not every actual stage sealed the restored committed state")
                host._session_stop_state = (apply["session"], headers["C"][0]["stops"])
                return {**expected, "session": apply["session"], "processed": manifest["timeline"]["processed"],
                        "pending_token": manifest["timeline"]["pending_token"], "weights_retained": True}
