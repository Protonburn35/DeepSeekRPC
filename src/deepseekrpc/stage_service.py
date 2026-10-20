"""Persistent real stage service, with CUDA work confined to one bounded actor."""
import threading

from .actor import ActorBusy, ActorInvalidated, SerializedActor
from .bootstrap import LoadedStage
from .control_rpc import ControlRPC
from .execution import StageExecution
from .jsonutil import ValidationError
from .planner import Role
from .prompt_wire import PromptSpec, receive_prompt
from .protocol import Kind
from .publication import PublicationLedger
from .request_settings import RequestSettings
from .runtime_messages import fields, input_from, output_from, receipts_from, window_from, wire
from .stage_data import StageDataMesh
from .terminal_sampling import TerminalSampler


COMMAND_KINDS = {
    "status": Kind.CAPABILITIES, "admit": Kind.PLAN_ACK, "reset": Kind.RESET,
    "step": Kind.STEP_BEGIN, "prepare": Kind.COMMIT, "finish": Kind.COMMIT,
    "seal": Kind.COMMIT, "target": Kind.BEGIN, "sample_seed": Kind.VERIFY_RESULT,
    "sample_begin": Kind.DRAFT, "normal_decode": Kind.DRAFT, "verify": Kind.VERIFY_RESULT,
    "output_prepare": Kind.COMMIT, "output_commit": Kind.COMMIT, "output_finalize": Kind.COMMIT,
    "fault": Kind.ABORT,
    "prefix_checkpoint": Kind.PREFIX_ACCEPT,
    "snapshot_prepare": Kind.SNAPSHOT_PREPARE, "snapshot_export": Kind.SNAPSHOT_SEAL,
    "snapshot_seal": Kind.SNAPSHOT_SEAL, "restore_prepare": Kind.RESTORE_PREPARE,
    "restore_validate": Kind.RESTORE_PREPARE, "restore_apply": Kind.RESTORE_APPLY,
    "restore_seal": Kind.RESTORE_ACK,
}


class StageService:
    def __init__(self, stage: LoadedStage, *, data_connections, control_connections, image_rows=None,
                 control_rpcs=None):
        if not isinstance(stage, LoadedStage) or stage.closed:
            raise ValidationError("stage service requires its real loaded Windows CUDA graph")
        if image_rows is not None and (stage.role != Role.HOST or not callable(image_rows)):
            raise ValidationError("only stage A may resolve original image rows for graph windows")
        self.execution = StageExecution(stage)
        self.data = StageDataMesh(self.execution, data_connections)
        self.mesh_nonce = self._validate_channels(data_connections, control_connections)
        peers = set(Role) - {stage.role}
        if set(control_connections) != peers:
            raise ValidationError("control mesh must contain exactly the other two stages")
        for peer, connection in control_connections.items():
            if ((connection.local, connection.peer, connection.plane, connection.run) !=
                    (stage.role, peer, "control", self.data.run)):
                raise ValidationError("control channel differs from its authenticated stage/run/plane")
        if len({id(c) for c in control_connections.values()}) != 2:
            raise ValidationError("control mesh contains a duplicate channel")
        self.image_rows = image_rows
        self._ready, self._settings, self._sampler, self._sampling, self._outputs = False, None, None, None, None
        self._stops = None
        self._reset_context = None
        self._snapshot = self._restore = self._snapshot_exported = None
        self._session_stop_state = None
        self._status_lock = threading.Lock()
        self._status = {}
        self._refresh_status()
        self.actor = SerializedActor(self._dispatch, interrupt=self._interrupt, fault=self._fault)
        if control_rpcs is not None:
            if (set(control_rpcs) != peers or any(not isinstance(rpc, ControlRPC) or
                    rpc.pump.connection is not control_connections[peer] or not rpc.started
                    for peer, rpc in control_rpcs.items())):
                raise ValidationError("prestarted controls must own exactly the admitted mesh connections")
            self.controls = dict(control_rpcs)
        else:
            self.controls = {peer: ControlRPC(connection, self._handle, on_failure=lambda _: self.actor.invalidate())
                             for peer, connection in control_connections.items()}

    @staticmethod
    def _validate_channels(data, controls):
        channels = [*data.values(), *controls.values()]
        nonces = {c.mesh for c in channels}
        if len(nonces) != 1 or any(c.closed for c in channels):
            raise ValidationError("all stage planes must belong to one live mesh incarnation")
        return next(iter(nonces))

    def rebind(self, mesh):
        """Replace interrupted sockets after the actor has quiesced CUDA work."""
        data, controls = mesh.resources()
        connections = {peer: rpc.pump.connection for peer, rpc in controls.items()}
        nonce = self._validate_channels(data, connections)
        if nonce is None or mesh.role != self.role or mesh.run != self.data.run:
            raise ValidationError("replacement mesh differs from the loaded stage's role/run")

        def install():
            if not mesh.complete:
                raise ConnectionError("replacement mesh failed before actor quiescence")
            self.data = StageDataMesh(self.execution, data)
            self.controls, self.mesh_nonce = controls, nonce
            self._ready = False
            self._reset_context = None
            self._refresh_status()
            mesh.attach(self)
            return {"rebound": True, "weights_retained": True}

        # The caller retires old gates/controls first. Invalidation schedules
        # fault before install, discarding old queued model actions.
        self.actor.invalidate()
        return self.actor.call(install)

    def start(self):
        for rpc in self.controls.values():
            if not rpc.started:
                rpc.start()
        return self

    @property
    def role(self):
        return self.execution.stage.role

    def _refresh_status(self):
        ledger = self.execution.ledger
        status = {"stage": int(self.role), "run": self.data.run, "identity": self.execution.stage.identity,
                  "status": "READY" if self._ready else "LOADED", "phase": ledger.phase,
                  "session": ledger.session, "request": ledger.request, "epoch": ledger.epoch,
                  "processed": ledger.processed, "pending_token": ledger.pending_token,
                  "stop_reason": ledger.stop_reason,
                  "history_digest": ledger.history_digest,
                  "prefix_offer": self.execution.prefix.offer,
                  "next_operation": ledger._next,
                  "settings_digest": self._settings.digest if self._settings is not None else None,
                  "minimum_next_epoch": max(ledger.epoch + 1, self.execution.stage.attention.next_reset_epoch)}
        if self._reset_context is not None:
            status.update(session=self._reset_context.session, request=self._reset_context.request,
                          epoch=self._reset_context.epoch, phase="resetting")
        with self._status_lock:
            self._status = status

    def status(self):
        with self._status_lock:
            status = dict(self._status)
        controls = getattr(self, "controls", {})
        transport_ready = (len(controls) == 2 and not any(c.closed for c in self.data.connections.values()) and
                           not any(rpc.pump.connection.closed for rpc in controls.values()))
        status["transport_ready"] = transport_ready
        if not transport_ready and status["status"] == "READY":
            status["status"] = "RECOVERING"
        return status

    def cancel(self, *, session, request, epoch):
        status = self.status()
        if (session, request, epoch) != (status["session"], status["request"], status["epoch"]):
            return {"cancelled": False}
        if status["phase"] == "faulted" or status["stop_reason"] is not None:
            return {"cancelled": True}
        self.actor.invalidate()
        return {"cancelled": True}

    def _interrupt(self):
        self.execution.request_cancel()
        # Closing a separate data socket interrupts a blocked receive without
        # acquiring the CUDA mutex or waiting for its frame deadline. Fresh
        # authenticated channels are paired after actor quiescence; model and
        # physically resident weight allocations remain owned.
        self.data.close()

    def _handle(self, frame, body, reply, *, mesh=None):
        if mesh is not None and mesh != self.mesh_nonce:
            reply(error="NOT_READY")
            return
        # Only A schedules model work. Other peers may maintain liveness or
        # report cancellation; neither path waits for the CUDA actor mutex.
        if frame.kind == Kind.HEARTBEAT:
            fields(body, ("action",))
            if body["action"] != "heartbeat":
                raise ValidationError("invalid heartbeat command")
            reply({"alive": True, **self.status()})
            return
        if frame.kind == Kind.CANCEL:
            fields(body, ("action",))
            if body["action"] != "cancel" or frame.sender != Role.HOST:
                raise ValidationError("only the coordinator may cancel a request")
            reply(self.cancel(session=frame.session, request=frame.request, epoch=frame.epoch))
            return
        action = body.get("action")
        if not isinstance(action, str) or COMMAND_KINDS.get(action) != frame.kind:
            reply(error="BAD_REQUEST")
            return
        if action == "status":
            fields(body, ("action",))
            reply(self.status())
            return
        if frame.sender != Role.HOST or self.role == Role.HOST:
            reply(error="PROTOCOL")
            return
        try:
            future = self.actor.submit(frame, body, expected_mesh=mesh)
        except ActorBusy:
            reply(error="BUSY")
            return
        except ConnectionError:
            reply(error="NOT_READY")
            return

        def done(value):
            try:
                result = value.result()
            except (ActorInvalidated, InterruptedError):
                code = "CANCELLED"
            except ValidationError:
                code = "BAD_REQUEST"
            except BaseException:
                code = "SESSION_FAULT"
            else:
                code = None
            try:
                reply(result if code is None else None, error=code)
            except BaseException:
                if mesh is None or mesh == self.mesh_nonce:
                    self.actor.invalidate()
        future.add_done_callback(done)

    def submit_local(self, body, *, prompt=None):
        if self.role != Role.HOST:
            raise ValidationError("only the local stage A coordinator may submit in-process model work")
        return self.actor.submit(None, body, prompt=prompt)

    def _fault(self):
        if self._sampling is not None and not self._sampling.closed:
            self._sampling.abort()
        self._sampling = None
        for operation in (self._snapshot, self._restore):
            if operation is not None:
                operation.close()
        self._snapshot = self._restore = self._snapshot_exported = None
        self.execution.fault()
        self._refresh_status()

    def _frame_context(self, frame, *, spec=None, window=None):
        if frame is None:
            return
        ledger = self.execution.ledger
        context = spec or ledger
        if (frame.session, frame.request, frame.epoch) != (context.session, context.request, context.epoch):
            raise ValidationError("control command belongs to a different request/session epoch")
        if window is not None:
            op = window.operation
            if (frame.operation, frame.transaction, frame.position, frame.rows) != (op.number, op.transaction, op.start, op.rows):
                raise ValidationError("control command differs from its admitted exact model window")

    def _controls(self):
        history = self.execution.history
        if self.execution.ledger.stop_reason is not None:
            raise ValidationError("this request has already committed EOS, a stop or its output/context limit")
        remaining = min(self._settings.max_tokens - int(history.generated_counts().sum()),
                        history.capacity - history.length)
        if remaining < 1:
            raise ValidationError("the output/context limit has already ended this request")
        return {"policy": self._settings.policy, "counts": history.counts(),
                "generated_counts": history.generated_counts(), "remaining": remaining,
                "eos_ids": frozenset(self._settings.eos_ids), "logit_bias": dict(self._settings.logit_bias),
                "stop_identity": self._stops.identity}

    def _terminal(self):
        if self.role != Role.TERMINAL or self._sampler is None:
            raise ValidationError("only stage C owns this request's target/draft sampling state")

    def _compact_decision(self):
        self._terminal()
        prepared = self._sampling.prepared()
        decision = self._sampling.decision
        return {"digest": prepared["digest"], "accepted_target_rows": decision.accepted_target_rows,
                "output_ids": list(decision.output_ids), "pending_token": decision.pending_token,
                "stop_reason": decision.stop_reason}

    def _validate_sampling(self, plan):
        decision = self._compact_decision()
        return (plan.sampling_digest == decision["digest"] and plan.output_ids == tuple(decision["output_ids"]) and
                plan.pending_token == decision["pending_token"] and plan.stop_reason == decision["stop_reason"] and
                plan.initial_seed == (self._sampling.seed_token is None))

    def _finalize_sampling(self, plan):
        if not self._validate_sampling(plan):
            raise ValidationError("published decision differs from stage C's actual sampling transaction")
        preview = self._stops.preview(plan.output_ids)
        if preview.token_count != len(plan.output_ids) or (preview.matched is not None) != (plan.stop_reason == "stop"):
            raise ValidationError("published stop state differs from the exact accepted output token prefix")
        self._stops.commit(preview)
        self._sampling.commit(plan.sampling_digest)
        self._sampling = None

    def _dispatch(self, frame, body, *, prompt=None, expected_mesh=None):
        if expected_mesh is not None and expected_mesh != self.mesh_nonce:
            raise ActorInvalidated("queued control work belongs to a retired mesh incarnation")
        action = body.get("action")
        try:
            if action.startswith(("snapshot_", "restore_")):
                return self._administrative(frame, body)
            if self._snapshot is not None or self._restore is not None:
                raise ValidationError("model state is frozen by an administrative snapshot transaction")
            if action == "admit":
                fields(body, ("action", "peers"))
                if not self.status()["transport_ready"]:
                    raise ValidationError("stage mesh requires fresh authenticated channels before readiness")
                peers = body["peers"]
                if not isinstance(peers, list) or len(peers) != 3:
                    raise ValidationError("readiness requires all three actual loaded stage identities")
                seen = set()
                for peer in peers:
                    fields(peer, ("stage", "run", "identity"))
                    from .jsonutil import integer
                    role = Role(integer(peer["stage"], "admitted peer stage", 0, 2))
                    if role in seen or (peer["run"], peer["identity"]) != (self.data.run, self.execution.stage.identity):
                        raise ValidationError("peer readiness differs in model/runtime/precision/plan identity")
                    seen.add(role)
                self._ready = True
                return {"ready": True, "identity": self.execution.stage.identity}
            if not self._ready:
                raise ValidationError("stage is loaded but has not admitted the complete matching mesh")
            if action == "reset":
                fields(body, ("action", "prompt", "settings", "prefix"))
                spec, settings = PromptSpec.from_wire(body["prompt"]), RequestSettings.from_wire(body["settings"])
                self._frame_context(frame, spec=spec)
                if spec.identity != self.execution.stage.identity:
                    raise ValidationError("prompt model/runtime identity differs from the actual loaded stage")
                if self._sampling is not None and not self._sampling.closed:
                    self._sampling.abort()
                self._sampling = None
                self._reset_context = spec
                self._refresh_status()
                # A coherent new reset may transfer metadata after a previous
                # cancelled request. A concurrent cancel still invalidates this
                # actor generation, so reset cannot silently resume that request.
                self.execution.stage.cancellation.clear()
                if self.role != Role.HOST:
                    prompt = receive_prompt(self.data.connections[Role.HOST], spec, identity=spec.identity,
                                            cancelled=self.execution.stage.cancellation.is_set)
                if prompt is None or (prompt.identity, prompt.digest, prompt.length) != (spec.identity, spec.history_digest, spec.tokens):
                    raise ValidationError("stage A reset requires the exact validated local full prompt")
                tokens, mask, images = prompt.export_prompt()
                result = self.execution.reset(session=spec.session, request=spec.request, epoch=spec.epoch,
                    tokens=tokens, text_mask=mask, images=images, expected_history=spec.history_digest, prefix=body["prefix"])
                self._reset_context = None
                self._settings = settings
                self._sampler = TerminalSampler(settings.seed) if self.role == Role.TERMINAL else None
                if self.role == Role.TERMINAL:
                    from .detokenize import StopTracker, TokenBytes
                    self._stops = StopTracker(TokenBytes(self.execution.stage.tokenizer), settings.stop)
                self._outputs = PublicationLedger(self.execution.ledger, self.execution.history,
                    **({"validate_sampling": self._validate_sampling, "finalize_sampling": self._finalize_sampling}
                       if self.role == Role.TERMINAL else {}))
                return {**result, "settings_digest": settings.digest}
            self._frame_context(frame)
            if action == "prefix_checkpoint":
                fields(body, ("action",))
                return {"prefix": self.execution.capture_prefix()}
            if action == "step":
                fields(body, ("action", "window", "inputs"))
                window, inputs = window_from(body["window"]), input_from(body["inputs"])
                self._frame_context(frame, window=window)
                cached = self.execution.begin(window, inputs)
                if cached is not None:
                    return {"replayed_receipt": wire(cached)}
                incoming = self.data.receive_dependencies(window)
                image_rows = None
                if self.role == Role.HOST and not all(inputs.text_mask):
                    if self.image_rows is None:
                        raise ValidationError("original image rows are unavailable for the exact CED window")
                    image_rows = self.image_rows(self.execution.history, window)
                result = self.execution.forward(incoming=incoming, image_rows=image_rows)
                self.data.send_outputs(window, result)
                proposals = None
                if window.phase == "draft" and self.role == Role.TERMINAL:
                    if self._sampling is None:
                        raise ValidationError("draft graph has no admitted terminal sampling work")
                    proposals = list(self.execution.propose(self._sampling))
                return {"operation_digest": window.operation.digest, "proposal_ids": proposals}
            if action == "prepare":
                fields(body, ("action", "accepted", "result_digest", "pending_token", "aborted"))
                self._frame_context(frame, window=self.execution._window)
                digest = self.execution.prepare(**{k: v for k, v in body.items() if k != "action"})
                return {"commit_digest": digest}
            if action == "finish":
                fields(body, ("action", "commit_digest"))
                self._frame_context(frame, window=self.execution._window)
                return {"receipt": wire(self.execution.finish(commit_digest=body["commit_digest"]))}
            if action in {"seal", "target"}:
                fields(body, ("action", "receipts", "commit_digest"))
                receipts = receipts_from(body["receipts"])
                if action == "target":
                    self.execution.activate_target(receipts, commit_digest=body["commit_digest"])
                else:
                    self.execution.seal(receipts, commit_digest=body["commit_digest"])
                return {"sealed": True}
            if action == "sample_seed":
                fields(body, ("action",))
                self._terminal()
                self._sampling = self.execution.sample_seed(self._sampler, **self._controls())
                self._sampling.apply_stop(self._stops)
                return self._compact_decision()
            if action == "sample_begin":
                fields(body, ("action",))
                self._terminal()
                if not self.execution.ledger.quiescent or self.execution.ledger.phase != "decode":
                    raise ValidationError("sampling requires a globally committed idle target boundary")
                self._sampling = self._sampler.begin_verification(seed_token=self.execution.ledger.pending_token, **self._controls())
                return {"sampling": True}
            if action == "normal_decode":
                fields(body, ("action",))
                self._terminal()
                if self._sampling is None:
                    raise ValidationError("normal decode has no admitted sampling transaction")
                self._sampling.use_normal_decode()
                return {"normal_decode": True}
            if action == "verify":
                fields(body, ("action",))
                self._terminal()
                if self._sampling is None:
                    raise ValidationError("target verification has no admitted sampling transaction")
                self.execution.verify(self._sampling)
                self._sampling.apply_stop(self._stops)
                return self._compact_decision()
            if action in {"output_prepare", "output_commit", "output_finalize"}:
                fields(body, ("action", "plan") if action == "output_prepare" else ("action", "plan", "receipts"))
                plan = output_from(body["plan"])
                if self._outputs is None:
                    raise ValidationError("output publication has no admitted request history")
                if action == "output_prepare":
                    receipt = self._outputs.prepare(plan)
                else:
                    receipts = receipts_from(body["receipts"], output=True)
                    method = self._outputs.commit if action == "output_commit" else self._outputs.finalize
                    receipt = method(plan, receipts)
                return {"receipt": wire(receipt)}
            if action == "fault":
                fields(body, ("action",))
                self._fault()
                return {"faulted": True, "weights_retained": True}
            raise ValidationError("unknown stage service command")
        except BaseException:
            # Some model resources may already have changed. Recovery uses a
            # coherent newer epoch and retained weights, never guessed rollback.
            self._reset_context = None
            self._fault()
            raise
        finally:
            self._refresh_status()

    def _administrative(self, frame, body):
        from .snapshot_archive import MAX_ARCHIVE
        from .snapshot_restore import StageRestore
        from .snapshot_state import StageSnapshot
        from .snapshot_wire import receive_archive, send_archive, template
        from .jsonutil import integer
        from .protocol import hex_id
        action = body["action"]
        if action == "snapshot_prepare":
            fields(body, ("action", "id"))
            self._frame_context(frame)
            if self._snapshot is not None or self._restore is not None:
                raise ValidationError("another snapshot transaction already owns this stage")
            with self.execution._compute():
                self._snapshot = StageSnapshot(self, body["id"])
            return self._snapshot.offer()
        if action == "restore_prepare":
            fields(body, ("action", "id", "bytes", "sha256", "header_sha256"))
            if self._snapshot is not None or self._restore is not None:
                raise ValidationError("another snapshot transaction already owns this stage")
            if not self.execution.ledger.quiescent and self.execution.ledger.phase not in {"empty", "faulted"}:
                raise ValidationError("restore cannot interrupt an uncommitted live GPU operation")
            self._restore = StageRestore(self, body["id"], total=body["bytes"], sha256=body["sha256"], header_sha256=body["header_sha256"])
            self.execution.stage.cancellation.clear()
            self._ready = False
            return {"prepared": True, "id": body["id"]}
        operation = self._snapshot if action.startswith("snapshot_") else self._restore
        if operation is None or body.get("id") != operation.id:
            raise ValidationError("snapshot command differs from its admitted frozen transaction")
        if action == "snapshot_export":
            fields(body, ("action", "id"))
            if self.role == Role.HOST:
                raise ValidationError("A writes its own snapshot locally through its serialized model owner")
            self._frame_context(frame)
            connection = self.data.connections[Role.HOST]
            ledger = self.execution.ledger
            transfer = template(connection, operation.id, sender=self.role,
                session=ledger.session, request=ledger.request, epoch=ledger.epoch)
            with self.execution._compute():
                self._snapshot_exported = send_archive(connection, transfer, operation.chunks(), total=operation.total,
                                                       cancelled=self.execution.stage.cancellation.is_set)
            return self._snapshot_exported
        if action == "snapshot_seal":
            fields(body, ("action", "id", "files"))
            if not isinstance(body["files"], dict) or set(body["files"]) != set("ABC") or self._snapshot_exported is None:
                raise ValidationError("snapshot seal requires three complete state stream acknowledgments")
            for record in body["files"].values():
                fields(record, ("bytes", "sha256", "header_sha256"))
                integer(record["bytes"], "sealed snapshot bytes", 1, MAX_ARCHIVE)
                hex_id(record["sha256"], 64, "sealed snapshot checksum")
                hex_id(record["header_sha256"], 64, "sealed snapshot inventory checksum")
            record = body["files"]["ABC"[int(self.role)]]
            if (record["bytes"], record["sha256"], record["header_sha256"]) != (
                    self._snapshot_exported["bytes"], self._snapshot_exported["sha256"], operation.header_sha256):
                raise ValidationError("snapshot seal differs from this stage's actual exported stream")
            operation.close()
            self._snapshot = self._snapshot_exported = None
            return {"sealed": True, "id": body["id"]}
        if action in {"restore_validate", "restore_apply"}:
            fields(body, ("action", "id", "session", "request", "epoch") if action == "restore_apply" else ("action", "id"))
            if self.role == Role.HOST:
                raise ValidationError("A restores its own host-owned stream through its serialized model owner")
            connection = self.data.connections[Role.HOST]
            transfer = template(connection, operation.id, sender=Role.HOST, session=frame.session,
                                request=frame.request, epoch=frame.epoch)
            chunks = receive_archive(connection, transfer, total=operation.total, sha256=operation.sha256,
                                     cancelled=self.execution.stage.cancellation.is_set)
            if action == "restore_validate":
                return operation.validate(chunks)
            with self.execution._compute():
                result = operation.apply(chunks, session=body["session"], request=body["request"], epoch=body["epoch"])
                return {**result, "stage": int(self.role)}
        if action == "restore_seal":
            fields(body, ("action", "id", "receipts"))
            receipts = body["receipts"]
            if not operation.applied or not isinstance(receipts, list) or len(receipts) != 3:
                raise ValidationError("restore readiness requires all three actual successful apply receipts")
            roles, common = set(), None
            for receipt in receipts:
                fields(receipt, ("applied", "id", "session", "request", "epoch", "history", "processed", "pending_token", "settings_digest", "next_operation", "stage"))
                role = integer(receipt["stage"], "restored receipt stage", 0, 2)
                value = {key: val for key, val in receipt.items() if key != "stage"}
                if role in roles or common is not None and common != value:
                    raise ValidationError("restore receipts disagree on the committed session")
                common = value
                roles.add(role)
            ledger = self.execution.ledger
            if (common["applied"] is not True or common["id"] != operation.id or
                    (common["session"], common["request"], common["epoch"], common["history"], common["processed"],
                     common["pending_token"], common["settings_digest"], common["next_operation"]) !=
                    (ledger.session, ledger.request, ledger.epoch, ledger.history_digest, ledger.processed,
                     ledger.pending_token, self._settings.digest, ledger._next)):
                raise ValidationError("restore readiness differs from this stage's actual applied state")
            self._ready = True
            operation.close()
            self._restore = None
            return {"restored": True, "id": body["id"], "history": ledger.history_digest, "epoch": ledger.epoch}
        raise ValidationError("unknown snapshot administrative action")

    def export_local_snapshot(self, write_stage):
        if self.role != Role.HOST or not callable(write_stage):
            raise ValidationError("only A may write its own serialized snapshot to host-owned storage")
        def export():
            if self._snapshot is None:
                raise ValidationError("A snapshot state is not sealed")
            with self.execution._compute():
                self._snapshot_exported = write_stage("A", self._snapshot.chunks())
            return self._snapshot_exported
        return self.actor.call(export)

    def restore_local(self, chunks, *, apply=None):
        if self.role != Role.HOST:
            raise ValidationError("only A may consume a host-owned snapshot archive locally")
        def restore():
            if self._restore is None:
                raise ValidationError("A restore transaction has not been admitted")
            if apply is None:
                return self._restore.validate(chunks)
            with self.execution._compute():
                result = self._restore.apply(chunks, **apply)
            self._refresh_status()
            return {**result, "stage": int(self.role)}
        return self.actor.call(restore)

    def close(self):
        self.actor.invalidate()
        self.data.close()
        for rpc in self.controls.values():
            rpc.close()
        self.actor.close()
        # Only after the actor has definitely quiesced may the persistent model
        # owner release graph/weights/locked memory.
        self.execution.stage.close()
