"""Real archive/filesystem/TLS boundaries; no model execution is simulated."""
from concurrent.futures import ThreadPoolExecutor
import io
import os

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Role
from deepseekrpc.protocol import MAX_PAYLOAD
from deepseekrpc.snapshot_archive import (MAGIC, MAX_HEADER, PREFIX, Record, SnapshotStore,
                                         archive_chunks, encode_header, locked_read, read_header)
from deepseekrpc.snapshot_wire import TileReader, receive_archive, send_archive, template
from deepseekrpc.transport import accept, connect, listen


def archive():
    record = Record("synthetic-metadata", "u8", (MAX_PAYLOAD + 157,))
    header = encode_header({"scope": "archive fixture; no model"}, [record])
    def chunks():
        yield b"a" * MAX_PAYLOAD
        yield b"b" * 157
    return lambda: archive_chunks(header, [record], {record.name: chunks}), len(header) + record.nbytes


def test_archive_tiles_round_trip_without_whole_state_buffering():
    chunks, total = archive()
    reader = TileReader(chunks())
    metadata, records, header, length = read_header(reader, expected_bytes=total)
    assert length == total and records == (Record("synthetic-metadata", "u8", (MAX_PAYLOAD + 157,)),)
    assert reader.read(MAX_PAYLOAD) == b"a" * MAX_PAYLOAD
    assert reader.read(157) == b"b" * 157
    reader.finish()
    with pytest.raises(ValidationError, match="oversized"):
        read_header(io.BytesIO(PREFIX.pack(MAGIC, 1, MAX_HEADER + 1)))
    with pytest.raises(ValidationError, match="record"):
        Record("../../outside", "u8", (1,))


def test_snapshot_is_published_only_after_three_complete_streams_and_corruption_is_detected(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots", identity="a" * 64)
    chunks, total = archive()
    with store.prepare(timeline={"fixture": True}, expected_bytes=3 * total) as transaction:
        transaction.write_stage("A", chunks())
        assert store.list() == []
        with pytest.raises(ValidationError, match="all three"):
            transaction.commit()
        for stage in "BC":
            transaction.write_stage(stage, chunks())
        manifest = transaction.commit()
    validated, headers = store.validate(manifest["id"])
    assert validated == manifest and set(headers) == set("ABC")
    assert len(store.list()) == 1
    with pytest.raises(ValidationError, match="compatibility"):
        SnapshotStore(store.root, identity="b" * 64).validate(manifest["id"])
    path = store.root / manifest["id"] / "stage-B.bin"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValidationError, match="checksum"):
        store.validate(manifest["id"])
    with pytest.raises(ValidationError, match="opaque"):
        store.validate("../outside")
    assert store.remove(manifest["id"])["deleted"] is True
    assert store.list() == []


def test_failed_snapshot_cleans_only_its_unpublished_generated_files(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots", identity="a" * 64)
    chunks, total = archive()
    with pytest.raises(RuntimeError, match="interrupted"):
        with store.prepare(timeline={}, expected_bytes=3 * total) as transaction:
            transaction.write_stage("B", chunks())
            raise RuntimeError("interrupted save fixture")
    assert {path.name for path in store.root.iterdir()} == {".owner"}
    assert store.list() == []


def test_actual_tls_snapshot_tiles_are_bound_to_transaction_and_complete_length(certificates):
    run, zero, identifier = "a" * 64, "0" * 32, "b" * 32
    chunks, total = archive()
    with listen("127.0.0.1", 0) as listener, ThreadPoolExecutor(max_workers=1) as pool:
        def server():
            with accept(listener, certificates[Role.MIDDLE], local=Role.MIDDLE, peer=Role.HOST,
                        run=run, plane="data", timeout=3) as connection:
                frame = template(connection, identifier, sender=Role.MIDDLE, session=zero, request=zero, epoch=1)
                return send_archive(connection, frame, chunks(), total=total)
        future = pool.submit(server)
        with connect(listener.getsockname(), certificates[Role.HOST], local=Role.HOST, peer=Role.MIDDLE,
                     run=run, plane="data", timeout=3) as connection:
            frame = template(connection, identifier, sender=Role.MIDDLE, session=zero, request=zero, epoch=1)
            reader = TileReader(receive_archive(connection, frame, total=total))
            _, records, header, _ = read_header(reader, expected_bytes=total)
            assert records[0].nbytes == MAX_PAYLOAD + 157
            assert reader.read(MAX_PAYLOAD) == b"a" * MAX_PAYLOAD
            assert reader.read(157) == b"b" * 157
            reader.finish()
            assert not connection.closed
        assert future.result(timeout=3)["bytes"] == total


def test_windows_snapshot_handle_prevents_write_and_delete_during_validation(tmp_path):
    path = tmp_path / "state.bin"
    path.write_bytes(b"synthetic")
    with locked_read(path) as stream:
        assert stream.read() == b"synthetic"
        if os.name == "nt":
            with pytest.raises(OSError):
                path.open("wb")
            with pytest.raises(OSError):
                path.unlink()
    assert path.read_bytes() == b"synthetic"
