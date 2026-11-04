"""Event schema, serialization and transports."""

import io
import json
import queue
import subprocess
import sys
import time
import unittest
from pathlib import Path

import events as ev

ROOT = Path(__file__).resolve().parent.parent


def sample(name=ev.START_REQUESTED, **fields):
    defaults = {
        ev.SELECTION_PENDING: dict(feature="ocr", label="OCR", previous_feature=None),
        ev.SELECTION_CANCELLED: dict(feature="ocr", reason="timeout"),
        ev.START_REQUESTED: dict(feature="ocr", label="OCR"),
        ev.STOP_REQUESTED: dict(feature="ocr", reason="open_palm"),
        ev.RETURN_TO_SELECTION: dict(previous_feature="ocr", reason="stopped"),
    }[name]
    defaults.update(fields)
    return ev.Event(name, 1234567890.12345, "thumbs_up", "FEATURE_ACTIVE", defaults)


class SchemaTests(unittest.TestCase):
    def test_common_fields_and_order(self):
        d = ev.to_dict(sample())
        self.assertEqual(list(d), ["schema_version", "source", "event", "gesture", "state",
                                   "feature", "label", "timestamp"])
        self.assertEqual(d["schema_version"], 1)
        self.assertEqual(d["source"], "mavis_gesture")
        self.assertEqual(d["timestamp"], 1234567890.123)

    def test_every_event_has_its_declared_fields(self):
        for name, keys in ev.EVENT_FIELDS.items():
            d = ev.to_dict(sample(name))
            for k in ("schema_version", "source", "event", "gesture", "state", "timestamp"):
                self.assertIn(k, d)
            for k in keys:
                self.assertIn(k, d)

    def test_missing_optional_field_serializes_as_null(self):
        e = ev.Event(ev.STOP_REQUESTED, 1.0, None, "SELECTING_FEATURE", {"feature": "ocr"})
        self.assertIsNone(ev.to_dict(e)["reason"])
        self.assertIsNone(ev.to_dict(e)["gesture"])

    def test_rejects_unknown_event_or_field(self):
        with self.assertRaises(ValueError):
            ev.Event("feature.exploded", 1.0, None, "X")
        with self.assertRaises(ValueError):
            ev.Event(ev.START_REQUESTED, 1.0, None, "X", {"colour": "red"})

    def test_json_is_compact_single_line(self):
        line = ev.to_json(sample())
        self.assertNotIn("\n", line)
        self.assertNotIn(" ", line)
        self.assertEqual(json.loads(line)["event"], "feature.start_requested")


class TransportTests(unittest.TestCase):
    def test_bus_fans_out_and_isolates_failures(self):
        bus, got = ev.EventBus(), []

        def broken(_):
            raise RuntimeError("consumer bug")
        bus.subscribe(broken)
        bus.subscribe(got.append)
        bus.publish(sample())
        self.assertEqual(len(got), 1)

    def test_unsubscribe(self):
        bus, got = ev.EventBus(), []
        off = bus.subscribe(got.append)
        off()
        bus.publish(sample())
        self.assertEqual(got, [])

    def test_queue_sink(self):
        sink = ev.QueueSink(queue.Queue())
        sink(sample())
        self.assertEqual(sink.queue.get_nowait().name, ev.START_REQUESTED)

    def test_jsonl_writer_one_object_per_line(self):
        buf = io.StringIO()
        w = ev.JsonlWriter(buf)
        names = list(ev.EVENT_FIELDS)
        for n in names:
            w(sample(n))
        w.close()
        lines = buf.getvalue().splitlines()
        self.assertEqual([json.loads(x)["event"] for x in lines], names)

    def test_cli_help(self):
        r = subprocess.run([sys.executable, str(ROOT / "mavis_gesture.py"), "--help"],
                           capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--jsonl", r.stdout)

    def test_real_timestamp_from_controller(self):
        from feature_controller import FeatureController
        from gesture_mapping import load_mapping
        out = FeatureController(load_mapping()).on_gesture("one_finger")
        self.assertLess(abs(out[0].timestamp - time.time()), 2.0)


if __name__ == "__main__":
    unittest.main()
