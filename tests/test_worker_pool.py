"""Contracts for bounded ephemeral NEXUS workers."""

import threading
import time
import unittest

from core.nexus.worker_pool import BoundedWorkerPool, WorkerPoolBusy


class WorkerPoolTests(unittest.TestCase):
    def test_registered_job_runs_on_json_copy(self):
        pool = BoundedWorkerPool(max_workers=1, max_queued=0)
        try:
            payload = {"items": [1, 2]}
            pool.register("echo", lambda value, cancel: value)
            lease = pool.submit("echo", payload)
            payload["items"].append(3)
            self.assertEqual(pool.result(lease.job_id, timeout=2), {"items": [1, 2]})
        finally:
            pool.close()

    def test_unknown_job_type_is_never_executed(self):
        pool = BoundedWorkerPool(max_workers=1)
        try:
            with self.assertRaises(PermissionError):
                pool.submit("not.registered", {})
        finally:
            pool.close()

    def test_queued_cancellation_releases_capacity(self):
        started = threading.Event()
        release = threading.Event()

        def blocker(payload, cancel):
            started.set()
            while not release.is_set() and not cancel.is_set():
                time.sleep(0.005)
            return payload["id"]

        pool = BoundedWorkerPool(max_workers=1, max_queued=1)
        try:
            pool.register("block", blocker)
            first = pool.submit("block", {"id": 1})
            self.assertTrue(started.wait(1))
            second = pool.submit("block", {"id": 2})
            with self.assertRaises(WorkerPoolBusy):
                pool.submit("block", {"id": 3})

            self.assertTrue(pool.cancel(second.job_id))

            third = None
            deadline = time.monotonic() + 2
            while third is None and time.monotonic() < deadline:
                try:
                    third = pool.submit("block", {"id": 3})
                except WorkerPoolBusy:
                    time.sleep(0.01)
            self.assertIsNotNone(third, "cancelled queued job leaked worker capacity")

            release.set()
            self.assertEqual(pool.result(first.job_id, timeout=2), 1)
            self.assertEqual(pool.result(third.job_id, timeout=2), 3)
        finally:
            release.set()
            pool.close()

    def test_timeout_marks_running_job_for_cooperative_cancellation(self):
        now = [10.0]
        started = threading.Event()

        def handler(payload, cancel):
            started.set()
            while not cancel.is_set():
                time.sleep(0.005)
            return "cancelled"

        pool = BoundedWorkerPool(
            max_workers=1,
            max_queued=0,
            default_timeout_seconds=2,
            clock=lambda: now[0],
        )
        try:
            pool.register("wait", handler)
            lease = pool.submit("wait", {})
            self.assertTrue(started.wait(1))
            now[0] = 13.0
            self.assertEqual(pool.cancel_expired(), (lease.job_id,))
            self.assertEqual(pool.result(lease.job_id, timeout=2), "cancelled")
        finally:
            pool.close()

    def test_payload_budget_is_bounded(self):
        pool = BoundedWorkerPool(max_workers=1)
        try:
            pool.register("echo", lambda value, cancel: value)
            with self.assertRaises(ValueError):
                pool.submit("echo", {"blob": "x" * (65 * 1024)})
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
