import asyncio
import unittest

from core.nexus.middleware import (
    MiddlewareCancelledError,
    MiddlewareContext,
    MiddlewareError,
    MiddlewareExecutionError,
    MiddlewarePipeline,
    MiddlewareSpec,
)


class NEXUSMiddlewareTests(unittest.TestCase):
    def test_middleware_wraps_terminal_in_order_and_traces_once(self):
        events = []

        async def first(context, call_next):
            events.append("first:before")
            value = await call_next()
            events.append("first:after")
            return value + "!"

        async def second(context, call_next):
            events.append("second:before")
            value = await call_next()
            events.append("second:after")
            return value.upper()

        async def terminal(context):
            events.append("terminal")
            return "jarvis"

        pipeline = MiddlewarePipeline((MiddlewareSpec("first", first), MiddlewareSpec("second", second)))
        result, trace = asyncio.run(pipeline.run(MiddlewareContext("model.reply", "standard"), terminal))
        self.assertEqual(result, "JARVIS!")
        self.assertEqual(events, ["first:before", "second:before", "terminal", "second:after", "first:after"])
        self.assertEqual({item.stage for item in trace}, {"first", "second", "terminal"})
        self.assertTrue(all(item.duration_ms >= 0 for item in trace))

    def test_middleware_cannot_call_next_twice(self):
        terminal_calls = []

        async def bad(context, call_next):
            await call_next()
            try:
                return await call_next()
            except MiddlewareError:
                return "must not mask the contract failure"

        async def terminal(context):
            terminal_calls.append(context.operation)
            return "ok"

        pipeline = MiddlewarePipeline((MiddlewareSpec("bad", bad),))
        with self.assertRaises(MiddlewareError) as caught:
            asyncio.run(pipeline.run(MiddlewareContext("tool.call", "mission"), terminal))
        self.assertEqual(terminal_calls, ["tool.call"])
        self.assertTrue(caught.exception.terminal_completed)
        self.assertFalse(caught.exception.retry_safe)
        self.assertIn("automatic retry is unsafe", str(caught.exception))

    def test_short_circuit_requires_explicit_application_registration(self):
        async def stop(context, call_next):
            return "denied"
        strict = MiddlewarePipeline((MiddlewareSpec("guard", stop),))
        with self.assertRaises(MiddlewareError):
            asyncio.run(strict.run(MiddlewareContext("tool.call", "mission"), lambda context: "unsafe"))
        guarded = MiddlewarePipeline((MiddlewareSpec("guard", stop, allow_short_circuit=True),))
        result, trace = asyncio.run(guarded.run(MiddlewareContext("tool.call", "mission"), lambda context: "unsafe"))
        self.assertEqual(result, "denied")
        self.assertEqual(trace[-1].outcome, "short_circuit")
        self.assertTrue(all(item.stage != "terminal" for item in trace))

    def test_handler_exception_is_redacted_and_identifies_stage(self):
        async def explode(context, call_next):
            raise RuntimeError("secret provider payload")
        pipeline = MiddlewarePipeline((MiddlewareSpec("telemetry", explode),))
        with self.assertRaises(MiddlewareExecutionError) as caught:
            asyncio.run(pipeline.run(MiddlewareContext("model.reply", "fast"), lambda context: "ok"))
        self.assertEqual(caught.exception.stage, "telemetry")
        self.assertEqual(caught.exception.error_type, "RuntimeError")
        self.assertNotIn("secret provider payload", str(caught.exception))
        self.assertTrue(caught.exception.retry_safe)

    def test_pipeline_is_bounded_and_names_are_unique(self):
        async def passthrough(context, call_next):
            return await call_next()
        with self.assertRaises(ValueError):
            MiddlewarePipeline(tuple(MiddlewareSpec(f"m{i}", passthrough) for i in range(33)))
        with self.assertRaises(ValueError):
            MiddlewarePipeline((MiddlewareSpec("same", passthrough), MiddlewareSpec("same", passthrough)))

    def test_context_metadata_is_immutable_and_bounded(self):
        source = {"owner": {"modes": ["local", {"verified": True}]}}
        context = MiddlewareContext("model.reply", "standard", metadata=source)
        source["owner"]["modes"][1]["verified"] = False
        source["owner"]["modes"].append("remote")
        self.assertEqual(context.metadata["owner"]["modes"][1]["verified"], True)
        self.assertEqual(context.metadata["owner"]["modes"], ("local", {"verified": True}))
        with self.assertRaises(TypeError):
            context.metadata["owner"] = "x"
        with self.assertRaises(TypeError):
            context.metadata["owner"]["modes"][1]["verified"] = False

    def test_context_normalizes_operation_and_route_and_rejects_ambiguous_tokens(self):
        context = MiddlewareContext("  Model.Reply  ", " STANDARD ")
        self.assertEqual(context.operation, "model.reply")
        self.assertEqual(context.route, "standard")
        for operation, route in (
            ("model reply", "standard"),
            ("model.reply", "standard/other"),
            ("mødel.reply", "standard"),
            ("model.reply\nnext", "standard"),
        ):
            with self.subTest(operation=operation, route=route), self.assertRaises(ValueError):
                MiddlewareContext(operation, route)

    def test_context_rejects_deep_large_cyclic_or_non_json_metadata(self):
        deeply_nested = {}
        cursor = deeply_nested
        for _ in range(12):
            cursor["next"] = {}
            cursor = cursor["next"]
        cyclic = []
        cyclic.append(cyclic)
        for metadata in (
            {"deep": deeply_nested},
            {"large": "x" * 33000},
            {"cumulative": ["x" * 1000] * 33},
            {"integer": 1 << 5000},
            {"cycle": cyclic},
            {"opaque": object()},
        ):
            with self.subTest(kind=next(iter(metadata))), self.assertRaises(ValueError):
                MiddlewareContext("model.reply", "standard", metadata=metadata)

    def test_call_next_rejects_concurrent_and_out_of_scope_tasks_before_terminal(self):
        terminal_calls = []

        async def terminal(context):
            terminal_calls.append(context.operation)
            return "unsafe"

        async def concurrent(context, call_next):
            return await asyncio.create_task(call_next())

        concurrent_pipeline = MiddlewarePipeline((MiddlewareSpec("concurrent", concurrent),))
        with self.assertRaises(MiddlewareError) as caught:
            asyncio.run(concurrent_pipeline.run(MiddlewareContext("tool.call", "mission"), terminal))
        self.assertTrue(caught.exception.retry_safe)
        self.assertEqual(terminal_calls, [])

        async def scenario():
            leaked = []

            async def detach(context, call_next):
                leaked.append(asyncio.create_task(call_next()))
                return "short"

            pipeline = MiddlewarePipeline((MiddlewareSpec("detach", detach, allow_short_circuit=True),))
            result, _ = await pipeline.run(MiddlewareContext("tool.call", "mission"), terminal)
            self.assertEqual(result, "short")
            with self.assertRaises(MiddlewareError) as detached:
                await leaked[0]
            self.assertTrue(detached.exception.retry_safe)

        asyncio.run(scenario())
        self.assertEqual(terminal_calls, [])

    def test_errors_after_terminal_make_retry_state_explicit(self):
        terminal_calls = []

        async def fail_after(context, call_next):
            await call_next()
            raise RuntimeError("post-processing failed")

        async def terminal(context):
            terminal_calls.append(context.operation)
            return "committed"

        pipeline = MiddlewarePipeline((MiddlewareSpec("postprocess", fail_after),))
        with self.assertRaises(MiddlewareExecutionError) as caught:
            asyncio.run(pipeline.run(MiddlewareContext("tool.call", "mission"), terminal))
        error = caught.exception
        self.assertEqual(terminal_calls, ["tool.call"])
        self.assertEqual(error.stage, "postprocess")
        self.assertTrue(error.terminal_completed)
        self.assertFalse(error.retry_safe)
        self.assertIn("automatic retry is unsafe", str(error))

    def test_terminal_failure_records_started_state_as_retry_unsafe(self):
        def terminal(context):
            raise RuntimeError("effect may already have happened")

        with self.assertRaises(MiddlewareExecutionError) as caught:
            asyncio.run(MiddlewarePipeline().run(MiddlewareContext("tool.call", "mission"), terminal))
        self.assertEqual(caught.exception.stage, "terminal")
        self.assertTrue(caught.exception.terminal_started)
        self.assertFalse(caught.exception.terminal_completed)
        self.assertFalse(caught.exception.retry_safe)

    def test_task_cancellation_preserves_retry_state_before_during_and_after_terminal(self):
        async def scenario(phase):
            entered = asyncio.Event()
            blocked = asyncio.Event()
            effects = []

            async def pause():
                entered.set()
                await blocked.wait()

            async def middleware(context, call_next):
                if phase == "before":
                    await pause()
                result = await call_next()
                if phase == "after":
                    await pause()
                return result

            async def terminal(context):
                effects.append("effect")
                if phase == "during":
                    await pause()
                return "committed"

            pipeline = MiddlewarePipeline((MiddlewareSpec("wrapper", middleware),))
            task = asyncio.create_task(pipeline.run(MiddlewareContext("tool.call", "mission"), terminal))
            await entered.wait()
            task.cancel("secret cancellation payload")
            with self.assertRaises(asyncio.CancelledError) as caught:
                await task
            error = caught.exception
            self.assertIsInstance(error, MiddlewareCancelledError)
            self.assertTrue(task.cancelled())
            self.assertEqual(error.terminal_started, phase != "before")
            self.assertEqual(error.terminal_completed, phase == "after")
            self.assertEqual(error.retry_safe, phase == "before")
            self.assertEqual(effects, [] if phase == "before" else ["effect"])
            self.assertNotIn("secret cancellation payload", str(error))

        for phase in ("before", "during", "after"):
            with self.subTest(phase=phase):
                asyncio.run(scenario(phase))

    def test_middleware_cannot_mask_terminal_cancellation(self):
        async def scenario(mask_with_error):
            effects = []

            async def middleware(context, call_next):
                try:
                    return await call_next()
                except asyncio.CancelledError:
                    if mask_with_error:
                        raise MiddlewareError("masking cancellation")
                    return "masked cancellation"

            async def terminal(context):
                effects.append("effect")
                raise asyncio.CancelledError()

            pipeline = MiddlewarePipeline((MiddlewareSpec("wrapper", middleware),))
            task = asyncio.create_task(pipeline.run(MiddlewareContext("tool.call", "mission"), terminal))
            with self.assertRaises(MiddlewareCancelledError) as caught:
                await task
            self.assertTrue(task.cancelled())
            self.assertEqual(effects, ["effect"])
            self.assertEqual(caught.exception.stage, "terminal")
            self.assertTrue(caught.exception.terminal_started)
            self.assertFalse(caught.exception.retry_safe)

        for mask_with_error in (False, True):
            with self.subTest(mask_with_error=mask_with_error):
                asyncio.run(scenario(mask_with_error))


if __name__ == "__main__":
    unittest.main()
