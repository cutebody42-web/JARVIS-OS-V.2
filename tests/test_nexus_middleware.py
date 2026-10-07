import asyncio
import unittest

from core.nexus.middleware import (
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
        async def bad(context, call_next):
            await call_next()
            return await call_next()
        pipeline = MiddlewarePipeline((MiddlewareSpec("bad", bad),))
        with self.assertRaises(MiddlewareError):
            asyncio.run(pipeline.run(MiddlewareContext("tool.call", "mission"), lambda context: "ok"))

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

    def test_pipeline_is_bounded_and_names_are_unique(self):
        async def passthrough(context, call_next):
            return await call_next()
        with self.assertRaises(ValueError):
            MiddlewarePipeline(tuple(MiddlewareSpec(f"m{i}", passthrough) for i in range(33)))
        with self.assertRaises(ValueError):
            MiddlewarePipeline((MiddlewareSpec("same", passthrough), MiddlewareSpec("same", passthrough)))

    def test_context_metadata_is_immutable_and_bounded(self):
        source = {"owner": "local"}
        context = MiddlewareContext("model.reply", "standard", metadata=source)
        source["owner"] = "changed"
        self.assertEqual(context.metadata["owner"], "local")
        with self.assertRaises(TypeError):
            context.metadata["owner"] = "x"


if __name__ == "__main__":
    unittest.main()
