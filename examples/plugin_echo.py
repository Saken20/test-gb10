"""Plugin template (interfaces of all stages: adapters.py). Config:

    llm: {type: plugin, plugin: examples/plugin_echo.py:EchoLLM, delay_s: 0.01}
"""
import asyncio


class EchoLLM:
    def __init__(self, stage_cfg, cfg, client):
        self.delay = float(stage_cfg.get("delay_s", 0.01))

    async def stream(self, messages):
        words = messages[-1]["content"].split()
        for i, w in enumerate(words):
            await asyncio.sleep(self.delay)
            yield (w if i == 0 else " " + w), None
        # usage is optional; without it bench.py counts stream chunks as tokens
        yield "", {"completion_tokens": len(words), "prompt_tokens": sum(len(m["content"].split()) for m in messages)}

    async def aclose(self):
        pass


class EchoNLU:
    def __init__(self, stage_cfg, cfg, client):
        self.keywords = stage_cfg.get("keywords") or {"hello": "greeting"}

    async def classify(self, text):
        low = text.lower()
        return next((intent for word, intent in self.keywords.items() if word in low), None)
