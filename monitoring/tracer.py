import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Span:
    name: str
    component: str
    node_id: str
    round_num: int
    start_ns: int
    duration_us: float
    tags: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    def close(self):
        self.duration_us = (time.time_ns() - self.start_ns) / 1_000


class Tracer:
    def __init__(self):
        self._roots: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name, component, node_id="", round_num=0, **tags):
        child = Span(
            name=name,
            component=component,
            node_id=node_id,
            round_num=round_num,
            start_ns=time.time_ns(),
            duration_us=0.0,
            tags=tags,
        )
        if self._stack:
            self._stack[-1].children.append(child)
        self._stack.append(child)
        try:
            yield
        finally:
            child.close()
            if len(self._stack) == 1:
                self._roots.append(self._stack.pop())
            else:
                self._stack.pop()

    @contextmanager
    def trace_async(self, coro_factory, name, component, node_id="", round_num=0, **tags):
        with self.span(name, component, node_id, round_num, **tags):
            yield

    def trace_summary(self) -> dict:
        result = {}
        for root in self._roots:
            _aggregate(root, result)
        return result

    def clear(self):
        self._roots.clear()
        self._stack.clear()

    def to_json(self, path: Optional[str] = None) -> str:
        data = [_span_to_dict(r) for r in self._roots]
        output = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output


def _aggregate(span: Span, acc: dict):
    key = f"{span.component}.{span.name}"
    if key not in acc:
        acc[key] = {"count": 0, "total_us": 0.0, "min_us": float("inf"), "max_us": 0.0}
    acc[key]["count"] += 1
    acc[key]["total_us"] += span.duration_us
    acc[key]["min_us"] = min(acc[key]["min_us"], span.duration_us)
    acc[key]["max_us"] = max(acc[key]["max_us"], span.duration_us)
    for c in span.children:
        _aggregate(c, acc)


def _span_to_dict(s: Span) -> dict:
    d = asdict(s)
    d["children"] = [_span_to_dict(c) for c in s.children]
    return d
