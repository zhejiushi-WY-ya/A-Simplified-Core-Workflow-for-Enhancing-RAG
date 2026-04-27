from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class MaterializationEvent:
    node_path: str
    event_type: str
    model: str
    timestamp: str
    cache_key: Optional[str] = None
    cache_hit: bool = False
    request_signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionTraceCollector:
    question_id: str
    question_text: str
    cluster_id: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: List[MaterializationEvent] = field(default_factory=list)

    def record(
        self,
        *,
        node_path: str,
        event_type: str,
        model: str,
        cache_key: Optional[str] = None,
        cache_hit: bool = False,
        request_signature: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.events.append(
            MaterializationEvent(
                node_path=node_path,
                event_type=event_type,
                model=model,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cache_key=cache_key,
                cache_hit=cache_hit,
                request_signature=request_signature,
                metadata=metadata or {},
            )
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "cluster_id": self.cluster_id,
            "started_at": self.started_at,
            "events": [asdict(event) for event in self.events],
        }

    def build_state_patch(self) -> Dict[str, Any]:
        refs = []
        hits = []
        for event in self.events:
            if not event.cache_key:
                continue
            ref = {
                "node_path": event.node_path,
                "cache_key": event.cache_key,
                "event_type": event.event_type,
                "model": event.model,
            }
            refs.append(ref)
            if event.cache_hit:
                hits.append(ref)
        return {
            "llm_cache_refs": refs,
            "llm_cache_hits": hits,
        }


_CURRENT_TRACE: ContextVar[Optional[QuestionTraceCollector]] = ContextVar(
    "wtb_current_question_trace",
    default=None,
)
_NODE_STACK: ContextVar[Tuple[str, ...]] = ContextVar("wtb_node_stack", default=())


@contextmanager
def question_trace(
    *,
    question_id: str,
    question_text: str,
    cluster_id: Optional[str] = None,
) -> Iterator[QuestionTraceCollector]:
    collector = QuestionTraceCollector(
        question_id=question_id,
        question_text=question_text,
        cluster_id=cluster_id,
    )
    trace_token = _CURRENT_TRACE.set(collector)
    stack_token = _NODE_STACK.set(())
    try:
        yield collector
    finally:
        _NODE_STACK.reset(stack_token)
        _CURRENT_TRACE.reset(trace_token)


@contextmanager
def trace_node(segment: str) -> Iterator[str]:
    current_stack = _NODE_STACK.get()
    new_stack = (*current_stack, segment)
    token = _NODE_STACK.set(new_stack)
    try:
        yield ".".join(new_stack)
    finally:
        _NODE_STACK.reset(token)


def current_node_path(default: str = "") -> str:
    stack = _NODE_STACK.get()
    if not stack:
        return default
    return ".".join(stack)


def record_materialization_event(
    *,
    event_type: str,
    model: str,
    cache_key: Optional[str] = None,
    cache_hit: bool = False,
    request_signature: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    collector = _CURRENT_TRACE.get()
    if collector is None:
        return
    collector.record(
        node_path=current_node_path(default="retrieval"),
        event_type=event_type,
        model=model,
        cache_key=cache_key,
        cache_hit=cache_hit,
        request_signature=request_signature,
        metadata=metadata,
    )


def build_state_patch() -> Dict[str, Any]:
    collector = _CURRENT_TRACE.get()
    if collector is None:
        return {}
    return collector.build_state_patch()
