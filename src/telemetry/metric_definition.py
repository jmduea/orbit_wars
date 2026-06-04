"""Telemetry metric definition types and factory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    group: str
    description: str
    record_kinds: frozenset[str]
    protected: bool = False
    internal_only: bool = False
    rollout_scalar_role: str | None = None


def metric(
    name: str,
    group: str,
    description: str,
    *,
    record_kinds: tuple[str, ...] = ("update",),
    protected: bool = False,
    internal_only: bool = False,
    rollout_scalar_role: str | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        group=group,
        description=description,
        record_kinds=frozenset(record_kinds),
        protected=protected,
        internal_only=internal_only,
        rollout_scalar_role=rollout_scalar_role,
    )
