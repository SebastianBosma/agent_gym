"""Trace parser module - converts raw traces to unified JSON format."""

from .parser import parse_traces, TraceSchema

__all__ = ["parse_traces", "TraceSchema"]

