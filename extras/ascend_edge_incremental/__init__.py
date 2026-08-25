"""Isolated Ascend310B light-weight class-incremental training workflow."""

from .protocol import EdgeProtocol, RoundSpec, load_protocol

__all__ = ["EdgeProtocol", "RoundSpec", "load_protocol"]
