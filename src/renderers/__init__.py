"""
Renderer implementations for the Mitsuba application.

This package contains various renderer implementations for different use cases.
"""

from .base import BaseRenderer
from .multi import MultiObjSceneRenderer

__all__ = ['BaseRenderer', 'MultiObjSceneRenderer']
