"""
Analytics module for trading system.

Provides performance attribution, risk analytics, and portfolio analysis tools.
"""

from .performance_attribution import (
    PerformanceAttributor,
    AttributionMetrics,
    create_performance_report
)

__all__ = [
    'PerformanceAttributor',
    'AttributionMetrics',
    'create_performance_report'
]
