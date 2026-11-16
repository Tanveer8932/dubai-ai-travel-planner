"""Deterministic validators shared by both modes.

Import rules from :mod:`dubai_planner.validation.rules` and shared limits from
:mod:`dubai_planner.validation.thresholds` (this package does not import them eagerly,
so ``tools`` can use the thresholds without an import cycle).
"""
