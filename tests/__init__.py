"""Offline test suite.

FORCE_RESEARCH (on by default in production) makes the graph run a real web search for
general-knowledge questions. Most graph tests mock the model but not the search tools,
so it is disabled here to keep the suite offline and deterministic. The forced-research
behaviour itself is tested explicitly in tests/test_regressions.py with mocked tools.
"""
from config import settings

settings.force_research = False
