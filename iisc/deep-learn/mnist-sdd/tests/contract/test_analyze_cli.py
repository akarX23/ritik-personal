"""T019 – Contract tests for analyze.py CLI.

Tests must FAIL before src/analyze.py is implemented.
"""

import pytest
import sys


def _get_parser():
    from src.analyze import build_parser
    return build_parser()


def test_results_required():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # no -r provided


def test_results_short_flag():
    parser = _get_parser()
    args = parser.parse_args(["-r", "/tmp/results"])
    assert args.results == "/tmp/results"


def test_results_long_flag():
    parser = _get_parser()
    args = parser.parse_args(["--results", "/tmp/results"])
    assert args.results == "/tmp/results"
