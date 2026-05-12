"""T011 – Contract tests for train.py CLI.

Validates all required arguments exist, defaults are correct, and that
--device is required (no default).

Tests must FAIL before src/train.py is implemented.
"""

import pytest
import argparse
import sys
from unittest.mock import patch


def _get_parser():
    from src.train import build_parser
    return build_parser()


def test_epochs_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.epochs == 10


def test_results_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.results == "./results"


def test_batch_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.batch == 64


def test_lr_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.lr == pytest.approx(0.001)


def test_data_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.data == "./data"


def test_device_required():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # no -d provided


def test_device_cpu_accepted():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.device == "cpu"


def test_device_xpu_accepted():
    parser = _get_parser()
    args = parser.parse_args(["-d", "xpu"])
    assert args.device == "xpu"


def test_short_flags_work():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu", "-e", "5", "-r", "/tmp/r", "-b", "32",
                               "-lr", "0.01", "-m", "/tmp/d"])
    assert args.epochs == 5
    assert args.results == "/tmp/r"
    assert args.batch == 32
    assert args.lr == pytest.approx(0.01)
    assert args.data == "/tmp/d"


def test_long_flags_work():
    parser = _get_parser()
    args = parser.parse_args(["--device", "cpu", "--epochs", "3"])
    assert args.device == "cpu"
    assert args.epochs == 3
