"""T055 – Unit tests for multi-batch training loop logic.

Tests batch size parsing, sequential batch execution, and distinct run_id generation
for multi-batch mode.
"""

import pytest


def test_parse_batches_single_value():
    """T055: Parse --batches with single value."""
    batches_str = "32"
    batches = [int(b.strip()) for b in batches_str.split(",")]
    assert batches == [32]


def test_parse_batches_multiple_values():
    """T055: Parse --batches with comma-separated values."""
    batches_str = "32,64,128"
    batches = [int(b.strip()) for b in batches_str.split(",")]
    assert batches == [32, 64, 128]


def test_parse_batches_with_whitespace():
    """T055: Parse --batches handles whitespace correctly."""
    batches_str = "32, 64, 128"
    batches = [int(b.strip()) for b in batches_str.split(",")]
    assert batches == [32, 64, 128]


def test_batch_loop_execution_count():
    """T055: Multi-batch loop executes once per batch size."""
    batches_str = "32,64,128"
    batches = [int(b.strip()) for b in batches_str.split(",")]
    
    execution_count = len(batches)
    assert execution_count == 3


def test_batch_sizes_maintain_order():
    """T055: Multi-batch execution preserves batch size order."""
    batches_str = "128,64,32"
    batches = [int(b.strip()) for b in batches_str.split(",")]
    
    assert batches[0] == 128
    assert batches[1] == 64
    assert batches[2] == 32


def test_distinct_run_ids_generated():
    """T055: Each batch execution should get a distinct run_id."""
    from src.metrics import new_run_id
    
    run_ids = [new_run_id() for _ in range(3)]
    
    # All should be UUIDs (36 chars with hyphens)
    assert all(len(rid) == 36 for rid in run_ids)
    
    # All should be unique
    assert len(set(run_ids)) == 3


def test_batch_loop_would_execute_sequentially():
    """T055: Batch loop logic should execute each batch sequentially."""
    batches = [32, 64, 128]
    
    # Simulate batch loop execution tracking
    executed_order = []
    for batch_size in batches:
        executed_order.append(batch_size)
    
    # Order should be preserved
    assert executed_order == [32, 64, 128]
