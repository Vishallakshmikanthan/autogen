"""Unit tests for deterministic RNG streams and seed separation."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.core.rng import SeedManager


@pytest.mark.unit
def test_seed_determinism():
    mgr_a = SeedManager(master_seed=12345)
    mgr_b = SeedManager(master_seed=12345)

    stream_a = mgr_a.spawn_stream("subsystem_alpha")
    stream_b = mgr_b.spawn_stream("subsystem_alpha")

    samples_a = stream_a.uniform(size=10)
    samples_b = stream_b.uniform(size=10)

    np.testing.assert_array_equal(samples_a, samples_b)


@pytest.mark.unit
def test_stream_independence():
    mgr = SeedManager(master_seed=42)
    stream_1 = mgr.spawn_stream("env")
    stream_2 = mgr.spawn_stream("agent")

    sample_1 = stream_1.integers(0, 1000, size=20)
    sample_2 = stream_2.integers(0, 1000, size=20)

    assert not np.array_equal(sample_1, sample_2)


@pytest.mark.unit
def test_train_eval_separation():
    mgr = SeedManager(master_seed=777)
    train_gen = mgr.get_train_generator(0)
    eval_gen = mgr.get_eval_generator(0)

    train_data = train_gen.normal(size=15)
    eval_data = eval_gen.normal(size=15)

    assert not np.array_equal(train_data, eval_data)


@pytest.mark.unit
def test_negative_seed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        SeedManager(master_seed=-1)


@pytest.mark.unit
def test_stream_cache_preserves_state():
    mgr = SeedManager(master_seed=99)
    stream_first = mgr.spawn_stream("cached_stream")
    val_1 = stream_first.random()

    stream_second = mgr.spawn_stream("cached_stream")
    assert stream_first is stream_second
    val_2 = stream_second.random()
    assert val_1 != val_2
