"""Mathematical tests for Value Decomposition (VDN, QMIX-Lite) and coordination (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.multiagent.coordination import (
    ContractNetProtocol,
    TaskAnnouncement,
    TaskBid,
    VickreyAuction,
)
from autogen.intelligence.multiagent.value_decomposition import (
    QMIXLite,
    ValueDecompositionNetworks,
)


@pytest.mark.mathematical
def test_vdn_individual_global_max_monotonicity():
    """Verifies that VDN satisfies the IGM condition across 1,000 random Q-value configurations."""
    rng = np.random.default_rng(42)

    for _ in range(1000):
        # 3 agents with 4 actions each
        tables = [rng.uniform(-10.0, 10.0, size=4) for _ in range(3)]
        assert ValueDecompositionNetworks.verify_igm_condition(tables) is True


@pytest.mark.mathematical
def test_vdn_single_agent_reduction():
    """Verifies that VDN reduces to standard single-agent value when N=1."""
    q_table = np.array([1.5, 4.2, -0.5, 2.1])
    joint_q = ValueDecompositionNetworks.joint_q([q_table[1]])
    best_act = ValueDecompositionNetworks.individual_best_actions([q_table])

    assert joint_q == 4.2
    assert best_act == (1,)


@pytest.mark.mathematical
def test_qmix_lite_monotonicity():
    """Verifies that QMIX-Lite satisfies dQ_tot / dQ_i >= 0 across diverse states."""
    rng = np.random.default_rng(123)
    qmix = QMIXLite(num_agents=3, state_dim=6, embed_dim=8, seed=42)

    for _ in range(100):
        state = rng.normal(0.0, 1.0, size=6)
        assert qmix.check_monotonicity(state, delta=0.5) is True


@pytest.mark.unit
def test_vickrey_auction_second_price():
    """Verifies that second-price sealed-bid auction awards highest bidder at second-highest price."""
    auction = VickreyAuction(item_name="foraging_permit", reserve_price=5.0)

    auction.submit_bid("agent_a", 15.0)
    auction.submit_bid("agent_b", 22.0)
    auction.submit_bid("agent_c", 18.0)

    winner, clearing_price = auction.resolve()
    assert winner == "agent_b"
    assert clearing_price == 18.0  # Second highest bid


@pytest.mark.unit
def test_contract_net_protocol_cost_minimization():
    """Verifies Contract-Net protocol awards task to lowest cost bidder."""
    cnet = ContractNetProtocol()
    task = TaskAnnouncement(task_id="t1", manager_id="mgr", description="scout_sector_north")
    cnet.announce_task(task)

    cnet.submit_bid(TaskBid(task_id="t1", contractor_id="c1", bid_cost=50.0))
    cnet.submit_bid(TaskBid(task_id="t1", contractor_id="c2", bid_cost=30.0))
    cnet.submit_bid(TaskBid(task_id="t1", contractor_id="c3", bid_cost=45.0))

    award = cnet.evaluate_and_award("t1")
    assert award is not None
    assert award.winner_contractor_id == "c2"
    assert award.agreed_cost == 30.0

