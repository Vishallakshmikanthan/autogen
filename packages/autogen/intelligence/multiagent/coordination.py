"""Coordination protocols: Contract-Net task allocation and Vickrey second-price auctions (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskAnnouncement:
    """Task specification announced by a manager agent."""

    task_id: str
    manager_id: str
    description: str
    requirements: Mapping[str, Any] = field(default_factory=dict)
    expiration_step: int = 10


@dataclass(frozen=True, slots=True)
class TaskBid:
    """Bid proposal submitted by a prospective contractor."""

    task_id: str
    contractor_id: str
    bid_cost: float
    capabilities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContractAward:
    """Contract assignment awarded by manager to winning contractor."""

    task_id: str
    manager_id: str
    winner_contractor_id: str
    agreed_cost: float


class ContractNetProtocol:
    """Contract-Net protocol managing decentralized task announcement, bidding, and awarding."""

    def __init__(self) -> None:
        self.active_announcements: dict[str, TaskAnnouncement] = {}
        self.received_bids: dict[str, list[TaskBid]] = {}
        self.awarded_contracts: dict[str, ContractAward] = {}

    def announce_task(self, announcement: TaskAnnouncement) -> None:
        """Publishes a new task for contractor bidding."""
        self.active_announcements[announcement.task_id] = announcement
        self.received_bids[announcement.task_id] = []

    def submit_bid(self, bid: TaskBid) -> bool:
        """Records contractor bid for an active announcement."""
        if bid.task_id not in self.active_announcements:
            return False
        self.received_bids[bid.task_id].append(bid)
        return True

    def evaluate_and_award(self, task_id: str) -> ContractAward | None:
        """Awards contract to the lowest-cost bidder (cost-minimizing task allocation)."""
        if task_id not in self.active_announcements:
            return None

        bids = self.received_bids.get(task_id, [])
        if not bids:
            return None

        # Best contractor is the one offering lowest execution cost
        best_bid = min(bids, key=lambda b: b.bid_cost)
        announcement = self.active_announcements[task_id]

        award = ContractAward(
            task_id=task_id,
            manager_id=announcement.manager_id,
            winner_contractor_id=best_bid.contractor_id,
            agreed_cost=best_bid.bid_cost,
        )
        self.awarded_contracts[task_id] = award
        del self.active_announcements[task_id]
        return award


class VickreyAuction:
    """Second-Price Sealed-Bid (Vickrey) Auction ensuring truthful dominant-strategy bidding."""

    def __init__(self, item_name: str, reserve_price: float = 0.0) -> None:
        self.item_name = item_name
        self.reserve_price = float(reserve_price)
        self.bids: dict[str, float] = {}

    def submit_bid(self, bidder_id: str, bid_amount: float) -> None:
        """Registers sealed bid from participant."""
        self.bids[bidder_id] = float(bid_amount)

    def resolve(self) -> tuple[str | None, float]:
        """Resolves auction: highest bidder wins, but pays the second-highest bid.

        Returns (winner_id, clearing_price).
        """
        valid_bids = [(bidder, amt) for bidder, amt in self.bids.items() if amt >= self.reserve_price]
        if not valid_bids:
            return None, self.reserve_price

        # Sort descending by bid amount
        sorted_bids = sorted(valid_bids, key=lambda x: x[1], reverse=True)
        winner_id, highest_bid = sorted_bids[0]

        if len(sorted_bids) > 1:
            second_highest_bid = sorted_bids[1][1]
            clearing_price = second_highest_bid
        else:
            clearing_price = self.reserve_price

        return winner_id, clearing_price
