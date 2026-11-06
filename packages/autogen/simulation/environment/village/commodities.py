"""Physical commodities, decay mechanics, and Walrasian/Vickrey double auction market."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CommodityType(str, Enum):
    """Tangible commodities circulating in the AUTOGEN Village economy."""

    WATER = "water"
    WHEAT = "wheat"
    FLOUR = "flour"
    BREAD = "bread"
    TIMBER = "timber"
    IRON_ORE = "iron_ore"
    FISH = "fish"
    BERRIES = "berries"
    HERBS = "herbs"
    HEALING_BROTH = "healing_broth"
    TOOLS = "tools"


class CommoditySpec(BaseModel):
    """Physical and economic specification of a commodity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commodity: CommodityType
    base_price: float = Field(gt=0.0)
    decay_rate: float = Field(ge=0.0, le=1.0, description="Per-step fractional spoilage rate")
    nutrition_value: float = Field(ge=0.0, description="Hunger saturation restored per unit")
    hydration_value: float = Field(ge=0.0, description="Hydration restored per unit")
    energy_boost: float = Field(ge=0.0, description="Immediate energy restored per unit")
    health_boost: float = Field(default=0.0, ge=0.0, description="Immediate health restored per unit")


COMMODITY_CATALOG: dict[CommodityType, CommoditySpec] = {
    CommodityType.WATER: CommoditySpec(
        commodity=CommodityType.WATER,
        base_price=1.5,
        decay_rate=0.0,
        nutrition_value=0.0,
        hydration_value=35.0,
        energy_boost=2.0,
    ),
    CommodityType.WHEAT: CommoditySpec(
        commodity=CommodityType.WHEAT,
        base_price=3.0,
        decay_rate=0.005,
        nutrition_value=30.0,
        hydration_value=0.0,
        energy_boost=8.0,
    ),
    CommodityType.FLOUR: CommoditySpec(
        commodity=CommodityType.FLOUR,
        base_price=4.2,
        decay_rate=0.003,
        nutrition_value=34.0,
        hydration_value=0.0,
        energy_boost=10.0,
    ),
    CommodityType.BREAD: CommoditySpec(
        commodity=CommodityType.BREAD,
        base_price=5.5,
        decay_rate=0.008,
        nutrition_value=45.0,
        hydration_value=4.0,
        energy_boost=16.0,
        health_boost=6.0,
    ),
    CommodityType.TIMBER: CommoditySpec(
        commodity=CommodityType.TIMBER,
        base_price=4.0,
        decay_rate=0.0,
        nutrition_value=0.0,
        hydration_value=0.0,
        energy_boost=0.0,
    ),
    CommodityType.IRON_ORE: CommoditySpec(
        commodity=CommodityType.IRON_ORE,
        base_price=5.2,
        decay_rate=0.0,
        nutrition_value=0.0,
        hydration_value=0.0,
        energy_boost=0.0,
    ),
    CommodityType.FISH: CommoditySpec(
        commodity=CommodityType.FISH,
        base_price=4.5,
        decay_rate=0.025,
        nutrition_value=40.0,
        hydration_value=5.0,
        energy_boost=12.0,
    ),
    CommodityType.BERRIES: CommoditySpec(
        commodity=CommodityType.BERRIES,
        base_price=2.2,
        decay_rate=0.015,
        nutrition_value=20.0,
        hydration_value=10.0,
        energy_boost=6.0,
    ),
    CommodityType.HERBS: CommoditySpec(
        commodity=CommodityType.HERBS,
        base_price=3.8,
        decay_rate=0.010,
        nutrition_value=8.0,
        hydration_value=5.0,
        energy_boost=10.0,
        health_boost=8.0,
    ),
    CommodityType.HEALING_BROTH: CommoditySpec(
        commodity=CommodityType.HEALING_BROTH,
        base_price=7.5,
        decay_rate=0.012,
        nutrition_value=25.0,
        hydration_value=22.0,
        energy_boost=28.0,
        health_boost=24.0,
    ),
    CommodityType.TOOLS: CommoditySpec(
        commodity=CommodityType.TOOLS,
        base_price=8.0,
        decay_rate=0.002,
        nutrition_value=0.0,
        hydration_value=0.0,
        energy_boost=0.0,
    ),
}


class CommodityInventory(BaseModel):
    """Immutable commodity inventory container with spoilage and valuation methods."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: dict[CommodityType, float] = Field(default_factory=dict)

    def get(self, commodity: CommodityType) -> float:
        """Returns current quantity of a commodity (0.0 if absent)."""
        return max(0.0, float(self.items.get(commodity, 0.0)))

    def with_delta(self, commodity: CommodityType, delta: float) -> CommodityInventory:
        """Returns a new CommodityInventory with quantity adjusted by delta (clamped at 0)."""
        updated = dict(self.items)
        new_val = max(0.0, self.get(commodity) + float(delta))
        if new_val <= 1e-6:
            updated.pop(commodity, None)
        else:
            updated[commodity] = round(new_val, 4)
        return CommodityInventory(items=updated)

    def apply_spoilage(self) -> CommodityInventory:
        """Applies per-step physical decay $\\gamma_{\\text{decay}}$ to perishable commodities."""
        updated: dict[CommodityType, float] = {}
        for comm, qty in self.items.items():
            if qty <= 1e-6:
                continue
            decay = COMMODITY_CATALOG[comm].decay_rate
            rem = qty * (1.0 - decay)
            if rem > 1e-4:
                updated[comm] = round(rem, 4)
        return CommodityInventory(items=updated)

    def total_valuation(self, clearing_prices: dict[CommodityType, float]) -> float:
        """Computes $\\sum_k p_k \\cdot I_{i, k}$ at local market clearing prices."""
        total = 0.0
        for comm, qty in self.items.items():
            price = clearing_prices.get(comm, COMMODITY_CATALOG[comm].base_price)
            total += max(0.0, qty) * max(0.0, price)
        return total

    def total_edible_units(self) -> float:
        """Returns total units of food (wheat, flour, bread, fish, berries, healing broth)."""
        return (
            self.get(CommodityType.WHEAT)
            + self.get(CommodityType.FLOUR)
            + self.get(CommodityType.BREAD)
            + self.get(CommodityType.FISH)
            + self.get(CommodityType.BERRIES)
            + self.get(CommodityType.HEALING_BROTH)
        )

    def to_plain_dict(self) -> dict[str, float]:
        """Serializes keys as string names for JSON transport."""
        return {k.value: round(v, 3) for k, v in sorted(self.items.items(), key=lambda x: x[0].value)}


class MarketBid(BaseModel):
    """Buy order submitted to the Central Market."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    buyer_id: str
    commodity: CommodityType
    quantity: float = Field(gt=0.0)
    max_unit_price: float = Field(gt=0.0)
    buyer_status: float = Field(default=0.1, ge=0.0, le=1.0)


class MarketAsk(BaseModel):
    """Sell order submitted to the Central Market."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seller_id: str
    commodity: CommodityType
    quantity: float = Field(gt=0.0)
    min_unit_price: float = Field(gt=0.0)
    is_price_gouging: bool = False


class TradeExecution(BaseModel):
    """Record of a matched trade in the Double Auction Market."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    buyer_id: str
    seller_id: str
    commodity: CommodityType
    quantity: float = Field(gt=0.0)
    clearing_unit_price: float = Field(gt=0.0)
    total_coins: float = Field(gt=0.0)
    status_discount_applied: float = Field(default=0.0, ge=0.0)


class DoubleAuctionMarket:
    """Deterministic double auction and scarcity price clearing engine.

    Matches highest bids with lowest asks. High societal status $\\mathcal{S}_i$
    grants priority tie-breaking and reduced market spread markup.
    """

    def __init__(self) -> None:
        self.clearing_prices: dict[CommodityType, float] = {
            c: spec.base_price for c, spec in COMMODITY_CATALOG.items()
        }
        self.price_history: list[dict[str, float]] = []
        self.recent_trades: list[TradeExecution] = []

    def get_price(self, commodity: CommodityType) -> float:
        """Returns current clearing price $p_k$."""
        return self.clearing_prices.get(commodity, COMMODITY_CATALOG[commodity].base_price)

    def clear_market(
        self,
        step: int,
        bids: list[MarketBid],
        asks: list[MarketAsk],
        scarcity_index: dict[CommodityType, float] | None = None,
    ) -> list[TradeExecution]:
        """Executes double auction clearing across all commodities for step $t$."""
        executions: list[TradeExecution] = []

        for comm in CommodityType:
            comm_bids = [b for b in bids if b.commodity == comm]
            comm_asks = [a for a in asks if a.commodity == comm]

            # Deterministic sort: highest max_unit_price, then highest societal status, then buyer_id
            comm_bids.sort(key=lambda b: (-b.max_unit_price, -b.buyer_status, b.buyer_id))
            # Deterministic sort: lowest min_unit_price, then seller_id
            comm_asks.sort(key=lambda a: (a.min_unit_price, a.seller_id))

            b_idx = 0
            a_idx = 0
            rem_bid_qty = comm_bids[0].quantity if comm_bids else 0.0
            rem_ask_qty = comm_asks[0].quantity if comm_asks else 0.0

            cleared_prices: list[float] = []

            while b_idx < len(comm_bids) and a_idx < len(comm_asks):
                bid = comm_bids[b_idx]
                ask = comm_asks[a_idx]

                if bid.buyer_id == ask.seller_id:
                    a_idx += 1
                    if a_idx < len(comm_asks):
                        rem_ask_qty = comm_asks[a_idx].quantity
                    continue

                if bid.max_unit_price < ask.min_unit_price:
                    break

                trade_qty = min(rem_bid_qty, rem_ask_qty)
                if trade_qty > 1e-4:
                    # Vickrey / Midpoint clearing with high-status buyer discount
                    midpoint = 0.5 * (bid.max_unit_price + ask.min_unit_price)
                    status_discount = min(0.15, 0.12 * bid.buyer_status)
                    unit_price = max(
                        ask.min_unit_price,
                        round(midpoint * (1.0 - status_discount), 4),
                    )
                    total_coins = round(trade_qty * unit_price, 4)

                    executions.append(
                        TradeExecution(
                            step=step,
                            buyer_id=bid.buyer_id,
                            seller_id=ask.seller_id,
                            commodity=comm,
                            quantity=round(trade_qty, 4),
                            clearing_unit_price=unit_price,
                            total_coins=total_coins,
                            status_discount_applied=round(status_discount, 4),
                        )
                    )
                    cleared_prices.append(unit_price)

                rem_bid_qty -= trade_qty
                rem_ask_qty -= trade_qty

                if rem_bid_qty <= 1e-4:
                    b_idx += 1
                    if b_idx < len(comm_bids):
                        rem_bid_qty = comm_bids[b_idx].quantity
                if rem_ask_qty <= 1e-4:
                    a_idx += 1
                    if a_idx < len(comm_asks):
                        rem_ask_qty = comm_asks[a_idx].quantity

            # Update clearing price based on order imbalance and scarcity
            base_p = COMMODITY_CATALOG[comm].base_price
            current_p = self.clearing_prices[comm]
            total_demand = sum(b.quantity for b in comm_bids)
            total_supply = sum(a.quantity for a in comm_asks)
            scarcity_mult = (
                scarcity_index.get(comm, 1.0) if scarcity_index is not None else 1.0
            )

            if cleared_prices:
                avg_cleared = sum(cleared_prices) / len(cleared_prices)
                target_p = 0.65 * current_p + 0.35 * avg_cleared
            else:
                imbalance = (total_demand - total_supply) / max(1.0, total_demand + total_supply)
                target_p = current_p * (1.0 + 0.08 * imbalance)

            # Anchor gently toward scarcity-adjusted fundamental value
            fundamental_p = base_p * scarcity_mult
            new_p = 0.85 * target_p + 0.15 * fundamental_p
            self.clearing_prices[comm] = round(max(0.5 * base_p, min(4.0 * base_p, new_p)), 4)

        self.recent_trades = executions[-50:]
        self.price_history.append({k.value: v for k, v in self.clearing_prices.items()})
        return executions
