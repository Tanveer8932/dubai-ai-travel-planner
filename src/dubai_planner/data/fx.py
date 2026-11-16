"""Currency conversion with AED as a first-class currency.

The ECB (and therefore Frankfurter) publishes no AED rate. The dirham has been pegged
to the US dollar at **1 USD = 3.6725 AED** since 1997, so every conversion goes through
USD: ``X → USD`` via Frankfurter, then ``USD → AED`` via the peg. USD↔AED needs no
network at all. Offline, a stored Frankfurter snapshot (``fx_rates.json``) is used.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DataKind, Provenance, quantize_aed, to_decimal
from dubai_planner.models.sources import FxRates

logger = logging.getLogger(__name__)

LATEST_URL = "https://api.frankfurter.dev/v1/latest"
AED_PER_USD = Decimal("3.6725")
PEG_PROVENANCE = Provenance(
    kind=DataKind.VERIFIED,
    source="Central Bank of the UAE — fixed AED/USD peg",
    source_url="https://www.centralbank.ae/en/",
    as_of=date(2026, 11, 15),
    note="Fixed peg of 3.6725 AED per USD in place since 1997.",
)
_PEGGED = {"USD": Decimal("1"), "AED": AED_PER_USD}


class FxClient:
    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient("frankfurter", ttl=timedelta(hours=24))

    def rates(self) -> FxRates:
        """Units per USD for all ECB currencies plus AED. Never raises for network issues."""
        try:
            resp = self.http.get_json(LATEST_URL, {"base": "USD"})
            data = resp.data
            rates = {str(k).upper(): to_decimal(v) for k, v in data["rates"].items()}
            rates.update(_PEGGED)
            return FxRates(
                rates=rates,
                provenance=Provenance(
                    kind=DataKind.API,
                    source="Frankfurter (ECB reference rates)",
                    source_url=LATEST_URL,
                    as_of=date.fromisoformat(data["date"]),
                    note="stale cache" if resp.stale else None,
                ),
            )
        except (SourceUnavailableError, KeyError, TypeError, ValueError) as exc:
            logger.info("frankfurter unavailable (%s); using seed snapshot", exc)
            snapshot = load_catalog().fx_rates
            return snapshot.model_copy(update={"rates": {**snapshot.rates, **_PEGGED}})

    def convert(self, amount: Decimal | float | int | str, from_ccy: str, to_ccy: str) -> Decimal:
        return convert(amount, from_ccy, to_ccy, client=self)


def convert(
    amount: Decimal | float | int | str,
    from_ccy: str,
    to_ccy: str,
    *,
    client: FxClient | None = None,
) -> Decimal:
    """Convert ``amount`` between currencies, rounded to 2 dp (half-up).

    >>> convert(100, "USD", "AED")
    Decimal('367.25')
    """
    value = to_decimal(amount)
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"invalid amount {amount!r}")
    src, dst = from_ccy.upper(), to_ccy.upper()
    if src == dst:
        return quantize_aed(value)
    if src in _PEGGED and dst in _PEGGED:  # pure peg: no network
        rates = _PEGGED
    else:
        rates = (client or FxClient()).rates().rates
    try:
        usd = value / rates[src]
        return quantize_aed(usd * rates[dst])
    except KeyError as exc:
        raise ValueError(f"unsupported currency {exc.args[0]!r}") from exc


def to_aed(amount: Decimal | float | int | str, currency: str, *, client: FxClient | None = None) -> Decimal:
    return convert(amount, currency, "AED", client=client)
