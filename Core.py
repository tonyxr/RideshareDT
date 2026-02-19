from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
"""
"""
End-to-end two-firm market simulator + Firm1 RL training.

Defaults aim for stable learning:
- large customers_per_step
- normalized reward + coefficient movement penalty
- Wasserstein trust region in policy update
"""

import argparse
import csv
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from MarketInteraction import MarketInteraction, RideContext
from Market_models import CoefficientOverrides
from GenerateAgent import GenerateAgent
from choice_models import ParametricChoiceModel, LLMChoiceModel, ChoiceResult
from pricing_models import FirmMetrics, FirmStaticPricer, FirmHeuristicPricer
from coeff_utils import set_coeff


def _parse_kv_floats(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad entry '{part}'. Use key=value, e.g. base_fare=2.8")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


class Core:
    def __init__(
        self,
        market_name: str,
        seed: int = 1000,
        choice_mode: str = "parametric",
        model_name: str = "gpt-4o-mini",
        firm1_mode: str = "heuristic",
        firm2_mode: str = "static",
        firm1_static_values: str = "",
        firm2_static_values: str = "",
        total_customers_pool: int = 20000,
    ):
        self.rng = np.random.default_rng(seed)
        self.market = MarketInteraction(city_name=market_name, seed=seed)
        self.market.set_market(market_name)
        self.market_name = market_name

        self.agent_gen = GenerateAgent(seed=seed, total_customers=total_customers_pool)

        # choice model
        self.choice_mode = choice_mode
        if choice_mode == "llm":
            self.choice_model = LLMChoiceModel(model_name=model_name)
        else:
            self.choice_model = ParametricChoiceModel(seed=seed)

        # firms
        self.firm1_mode = firm1_mode
        self.firm2_mode = firm2_mode

        # apply static overrides (if any)
        f1_vals = _parse_kv_floats(firm1_static_values)
        f2_vals = _parse_kv_floats(firm2_static_values)

        for k, v in f1_vals.items():
            set_coeff(self.market.curr_market, self.firm1.overrides, k, v)
        for k, v in f2_vals.items():
            set_coeff(self.market.curr_market, self.firm2.overrides, k, v)

        # last batch summaries (optional; can be logged)
        self.airport_rate_last = self.market.airport_prob
        self.mean_distance_last = 4.0
        
        self.opt_keys = ["base_fare", "per_minute"]
        

    @staticmethod
    def estimate_duration(miles: float, hod: int) -> float:
        mph = 18.0 if (7 <= hod < 10 or 16 <= hod < 19) else 25.0
        return max(5.0, 60.0 * miles / max(8.0, mph))

    def simulate_batch(
        self,
        day_of_week: int,
        weather: str,
        hour: int,
        customers_per_step: int,
    ) -> Tuple[List[Dict[str, Any]], FirmMetrics, FirmMetrics, float, float, float]:
        rows: List[Dict[str, Any]] = []
        firm1 = FirmMetrics()
        firm2 = FirmMetrics()
        gaps: List[float] = []

        airport_count = 0
        dist_sum = 0.0

        for _ in range(customers_per_step):
            profile = self.agent_gen.sample_profile()

            # trip-specific distance (scenario-side)
            travel_distance = round(float(self.rng.exponential(4.0)), 2)

            airport = self.market.sample_airport_flag()
            service = self.market.sample_service()
            airport_count += int(airport)
            dist_sum += float(travel_distance)

            duration = self.estimate_duration(travel_distance, hour)

            ctx = RideContext(
                day_of_week=day_of_week,
                weather=weather,
                hour=hour,
                airport=airport,
                service=service,
            )

            p1 = self.market.quote_price(travel_distance, duration, ctx, overrides=self.firm1.overrides)
            p2 = self.market.quote_price(travel_distance, duration, ctx, overrides=self.firm2.overrides)

            gaps.append(p2 - p1)

            scenario = {
                "City": self.market_name,
                "DistanceMiles": float(travel_distance),
                "DurationMinutes": float(round(duration, 2)),
                "DayOfWeek": int(day_of_week),
                "Hour": int(hour),
                "Weather": str(weather),
                "Airport": bool(airport),
                "Service": str(service),
            }

            choice_res: ChoiceResult = self.choice_model.choose(profile, scenario, p1, p2)
            choice = choice_res.choice

            firm1.total += 1
            firm2.total += 1
            if choice == "Firm1":
                firm1.wins += 1
                firm1.revenue += float(p1)
            else:
                firm2.wins += 1
                firm2.revenue += float(p2)

            rows.append({
                "City": self.market_name,
                "DayOfWeek": day_of_week,
                "Weather": weather,
                "Hour": hour,
                "Airport": airport,
                "Service": service,
                "TravelDistance": travel_distance,
                "Price_Firm1": p1,
                "Price_Firm2": p2,
                "Choice": choice,
                "ReasonCodes": ",".join(choice_res.reason_codes),
                "ShortReason": choice_res.short_reason,
                **profile,
            })

        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        airport_rate = float(airport_count / max(1, customers_per_step))
        mean_dist = float(dist_sum / max(1, customers_per_step))
        return rows, firm1, firm2, mean_gap, airport_rate, mean_dist

    def run(self, days: int, timesteps_per_day: int, customers_per_step: int) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []

        for d in range(days):
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(timesteps_per_day)]

            # day accumulators for logging
            share_sum = 0.0
            revpr_sum = 0.0
            gap_sum = 0.0

            for t in range(timesteps_per_day):
                hour = hours[t]
                base = self.market.curr_market

                # Firm actions (heuristic or static)
                if self.firm1_mode == "heuristic":
                    self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)
                if self.firm2_mode == "heuristic":
                    self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

                rows, m1, m2, mean_gap, airport_rate, mean_dist = self.simulate_batch(
                    day_of_week=day_ctx.day_of_week,
                    weather=day_ctx.weather,
                    hour=hour,
                    customers_per_step=customers_per_step,
                )
                all_rows.extend(rows)

                # update heuristic memory (only if heuristic)
                if self.firm1_mode == "heuristic":
                    self.firm1.update(metrics=m1, price_gap_mean=-mean_gap)  # note sign: Firm1 - Firm2
                if self.firm2_mode == "heuristic":
                    self.firm2.update(metrics=m2, price_gap_mean=mean_gap)   # Firm2 - Firm1

                share_sum += float(m1.share)
                revpr_sum += float(m1.rev_per_request)
                gap_sum += float(mean_gap)

                self.airport_rate_last = airport_rate
                self.mean_distance_last = mean_dist

            # print every ~10% of days
            k = max(1, days // 10)
            if (d + 1) % k == 0 or (d + 1) == 1 or (d + 1) == days:
                avg_share = share_sum / max(1, timesteps_per_day)
                avg_revpr = revpr_sum / max(1, timesteps_per_day)
                avg_gap = gap_sum / max(1, timesteps_per_day)
                print(
                    f"[Day {d+1}/{days}] avg_share(F1)={avg_share:.3f} avg_revPR(F1)=${avg_revpr:.2f} "
                    f"avg_gap(F2-F1)=${avg_gap:.2f}"
                )

        return all_rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to write.")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="Seattle")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--choice_mode", type=str, default="parametric", choices=["parametric", "llm"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=str, default="market_runs.csv")

    parser.add_argument("--firm1_mode", type=str, default="heuristic", choices=["heuristic", "static"])
    parser.add_argument("--firm2_mode", type=str, default="static", choices=["heuristic", "static"])

    parser.add_argument("--firm1_static_values", type=str, default="")
    parser.add_argument("--firm2_static_values", type=str, default="")

    parser.add_argument("--pool", type=int, default=20000, help="Static customer pool size.")

    args = parser.parse_args()

    core = Core(
        market_name=args.market,
        seed=args.seed,
        choice_mode=args.choice_mode,
        model_name=args.model,
        firm1_mode=args.firm1_mode,
        firm2_mode=args.firm2_mode,
        firm1_static_values=args.firm1_static_values,
        firm2_static_values=args.firm2_static_values,
        total_customers_pool=args.pool,
    )

    rows = core.run(days=args.days, timesteps_per_day=args.timesteps, customers_per_step=args.customers)
    _write_csv(args.out, rows)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()