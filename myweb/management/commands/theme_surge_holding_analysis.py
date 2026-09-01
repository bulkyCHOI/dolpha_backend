"""급등테마주 보유기간 관찰 리포트 (P0 — 코드 변경 없이 현황 측정).

실제 체결된 급등테마주 매매(BUY→SELL)를 라운드 단위로 묶어
  · 실제 보유시간 / 실현 손익률 / 청산 사유 분포
  · "만약 당일 청산하지 않고 D+1 ~ D+3 종가에 청산했다면" 반사실(counterfactual) 손익
  · 익일 시가 갭 분포 (오버나이트 리스크 실측)
을 집계한다. 청산 로직을 바꾸기 전에 "며칠 보유가 실제로 유리한가"를 수치로 확인하는 용도.

사용법:
    python manage.py theme_surge_holding_analysis                    # 최근 60일
    python manage.py theme_surge_holding_analysis --days 90 --user alice
    python manage.py theme_surge_holding_analysis --days 90 --csv /tmp/holding.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from django.core.management.base import BaseCommand
from django.utils import timezone as tz

from myweb.models import StockOHLCV, TradeEntry, User

STRATEGY_TYPE = "theme_surge"
FLAT_QTY_EPS = 1e-6           # 보유수량이 이 값 이하이면 청산 완료로 간주
COUNTERFACTUAL_HORIZONS = (0, 1, 2, 3, 5)   # 진입일(D0) 및 이후 실거래일 종가 청산 시나리오
GAP_DOWN_ALERT_PCT = -5.0
GAP_UP_ALERT_PCT = 10.0


@dataclass
class Round:
    """한 종목의 진입(들)부터 전량 청산까지를 묶은 매매 라운드."""

    user: str
    stock_code: str
    stock_name: str
    entry_at: dt.datetime
    exit_at: dt.datetime
    buy_qty: float = 0.0
    buy_amount: float = 0.0
    sell_qty: float = 0.0
    sell_amount: float = 0.0
    exit_reasons: list[str] = field(default_factory=list)

    @property
    def entry_vwap(self) -> float:
        return self.buy_amount / self.buy_qty if self.buy_qty else 0.0

    @property
    def exit_vwap(self) -> float:
        return self.sell_amount / self.sell_qty if self.sell_qty else 0.0

    @property
    def realized_pct(self) -> float:
        base = self.entry_vwap
        return (self.exit_vwap / base - 1.0) * 100.0 if base else 0.0

    @property
    def holding_minutes(self) -> float:
        return (self.exit_at - self.entry_at).total_seconds() / 60.0

    @property
    def is_intraday(self) -> bool:
        return self.entry_at.date() == self.exit_at.date()

    @property
    def calendar_days(self) -> int:
        return (self.exit_at.date() - self.entry_at.date()).days

    @property
    def exit_reason_label(self) -> str:
        if not self.exit_reasons:
            return "(사유 없음)"
        # 마지막 청산 조각의 사유를 대표값으로
        raw = self.exit_reasons[-1]
        return raw.split(" — ")[0].split(" (")[0].split(":")[0].strip()[:40] or "(사유 없음)"


def _amount(entry: TradeEntry) -> float:
    amt = float(entry.filled_amount or 0.0)
    if amt > 0:
        return amt
    return float(entry.filled_price or 0.0) * float(entry.filled_quantity or 0.0)


def _reason(entry: TradeEntry) -> str:
    return (entry.note or entry.get_entry_type_display() or "").strip()


def build_rounds(entries: Iterable[TradeEntry]) -> tuple[list[Round], int]:
    """(user, stock_code, trading_config) 별로 체결을 순회하며 라운드를 만든다.

    반환: (완결된 라운드 목록, 미청산 상태로 남은 라운드 수)
    """
    groups: dict[tuple, list[TradeEntry]] = {}
    for e in entries:
        groups.setdefault((e.user_id, e.stock_code, e.trading_config_id), []).append(e)

    rounds: list[Round] = []
    open_rounds = 0

    for fills in groups.values():
        fills.sort(key=lambda e: e.filled_at or e.ordered_at or e.created_at)
        cur: Round | None = None
        qty = 0.0

        for e in fills:
            fq = float(e.filled_quantity or 0.0)
            if fq <= 0:
                continue
            when = e.filled_at or e.ordered_at or e.created_at

            if e.trade_type == "BUY":
                if cur is None:
                    cur = Round(
                        user=e.user.username if e.user_id else "?",
                        stock_code=e.stock_code,
                        stock_name=e.stock_name,
                        entry_at=when,
                        exit_at=when,
                    )
                cur.buy_qty += fq
                cur.buy_amount += _amount(e)
                qty += fq
            elif e.trade_type == "SELL":
                if cur is None:
                    continue  # 관찰 창 밖에서 진입한 포지션의 매도 — 건너뜀
                sell_fq = min(fq, qty) if qty > 0 else fq
                cur.sell_qty += sell_fq
                cur.sell_amount += _amount(e) * (sell_fq / fq if fq else 1.0)
                cur.exit_at = when
                cur.exit_reasons.append(_reason(e))
                qty -= sell_fq
                if qty <= FLAT_QTY_EPS:
                    if cur.sell_qty > 0:
                        rounds.append(cur)
                    cur = None
                    qty = 0.0

        if cur is not None and cur.buy_qty > 0:
            open_rounds += 1

    return rounds, open_rounds


def _daily_bars(code: str, start: dt.date, horizon: int) -> list[tuple[dt.date, float, float]]:
    """진입일 이후 실거래일(open>0)의 (date, open, close) 를 최대 horizon+1개 반환."""
    end = start + dt.timedelta(days=horizon + 20)  # 주말/휴장 여유
    rows = (
        StockOHLCV.objects.filter(
            code_id=code, date__gte=start, date__lte=end, open__gt=0
        )
        .order_by("date")
        .values_list("date", "open", "close")[: horizon + 1]
    )
    return [(d, float(o), float(c)) for d, o, c in rows]


@dataclass
class Counterfactual:
    horizon_pct: dict[int, list[float]] = field(
        default_factory=lambda: {h: [] for h in COUNTERFACTUAL_HORIZONS}
    )
    overnight_gap_pct: list[float] = field(default_factory=list)
    missing_price: int = 0


def compute_counterfactual(rounds: list[Round]) -> Counterfactual:
    """당일 청산된 라운드에 대해 D0~D+N 종가 청산 시 손익률을 계산."""
    cf = Counterfactual()
    for r in rounds:
        if not r.is_intraday or r.entry_vwap <= 0:
            continue
        bars = _daily_bars(r.stock_code, r.entry_at.date(), max(COUNTERFACTUAL_HORIZONS))
        if not bars:
            cf.missing_price += 1
            continue
        base = r.entry_vwap
        for h in COUNTERFACTUAL_HORIZONS:
            if h < len(bars):
                close_h = bars[h][2]
                cf.horizon_pct[h].append((close_h / base - 1.0) * 100.0)
        if len(bars) >= 2:
            close_d0, open_d1 = bars[0][2], bars[1][1]
            if close_d0 > 0:
                cf.overnight_gap_pct.append((open_d1 / close_d0 - 1.0) * 100.0)
    return cf


def _stats(values: list[float]) -> str:
    if not values:
        return "표본 없음"
    win = sum(1 for v in values if v > 0) / len(values) * 100.0
    return (
        f"n={len(values):>4}  평균 {statistics.mean(values):+6.2f}%  "
        f"중앙 {statistics.median(values):+6.2f}%  승률 {win:5.1f}%  "
        f"최저 {min(values):+6.2f}%  최고 {max(values):+6.2f}%"
    )


class Command(BaseCommand):
    help = "급등테마주 실제 매매의 보유기간·손익과 며칠 보유 시 반사실 손익을 집계합니다."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--days", type=int, default=60, help="집계 기간(일)")
        parser.add_argument("--user", type=str, default=None, help="특정 사용자만")
        parser.add_argument("--csv", type=str, default=None, help="라운드 상세 CSV 경로")

    def handle(self, *args, **options) -> None:
        days: int = options["days"]
        username: str | None = options["user"]
        since = tz.now() - dt.timedelta(days=days)

        qs = (
            TradeEntry.objects.filter(
                trading_config__strategy_type=STRATEGY_TYPE,
                trade_type__in=("BUY", "SELL"),
                filled_quantity__gt=0,
                created_at__gte=since,
            )
            .select_related("user")
        )
        if username:
            try:
                qs = qs.filter(user=User.objects.get(username=username))
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"사용자 '{username}' 없음"))
                return

        entries = list(qs)
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"\n=== 급등테마주 보유기간 관찰 ({since.date()} ~ 오늘, {days}일) ===\n"
            )
        )
        if not entries:
            self.stdout.write(
                "체결 기록 없음 — trading_config.strategy_type='theme_surge' 매매가 있는지, "
                "--user 지정이 맞는지 확인하세요.\n"
            )
            return

        rounds, open_rounds = build_rounds(entries)
        if not rounds:
            self.stdout.write(f"완결된 매매 라운드 없음 (미청산 {open_rounds}건)\n")
            return

        rounds.sort(key=lambda r: r.entry_at)
        self._section_actual(rounds, open_rounds)
        self._section_exit_reasons(rounds)
        self._section_counterfactual(rounds)
        self._section_params()

        if options["csv"]:
            self._write_csv(options["csv"], rounds)
            self.stdout.write(self.style.SUCCESS(f"\nCSV 저장: {options['csv']}\n"))

    # ── 실제 매매 결과 ────────────────────────────────────────────
    def _section_actual(self, rounds: list[Round], open_rounds: int) -> None:
        intraday = [r for r in rounds if r.is_intraday]
        overnight = [r for r in rounds if not r.is_intraday]

        self.stdout.write(self.style.HTTP_INFO("① 실제 매매 결과"))
        self.stdout.write(
            f"   완결 라운드          : {len(rounds)}건  (미청산 {open_rounds}건)"
        )
        self.stdout.write(
            f"   기간                : {rounds[0].entry_at.date()} ~ {rounds[-1].exit_at.date()}"
        )
        self.stdout.write(
            f"   전체 손익           : {_stats([r.realized_pct for r in rounds])}"
        )
        self.stdout.write(
            f"   당일 청산 ({len(intraday):>3}건) : {_stats([r.realized_pct for r in intraday])}"
        )
        self.stdout.write(
            f"   익일+ 보유 ({len(overnight):>3}건) : {_stats([r.realized_pct for r in overnight])}"
        )
        if overnight:
            buckets: Counter[str] = Counter()
            for r in overnight:
                key = f"{r.calendar_days}일 경과" if r.calendar_days <= 5 else "6일+"
                buckets[key] += 1
            dist = "  ".join(f"{k} {v}건" for k, v in sorted(buckets.items()))
            self.stdout.write(f"     └ 보유일 분포: {dist}")
        med_min = statistics.median([r.holding_minutes for r in intraday]) if intraday else 0
        self.stdout.write(f"   당일 라운드 보유시간 중앙값 : {med_min:.0f}분\n")

    # ── 청산 사유 ────────────────────────────────────────────────
    def _section_exit_reasons(self, rounds: list[Round]) -> None:
        by_reason: dict[str, list[float]] = {}
        for r in rounds:
            by_reason.setdefault(r.exit_reason_label, []).append(r.realized_pct)

        self.stdout.write(self.style.HTTP_INFO("② 청산 사유별 손익"))
        for reason, pcts in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            win = sum(1 for v in pcts if v > 0) / len(pcts) * 100.0
            self.stdout.write(
                f"   {reason:<24} {len(pcts):>4}건  평균 {statistics.mean(pcts):+6.2f}%  승률 {win:5.1f}%"
            )
        self.stdout.write("")

    # ── 반사실: 며칠 보유 ────────────────────────────────────────
    def _section_counterfactual(self, rounds: list[Round]) -> None:
        cf = compute_counterfactual(rounds)
        intraday = [r for r in rounds if r.is_intraday and r.entry_vwap > 0]
        actual = [r.realized_pct for r in intraday]

        self.stdout.write(
            self.style.HTTP_INFO("③ 반사실 — 당일 청산분을 D+N 종가까지 보유했다면")
        )
        if cf.missing_price:
            self.stdout.write(
                f"   (일봉 데이터 없어 제외된 라운드: {cf.missing_price}건)"
            )
        self.stdout.write(f"   실제 당일청산         : {_stats(actual)}")
        for h in COUNTERFACTUAL_HORIZONS:
            label = "D0 종가(당일 종가청산)" if h == 0 else f"D+{h} 종가"
            vals = cf.horizon_pct[h]
            line = f"   {label:<20} : {_stats(vals)}"
            if vals and actual:
                delta = statistics.mean(vals) - statistics.mean(actual)
                line += f"   (실제 대비 {delta:+.2f}%p)"
            self.stdout.write(line)

        gaps = cf.overnight_gap_pct
        self.stdout.write("\n   익일 시가 갭 (D0 종가 → D+1 시가):")
        if gaps:
            gaps_sorted = sorted(gaps)
            p10 = gaps_sorted[int(len(gaps) * 0.10)]
            p90 = gaps_sorted[int(len(gaps) * 0.90)]
            down = sum(1 for g in gaps if g <= GAP_DOWN_ALERT_PCT)
            up = sum(1 for g in gaps if g >= GAP_UP_ALERT_PCT)
            self.stdout.write(
                f"     n={len(gaps)}  평균 {statistics.mean(gaps):+.2f}%  "
                f"중앙 {statistics.median(gaps):+.2f}%  P10 {p10:+.2f}%  P90 {p90:+.2f}%"
            )
            self.stdout.write(
                f"     {GAP_DOWN_ALERT_PCT:+.0f}% 이하 갭다운 {down}건  "
                f"{GAP_UP_ALERT_PCT:+.0f}% 이상 갭업 {up}건"
            )
        else:
            self.stdout.write("     표본 없음")
        self.stdout.write("")

    def _section_params(self) -> None:
        from dolpha.theme_surge import config as cfg

        self.stdout.write(self.style.HTTP_INFO("현재 청산 파라미터 (참고)"))
        rows = [
            ("DEFAULT_FORCE_EXIT_TIME", cfg.DEFAULT_FORCE_EXIT_TIME, "당일 강제청산 시각"),
            ("DEFAULT_EXIT_STAGES", cfg.DEFAULT_EXIT_STAGES, "분할 익절 차수(T배수, 비율)"),
            ("DEFAULT_TRAILING_START_T", cfg.DEFAULT_TRAILING_START_T, "트레일링 시작 T배수"),
            ("DEFAULT_TRAILING_BAR_UNIT", cfg.DEFAULT_TRAILING_BAR_UNIT, "트레일링 봉 단위"),
        ]
        for name, val, desc in rows:
            self.stdout.write(f"   {name:<26} = {str(val):<26} {desc}")
        self.stdout.write("")

    def _write_csv(self, path: str, rounds: list[Round]) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "user", "stock_code", "stock_name",
                    "entry_at", "exit_at", "holding_minutes", "calendar_days",
                    "is_intraday", "entry_vwap", "exit_vwap", "realized_pct",
                    "exit_reason",
                ]
            )
            for r in rounds:
                w.writerow(
                    [
                        r.user, r.stock_code, r.stock_name,
                        r.entry_at.isoformat(), r.exit_at.isoformat(),
                        f"{r.holding_minutes:.0f}", r.calendar_days,
                        r.is_intraday, f"{r.entry_vwap:.2f}", f"{r.exit_vwap:.2f}",
                        f"{r.realized_pct:.3f}", r.exit_reason_label,
                    ]
                )
