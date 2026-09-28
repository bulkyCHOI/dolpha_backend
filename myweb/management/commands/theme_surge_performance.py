"""급등테마주 성과 리포트 — R 배수·MFE·조건별 분해·섀도 필터 검증 (docs/13 Phase 0).

theme_surge_holding_analysis 가 '보유기간'을 본다면 이 리포트는 '어떤 거래가 돈을 벌고
잃었나'를 본다. 체결(BUY→SELL) 라운드에 진입 시그널(ThemeEntrySignal)과 1분봉을 붙여

  · 기대값: 세전/비용 차감 손익률, R 배수, 손익비
  · 손절 품질: 손절 R, 갭 손절, 손실 거래의 MFE(돌파 실패형인지)
  · 조건별 분해: 당일 몇 번째 거래, 직전 거래 손실 여부, 테마 등락률, 거래량비, 추격폭, 진입 시각, 이월
  · 섀도 필터 검증: gate_flags.quality 에 '걸렸을' 거래 vs 통과 거래 성과
  · 게이트 차단 신호: 서킷브레이커·진입 마감으로 막힌 신호의 당일 종가 반사실 R

사용법:
    python manage.py theme_surge_performance                  # 최근 60일
    python manage.py theme_surge_performance --days 30 --user alice --cost 0.25
    python manage.py theme_surge_performance --csv /tmp/perf.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass

from django.core.management.base import BaseCommand
from django.utils import timezone as tz

from myweb.management.commands.theme_surge_holding_analysis import Round, build_rounds
from myweb.models import StockMinuteOhlcv, ThemeEntrySignal, ThemeSnapshot, TradeEntry, User

STRATEGY_TYPE = "theme_surge"
DEFAULT_COST_PCT = 0.23          # 왕복 거래비용(세금+수수료) 가정치(%)
SIGNAL_MATCH_WINDOW_MIN = 3      # 매수 체결 시각과 진입 시그널 시각의 허용 오차(분)
GATE_REASON_PREFIXES = ("신규 진입 마감", "서킷브레이커", "진입 품질 필터")


@dataclass(frozen=True)
class RoundRow:
    """리포트 한 줄 = 한 라운드 + 진입 맥락."""

    user: str
    code: str
    name: str
    entry_at: str
    exit_at: str
    pnl_pct: float
    r: float | None
    risk_pct: float | None
    mfe_r: float | None           # 진입~청산 사이 최대 유리 이동(R)
    close_r: float | None         # 진입일 마지막 1분봉 종가 기준 평가손익(R)
    chase_pct: float | None       # 평균 체결가의 전고점 대비 괴리(%)
    volume_ratio: float | None
    theme: str
    theme_fluct: float | None
    hour: int
    day_seq: int                  # 당일 몇 번째 라운드
    after_loss: bool | None       # 직전(이미 청산된) 당일 라운드가 손실이었나
    overnight: bool
    exit_reason: str
    shadow_failed: str            # 섀도 필터에 걸린 항목(콤마 구분), 기록 없으면 "-"
    amount: float


def _executed_signal(r: Round) -> ThemeEntrySignal | None:
    return (
        ThemeEntrySignal.objects.filter(
            stock_code=r.stock_code,
            date=tz.localtime(r.entry_at).date(),
            executed=True,
            checked_at__lte=r.entry_at + dt.timedelta(minutes=SIGNAL_MATCH_WINDOW_MIN),
        )
        .order_by("-checked_at")
        .first()
    )


def _bars(code: str, start: dt.datetime, end: dt.datetime) -> list[tuple]:
    return list(
        StockMinuteOhlcv.objects.filter(
            stock_code=code, bar_datetime__gt=start, bar_datetime__lte=end
        )
        .order_by("bar_datetime")
        .values_list("bar_datetime", "high", "close")
    )


def _theme_fluct(sig: ThemeEntrySignal) -> float | None:
    """시그널에 기록된 값 우선, 없으면(Phase 0 이전 데이터) 스냅샷에서 역산."""
    if sig.theme_fluctuation is not None:
        return sig.theme_fluctuation
    snap = (
        ThemeSnapshot.objects.filter(
            date=sig.date, tics_id=sig.tics_id,
            slot_time__lte=tz.localtime(sig.checked_at).time(),
        )
        .order_by("-slot_time")
        .values_list("fluctuation_rate", flat=True)
        .first()
    )
    return float(snap) if snap is not None else None


def _to_row(r: Round, day_seq: int, after_loss: bool | None) -> RoundRow:
    sig = _executed_signal(r)
    entry = tz.localtime(r.entry_at)
    stop = sig.pullback_low if sig else None
    risk = (r.entry_vwap - stop) if stop else None
    has_risk = bool(risk and risk > 0)

    mfe_r = close_r = None
    if has_risk:
        day_end = entry.replace(hour=15, minute=30, second=0, microsecond=0)
        bars = _bars(r.stock_code, r.entry_at, day_end)
        held = [b for b in bars if b[0] <= r.exit_at] or bars[:1]
        if held:
            mfe_r = (max(b[1] for b in held) - r.entry_vwap) / risk
        if bars:
            close_r = (bars[-1][2] - r.entry_vwap) / risk

    shadow = "-"
    if sig and sig.gate_flags:
        shadow = ",".join((sig.gate_flags.get("quality") or {}).get("failed") or []) or "통과"

    return RoundRow(
        user=r.user, code=r.stock_code, name=r.stock_name,
        entry_at=entry.strftime("%m-%d %H:%M"),
        exit_at=tz.localtime(r.exit_at).strftime("%m-%d %H:%M"),
        pnl_pct=r.realized_pct,
        r=(r.exit_vwap - r.entry_vwap) / risk if has_risk else None,
        risk_pct=risk / r.entry_vwap * 100.0 if has_risk else None,
        mfe_r=mfe_r, close_r=close_r,
        chase_pct=(r.entry_vwap / sig.prev_high - 1.0) * 100.0 if sig and sig.prev_high else None,
        volume_ratio=sig.volume_ratio if sig else None,
        theme=sig.theme_name if sig else "",
        theme_fluct=_theme_fluct(sig) if sig else None,
        hour=entry.hour, day_seq=day_seq, after_loss=after_loss,
        overnight=not r.is_intraday, exit_reason=r.exit_reason_label,
        shadow_failed=shadow, amount=r.buy_amount,
    )


def build_rows(rounds: list[Round]) -> list[RoundRow]:
    """라운드에 당일 순번·직전 손실 여부를 붙여 RoundRow 로 변환한다."""
    by_day: dict[tuple, list[Round]] = defaultdict(list)
    for r in rounds:
        by_day[(r.user, tz.localtime(r.entry_at).date())].append(r)

    rows: list[RoundRow] = []
    for day_rounds in by_day.values():
        day_rounds.sort(key=lambda r: r.entry_at)
        for i, r in enumerate(day_rounds):
            closed = [p for p in day_rounds[:i] if p.exit_at <= r.entry_at]
            after_loss = (closed[-1].realized_pct <= 0) if closed else None
            rows.append(_to_row(r, i + 1, after_loss))
    return sorted(rows, key=lambda x: x.entry_at)


class Command(BaseCommand):
    help = "급등테마주 성과 리포트 (R·MFE·조건별 분해·섀도 필터 검증)"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=60, help="조회 기간(일), 기본 60")
        parser.add_argument("--user", type=str, default=None, help="특정 유저만 (username)")
        parser.add_argument("--cost", type=float, default=DEFAULT_COST_PCT,
                            help=f"왕복 거래비용 가정(%%), 기본 {DEFAULT_COST_PCT}")
        parser.add_argument("--csv", type=str, default=None, help="라운드별 상세 CSV 경로")

    def handle(self, *args, **options) -> None:
        since = tz.now() - dt.timedelta(days=options["days"])
        qs = TradeEntry.objects.filter(
            trading_config__strategy_type=STRATEGY_TYPE,
            trade_type__in=("BUY", "SELL"), filled_quantity__gt=0, created_at__gte=since,
        ).select_related("user")
        signals = ThemeEntrySignal.objects.filter(checked_at__gte=since)
        if options["user"]:
            try:
                user = User.objects.get(username=options["user"])
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"사용자 '{options['user']}' 없음"))
                return
            qs, signals = qs.filter(user=user), signals.filter(user=user)

        rounds, open_rounds = build_rounds(qs)
        rows = build_rows(rounds)
        self.cost = options["cost"]
        out = self.stdout.write

        out(f"\n=== 급등테마주 성과 리포트 (최근 {options['days']}일, 비용 {self.cost:g}% 차감) ===")
        if not rows:
            out("  완결 라운드 없음")
            return
        out(f"완결 라운드 {len(rows)}건 · 미청산 {open_rounds}건 · "
            f"{rows[0].entry_at} ~ {rows[-1].entry_at}")

        self._summary(rows)
        self._breakdowns(rows)
        self._gate_blocked(signals)

        if options["csv"]:
            with open(options["csv"], "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(r) for r in rows)
            out(f"\nCSV 저장: {options['csv']}")

    # ── 출력 ────────────────────────────────────────────────
    def _line(self, label: str, rows: list[RoundRow]) -> None:
        if not rows:
            self.stdout.write(f"   {label:30s} 표본 없음")
            return
        pnl = [r.pnl_pct for r in rows]
        rs = [r.r for r in rows if r.r is not None]
        wins = sum(1 for p in pnl if p > 0)
        avg_r = f"{statistics.mean(rs):+.2f}R" if rs else "  -  "
        self.stdout.write(
            f"   {label:30s} n={len(rows):3d}  세전 {statistics.mean(pnl):+.2f}%"
            f"  순 {statistics.mean(pnl) - self.cost:+.2f}%  {avg_r}"
            f"  승률 {wins / len(rows) * 100:3.0f}%  중앙 {statistics.median(pnl):+.2f}%"
        )

    def _summary(self, rows: list[RoundRow]) -> None:
        out = self.stdout.write
        wins = [r for r in rows if r.pnl_pct > 0]
        losses = [r for r in rows if r.pnl_pct <= 0]
        out("\n① 기대값")
        self._line("전체", rows)
        if wins and losses:
            avg_w = statistics.mean(r.pnl_pct for r in wins)
            avg_l = statistics.mean(r.pnl_pct for r in losses)
            out(f"   평균 수익 {avg_w:+.2f}% / 평균 손실 {avg_l:+.2f}% → 손익비 {-avg_w / avg_l:.2f}")
        risks = [r.risk_pct for r in rows if r.risk_pct]
        if risks:
            out(f"   평균 손절폭(1R) {statistics.mean(risks):.2f}%")

        out("\n② 손절 품질")
        stops = [r for r in rows if r.exit_reason.startswith("손절") and r.r is not None]
        if stops:
            gap = [r for r in stops if r.r < -1.2]
            out(f"   손절 {len(stops)}건 평균 {statistics.mean(r.r for r in stops):+.2f}R"
                f" · −1.2R 초과 미끄러짐 {len(gap)}건 "
                + ", ".join(f"{r.name}({r.r:+.2f}R)" for r in gap))
        with_mfe = [r for r in losses if r.mfe_r is not None]
        if with_mfe:
            ran = sum(1 for r in with_mfe if r.mfe_r >= 0.5)
            out(f"   손실 거래 중 +0.5R 이상 갔던 거래 {ran}/{len(with_mfe)}"
                "  (낮을수록 '돌파 실패형' — 청산보다 진입 필터가 레버)")

    def _breakdowns(self, rows: list[RoundRow]) -> None:
        out = self.stdout.write
        groups = [
            ("③ 당일 순번", [
                ("당일 첫 거래", lambda r: r.day_seq == 1),
                ("2번째 이후", lambda r: r.day_seq > 1),
                ("  └ 직전 거래 손실 후", lambda r: r.after_loss is True),
                ("  └ 직전 거래 수익 후", lambda r: r.after_loss is False),
            ]),
            ("④ 진입 시점 테마 등락률", [
                ("< 4%", lambda r: r.theme_fluct is not None and r.theme_fluct < 4),
                ("4 ~ 6%", lambda r: r.theme_fluct is not None and 4 <= r.theme_fluct < 6),
                (">= 6%", lambda r: r.theme_fluct is not None and r.theme_fluct >= 6),
            ]),
            ("⑤ 돌파봉 거래량비", [
                ("< 2배", lambda r: r.volume_ratio is not None and r.volume_ratio < 2),
                ("2 ~ 4배", lambda r: r.volume_ratio is not None and 2 <= r.volume_ratio <= 4),
                ("> 4배", lambda r: r.volume_ratio is not None and r.volume_ratio > 4),
            ]),
            ("⑥ 체결가의 전고점 대비 괴리", [
                ("< 0.5%", lambda r: r.chase_pct is not None and r.chase_pct < 0.5),
                ("0.5 ~ 2%", lambda r: r.chase_pct is not None and 0.5 <= r.chase_pct <= 2),
                ("> 2%", lambda r: r.chase_pct is not None and r.chase_pct > 2),
            ]),
            ("⑦ 진입 시각", [
                ("09시", lambda r: r.hour == 9),
                ("10~11시", lambda r: 10 <= r.hour <= 11),
                ("12~13시", lambda r: 12 <= r.hour <= 13),
                ("14시 이후", lambda r: r.hour >= 14),
            ]),
            ("⑧ 오버나이트 (진입일 종가 평가손익 기준)", [
                ("당일 청산", lambda r: not r.overnight),
                ("이월 · 종가 수익", lambda r: r.overnight and r.close_r is not None and r.close_r > 0),
                ("이월 · 종가 손실", lambda r: r.overnight and r.close_r is not None and r.close_r <= 0),
            ]),
            ("⑨ 섀도 필터 (gate_flags 기록분만)", [
                ("필터 모두 통과", lambda r: r.shadow_failed == "통과"),
                ("테마 등락률 미달", lambda r: "theme_min_fluctuation" in r.shadow_failed),
                ("거래량 과열", lambda r: "breakout_volume_max" in r.shadow_failed),
                ("추격", lambda r: "chase_max" in r.shadow_failed),
            ]),
        ]
        for title, buckets in groups:
            out(f"\n{title}")
            for label, pred in buckets:
                self._line(label, [r for r in rows if pred(r)])

    def _gate_blocked(self, signals) -> None:
        """게이트에 막힌 신호(종목·일자별 첫 건)의 당일 종가 반사실 R."""
        out = self.stdout.write
        out("\n⑩ 게이트 차단 신호 (막지 않았다면 — 당일 종가 청산 가정)")
        blocked = signals.filter(passed=False, has_pullback=True, has_breakout=True).order_by("checked_at")
        firsts: dict[tuple, ThemeEntrySignal] = {}
        for sig in blocked:
            if sig.reason.startswith(GATE_REASON_PREFIXES):
                firsts.setdefault((sig.user_id, sig.date, sig.stock_code), sig)
        if not firsts:
            out("   기록 없음 (게이트 도입 이후 데이터가 쌓이면 표시)")
            return

        by_kind: dict[str, list[float]] = defaultdict(list)
        for sig in firsts.values():
            risk = sig.price - (sig.pullback_low or 0)
            if not sig.pullback_low or risk <= 0:
                continue
            at = tz.localtime(sig.checked_at)
            bars = _bars(sig.stock_code, sig.checked_at, at.replace(hour=15, minute=30, second=0))
            if bars:
                kind = next(p for p in GATE_REASON_PREFIXES if sig.reason.startswith(p))
                by_kind[kind].append((bars[-1][2] - sig.price) / risk)
        for kind, rs in by_kind.items():
            wins = sum(1 for x in rs if x > 0)
            out(f"   {kind:14s} n={len(rs):3d}  평균 {statistics.mean(rs):+.2f}R"
                f"  플러스 {wins}/{len(rs)}  (음수면 게이트가 손실을 막은 것)")
