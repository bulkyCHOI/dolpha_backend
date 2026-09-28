"""급등테마주 신규 진입 리스크 게이트 + 진입 품질 섀도 필터 (docs/13 Phase 0·1).

패턴 판정(entry.py)이 '진입 신호가 떴는가'를 본다면, 이 모듈은 '지금 그 신호를
받아도 되는가'를 본다. 실측 39라운드에서 손실은 대부분 진입 직후의 돌파 실패였고,
그 비중은 당일 첫 손실 이후·14시 이후·약한 테마에서 컸다.

    리스크 게이트 (항상 적용, 유저 설정으로 조정)
      - 신규 진입 마감 시각
      - 당일 손실 라운드 수 한도 (서킷브레이커)
      - 당일 실현손실 합계 한도 (계좌 대비 %)

    진입 품질 필터 (config.ENTRY_FILTER_MODE — 기본 "shadow")
      - 진입 시점 테마 등락률 하한
      - 돌파봉 거래량비 상한
      - 추격 상한 (현재가의 전고점 대비 괴리)

DB 조회는 하지 않는 순수 함수만 둔다. 집계는 TradingEngine 이 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as time_cls

from .config import (
    DEFAULT_DAILY_MAX_LOSS_PCT,
    DEFAULT_DAILY_MAX_LOSSES,
    DEFAULT_ENTRY_CUTOFF_TIME,
    ENTRY_FILTER_MODE,
    FILTER_BREAKOUT_VOLUME_RATIO_MAX,
    FILTER_CHASE_MAX_PCT,
    FILTER_THEME_MIN_FLUCTUATION_PCT,
)

FILTER_MODES = ("shadow", "enforce")


@dataclass(frozen=True)
class EntryGateSettings:
    """유저의 신규 진입 게이트 설정 (TradingDefaults 에서 읽어 정규화)."""

    entry_cutoff: time_cls
    daily_max_losses: int        # 0 이면 미사용
    daily_max_loss_pct: float    # 0 이면 미사용


@dataclass(frozen=True)
class DailyStats:
    """당일 급등테마주 매매 집계."""

    rounds_started: int     # 오늘 첫 매수가 체결된 라운드 수 (이월분 제외)
    realized_losses: int    # 오늘 손실로 끝난 라운드 수 (이월분 포함)
    realized_pnl: float     # 오늘 실현손익 합계(원)


@dataclass(frozen=True)
class GateResult:
    """게이트·필터 판정 결과."""

    blocked: bool                 # 진입을 막는가
    reason: str                   # 막았다면 그 사유 (아니면 "")
    flags: dict                   # ThemeEntrySignal.gate_flags 에 그대로 저장


def load_entry_gate_settings(defaults) -> EntryGateSettings:
    """TradingDefaults 에서 게이트 설정을 읽는다. 없거나 비어 있으면 기본값."""
    if defaults is None:
        return EntryGateSettings(
            entry_cutoff=DEFAULT_ENTRY_CUTOFF_TIME,
            daily_max_losses=DEFAULT_DAILY_MAX_LOSSES,
            daily_max_loss_pct=DEFAULT_DAILY_MAX_LOSS_PCT,
        )

    max_losses = getattr(defaults, "theme_surge_daily_max_losses", None)
    max_loss_pct = getattr(defaults, "theme_surge_daily_max_loss_pct", None)
    return EntryGateSettings(
        entry_cutoff=getattr(defaults, "theme_surge_entry_cutoff", None)
        or DEFAULT_ENTRY_CUTOFF_TIME,
        daily_max_losses=max(0, int(DEFAULT_DAILY_MAX_LOSSES if max_losses is None else max_losses)),
        daily_max_loss_pct=max(
            0.0, float(DEFAULT_DAILY_MAX_LOSS_PCT if max_loss_pct is None else max_loss_pct)
        ),
    )


def check_risk_gates(
    settings: EntryGateSettings,
    now_time: time_cls,
    stats: DailyStats,
    confirmed_capital: float | None,
) -> GateResult:
    """신규 진입 리스크 게이트를 판정한다.

    confirmed_capital 이 없으면(계좌 조회 실패) 손실 % 한도만 건너뛴다 —
    건수 한도와 시각 한도는 그대로 적용된다.
    """
    loss_pct = None
    if confirmed_capital and confirmed_capital > 0:
        loss_pct = -stats.realized_pnl / confirmed_capital * 100.0

    flags = {
        "entry_cutoff": settings.entry_cutoff.strftime("%H:%M"),
        "realized_losses": stats.realized_losses,
        "daily_max_losses": settings.daily_max_losses,
        "realized_pnl": round(stats.realized_pnl),
        "realized_loss_pct": round(loss_pct, 3) if loss_pct is not None else None,
        "daily_max_loss_pct": settings.daily_max_loss_pct,
    }

    reason = ""
    if now_time >= settings.entry_cutoff:
        reason = f"신규 진입 마감 ({settings.entry_cutoff:%H:%M} 이후)"
    elif settings.daily_max_losses and stats.realized_losses >= settings.daily_max_losses:
        reason = (
            f"서킷브레이커 — 당일 손실 {stats.realized_losses}회"
            f" ≥ 한도 {settings.daily_max_losses}회"
        )
    elif (
        settings.daily_max_loss_pct
        and loss_pct is not None
        and loss_pct >= settings.daily_max_loss_pct
    ):
        reason = (
            f"서킷브레이커 — 당일 실현손실 {loss_pct:.2f}%"
            f" ≥ 한도 {settings.daily_max_loss_pct:g}%"
        )

    return GateResult(blocked=bool(reason), reason=reason, flags={**flags, "blocked_by": reason})


def evaluate_quality_filters(
    theme_fluctuation: float | None,
    volume_ratio: float | None,
    current_price: float,
    prev_high: float | None,
    mode: str = ENTRY_FILTER_MODE,
) -> GateResult:
    """진입 품질 필터를 판정한다. 값이 없는 항목은 판정 불가(None)로 두고 통과시킨다.

    mode="shadow" 면 걸려도 blocked=False — 기록만 남긴다.
    """
    chase_pct = None
    if prev_high and prev_high > 0 and current_price > 0:
        chase_pct = (current_price / prev_high - 1.0) * 100.0

    checks = {
        "theme_min_fluctuation": _check(
            theme_fluctuation, FILTER_THEME_MIN_FLUCTUATION_PCT,
            lambda v, lim: v >= lim,
        ),
        "breakout_volume_max": _check(
            volume_ratio, FILTER_BREAKOUT_VOLUME_RATIO_MAX,
            lambda v, lim: v <= lim,
        ),
        "chase_max": _check(
            chase_pct, FILTER_CHASE_MAX_PCT,
            lambda v, lim: v <= lim,
        ),
    }
    failed = [name for name, c in checks.items() if c["pass"] is False]
    mode = mode if mode in FILTER_MODES else "shadow"

    reason = ""
    if failed and mode == "enforce":
        reason = "진입 품질 필터 — " + ", ".join(_FILTER_LABELS[n](checks[n]) for n in failed)

    return GateResult(
        blocked=bool(reason),
        reason=reason,
        flags={"mode": mode, "checks": checks, "failed": failed},
    )


_FILTER_LABELS = {
    "theme_min_fluctuation": lambda c: f"테마 등락률 {c['value']:.2f}% < {c['limit']:g}%",
    "breakout_volume_max": lambda c: f"돌파 거래량비 {c['value']:.2f} > {c['limit']:g}",
    "chase_max": lambda c: f"전고점 대비 +{c['value']:.2f}% > {c['limit']:g}% (추격)",
}


def _check(value, limit, ok) -> dict:
    if value is None:
        return {"value": None, "limit": limit, "pass": None}
    return {"value": round(float(value), 3), "limit": limit, "pass": bool(ok(float(value), limit))}
