"""오버나이트 보유 판정 — 강제청산 시각의 수급 4조건.

강제청산 시각에 아래 조건 중 유저가 지정한 개수(min_count) 이상이 충족되면
잔량을 익일로 이월하고, 미달이면 예정대로 전량 청산한다.

KIS 매매동향 API는 장중(~15:30 KST)에만 조회 가능하므로:
  - 장중(강제청산 시각)에는 `evaluate_overnight_signal` 이 실시간 조회로 판정하고,
  - 장 마감 후 확정(`exit_finalizer`)에는 15:29 에 저장해 둔 `InvestorFlowSnapshot`
    을 넘겨 `evaluate_overnight_signal_from_snapshot` 이 같은 규칙으로 판정한다.
조회가 하나라도 실패하거나 응답이 비어 있으면 available=False 로 반환하며,
호출부는 이를 미충족으로 처리한다(불확실하면 청산 = 보수적).

조건 키:
    foreign       — 외국계 회원사 합계 순매수 > 0
    institution   — 당일 기관 순매수 > 0 (장중엔 잠정 집계라 0으로 나올 수 있음)
    program       — 당일 프로그램 누적 순매수 > 0
    shinhan_top5  — 신한투자증권이 매수 상위 5개사 안에 있음

이 모듈은 KIS 조회와 순수 판정만 담당하고, 주문 실행은 TradingEngine 이 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls, datetime
from types import MappingProxyType
from typing import Mapping, Sequence

from .config import (
    DEFAULT_OVERNIGHT_MIN_COUNT,
    OVERNIGHT_CONDITION_KEYS,
)

CONDITION_LABELS: dict[str, str] = {
    "foreign": "외국인 순매수",
    "institution": "기관 순매수",
    "program": "프로그램 순매수",
    "shinhan_top5": "신한증권 매수상위5",
}

# 회원사명 표기가 시기·엔드포인트에 따라 다르다. KIS 회원사 응답은 '신한증권'으로,
# 정식 사명은 '신한투자증권'(2021년 '신한금융투자'에서 변경). 셋 다 확인한다.
_SHINHAN_ALIASES = ("신한투자증권", "신한금융투자", "신한증권")


@dataclass(frozen=True)
class OvernightSignal:
    """오버나이트 이월 판정 결과."""

    available: bool                 # 평가한 모든 조건을 오류 없이 조회했는가
    met: Mapping[str, bool]         # {조건키: 충족 여부} — 평가한 조건만 담는다 (불변)
    met_count: int                  # 충족된 조건 수
    required: int                   # 이월에 필요한 충족 수
    should_hold: bool               # 익일 이월 여부 (available 이고 met_count >= required)
    detail: str                     # 사람이 읽을 수 있는 요약


def normalize_conditions(conditions: object) -> list[str]:
    """유저 설정에서 유효한 조건 키만 원래 순서대로(중복 제거) 추린다.

    - None / 리스트가 아님  → 미설정으로 보고 전체 조건 반환
    - 빈 리스트             → 명시적 '조건 없음' 으로 그대로 [] 반환
      (호출부는 조건이 없으면 이월하지 않는다)
    """
    if conditions is None or not isinstance(conditions, (list, tuple)):
        return list(OVERNIGHT_CONDITION_KEYS)
    seen: set[str] = set()
    return [
        k for k in conditions
        if k in OVERNIGHT_CONDITION_KEYS and not (k in seen or seen.add(k))
    ]


def _today_yyyymmdd() -> str:
    from pytz import timezone as pytz_tz

    return datetime.now(pytz_tz("Asia/Seoul")).strftime("%Y%m%d")


def _institution_net(rows: list[dict], target: str) -> int | None:
    """target(YYYYMMDD) 일자 기관 순매수(주). 해당 행이 없으면 None."""
    for row in rows or []:
        if str(row.get("date", "")) == target:
            return int(row.get("orgn_qty", 0) or 0)
    return None


def _program_net_today(rows: list[dict]) -> int | None:
    """당일 프로그램 누적 순매수(주).

    comp-program-trade-today 는 당일 시간대별 데이터만 반환하고
    ntby_qty(whol_smtn_ntby_qty) 는 그 시점까지의 누적 합계이므로 시간이 가장
    늦은 행의 값을 취한다. 행이 없으면 None."""
    if not rows:
        return None
    latest = max(rows, key=lambda r: str(r.get("time", "")))
    return int(latest.get("ntby_qty", 0) or 0)


def _member_ok(member: object) -> bool:
    return isinstance(member, dict) and "buy" in member


def _evaluate(
    keys: list[str],
    required_n: int,
    *,
    member: dict | None,
    member_error: str | None,
    investor_rows: list[dict],
    program_rows: list[dict],
    inst_target: str,
    source: str,
) -> OvernightSignal:
    """조회/스냅샷에서 뽑은 원자료로 순수 판정한다."""
    if not keys:
        return OvernightSignal(
            available=False, met=MappingProxyType({}), met_count=0,
            required=0, should_hold=False, detail="평가할 조건이 선택되지 않음",
        )

    met: dict[str, bool] = {}
    errors: list[str] = []
    if member_error:
        errors.append(member_error)

    if "foreign" in keys:
        if member is None:
            met["foreign"] = False
        else:
            net = int((member.get("foreign") or {}).get("ntby_qty", 0) or 0)
            met["foreign"] = net > 0

    if "shinhan_top5" in keys:
        if member is None:
            met["shinhan_top5"] = False
        else:
            buy_names = [str(row.get("name", "")) for row in (member.get("buy") or [])]
            met["shinhan_top5"] = any(
                any(alias in name for alias in _SHINHAN_ALIASES) for name in buy_names
            )

    if "institution" in keys:
        net = _institution_net(investor_rows, inst_target)
        if net is None:
            errors.append("당일 기관 순매수 데이터 없음")
        met["institution"] = net is not None and net > 0

    if "program" in keys:
        net = _program_net_today(program_rows)
        if net is None:
            errors.append("당일 프로그램 순매수 데이터 없음")
        met["program"] = net is not None and net > 0

    met_count = sum(1 for value in met.values() if value)
    available = len(met) == len(keys) and not errors
    should_hold = available and met_count >= required_n

    summary = ", ".join(
        f"{CONDITION_LABELS[k]}{'✓' if met.get(k) else '✗'}" for k in keys
    )
    detail = f"[{source}] {summary} → {met_count}/{required_n} 충족"
    if errors:
        detail += f" (조회오류: {'; '.join(errors)})"

    return OvernightSignal(
        available=available,
        met=MappingProxyType(dict(met)),
        met_count=met_count,
        required=required_n,
        should_hold=should_hold,
        detail=detail,
    )


def evaluate_overnight_signal(
    stock_code: str,
    conditions: Sequence[str] | None,
    required: int | None,
) -> OvernightSignal:
    """종목의 수급 조건을 KIS 실시간 조회로 판정한다 (장중 강제청산 시각용)."""
    keys = normalize_conditions(conditions)
    if not keys:
        return _evaluate(keys, 0, member=None, member_error=None,
                         investor_rows=[], program_rows=[], inst_target="", source="실시간")

    try:
        required_n = int(required)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        required_n = DEFAULT_OVERNIGHT_MIN_COUNT
    required_n = max(1, min(required_n, len(keys)))

    from dolpha.kis.investor_flow import (
        GetInvestorToday,
        GetMemberFirmTrading,
        GetProgramTradeToday,
    )

    member: dict | None = None
    member_error: str | None = None
    if "foreign" in keys or "shinhan_top5" in keys:
        try:
            fetched = GetMemberFirmTrading(stock_code)
        except Exception as e:  # noqa: BLE001
            member_error = f"회원사 조회 실패: {e}"
        else:
            if _member_ok(fetched):
                member = fetched
            else:
                member_error = "회원사 응답이 비어 있음"

    investor_rows: list[dict] = []
    if "institution" in keys:
        try:
            investor_rows = (GetInvestorToday(stock_code) or {}).get("rows") or []
        except Exception:  # noqa: BLE001 — 빈 목록 → '데이터 없음'으로 미충족 처리
            investor_rows = []

    program_rows: list[dict] = []
    if "program" in keys:
        try:
            program_rows = (GetProgramTradeToday(stock_code) or {}).get("rows") or []
        except Exception:  # noqa: BLE001
            program_rows = []

    return _evaluate(
        keys, required_n,
        member=member, member_error=member_error,
        investor_rows=investor_rows, program_rows=program_rows,
        inst_target=_today_yyyymmdd(), source="실시간",
    )


def evaluate_overnight_signal_from_snapshot(
    snapshot,
    conditions: Sequence[str] | None,
    required: int | None,
    for_date: date_cls,
) -> OvernightSignal:
    """15:29 에 저장해 둔 InvestorFlowSnapshot 으로 판정한다 (장 마감 후 확정용).

    snapshot 이 None 이면 available=False.
    """
    keys = normalize_conditions(conditions)
    if not keys:
        return _evaluate(keys, 0, member=None, member_error=None,
                         investor_rows=[], program_rows=[], inst_target="", source="스냅샷")

    try:
        required_n = int(required)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        required_n = DEFAULT_OVERNIGHT_MIN_COUNT
    required_n = max(1, min(required_n, len(keys)))

    if snapshot is None:
        return OvernightSignal(
            available=False, met=MappingProxyType({}), met_count=0,
            required=required_n, should_hold=False,
            detail="[스냅샷] 15:29 매매동향 스냅샷 없음",
        )

    member_raw = snapshot.member_firm or {}
    member = member_raw if _member_ok(member_raw) else None
    member_error = None if member else "회원사 스냅샷이 비어 있음"

    investor_rows = (snapshot.investor_today or {}).get("rows") or []
    program_rows = (snapshot.program_trade or {}).get("rows") or []

    return _evaluate(
        keys, required_n,
        member=member, member_error=member_error,
        investor_rows=investor_rows, program_rows=program_rows,
        inst_target=for_date.strftime("%Y%m%d"), source="스냅샷",
    )
