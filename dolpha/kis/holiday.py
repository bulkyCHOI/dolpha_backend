"""KIS API — 국내 개장일(휴장일) 조회.

한국거래소 휴장일은 주말뿐 아니라 법정공휴일·대체공휴일·임시공휴일까지 포함하므로
요일만으로는 판정할 수 없다. KIS '국내휴장일조회'(CTCA0903R)의 `opnd_yn`(개장일 여부)을
사용해 정확히 판정한다.

    2026-08-14 개장일=Y   2026-08-15 개장일=N (광복절)
    2026-08-17 개장일=N (대체공휴일)   2026-08-18 개장일=Y

한 번 호출하면 기준일부터 수십 일치가 반환되므로 모듈 레벨 캐시에 일괄 적재하고,
캐시에 없는 날짜를 물어볼 때만 재조회한다(실사용상 하루 1회 수준).

주의: 이 엔드포인트는 REAL 도메인에서만 제공된다. REAL 키가 없거나 조회가 실패하면
      '주말만 휴장'이라는 보수적 폴백을 쓴다. 수집을 아예 멈추면 실제 개장일에
      데이터 구멍이 생겨 더 나쁘기 때문이다.
"""

from __future__ import annotations

import warnings
from datetime import date as date_cls, datetime, timedelta

import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

_PATH = "uapi/domestic-stock/v1/quotations/chk-holiday"
_TR_ID = "CTCA0903R"
_TIMEOUT_SEC = 10

# {date: 개장일 여부} — 프로세스 수명 동안 유지
_cache: dict[date_cls, bool] = {}
# 같은 날짜로 반복 실패할 때 매 호출마다 API를 두드리지 않도록 실패 시점을 기록
_last_failure_at: datetime | None = None
_FAILURE_COOLDOWN_SEC = 300


def is_trading_day(target: date_cls | None = None) -> bool:
    """해당 날짜가 국내 증시 개장일인지 반환한다.

    Args:
        target: 조회할 날짜 (기본: 오늘, KST)

    Returns:
        개장일이면 True. 조회 실패 시 주말 여부만으로 판정한다.
    """
    day = target or _today_kst()

    if day in _cache:
        return _cache[day]

    if _should_skip_fetch():
        return _weekend_fallback(day)

    try:
        fetched = _fetch_holidays(day)
    except Exception as e:
        _mark_failure()
        print(f"[개장일] KIS 휴장일 조회 실패 ({day}) — 주말 기준으로 폴백: {e}")
        return _weekend_fallback(day)

    if not fetched:
        _mark_failure()
        print(f"[개장일] KIS 휴장일 응답 비어 있음 ({day}) — 주말 기준으로 폴백")
        return _weekend_fallback(day)

    _cache.update(fetched)
    return _cache.get(day, _weekend_fallback(day))


def refresh_holiday_cache(start: date_cls | None = None) -> int:
    """기준일부터의 개장일 정보를 강제로 다시 채운다.

    Returns:
        캐시에 적재된 날짜 수 (실패 시 0)
    """
    global _last_failure_at

    day = start or _today_kst()
    try:
        fetched = _fetch_holidays(day)
    except Exception as e:
        print(f"[개장일] 캐시 갱신 실패: {e}")
        return 0

    _cache.update(fetched)
    _last_failure_at = None
    return len(fetched)


def clear_holiday_cache() -> None:
    """캐시를 비운다 (테스트용)."""
    global _last_failure_at
    _cache.clear()
    _last_failure_at = None


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _today_kst() -> date_cls:
    from pytz import timezone as pytz_tz

    return datetime.now(pytz_tz("Asia/Seoul")).date()


def _weekend_fallback(day: date_cls) -> bool:
    """조회 불가 시 폴백 — 주말만 휴장으로 본다."""
    return day.weekday() < 5


def _should_skip_fetch() -> bool:
    """직전 실패 후 쿨다운 중이면 재조회를 건너뛴다."""
    if _last_failure_at is None:
        return False
    return (datetime.now() - _last_failure_at).total_seconds() < _FAILURE_COOLDOWN_SEC


def _mark_failure() -> None:
    global _last_failure_at
    _last_failure_at = datetime.now()


def _fetch_holidays(start: date_cls) -> dict[date_cls, bool]:
    """기준일 이후 개장일 정보를 조회해 {date: 개장일여부} 로 반환한다.

    KIS는 기준일부터 수십 일치를 한 번에 내려준다. 페이지네이션(tr_cont)은
    쓰지 않고 첫 페이지만 사용한다 — 스캔은 '오늘' 판정만 필요하기 때문이다.
    """
    url = f"{get_url_base('REAL')}/{_PATH}"
    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")
    params = {
        "BASS_DT": start.strftime("%Y%m%d"),
        "CTX_AREA_NK": "",
        "CTX_AREA_FK": "",
    }

    res = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT_SEC, verify=False)
    if res.status_code != 200:
        raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")

    body = res.json()
    # rt_cd=0 정상, rt_cd=1 은 '다음 페이지 있음'이며 데이터 자체는 유효
    if body.get("rt_cd") not in ("0", "1"):
        raise RuntimeError(f"rt_cd={body.get('rt_cd')}: {body.get('msg1', '')[:200]}")

    result: dict[date_cls, bool] = {}
    for row in body.get("output", []):
        raw = row.get("bass_dt")
        if not raw or len(raw) != 8:
            continue
        try:
            parsed = date_cls(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            continue
        result[parsed] = row.get("opnd_yn") == "Y"

    return result


def next_trading_day(after: date_cls | None = None, max_lookahead: int = 14) -> date_cls | None:
    """기준일 다음의 최초 개장일을 반환한다. 못 찾으면 None."""
    cursor = (after or _today_kst()) + timedelta(days=1)
    for _ in range(max_lookahead):
        if is_trading_day(cursor):
            return cursor
        cursor += timedelta(days=1)
    return None
