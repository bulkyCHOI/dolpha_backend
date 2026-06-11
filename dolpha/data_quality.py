"""
데이터 품질 검증, 기업이벤트(무상감자, 주식분할 등) 보정, 분봉 백필 유틸리티

거래정지 / 기업이벤트로 인한 RS 계산 오류를 방지하기 위한 3가지 레이어:
  Layer 1 — 수정주가 재수집:  기업이벤트 감지 시 전체 OHLCV 히스토리 재수집
  Layer 2 — 거래정지 필터:    연속 N일 volume=0 종목을 RS 계산 유니버스에서 제외
  Layer 3 — 이상 수익률 감지: RS 윈도우 내 비정상 가격 변동 시 RS 무효(-1) 처리
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────

# 단일 거래일 변동률(change 필드)이 이 값을 초과하면 기업이벤트로 판단
# 예: 30:1 무상감자 → change ≈ 29.0 (2900%). 5.0(500%)으로 설정해 정상 급등과 구분.
CORPORATE_ACTION_CHANGE_THRESHOLD: float = 5.0

# 연속 N 거래일 volume=0 이면 거래정지로 판단
HALT_CONSECUTIVE_DAYS: int = 3

# 거래정지 해제 후 RS 계산에서 제외할 냉각 기간 (거래일 기준)
HALT_COOLDOWN_TRADING_DAYS: int = 10

# 수정주가 재수집 시 조회할 과거 기간 (일수, 약 1.5년)
REFETCH_LOOKBACK_DAYS: int = 540


# ── Layer 2: 거래정지 감지 ──────────────────────────────────────────────────


def is_trading_halted(ohlcv_queryset, target_date: date) -> bool:
    """
    target_date 포함 최근 HALT_CONSECUTIVE_DAYS 거래일이 모두 volume=0이면 거래정지로 판단.
    """
    recent = list(
        ohlcv_queryset.filter(date__lte=target_date)
        .order_by("-date")
        .values("volume")[:HALT_CONSECUTIVE_DAYS]
    )
    if len(recent) < HALT_CONSECUTIVE_DAYS:
        return False
    return all(row["volume"] == 0 for row in recent)


def is_in_halt_cooldown(ohlcv_queryset, target_date: date) -> bool:
    """
    최근 HALT_COOLDOWN_TRADING_DAYS 거래일 내에 연속 거래정지 구간이 있었으면 True.
    재상장 직후 수정주가가 DB에 반영되기 전 RS 오염을 방지한다.
    """
    window_size = HALT_COOLDOWN_TRADING_DAYS + HALT_CONSECUTIVE_DAYS
    window = list(
        ohlcv_queryset.filter(date__lte=target_date)
        .order_by("-date")
        .values("volume")[:window_size]
    )
    if len(window) < HALT_CONSECUTIVE_DAYS:
        return False

    streak = 0
    for row in window:
        if row["volume"] == 0:
            streak += 1
            if streak >= HALT_CONSECUTIVE_DAYS:
                return True
        else:
            streak = 0
    return False


def should_skip_rs(ohlcv_queryset, target_date: date) -> tuple[bool, str]:
    """
    Layer 2 통합 진입점.
    RS 계산을 건너뛰어야 하면 (True, 사유) 반환.
    """
    if is_trading_halted(ohlcv_queryset, target_date):
        return True, "거래정지 중"
    if is_in_halt_cooldown(ohlcv_queryset, target_date):
        return True, f"거래정지 해제 후 {HALT_COOLDOWN_TRADING_DAYS}일 냉각 중"
    return False, ""


# ── Layer 3: 이상 수익률 감지 ────────────────────────────────────────────────


def has_corporate_action_in_rs_window(
    ohlcv_queryset,
    target_date: date,
    period_days: int,
) -> bool:
    """
    Layer 3: RS 계산 윈도우(period_days) 내에 기업이벤트성 이상 변동률이 있으면 True.
    CORPORATE_ACTION_CHANGE_THRESHOLD를 초과하는 change 값을 탐지한다.
    """
    lookback_start = target_date - timedelta(days=int(period_days * 1.6))
    return ohlcv_queryset.filter(
        date__gt=lookback_start,
        date__lte=target_date,
        change__gt=CORPORATE_ACTION_CHANGE_THRESHOLD,
    ).exists()


# ── Layer 1: 수정주가 재수집 ─────────────────────────────────────────────────

# DART 공시에서 탐지할 기업이벤트 키워드
CORPORATE_ACTION_KEYWORDS: tuple[str, ...] = ("감자", "주식분할", "주식병합", "무상증자")

# 탐지 키워드 정규식 (| 구분 OR 패턴)
_KEYWORD_PATTERN: str = "|".join(CORPORATE_ACTION_KEYWORDS)

# DART 공시 조회 기간 (일)
DART_LOOKBACK_DAYS: int = 60

# 당일 중복 API 호출 방지 캐시: {(code, date_str): bool}
_dart_check_cache: dict[tuple[str, str], bool] = {}


def check_dart_corporate_action(code: str, lookback_days: int = DART_LOOKBACK_DAYS) -> bool:
    """
    DART 공시 목록에서 기업이벤트(감자/분할/병합/무상증자) 여부를 확인.

    - 동일 종목·날짜 조합은 캐시하여 중복 API 호출 방지
    - DART 지원 종목(KR)에서만 사용; US 종목은 detect_corporate_action_from_saved 사용
    - 반환: 공시 감지 시 True, 없거나 오류 시 False
    """
    from datetime import date as _date

    today_str = _date.today().isoformat()
    cache_key = (code, today_str)
    if cache_key in _dart_check_cache:
        return _dart_check_cache[cache_key]

    try:
        from dolpha.dart_parallel import _get_dart

        dart = _get_dart()
        start_str = (_date.today() - timedelta(days=lookback_days)).isoformat()
        df = dart.list(code, start=start_str, end=today_str)

        if df is None or len(df) == 0:
            _dart_check_cache[cache_key] = False
            return False

        matched = df[df["report_nm"].str.contains(_KEYWORD_PATTERN, na=False)]
        hit = not matched.empty
        _dart_check_cache[cache_key] = hit

        if hit:
            names = matched["report_nm"].tolist()
            logger.warning("[data_quality] DART 기업이벤트 공시 감지: %s → %s", code, names)

        return hit

    except Exception:
        logger.error("[data_quality] DART 공시 조회 오류: %s", code, exc_info=True)
        _dart_check_cache[cache_key] = False
        return False


def detect_corporate_action_from_saved(stock_ohlcv_list: list) -> bool:
    """
    방금 저장된 OHLCV 레코드 중 기업이벤트성 변동률이 있으면 True.
    DART 미지원 시장(US 등)의 폴백으로 사용.
    """
    return any(
        abs(getattr(obj, "change", 0.0) or 0.0) > CORPORATE_ACTION_CHANGE_THRESHOLD
        for obj in stock_ohlcv_list
    )


def backfill_minute_bars(stock_code: str, target_date: Optional[date] = None) -> int:
    """
    당일(또는 지정 날짜)의 누락된 분봉 데이터를 KIS API로 채웁니다.

    - 정규장(09:00~15:30) 내 전체 봉을 KIS API로 재조회
    - DB에 없는 봉(누락)만 INSERT, 이미 있는 봉은 UPDATE (update_or_create)
    - 오늘 날짜이면 현재 시각까지만 조회 (미래 시각 참조 방지)

    Args:
        stock_code   : 종목코드 (6자리)
        target_date  : 백필 대상 날짜 (None이면 오늘)

    Returns:
        upsert된 봉 수
    """
    from myweb.models import StockMinuteOhlcv
    from dolpha.kis.minute import GetMinuteOhlcvKR

    if target_date is None:
        target_date = date.today()

    today = date.today()
    if target_date == today:
        # 오늘: 현재 시각(마지막 완성된 봉)까지만 조회
        end_hhmmss = min(datetime.now().strftime("%H%M%S"), "153000")
        if end_hhmmss < "090000":
            # 장 시작 전 — 조회 불필요
            return 0
    else:
        end_hhmmss = "153000"

    logger.info("[backfill] %s %s 백필 시작 (end=%s)", stock_code, target_date, end_hhmmss)

    try:
        bars = GetMinuteOhlcvKR(stock_code, end_hhmmss=end_hhmmss)
    except Exception:
        logger.error("[backfill] KIS API 조회 실패: %s %s", stock_code, target_date, exc_info=True)
        return 0

    if not bars:
        logger.info("[backfill] %s %s KIS 반환 데이터 없음", stock_code, target_date)
        return 0

    # target_date 날짜의 봉만 필터 (GetMinuteOhlcvKR은 당일만 반환하지만 안전 확인)
    target_prefix = target_date.strftime("%Y-%m-%d")
    bars = [b for b in bars if b.get("datetime", "").startswith(target_prefix)]

    upsert_count = 0
    for bar in bars:
        if (bar.get("volume") or 0) <= 0:
            continue
        raw_dt = bar.get("datetime", "")
        try:
            # naive datetime: Django USE_TZ=True 환경에서 settings.TIME_ZONE(KST)으로 해석
            bar_dt_naive = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
            StockMinuteOhlcv.objects.update_or_create(
                stock_code=stock_code,
                bar_datetime=bar_dt_naive,
                defaults={
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                },
            )
            upsert_count += 1
        except Exception:
            logger.warning("[backfill] upsert 실패: %s %s", stock_code, raw_dt, exc_info=True)

    logger.info("[backfill] %s %s: %d봉 upsert 완료", stock_code, target_date, upsert_count)
    return upsert_count


def refetch_adjusted_history(company, area: str = "KR") -> int:
    """
    Layer 1: 기업이벤트 발생 종목의 OHLCV 전체 히스토리를 수정주가 기준으로 재수집.

    - DB의 해당 종목 OHLCV 전체 삭제 후 REFETCH_LOOKBACK_DAYS 분량 재수집
    - adj_ok="1" (수정주가) 사용
    - 반환값: 재수집된 레코드 수 (오류 시 0)
    """
    from datetime import date as _date
    import traceback

    from django.db import transaction

    from dolpha import stockCommon as Common
    from myweb.models import StockOHLCV

    try:
        end_date = _date.today().isoformat()
        start_date = (_date.today() - timedelta(days=REFETCH_LOOKBACK_DAYS)).isoformat()

        logger.warning(
            "[data_quality] 기업이벤트 감지 → 수정주가 재수집: %s (%s) %s ~ %s",
            company.code, getattr(company, "name", ""), start_date, end_date,
        )

        df = Common.GetOhlcv(area, company.code, start_date=start_date, end_date=end_date, adj_ok="1")
        if df is None or len(df) == 0:
            logger.warning("[data_quality] 재수집 데이터 없음: %s", company.code)
            return 0

        import pandas as pd

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        df = df.fillna({"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0, "change": 0.0})

        records = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if i > 0:
                prev_close = float(df.iloc[i - 1]["close"])
                change_rate = (float(row["close"]) - prev_close) / prev_close if prev_close > 0 else 0.0
            else:
                change_rate = 0.0

            records.append(StockOHLCV(
                code=company,
                date=idx.date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                change=change_rate,
            ))

        with transaction.atomic():
            StockOHLCV.objects.filter(code=company).delete()
            StockOHLCV.objects.bulk_create(records)

        logger.info("[data_quality] 재수집 완료: %s → %d건", company.code, len(records))
        return len(records)

    except Exception:
        logger.error("[data_quality] 재수집 오류: %s", company.code, exc_info=True)
        return 0
