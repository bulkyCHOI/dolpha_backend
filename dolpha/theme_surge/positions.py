"""급등테마주 자동매매 현황 조립.

화면 최상단에 "지금 내 돈이 어디에 들어가 있고, 왜 들어갔고, 다음 후보는 왜 아직
안 샀는가"를 보여주기 위해 흩어진 소스를 하나로 합친다.

    TradingConfig(theme_surge)  전략 설정 (손절/익절/진입차수)
    TradeEntry(BUY/FILLED)      진입 차수·평단·진입 시각
    KIS GetMyStockList()        실계좌 수량·평단·현재가·손익
    ThemeLeaderCandidate        종목이 속한 테마명·선정 점수
    ThemeEntrySignal            진입 사유 / 미진입 사유(3조건 충족 현황)

기존 /api/mypage/trading-status 로는 테마명·진입 사유를 알 수 없어 전용으로 만든다.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime

from pytz import timezone as pytz_tz

_KST = pytz_tz("Asia/Seoul")

STRATEGY = "theme_surge"


def build_positions(user, target_date: date_cls | None = None) -> dict:
    """유저의 급등테마주 보유 포지션과 대기 후보를 조립한다.

    Args:
        user:        조회 대상 유저
        target_date: 테마·시그널 매칭 기준일 (기본: 오늘 KST)

    Returns:
        {"positions": [...], "watching": [...], "summary": {...}}
    """
    from myweb.models import TradingConfig

    day = target_date or datetime.now(_KST).date()

    configs = list(
        TradingConfig.objects.filter(user=user, strategy_type=STRATEGY, is_active=True)
    )
    if not configs:
        return _empty()

    codes = [c.stock_code for c in configs]
    holdings = _kis_holdings(codes)
    themes = _theme_by_code(day, codes)
    signals = _latest_signal_by_code(user, day, codes)
    entries = _entry_stats(user, codes)

    positions: list[dict] = []
    watching: list[dict] = []

    for config in configs:
        code = config.stock_code
        holding = holdings.get(code)
        entry = entries.get(code, {})

        if holding and holding["quantity"] > 0:
            positions.append(_position_row(config, holding, entry, themes.get(code), signals.get(code)))
        else:
            watching.append(_watching_row(config, themes.get(code), signals.get(code)))

    positions.sort(key=lambda p: p["profit_loss_rate"], reverse=True)
    watching.sort(key=lambda w: w["conditions_met"], reverse=True)

    return {
        "positions": positions,
        "watching": watching,
        "summary": _summary(positions, watching),
    }


# ──────────────────────────────────────────────────────────────
# 행 조립
# ──────────────────────────────────────────────────────────────

def _position_row(config, holding: dict, entry: dict, theme: dict | None, signal) -> dict:
    """보유 중인 포지션 1행."""
    avg_price = holding["avg_price"]
    stop_pct = config.stop_loss or 0
    take_pct = config.take_profit or 0

    return {
        "stock_code": config.stock_code,
        "stock_name": holding.get("stock_name") or config.stock_name,
        "theme_name": (theme or {}).get("theme_name", ""),
        "entry_reason": _signal_reason(signal),
        "quantity": holding["quantity"],
        "avg_price": avg_price,
        "current_price": holding["current_price"],
        "profit_loss_amount": holding["profit_loss_amount"],
        "profit_loss_rate": round(holding["profit_loss_rate"], 2),
        "entry_count": entry.get("count", 0),
        "max_entries": (config.pyramiding_count or 0) + 1,
        "stop_price": round(avg_price * (1 - stop_pct / 100)) if stop_pct else None,
        "target_price": round(avg_price * (1 + take_pct / 100)) if take_pct else None,
        "entered_at": entry.get("first_at"),
    }


def _watching_row(config, theme: dict | None, signal) -> dict:
    """아직 진입하지 못한 후보 1행 — '왜 안 샀는가'가 핵심."""
    met = 0
    if signal is not None:
        met = sum(
            [signal.has_pullback, signal.has_breakout, signal.has_foreign_buying]
        )

    return {
        "stock_code": config.stock_code,
        "stock_name": config.stock_name,
        "theme_name": (theme or {}).get("theme_name", ""),
        "score": (theme or {}).get("score"),
        "has_pullback": bool(signal and signal.has_pullback),
        "has_breakout": bool(signal and signal.has_breakout),
        "has_foreign_buying": bool(signal and signal.has_foreign_buying),
        "conditions_met": met,
        "last_reason": _signal_reason(signal) or "아직 판정 이력이 없습니다.",
        "checked_at": (
            signal.checked_at.astimezone(_KST).strftime("%H:%M") if signal else None
        ),
    }


def _summary(positions: list[dict], watching: list[dict]) -> dict:
    total_pl = sum(p["profit_loss_amount"] for p in positions)
    total_cost = sum(p["avg_price"] * p["quantity"] for p in positions)
    return {
        "position_count": len(positions),
        "watching_count": len(watching),
        "total_profit_loss": total_pl,
        "total_profit_rate": round(total_pl / total_cost * 100, 2) if total_cost else 0.0,
    }


def _empty() -> dict:
    return {
        "positions": [],
        "watching": [],
        "summary": {
            "position_count": 0,
            "watching_count": 0,
            "total_profit_loss": 0,
            "total_profit_rate": 0.0,
        },
    }


def _signal_reason(signal) -> str:
    return signal.reason if signal is not None else ""


# ──────────────────────────────────────────────────────────────
# 소스 조회
# ──────────────────────────────────────────────────────────────

def _kis_holdings(codes: list[str]) -> dict[str, dict]:
    """KIS 실계좌 보유 현황을 {code: {...}} 로 반환. 조회 실패 시 빈 dict."""
    try:
        from dolpha.kis.trade import GetMyStockList

        rows = GetMyStockList()
    except Exception as e:
        print(f"[급등테마] 보유 현황 조회 실패: {e}")
        return {}

    wanted = set(codes)
    result: dict[str, dict] = {}
    for row in rows:
        code = row.get("StockCode")
        if code not in wanted:
            continue
        try:
            result[code] = {
                "stock_name": row.get("StockName", ""),
                "quantity": int(row["StockAmt"]),
                "avg_price": round(float(row["StockAvgPrice"])),
                "current_price": round(float(row["StockNowPrice"])),
                "profit_loss_amount": round(float(row["StockRevenueMoney"])),
                "profit_loss_rate": float(row["StockRevenueRate"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return result


def _theme_by_code(day: date_cls, codes: list[str]) -> dict[str, dict]:
    """종목별 소속 테마 (당일 최고 점수 기준)."""
    from myweb.models import ThemeLeaderCandidate

    result: dict[str, dict] = {}
    rows = ThemeLeaderCandidate.objects.filter(
        date=day, stock_code__in=codes
    ).order_by("-score")

    for row in rows:
        if row.stock_code in result:
            continue  # score 내림차순이므로 첫 행이 최고 점수
        result[row.stock_code] = {
            "theme_name": row.theme_name,
            "tics_id": row.tics_id,
            "score": round(row.score, 4),
        }

    return result


def _latest_signal_by_code(user, day: date_cls, codes: list[str]) -> dict:
    """종목별 가장 최근 진입 판정 시그널."""
    from myweb.models import ThemeEntrySignal

    result = {}
    rows = ThemeEntrySignal.objects.filter(
        user=user, date=day, stock_code__in=codes
    ).order_by("-checked_at")

    for row in rows:
        if row.stock_code in result:
            continue
        result[row.stock_code] = row

    return result


def _entry_stats(user, codes: list[str]) -> dict[str, dict]:
    """종목별 체결 진입 차수와 최초 진입 시각."""
    from myweb.models import TradeEntry

    rows = TradeEntry.objects.filter(
        user=user, stock_code__in=codes, trade_type="BUY", status="FILLED"
    ).order_by("filled_at")

    result: dict[str, dict] = {}
    for row in rows:
        stat = result.setdefault(row.stock_code, {"count": 0, "first_at": None})
        stat["count"] += 1
        if stat["first_at"] is None and row.filled_at:
            stat["first_at"] = row.filled_at.astimezone(_KST).strftime("%m-%d %H:%M")

    return result
