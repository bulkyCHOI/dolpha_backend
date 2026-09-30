"""텔레그램 매매 체결 알림.

trading_engine의 매수/매도 체결 시 텔레그램으로 실시간 알림을 전송합니다.
- fire-and-forget 스레드 기반 비동기 전송(매매 로직에 영향 없음)
- 포맷 함수와 전송 함수 분리(테스트 가능)
- 예외는 내부 처리(로그 기록만, 매매 로직에 전파 없음)
"""

import logging
import os
import threading
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_URL = "https://api.telegram.org/bot{}/sendMessage"

_NOTIFIER_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _format_number(value: int | float | None) -> str:
    if value is None:
        return "0"
    return f"{int(value):,}"


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "0.00%"
    return f"{value:+.2f}%"


def _format_integer_percent(value: float | None) -> str:
    if value is None:
        return "0"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.1f}"


def _format_timestamp(dt: datetime | None) -> str:
    if dt is None:
        return "알 수 없음"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_pnl_line(
    profit_loss: float | None,
    profit_loss_pct: float | None,
    remain_qty: int = 0,
) -> str:
    """손익 라인 구성 헬퍼."""
    if profit_loss is not None:
        pnl_value = int(profit_loss)
        pnl_str_num = f"+{_format_number(pnl_value)}" if pnl_value >= 0 else _format_number(pnl_value)
        pnl_str = f"실현손익: {pnl_str_num}원 ({_format_percentage(profit_loss_pct)})"
        if remain_qty > 0:
            return f"{pnl_str} · 잔여 {_format_number(remain_qty)}주"
        return pnl_str
    else:
        if remain_qty > 0:
            return f"실현손익: 확인불가 · 잔여 {_format_number(remain_qty)}주"
        return "실현손익: 확인불가"


def format_buy(
    stock_name: str, stock_code: str, strategy_display_name: str, qty: int, price: float,
    theme_name: str | None = None, is_virtual: bool = False, timestamp: datetime | None = None,
) -> str:
    """매수 메시지 포맷."""
    amount = price * qty
    emoji_prefix = "🟢 [모의]" if is_virtual else "🟢"
    theme_line = f" · 테마: {theme_name}" if theme_name else ""

    return (
        f"{emoji_prefix} [매수] {stock_name} ({stock_code})\n"
        f"전략: {strategy_display_name}{theme_line}\n"
        f"수량: {_format_number(qty)}주 @ {_format_number(int(price))}원\n"
        f"금액: {_format_number(int(amount))}원\n"
        f"시각: {_format_timestamp(timestamp)}"
    )


def format_sell(
    stock_name: str, stock_code: str, qty: int, price: float,
    avg_price: float | None, profit_loss: float | None, profit_loss_pct: float | None,
    reason: str = "", remain_qty: int = 0, is_virtual: bool = False,
    timestamp: datetime | None = None,
) -> str:
    """전량 매도 메시지 포맷."""
    if profit_loss is not None:
        emoji = "🔴" if profit_loss >= 0 else "🔵"
        result = "[매도·익절]" if profit_loss >= 0 else "[매도·손절]"
    else:
        emoji, result = "⚪", "[매도]"
    emoji_prefix = f"{emoji} [모의]" if is_virtual else emoji

    msg = f"{emoji_prefix} {result} {stock_name} ({stock_code})\n"
    if reason:
        msg += f"사유: {reason}\n"
    msg += f"수량: {_format_number(qty)}주 @ {_format_number(int(price))}원"
    if avg_price:
        msg += f" (평단 {_format_number(int(avg_price))}원)"
    msg += f"\n{_build_pnl_line(profit_loss, profit_loss_pct, remain_qty)}\n"
    msg += f"시각: {_format_timestamp(timestamp)}"
    return msg


def format_partial_sell(
    stock_name: str, stock_code: str, sell_qty: int, holding_qty: int, price: float,
    avg_price: float | None, profit_loss: float | None, profit_loss_pct: float | None,
    sell_pct: float, reason: str = "", is_virtual: bool = False, timestamp: datetime | None = None,
) -> str:
    """분할 매도 메시지 포맷."""
    if profit_loss is not None:
        emoji = "🩷" if profit_loss >= 0 else "🔵"
    else:
        emoji = "⚪"
    emoji_prefix = f"{emoji} [모의]" if is_virtual else emoji

    msg = f"{emoji_prefix} [분할매도 {_format_integer_percent(sell_pct)}%] {stock_name} ({stock_code})\n"
    if reason:
        msg += f"사유: {reason}\n"
    msg += f"수량: {_format_number(sell_qty)}주 / {_format_number(holding_qty)}주 @ {_format_number(int(price))}원"
    if avg_price:
        msg += f" (평단 {_format_number(int(avg_price))}원)"
    msg += f"\n{_build_pnl_line(profit_loss, profit_loss_pct, holding_qty - sell_qty)}\n"
    msg += f"시각: {_format_timestamp(timestamp)}"
    return msg


def format_failure(
    order_type: str, stock_name: str, stock_code: str, error_msg: str = "",
    is_virtual: bool = False, timestamp: datetime | None = None,
) -> str:
    """주문 실패 메시지 포맷."""
    emoji_prefix = "⚠️ [모의]" if is_virtual else "⚠️"
    error_display = error_msg if error_msg else "알 수 없음"

    return (
        f"{emoji_prefix} [{order_type} 실패] {stock_name} ({stock_code})\n"
        f"원인: {error_display}\n"
        f"시각: {_format_timestamp(timestamp)}"
    )


def _send_message(message: str) -> None:
    """텔레그램 메시지 전송 (스레드 기반)."""
    if not _NOTIFIER_ENABLED:
        return

    if len(message) > 4096:
        message = message[:4093] + "..."

    def _send_async():
        try:
            url = TELEGRAM_API_URL.format(TELEGRAM_BOT_TOKEN)
            response = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=5,
            )
            if response.status_code != 200:
                logger.warning(f"텔레그램 전송 실패: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("텔레그램 API 타임아웃")
        except requests.exceptions.RequestException:
            logger.warning("텔레그램 네트워크 오류")
        except Exception:
            logger.warning("텔레그램 전송 중 오류")

    thread = threading.Thread(target=_send_async, daemon=True)
    thread.start()


def _send_with_builder(build_fn) -> None:
    """메시지 빌더를 받아 백그라운드 스레드에서 실행 후 전송."""
    if not _NOTIFIER_ENABLED:
        return

    def _build_and_send():
        try:
            message = build_fn()
            if message:
                _send_message(message)
        except Exception:
            logger.warning("메시지 빌더 실행 중 오류")
        finally:
            from django.db import connection
            connection.close()

    thread = threading.Thread(target=_build_and_send, daemon=True)
    thread.start()


def _get_is_virtual(user, strategy_type: str) -> bool:
    """계좌 유형 조회."""
    try:
        from dolpha.strategy_account import resolve_account
        account = resolve_account(user, strategy_type)
        return account.account_type == "VIRTUAL"
    except Exception:
        logger.warning("계좌 유형 조회 실패")
        return False


def _get_theme_name(stock_code: str) -> str | None:
    """테마명 조회 (theme_surge만)."""
    try:
        from dolpha.theme_surge.entry_context import load_theme_context
        from django.utils import timezone as tz
        context = load_theme_context(stock_code, tz.now())
        return context.theme_name
    except Exception:
        return None


def notify_buy_async(stock_name, stock_code, strategy_type, qty, price, user, config, timestamp=None):
    """매수 체결 알림 전송."""
    _send_with_builder(lambda: format_buy(
        stock_name, stock_code, config.get_strategy_type_display(), qty, price,
        _get_theme_name(stock_code) if strategy_type == "theme_surge" else None,
        _get_is_virtual(user, strategy_type), timestamp))


def notify_sell_async(stock_name, stock_code, qty, price, avg_price, profit_loss,
                      profit_loss_pct, reason, remain_qty, user, config, timestamp=None):
    """전량 매도 체결 알림 전송."""
    _send_with_builder(lambda: format_sell(stock_name, stock_code, qty, price, avg_price,
        profit_loss, profit_loss_pct, reason, remain_qty, _get_is_virtual(user, config.strategy_type), timestamp))


def notify_partial_sell_async(stock_name, stock_code, sell_qty, holding_qty, price, avg_price,
                              profit_loss, profit_loss_pct, sell_pct, reason, user, config, timestamp=None):
    """분할 매도 체결 알림 전송."""
    _send_with_builder(lambda: format_partial_sell(stock_name, stock_code, sell_qty, holding_qty,
        price, avg_price, profit_loss, profit_loss_pct, sell_pct, reason, _get_is_virtual(user, config.strategy_type), timestamp))


def notify_failure_async(order_type, stock_name, stock_code, error_msg, user, config, timestamp=None):
    """주문 실패 알림 전송."""
    _send_with_builder(lambda: format_failure(order_type, stock_name, stock_code, error_msg,
        _get_is_virtual(user, config.strategy_type), timestamp))
