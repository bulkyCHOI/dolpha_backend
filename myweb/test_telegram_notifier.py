"""텔레그램 매매 체결 알림 포맷 함수 단위 테스트 (Django TestCase)."""

from datetime import datetime
from unittest.mock import patch, MagicMock

from django.test import TestCase, SimpleTestCase
from django.utils import timezone as tz

from dolpha import telegram_notifier


class FormatBuyTests(SimpleTestCase):
    """매수 메시지 포맷 테스트."""

    def test_buy_with_theme(self):
        """일반 매수 (모의계좌 아님, 테마 있음)."""
        msg = telegram_notifier.format_buy(
            stock_name="삼성전자",
            stock_code="005930",
            strategy_display_name="MTT (Minervini Trend Template)",
            qty=10,
            price=70000,
            theme_name="반도체",
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 10, 30, 45)
        )
        self.assertIn("🟢 [매수]", msg)
        self.assertIn("삼성전자 (005930)", msg)
        self.assertIn("MTT (Minervini Trend Template)", msg)
        self.assertIn("테마: 반도체", msg)
        self.assertIn("10주 @ 70,000원", msg)
        self.assertIn("700,000원", msg)
        self.assertNotIn("[모의]", msg)

    def test_buy_virtual_no_theme(self):
        """모의계좌 매수 (테마 없음)."""
        msg = telegram_notifier.format_buy(
            stock_name="SK하이닉스",
            stock_code="000660",
            strategy_display_name="High Tight Flag",
            qty=5,
            price=150000,
            is_virtual=True,
            timestamp=datetime(2026, 9, 30, 11, 0, 0)
        )
        self.assertIn("🟢 [모의] [매수]", msg)
        self.assertIn("SK하이닉스 (000660)", msg)
        self.assertNotIn("테마:", msg)
        self.assertIn("750,000원", msg)


class FormatSellTests(SimpleTestCase):
    """전량 매도 메시지 포맷 테스트."""

    def test_sell_profit(self):
        """익절 (이익)."""
        msg = telegram_notifier.format_sell(
            stock_name="NAVER",
            stock_code="035420",
            qty=20,
            price=410000,
            avg_price=400000,
            profit_loss=200000,
            profit_loss_pct=2.50,
            reason="익절",
            remain_qty=0,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 14, 30, 0)
        )
        self.assertIn("🔴 [매도·익절]", msg)
        self.assertIn("NAVER (035420)", msg)
        self.assertIn("익절", msg)
        self.assertIn("20주 @ 410,000원", msg)
        self.assertIn("평단 400,000원", msg)
        self.assertIn("+200,000원", msg)
        self.assertIn("+2.50%", msg)

    def test_sell_loss(self):
        """손절 (손실)."""
        msg = telegram_notifier.format_sell(
            stock_name="카카오",
            stock_code="035720",
            qty=30,
            price=90000,
            avg_price=100000,
            profit_loss=-300000,
            profit_loss_pct=-10.00,
            reason="손절",
            remain_qty=0,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 15, 0, 0)
        )
        self.assertIn("🔵 [매도·손절]", msg)
        self.assertIn("카카오 (035720)", msg)
        self.assertIn("-300,000원", msg)
        self.assertIn("-10.00%", msg)

    def test_sell_unknown_pnl(self):
        """매도 (손익 알 수 없음)."""
        msg = telegram_notifier.format_sell(
            stock_name="현대차",
            stock_code="005380",
            qty=10,
            price=220000,
            avg_price=None,
            profit_loss=None,
            profit_loss_pct=None,
            reason="강제청산",
            remain_qty=0,
            is_virtual=True,
            timestamp=datetime(2026, 9, 30, 16, 0, 0)
        )
        self.assertIn("⚪ [모의] [매도]", msg)
        self.assertIn("강제청산", msg)
        self.assertIn("확인불가", msg)

    def test_sell_no_reason(self):
        """사유 생략."""
        msg = telegram_notifier.format_sell(
            stock_name="삼성전자",
            stock_code="005930",
            qty=10,
            price=70000,
            avg_price=68000,
            profit_loss=20000,
            profit_loss_pct=2.94,
            reason="",
            remain_qty=0,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 14, 0, 0)
        )
        lines = msg.split("\n")
        # 첫 두 줄에 사유가 없어야 함
        self.assertFalse(any("사유:" in line for line in lines[:2]))


class FormatPartialSellTests(SimpleTestCase):
    """분할 매도 메시지 포맷 테스트."""

    def test_partial_sell_profit(self):
        """분할 매도 (이익)."""
        msg = telegram_notifier.format_partial_sell(
            stock_name="LG화학",
            stock_code="051910",
            sell_qty=25,
            holding_qty=100,
            price=500000,
            avg_price=480000,
            profit_loss=500000,
            profit_loss_pct=4.17,
            sell_pct=25,
            reason="부분익절",
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 13, 0, 0)
        )
        self.assertIn("🩷 [분할매도 25%]", msg)
        self.assertIn("LG화학 (051910)", msg)
        self.assertIn("25주 / 100주", msg)
        self.assertIn("잔여 75주", msg)
        self.assertIn("+500,000원", msg)

    def test_partial_sell_float_pct(self):
        """분할 매도 (소수 비율)."""
        msg = telegram_notifier.format_partial_sell(
            stock_name="포스코",
            stock_code="005490",
            sell_qty=16,
            holding_qty=100,
            price=70000,
            avg_price=75000,
            profit_loss=-80000,
            profit_loss_pct=-10.67,
            sell_pct=16.0,
            reason="리밸런싱",
            is_virtual=True,
            timestamp=datetime(2026, 9, 30, 12, 30, 0)
        )
        self.assertIn("🔵 [모의] [분할매도 16%]", msg)
        self.assertIn("-80,000원", msg)


class FormatFailureTests(SimpleTestCase):
    """주문 실패 메시지 포맷 테스트."""

    def test_failure_buy(self):
        """매수 실패."""
        msg = telegram_notifier.format_failure(
            order_type="매수",
            stock_name="삼성전자",
            stock_code="005930",
            error_msg="가용현금 부족",
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 10, 15, 0)
        )
        self.assertIn("⚠️ [매수 실패]", msg)
        self.assertIn("삼성전자 (005930)", msg)
        self.assertIn("가용현금 부족", msg)

    def test_failure_sell(self):
        """매도 실패 (모의계좌)."""
        msg = telegram_notifier.format_failure(
            order_type="매도",
            stock_name="SK하이닉스",
            stock_code="000660",
            error_msg="네트워크 오류",
            is_virtual=True,
            timestamp=datetime(2026, 9, 30, 14, 30, 0)
        )
        self.assertIn("⚠️ [모의] [매도 실패]", msg)
        self.assertIn("네트워크 오류", msg)

    def test_failure_no_error_msg(self):
        """오류 메시지 없음."""
        msg = telegram_notifier.format_failure(
            order_type="분할매도",
            stock_name="NAVER",
            stock_code="035420",
            error_msg="",
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 15, 0, 0)
        )
        self.assertIn("⚠️ [분할매도 실패]", msg)
        self.assertIn("알 수 없음", msg)


class SendMessageTests(SimpleTestCase):
    """메시지 전송 테스트."""

    @patch.dict("os.environ", {}, clear=False)
    def test_notifier_disabled_when_no_token(self):
        """토큰 미설정 시 requests.post 미호출."""
        # telegram_notifier 모듈 재로드는 어렵으므로, 직접 테스트
        import dolpha.telegram_notifier as notifier_module

        # _NOTIFIER_ENABLED 확인 (토큰이 설정되지 않으면 False)
        # 실제로는 모듈 로드 시 결정되므로, mock으로 _send_message 테스트
        with patch.object(notifier_module, "_NOTIFIER_ENABLED", False):
            with patch("requests.post") as mock_post:
                notifier_module._send_message("test message")
                # 토큰 비활성화 시 post 호출 없음
                mock_post.assert_not_called()

    @patch("requests.post")
    def test_request_exception_not_propagated(self, mock_post):
        """requests 예외가 전파되지 않음."""
        import dolpha.telegram_notifier as notifier_module

        mock_post.side_effect = Exception("Network error")

        with patch.object(notifier_module, "_NOTIFIER_ENABLED", True):
            with patch.object(notifier_module, "TELEGRAM_BOT_TOKEN", "token"):
                with patch.object(notifier_module, "TELEGRAM_CHAT_ID", "chat_id"):
                    # 예외가 발생하지 않아야 함
                    try:
                        notifier_module._send_message("test message")
                    except Exception as e:
                        self.fail(f"Exception propagated: {e}")

    @patch("dolpha.telegram_notifier.logger")
    @patch("requests.post")
    def test_token_not_in_log(self, mock_post, mock_logger):
        """로그에 토큰 문자열 포함 안 함."""
        import dolpha.telegram_notifier as notifier_module

        mock_post.side_effect = Exception("Network error")

        with patch.object(notifier_module, "_NOTIFIER_ENABLED", True):
            with patch.object(notifier_module, "TELEGRAM_BOT_TOKEN", "secret_token_12345"):
                with patch.object(notifier_module, "TELEGRAM_CHAT_ID", "secret_chat"):
                    notifier_module._send_message("test message")

                    # warning 로그에서 토큰 확인
                    for call in mock_logger.warning.call_args_list:
                        log_msg = str(call)
                        self.assertNotIn("secret_token_12345", log_msg)
                        self.assertNotIn("secret_chat", log_msg)


class FormatEdgeCasesTests(SimpleTestCase):
    """엣지 케이스 테스트."""

    def test_zero_profit_is_profit(self):
        """0 손익은 익절로 분류 (손익>=0)."""
        msg = telegram_notifier.format_sell(
            stock_name="SK하이닉스",
            stock_code="000660",
            qty=10,
            price=150000,
            avg_price=150000,
            profit_loss=0,
            profit_loss_pct=0.00,
            reason="청산",
            remain_qty=0,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 14, 0, 0)
        )
        self.assertIn("🔴 [매도·익절]", msg)
        self.assertIn("+0원", msg)

    def test_large_amount(self):
        """매우 큰 금액."""
        msg = telegram_notifier.format_buy(
            stock_name="삼성전자",
            stock_code="005930",
            strategy_display_name="MTT (Minervini Trend Template)",
            qty=10000,
            price=70000,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 10, 0, 0)
        )
        self.assertIn("10,000주", msg)
        self.assertIn("700,000,000원", msg)

    def test_float_sell_pct(self):
        """분할 비율 소수."""
        msg = telegram_notifier.format_partial_sell(
            stock_name="LG화학",
            stock_code="051910",
            sell_qty=16,
            holding_qty=100,
            price=500000,
            avg_price=480000,
            profit_loss=320000,
            profit_loss_pct=3.33,
            sell_pct=16.7,
            is_virtual=False,
            timestamp=datetime(2026, 9, 30, 13, 0, 0)
        )
        self.assertIn("[분할매도 16.7%]", msg)
