"""
부분체결 처리 테스트 (TradingEngine)

주문 후 실제 체결수량 확인 및 TradeEntry 업데이트 로직을 검증합니다.
"""

import unittest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime
from django.utils import timezone as tz
from django.test import TestCase

from myweb.models import User, TradingConfig, TradeEntry, TradingSummary
from dolpha.trading_engine import TradingEngine


class PartialFillReconciliationTest(TestCase):
    """부분체결 인식 및 재주문 테스트"""

    def setUp(self):
        """테스트 사용자, 설정, 엔진 생성"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass"
        )
        self.config = TradingConfig.objects.create(
            user=self.user,
            stock_code="005930",
            stock_name="삼성전자",
            strategy_type="test_strategy",
            is_active=True,
        )
        self.engine = TradingEngine(self.user)
        self.stock_code = "005930"

    # ── 공통 헬퍼 ──────────────────────────────────────────

    def _fill(self, order_no, ordered, filled, remain, cancelled=False, price=50000):
        return {
            "order_no": order_no, "ordered_qty": ordered, "filled_qty": filled,
            "remain_qty": remain, "avg_price": price if filled else 0,
            "filled_amount": price * filled, "cancelled": cancelled, "org_no": "00950",
        }

    def _patches(self):
        """외부 I/O(KIS, sleep, 알림, 로그)를 막는 공통 패치. 실주문은 절대 나가지 않는다."""
        cred = MagicMock(token_cache_key="acct-A")
        return [
            patch.object(self.engine, "_credential_for", return_value=cred),
            patch("dolpha.trading_engine.time.sleep"),
            patch("dolpha.trading_engine.telegram_notifier"),
            patch("dolpha.trading_engine.order_log"),
            patch.object(self.engine, "get_atr", return_value=None),
        ]

    def _enter(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        for p in self._patches():
            stack.enter_context(p)

    def _make_buy_entry(self, qty=118):
        return TradeEntry.objects.create(
            user=self.user, trading_config=self.config, stock_code=self.stock_code,
            stock_name="삼성전자", trade_type="BUY", entry_type="INITIAL",
            order_no="11111111", order_quantity=qty, order_price=Decimal("50000"),
            filled_quantity=qty, filled_price=Decimal("50000"),
            filled_amount=Decimal(50000 * qty), status="FILLED",
            ordered_at=tz.now(), filled_at=tz.now(),
        )

    # ── 매수 ──────────────────────────────────────────────

    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    @patch("dolpha.kis.trade.MakeBuyMarketOrder")
    @patch("dolpha.kis.trade.GetBalance")
    def test_buy_order_partial_fill_reconciliation(
        self, mock_balance, mock_make_buy, mock_get_fill, mock_cancel
    ):
        """BUY 부분체결: 100주 주문, 70주 체결 → 잔량 30주 취소, 장부 70주"""
        self._enter()
        mock_balance.return_value = {"RemainMoney": 10_000_000}
        mock_make_buy.return_value = {"OrderNum": "123", "OrderNum2": "12345678", "OrderTime": "093000"}
        mock_get_fill.side_effect = [
            self._fill("12345678", 100, 70, 30),                  # 체결 확인
            self._fill("12345678", 100, 70, 0, cancelled=True),   # 취소 후 최종
        ]
        mock_cancel.return_value = True

        result = self.engine.execute_buy_order(self.config, amount=5_000_000, current_price=50000)

        self.assertTrue(result)
        mock_cancel.assert_called_once()
        self.assertEqual(mock_cancel.call_args.args[0], "12345678")
        entry = TradeEntry.objects.get(trading_config=self.config, trade_type="BUY")
        self.assertEqual(entry.filled_quantity, 70)
        self.assertEqual(entry.order_quantity, 70)  # 잔량 취소 반영
        self.assertEqual(entry.status, "FILLED")
        self.assertIn("부분체결", entry.note)
        self.assertIn("잔량 30주 취소", entry.note)
        self.assertEqual(self.engine._config_net_position(self.config).qty, 70)

    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    @patch("dolpha.kis.trade.MakeBuyMarketOrder")
    @patch("dolpha.kis.trade.GetBalance")
    def test_buy_late_fill_during_cancel_is_recorded(
        self, mock_balance, mock_make_buy, mock_get_fill, mock_cancel
    ):
        """취소 직전에 추가 체결된 수량(70→90)도 장부에 반영된다"""
        self._enter()
        mock_balance.return_value = {"RemainMoney": 10_000_000}
        mock_make_buy.return_value = {"OrderNum": "123", "OrderNum2": "12345678", "OrderTime": "093000"}
        mock_get_fill.side_effect = [
            self._fill("12345678", 100, 70, 30),
            self._fill("12345678", 100, 90, 0, cancelled=True),
        ]
        mock_cancel.return_value = True

        self.assertTrue(self.engine.execute_buy_order(self.config, 5_000_000, 50000))
        self.assertEqual(self.engine._config_net_position(self.config).qty, 90)

    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    @patch("dolpha.kis.trade.MakeBuyMarketOrder")
    @patch("dolpha.kis.trade.GetBalance")
    def test_buy_cancel_failure_keeps_partial_for_later_reconcile(
        self, mock_balance, mock_make_buy, mock_get_fill, mock_cancel
    ):
        """취소 실패: 주문수량을 유지해 사후 재조회(_reconcile_pending_buy_fills) 대상으로 남긴다"""
        self._enter()
        mock_balance.return_value = {"RemainMoney": 10_000_000}
        mock_make_buy.return_value = {"OrderNum": "123", "OrderNum2": "12345678", "OrderTime": "093000"}
        mock_get_fill.return_value = self._fill("12345678", 100, 70, 30)
        mock_cancel.return_value = False

        self.assertTrue(self.engine.execute_buy_order(self.config, 5_000_000, 50000))
        entry = TradeEntry.objects.get(trading_config=self.config, trade_type="BUY")
        self.assertEqual((entry.filled_quantity, entry.order_quantity), (70, 100))
        self.assertEqual(entry.status, "FILLED")

    @patch("dolpha.kis.trade.GetOrderFill")
    @patch("dolpha.kis.trade.MakeBuyMarketOrder")
    @patch("dolpha.kis.trade.GetBalance")
    def test_buy_fill_lookup_failure_falls_back_to_filled_estimate(
        self, mock_balance, mock_make_buy, mock_get_fill
    ):
        """체결 조회 실패: SUBMITTED/0 으로 남기면 보유분이 장부에서 사라지므로 추정 FILLED"""
        self._enter()
        mock_balance.return_value = {"RemainMoney": 10_000_000}
        mock_make_buy.return_value = {"OrderNum": "123", "OrderNum2": "12345678", "OrderTime": "093000"}
        mock_get_fill.side_effect = RuntimeError("network down")

        self.assertTrue(self.engine.execute_buy_order(self.config, 5_000_000, 50000))
        entry = TradeEntry.objects.get(trading_config=self.config, trade_type="BUY")
        self.assertEqual(entry.status, "FILLED")
        self.assertEqual(entry.filled_quantity, 100)
        self.assertIn("체결 미확인", entry.note)
        self.assertEqual(self.engine._buy_entries(self.config).count(), 1)

    # ── 매도 ──────────────────────────────────────────────

    def _sell_fill_side_effect(self, fills):
        """order_no 별로 순서대로 fill 을 돌려주는 GetOrderFill side_effect"""
        queues = {k: list(v) for k, v in fills.items()}

        def _f(order_no, *args, **kwargs):
            q = queues[order_no]
            return q.pop(0) if len(q) > 1 else q[0]
        return _f

    @patch("dolpha.kis.trade.GetMyStockList")
    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.MakeSellMarketOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    def test_sell_partial_cancels_remainder_then_reorders_only_actual_remaining(
        self, mock_get_fill, mock_make_sell, mock_cancel, mock_stocks
    ):
        """118주 중 43주만 기록된 사고 재현: 원주문 잔량을 취소하고 실제 남은 75주만 재주문한다"""
        self._enter()
        self._make_buy_entry(118)
        mock_make_sell.side_effect = [
            {"OrderNum": "1", "OrderNum2": "S1", "OrderTime": "100000"},
            {"OrderNum": "1", "OrderNum2": "S2", "OrderTime": "100001"},
        ]
        mock_cancel.return_value = True
        mock_get_fill.side_effect = self._sell_fill_side_effect({
            "S1": [
                self._fill("S1", 118, 43, 75),                    # 체결 확인
                self._fill("S1", 118, 43, 0, cancelled=True),     # 취소 후
            ],
            "S2": [self._fill("S2", 75, 75, 0)],
        })
        # 재주문 직전 계좌 75주, 최종 비활성화 시점 0주
        mock_stocks.side_effect = [
            [{"StockCode": self.stock_code, "StockAmt": 75, "StockAvgPrice": 50000}],
            [],
        ]

        ok = self.engine.execute_sell_order(
            self.config, reason="손절", current_price=51000,
            holding_info={"qty": 118, "avg_price": 50000.0},
        )

        self.assertTrue(ok)
        # 원주문(S1) 취소가 재주문(S2)보다 먼저, 재주문 수량은 75주
        mock_cancel.assert_called_once()
        self.assertEqual(mock_cancel.call_args.args[0], "S1")
        self.assertEqual([c.args[1] for c in mock_make_sell.call_args_list], [118, 75])
        s1 = TradeEntry.objects.get(order_no="S1")
        self.assertEqual((s1.filled_quantity, s1.order_quantity, s1.status), (43, 43, "FILLED"))
        total_sold = sum(
            e.filled_quantity for e in TradeEntry.objects.filter(trade_type="SELL")
        )
        self.assertEqual(total_sold, 118)  # 이중 매도 없음
        self.config.refresh_from_db()
        self.assertFalse(self.config.is_active)

    @patch("dolpha.kis.trade.GetMyStockList")
    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.MakeSellMarketOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    def test_sell_cancel_failure_does_not_reorder(
        self, mock_get_fill, mock_make_sell, mock_cancel, mock_stocks
    ):
        """원주문 잔량 취소에 실패하면 재주문하지 않고 설정을 활성 상태로 유지한다"""
        self._enter()
        self._make_buy_entry(118)
        mock_make_sell.return_value = {"OrderNum": "1", "OrderNum2": "S1", "OrderTime": "100000"}
        mock_cancel.return_value = False
        mock_get_fill.return_value = self._fill("S1", 118, 43, 75)

        ok = self.engine.execute_sell_order(
            self.config, reason="손절", current_price=51000,
            holding_info={"qty": 118, "avg_price": 50000.0},
        )

        self.assertTrue(ok)
        mock_make_sell.assert_called_once()
        mock_stocks.assert_not_called()
        self.config.refresh_from_db()
        self.assertTrue(self.config.is_active)

    @patch("dolpha.kis.trade.GetMyStockList")
    @patch("dolpha.kis.trade.CancelOrder")
    @patch("dolpha.kis.trade.MakeSellMarketOrder")
    @patch("dolpha.kis.trade.GetOrderFill")
    def test_retry_never_sells_more_than_account_holding(
        self, mock_get_fill, mock_make_sell, mock_cancel, mock_stocks
    ):
        """계좌 실보유(30주)가 재주문 필요 수량(75주)보다 적으면 30주만 주문한다"""
        self._enter()
        mock_stocks.side_effect = [
            [{"StockCode": self.stock_code, "StockAmt": 30, "StockAvgPrice": 50000}],
            [],  # 30주 매도 후 계좌 0
        ]
        mock_make_sell.return_value = {"OrderNum": "1", "OrderNum2": "R1", "OrderTime": "100000"}
        mock_get_fill.return_value = self._fill("R1", 30, 30, 0)

        unresolved = self.engine._retry_sell_order(self.config, 75, 51000, "STOP_LOSS", "손절")

        mock_make_sell.assert_called_once()
        self.assertEqual(mock_make_sell.call_args.args[1], 30)  # 75가 아니라 30
        mock_cancel.assert_not_called()
        self.assertFalse(unresolved)  # 이후 계좌 잔량 0 → 더 팔 것 없음

    @patch("dolpha.kis.trade.GetMyStockList")
    @patch("dolpha.kis.trade.MakeSellMarketOrder")
    def test_retry_skips_when_account_has_nothing_or_lookup_fails(self, mock_make_sell, mock_stocks):
        self._enter()
        mock_stocks.return_value = []
        self.assertFalse(self.engine._retry_sell_order(self.config, 75, 51000, "STOP_LOSS", "손절"))
        mock_stocks.side_effect = RuntimeError("down")
        self.assertTrue(self.engine._retry_sell_order(self.config, 75, 51000, "STOP_LOSS", "손절"))
        mock_make_sell.assert_not_called()

    @patch("dolpha.kis.trade.GetMyStockList")
    def test_account_qty_subtracts_only_same_account_other_strategies(self, mock_stocks):
        """같은 계좌의 다른 활성 전략 몫만 차감한다 (다른 계좌 전략은 무시)"""
        self._enter()
        other_same = TradingConfig.objects.create(
            user=self.user, stock_code=self.stock_code, stock_name="삼성전자",
            strategy_type="other_same", is_active=True,
        )
        other_diff = TradingConfig.objects.create(
            user=self.user, stock_code=self.stock_code, stock_name="삼성전자",
            strategy_type="other_diff", is_active=True,
        )
        for cfg in (other_same, other_diff):
            TradeEntry.objects.create(
                user=self.user, trading_config=cfg, stock_code=self.stock_code,
                stock_name="삼성전자", trade_type="BUY", entry_type="INITIAL",
                order_no=f"B{cfg.pk}", order_quantity=40, order_price=Decimal("50000"),
                filled_quantity=40, filled_price=Decimal("50000"),
                filled_amount=Decimal("2000000"), status="FILLED",
                ordered_at=tz.now(), filled_at=tz.now(),
            )

        def cred_for(cfg):
            key = "acct-B" if cfg.strategy_type == "other_diff" else "acct-A"
            return MagicMock(token_cache_key=key)

        stocks = [{"StockCode": self.stock_code, "StockAmt": 100, "StockAvgPrice": 50000}]
        with patch.object(self.engine, "_credential_for", side_effect=cred_for):
            # 계좌 100주 - 같은 계좌 other_same 40주 = 60 (other_diff 는 다른 계좌라 차감 안 함)
            self.assertEqual(self.engine._get_account_qty_for_strategy(self.config, stocks), 60)

    @patch("dolpha.kis.trade.GetMyStockList")
    def test_deactivate_config_checks_account_qty(self, mock_get_stocks):
        """비활성화 전 계좌 실보유 확인"""
        # 계좌에 종목이 있는 경우
        mock_get_stocks.return_value = [
            {"StockCode": self.stock_code, "StockAmt": 50, "StockAvgPrice": 50000}
        ]

        with patch.object(self.engine, "_credential_for") as mock_cred:
            mock_cred.return_value = MagicMock()
            result = self.engine._deactivate_config(self.config)

        self.assertFalse(result)  # 비활성화 거부
        self.assertTrue(self.config.is_active)  # 여전히 활성

        # 계좌에 종목이 없는 경우
        mock_get_stocks.return_value = []

        with patch.object(self.engine, "_credential_for") as mock_cred:
            mock_cred.return_value = MagicMock()
            result = self.engine._deactivate_config(self.config)

        self.assertTrue(result)  # 비활성화 성공
        self.assertFalse(self.config.is_active)  # 비활성화됨

    def test_update_trading_summary_closed_status_with_filled_qty(self):
        """CLOSED 판정: buy_filled_qty == sell_filled_qty"""
        # BUY 100주 체결
        buy_entry = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="INITIAL",
            order_no="11111111",
            order_quantity=100,
            order_price=Decimal("50000"),
            filled_quantity=100,
            filled_price=Decimal("50000"),
            filled_amount=Decimal("5000000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        # SELL 50주 체결
        sell_entry1 = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="SELL",
            entry_type="EXIT_PARTIAL",
            order_no="22222222",
            order_quantity=50,
            order_price=Decimal("51000"),
            filled_quantity=50,
            filled_price=Decimal("51000"),
            filled_amount=Decimal("2550000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        with patch.object(self.engine, "_buy_entries") as mock_buy_entries:
            mock_buy_entries.return_value = [buy_entry]

        # 50주만 매도 → HOLDING
        self.engine._update_trading_summary(self.config, self.stock_code, "삼성전자")
        summary = TradingSummary.objects.filter(
            user=self.user, stock_code=self.stock_code
        ).first()
        self.assertIsNotNone(summary)
        self.assertEqual(summary.final_status, "HOLDING")

        # SELL 50주 추가 체결 → CLOSED
        sell_entry2 = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="SELL",
            entry_type="EXIT_FULL",
            order_no="33333333",
            order_quantity=50,
            order_price=Decimal("51500"),
            filled_quantity=50,
            filled_price=Decimal("51500"),
            filled_amount=Decimal("2575000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        with patch.object(self.engine, "_buy_entries") as mock_buy_entries:
            mock_buy_entries.return_value = [buy_entry]

        self.engine._update_trading_summary(self.config, self.stock_code, "삼성전자")
        summary.refresh_from_db()
        self.assertEqual(summary.final_status, "CLOSED")

    def test_config_net_position_with_partial_sells(self):
        """순보유 계산: 부분매도 여러 건 처리"""
        # BUY 100주
        buy_entry = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="INITIAL",
            order_no="11111111",
            order_quantity=100,
            order_price=Decimal("50000"),
            filled_quantity=100,
            filled_price=Decimal("50000"),
            filled_amount=Decimal("5000000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        # SELL 30주 (부분체결)
        sell1 = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="SELL",
            entry_type="EXIT_PARTIAL",
            order_no="22222222",
            order_quantity=30,
            order_price=Decimal("51000"),
            filled_quantity=20,  # 30주 중 20주만 체결
            filled_price=Decimal("51000"),
            filled_amount=Decimal("1020000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        # SELL 10주 (재주문)
        sell2 = TradeEntry.objects.create(
            user=self.user,
            trading_config=self.config,
            stock_code=self.stock_code,
            stock_name="삼성전자",
            trade_type="SELL",
            entry_type="EXIT_PARTIAL",
            order_no="33333333",
            order_quantity=10,
            order_price=Decimal("51200"),
            filled_quantity=10,  # 재주문 10주 모두 체결
            filled_price=Decimal("51200"),
            filled_amount=Decimal("512000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        with patch.object(self.engine, "_buy_entries") as mock_buy_entries:
            mock_buy_entries.return_value = [buy_entry]

        pos = self.engine._config_net_position(self.config)
        # 100 - 20 - 10 = 70
        self.assertEqual(pos.qty, 70)
        self.assertEqual(pos.avg_price, 50000)  # 평단가 유지


class CommonRiskManagementTest(TestCase):
    """공통 리스크 관리 (단일 종목 20% 비중 상한 & 피라미딩 리스크 중립화) 테스트"""

    def setUp(self):
        self.user = User.objects.create_user(
            username="riskuser", email="risk@example.com", password="testpass"
        )
        self.engine = TradingEngine(self.user)

    def test_position_cap_pct_for_all_strategies(self):
        """모든 전략(htf, mtt, weekly_high, theme_surge 등)에 기본 20.0% 상한 적용 확인"""
        from dolpha.theme_surge.config import DEFAULT_MAX_POSITION_PCT

        for strat in ["htf", "mtt", "fifty_day_high", "weekly_high", "theme_surge"]:
            config = TradingConfig.objects.create(
                user=self.user,
                stock_code="005930",
                stock_name="삼성전자",
                strategy_type=strat,
                is_active=True,
            )
            cap = self.engine._position_cap_pct(config)
            self.assertEqual(cap, DEFAULT_MAX_POSITION_PCT)
            self.assertEqual(cap, 20.0)

    def test_pyramiding_risk_neutral_stop_theme_surge(self):
        """theme_surge 피라미딩 시 손절가(theme_pullback_low)는 진입 눌림저점 그대로 유지되는지 검증"""
        config = TradingConfig.objects.create(
            user=self.user,
            stock_code="005930",
            stock_name="삼성전자",
            strategy_type="theme_surge",
            trading_mode="manual",
            theme_pullback_low=48000.0,
            is_active=True,
        )
        # 1차 매수: 50,000원에 10주
        entry1 = TradeEntry.objects.create(
            user=self.user,
            trading_config=config,
            stock_code="005930",
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="INITIAL",
            order_no="10001",
            order_quantity=10,
            filled_quantity=10,
            filled_price=Decimal("50000"),
            filled_amount=Decimal("500000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )
        # 2차 매수(피라미딩): 55,000원에 10주 -> 신규 평단 52,500원
        entry2 = TradeEntry.objects.create(
            user=self.user,
            trading_config=config,
            stock_code="005930",
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="PYRAMIDING",
            order_no="10002",
            order_quantity=10,
            filled_quantity=10,
            filled_price=Decimal("55000"),
            filled_amount=Decimal("550000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        fill = {"avg_price": 55000, "filled_qty": 10}
        self.engine._apply_risk_neutral_pyramiding_stop(config, entry2, fill)

        config.refresh_from_db()
        entry2.refresh_from_db()
        # 급등테마주는 본전 상향 없이 눌림저점 48,000원 유지
        self.assertEqual(config.theme_pullback_low, 48000.0)
        self.assertIsNone(entry2.stop_price)

    def test_pyramiding_risk_neutral_stop_manual(self):
        """manual 모드 피라미딩 시 stop_loss %가 본전(0.0%)으로 상향 조정되는지 검증"""
        config = TradingConfig.objects.create(
            user=self.user,
            stock_code="005930",
            stock_name="삼성전자",
            strategy_type="mtt",
            trading_mode="manual",
            stop_loss=8.0,
            is_active=True,
        )
        # 1차 매수: 50,000원에 10주
        entry1 = TradeEntry.objects.create(
            user=self.user,
            trading_config=config,
            stock_code="005930",
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="INITIAL",
            order_no="20001",
            order_quantity=10,
            filled_quantity=10,
            filled_price=Decimal("50000"),
            filled_amount=Decimal("500000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )
        # 2차 매수(피라미딩): 52,000원에 10주 -> 신규 평단 51,000원
        entry2 = TradeEntry.objects.create(
            user=self.user,
            trading_config=config,
            stock_code="005930",
            stock_name="삼성전자",
            trade_type="BUY",
            entry_type="PYRAMIDING",
            order_no="20002",
            order_quantity=10,
            filled_quantity=10,
            filled_price=Decimal("52000"),
            filled_amount=Decimal("520000"),
            status="FILLED",
            ordered_at=tz.now(),
            filled_at=tz.now(),
        )

        fill = {"avg_price": 52000, "filled_qty": 10}
        self.engine._apply_risk_neutral_pyramiding_stop(config, entry2, fill)

        config.refresh_from_db()
        entry2.refresh_from_db()
        # 목표 손절가는 max(50,000, 51,000) = 51,000원 (신규 평단가 Break-even).
        # 신규 평단가 이상 방어이므로 stop_loss는 0.0%로 상향
        self.assertEqual(config.stop_loss, 0.0)
        self.assertEqual(entry2.stop_price, Decimal("51000.00"))


if __name__ == "__main__":
    unittest.main()



class GetOrderFillCancelParsingTest(TestCase):
    """원주문 행은 cncl_yn=N 이고 cncl_cfrm_qty 로만 취소가 표시된다 (2026-10-19 삼성SDI 잔량 오판)."""

    @patch("dolpha.kis.trade.requests.get")
    @patch("dolpha.kis.trade.GetHeaders", return_value={})
    @patch("dolpha.kis.trade._account_params", return_value={})
    @patch("dolpha.kis.trade._sleep")
    @patch("dolpha.kis.trade.resolve_credential")
    def test_cancel_confirmed_qty_marks_cancelled(self, _cred, _sleep, _acct, _hdr, mock_get):
        from dolpha.kis import trade as KIS

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "rt_cd": "0",
            "output1": [{
                "odno": "0000023393", "pdno": "006400", "ord_gno_brno": "00950",
                "ord_qty": "88", "tot_ccld_qty": "30", "tot_ccld_amt": "17027500",
                "rmn_qty": "0", "cncl_yn": "N", "cncl_cfrm_qty": "58",
            }],
        }
        fill = KIS.GetOrderFill("0000023393", stock_code="006400", order_date="20261019")
        self.assertTrue(fill["cancelled"])
        self.assertEqual(fill["remain_qty"], 0)
        self.assertEqual(fill["filled_qty"], 30)


class RefetchSellProfitLossTest(TestCase):
    """재조회로 매도 체결수량이 늘면 손익도 늘어난 수량 기준으로 다시 계산돼야 한다 (2026-10-19 삼성SDI)."""

    def setUp(self):
        self.user = User.objects.create_user(username="pl_user", password="pw")
        self.engine = TradingEngine(self.user)

    @patch("dolpha.kis.trade.GetOrderFill")
    def test_profit_loss_rescaled_on_refetch(self, mock_fill):
        config = TradingConfig.objects.create(
            user=self.user, stock_code="006400", stock_name="삼성SDI",
            strategy_type="theme_surge", is_active=True,
        )
        # 최초 확인 시 2주 @567,750 체결, 평단 568,739 → 손익 -1,978
        entry = TradeEntry.objects.create(
            user=self.user, trading_config=config, stock_code="006400", stock_name="삼성SDI",
            trade_type="SELL", entry_type="STOP_LOSS", order_no="23393",
            order_quantity=88, filled_quantity=2, filled_price=Decimal("567750"),
            filled_amount=Decimal("1135500"), profit_loss=Decimal("-1978"),
            status="FILLED", ordered_at=tz.now(), filled_at=tz.now(),
        )
        mock_fill.return_value = {
            "filled_qty": 30, "avg_price": 567583, "filled_amount": 17027500,
            "remain_qty": 0, "cancelled": True,
        }
        with patch.object(TradingEngine, "_credential_for", return_value=None):
            self.assertTrue(self.engine._refetch_and_apply_fill(config, entry))

        entry.refresh_from_db()
        self.assertEqual(entry.filled_quantity, 30)
        # 30주 × (567,583 - 568,739) ≈ -34,680
        self.assertAlmostEqual(float(entry.profit_loss), 17027500 - 568739 * 30, delta=1)
        self.assertAlmostEqual(entry.profit_loss_percent, -0.2033, places=2)
