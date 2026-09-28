"""급등테마주 리스크 게이트·섀도 필터·이월 수익쿠션 테스트 (docs/13 Phase 0·1).

- entry_gates: 신규 진입 마감·서킷브레이커(손실 횟수/손실 %)·품질 필터(섀도/강제)
- entry_context.load_daily_stats: 라운드 단위 손실 집계
- overnight: unrealized_r / apply_profit_cushion
- TradingEngine: 게이트가 진입 신호를 막고 맥락을 기록하는지, 이월 포지션 본전 손절
"""

from datetime import datetime, time, timedelta
from decimal import Decimal
from types import MappingProxyType
from unittest.mock import patch

import pytz
from django.test import TestCase
from django.utils import timezone as tz

from dolpha.theme_surge.entry import EntryDecision
from dolpha.theme_surge.entry_gates import (
    DailyStats,
    EntryGateSettings,
    check_risk_gates,
    evaluate_quality_filters,
    load_entry_gate_settings,
)
from dolpha.theme_surge.overnight import (
    OvernightSignal,
    apply_profit_cushion,
    unrealized_r,
)

KST = pytz.timezone("Asia/Seoul")
_GATES = EntryGateSettings(entry_cutoff=time(14, 0), daily_max_losses=1, daily_max_loss_pct=1.5)
_NO_TRADES = DailyStats(rounds_started=0, realized_losses=0, realized_pnl=0.0)


class RiskGateTests(TestCase):
    def test_passes_when_nothing_happened(self):
        result = check_risk_gates(_GATES, time(10, 0), _NO_TRADES, 100_000_000)
        self.assertFalse(result.blocked)
        self.assertEqual(result.reason, "")

    def test_blocks_at_or_after_cutoff(self):
        result = check_risk_gates(_GATES, time(14, 0), _NO_TRADES, 100_000_000)
        self.assertTrue(result.blocked)
        self.assertIn("신규 진입 마감", result.reason)

    def test_blocks_after_first_loss(self):
        stats = DailyStats(rounds_started=1, realized_losses=1, realized_pnl=-300_000)
        result = check_risk_gates(_GATES, time(10, 0), stats, 100_000_000)
        self.assertTrue(result.blocked)
        self.assertIn("당일 손실 1회", result.reason)

    def test_blocks_on_loss_pct(self):
        gates = EntryGateSettings(time(14, 0), daily_max_losses=0, daily_max_loss_pct=1.5)
        stats = DailyStats(rounds_started=3, realized_losses=3, realized_pnl=-1_600_000)
        result = check_risk_gates(gates, time(10, 0), stats, 100_000_000)
        self.assertTrue(result.blocked)
        self.assertIn("실현손실 1.60%", result.reason)

    def test_zero_limits_disable_breaker(self):
        gates = EntryGateSettings(time(15, 30), daily_max_losses=0, daily_max_loss_pct=0.0)
        stats = DailyStats(rounds_started=5, realized_losses=5, realized_pnl=-9_000_000)
        self.assertFalse(check_risk_gates(gates, time(10, 0), stats, 100_000_000).blocked)

    def test_missing_capital_skips_only_pct_limit(self):
        gates = EntryGateSettings(time(14, 0), daily_max_losses=0, daily_max_loss_pct=1.5)
        stats = DailyStats(rounds_started=2, realized_losses=2, realized_pnl=-5_000_000)
        result = check_risk_gates(gates, time(10, 0), stats, None)
        self.assertFalse(result.blocked)
        self.assertIsNone(result.flags["realized_loss_pct"])

    def test_load_settings_defaults_and_zero(self):
        self.assertEqual(load_entry_gate_settings(None).daily_max_losses, 1)

        class _Defaults:
            theme_surge_daily_max_losses = 0
            theme_surge_daily_max_loss_pct = 0.0
            theme_surge_entry_cutoff = time(15, 0)

        loaded = load_entry_gate_settings(_Defaults())
        self.assertEqual(loaded.daily_max_losses, 0)   # 0 은 '미사용'으로 보존
        self.assertEqual(loaded.daily_max_loss_pct, 0.0)
        self.assertEqual(loaded.entry_cutoff, time(15, 0))


class QualityFilterTests(TestCase):
    def test_shadow_records_failures_without_blocking(self):
        result = evaluate_quality_filters(3.2, 5.0, 10_300, 10_000, mode="shadow")
        self.assertFalse(result.blocked)
        self.assertEqual(
            result.flags["failed"], ["theme_min_fluctuation", "breakout_volume_max", "chase_max"]
        )
        self.assertEqual(result.flags["checks"]["chase_max"]["value"], 3.0)

    def test_enforce_blocks_with_reason(self):
        result = evaluate_quality_filters(3.2, 2.0, 10_100, 10_000, mode="enforce")
        self.assertTrue(result.blocked)
        self.assertIn("테마 등락률 3.20% < 4%", result.reason)

    def test_unknown_values_pass(self):
        result = evaluate_quality_filters(None, None, 10_000, None, mode="enforce")
        self.assertFalse(result.blocked)
        self.assertIsNone(result.flags["checks"]["theme_min_fluctuation"]["pass"])


class ProfitCushionTests(TestCase):
    def _signal(self, should_hold=True):
        return OvernightSignal(
            available=True, met=MappingProxyType({"program": True}), met_count=1,
            required=1, should_hold=should_hold, detail="[실시간] 프로그램 순매수✓ → 1/1 충족",
        )

    def test_unrealized_r(self):
        self.assertAlmostEqual(unrealized_r(100.0, 104.0, 95.0), 0.8)
        self.assertIsNone(unrealized_r(100.0, 104.0, None))
        self.assertIsNone(unrealized_r(100.0, 104.0, 100.0))  # 손절가=평단 → R 산출 불가

    def test_cushion_met_keeps_hold(self):
        result = apply_profit_cushion(self._signal(), 0.8, 0.5)
        self.assertTrue(result.should_hold)
        self.assertIn("수익쿠션 +0.80R✓", result.detail)

    def test_cushion_short_cancels_hold(self):
        result = apply_profit_cushion(self._signal(), 0.2, 0.5)
        self.assertFalse(result.should_hold)
        self.assertEqual(result.met_count, 1)   # 수급 판정 결과는 보존

    def test_unknown_r_cancels_hold(self):
        self.assertFalse(apply_profit_cushion(self._signal(), None, 0.5).should_hold)

    def test_supply_unmet_stays_unmet(self):
        self.assertFalse(apply_profit_cushion(self._signal(False), 3.0, 0.5).should_hold)


class _EngineFixture(TestCase):
    def setUp(self):
        from django.contrib.auth import get_user_model
        from myweb.models import TradingConfig, TradingDefaults
        from dolpha.trading_engine import TradingEngine

        self.user = get_user_model().objects.create(username="gate_tester")
        TradingDefaults.objects.create(user=self.user, theme_surge_enabled=True)
        self.config = TradingConfig.objects.create(
            user=self.user, stock_code="067310", stock_name="하나마이크론",
            trading_mode="manual", strategy_type="theme_surge", is_active=True,
        )
        self.engine = TradingEngine(user=self.user)

    def _fill(self, config, trade_type, entry_type, when, profit_loss=None):
        from myweb.models import TradeEntry

        return TradeEntry.objects.create(
            user=self.user, trading_config=config, stock_code=config.stock_code,
            stock_name=config.stock_name, trade_type=trade_type, entry_type=entry_type,
            order_quantity=10, filled_quantity=10, filled_price=Decimal("1000"),
            status="FILLED", filled_at=when,
            profit_loss=Decimal(str(profit_loss)) if profit_loss is not None else None,
        )


class DailyStatsTests(_EngineFixture):
    def test_counts_rounds_by_terminal_sell(self):
        from myweb.models import TradingConfig
        from dolpha.theme_surge.entry_context import load_daily_stats

        now = tz.localtime().replace(hour=13, minute=0)
        other = TradingConfig.objects.create(
            user=self.user, stock_code="010170", stock_name="대한광통신",
            trading_mode="manual", strategy_type="theme_surge", is_active=True,
        )
        at = lambda h, m: now.replace(hour=h, minute=m)  # noqa: E731

        # 라운드1: 분할익절 +50만 후 손절 −20만 → 라운드 합계 +30만 (손실 아님)
        self._fill(self.config, "BUY", "INITIAL", at(9, 30))
        self._fill(self.config, "SELL", "EXIT_PARTIAL", at(9, 50), 500_000)
        self._fill(self.config, "SELL", "STOP_LOSS", at(10, 10), -200_000)
        # 라운드2: 전일 이월분 손절 −40만 (오늘 BUY 없음, 손실로 집계)
        self._fill(other, "SELL", "STOP_LOSS", at(9, 5), -400_000)
        # 어제 체결은 제외
        self._fill(other, "SELL", "STOP_LOSS", at(9, 5) - timedelta(days=1), -999_999)

        stats = load_daily_stats(self.user, now)
        self.assertEqual(stats.rounds_started, 1)
        self.assertEqual(stats.realized_losses, 1)
        self.assertAlmostEqual(stats.realized_pnl, -100_000)


class EngineGateWiringTests(_EngineFixture):
    _PASSED = EntryDecision(
        passed=True, reason="눌림 → 돌파 (외국인 필터 미사용)", prev_high=10_000.0,
        pullback_low=9_800.0, pullback_pct=2.0, volume_ratio=2.0,
        has_pullback=True, has_breakout=True,
    )

    def _run_entry(self, hour, minute):
        from myweb.models import ThemeEntrySignal

        now = KST.localize(datetime.combine(tz.localdate(), time(hour, minute)))
        with patch("dolpha.theme_surge.check_theme_surge_entry", return_value=self._PASSED), \
                patch("dolpha.trading_engine.tz.localtime", return_value=now):
            passed = self.engine.check_theme_surge_entry(self.config, 10_050.0)
        return passed, ThemeEntrySignal.objects.filter(user=self.user).latest("checked_at")

    def test_signal_passes_before_cutoff_and_records_context(self):
        passed, signal = self._run_entry(10, 30)
        self.assertTrue(passed)
        self.assertEqual(signal.day_trade_seq, 1)
        self.assertEqual(signal.day_realized_losses, 0)
        from dolpha.theme_surge.config import ENTRY_FILTER_MODE
        self.assertEqual(signal.gate_flags["quality"]["mode"], ENTRY_FILTER_MODE)
        self.assertEqual(signal.gate_flags["risk"]["blocked_by"], "")

    def test_signal_blocked_after_cutoff(self):
        passed, signal = self._run_entry(14, 5)
        self.assertFalse(passed)
        self.assertFalse(signal.passed)
        self.assertTrue(signal.reason.startswith("신규 진입 마감"))
        self.assertIn("신호: 눌림", signal.reason)   # 막힌 원래 신호 보존

    def test_signal_blocked_after_first_loss(self):
        now = KST.localize(datetime.combine(tz.localdate(), time(9, 40)))
        self._fill(self.config, "BUY", "INITIAL", now)
        self._fill(self.config, "SELL", "STOP_LOSS", now + timedelta(minutes=20), -300_000)
        passed, signal = self._run_entry(10, 30)
        self.assertFalse(passed)
        self.assertIn("서킷브레이커", signal.reason)
        self.assertEqual(signal.day_realized_losses, 1)


class EffectiveStopTests(_EngineFixture):
    def _settings(self, breakeven=True):
        from dolpha.theme_surge.exit_rules import load_exit_settings

        settings = load_exit_settings(None)
        return settings.__class__(**{**settings.__dict__, "overnight_breakeven_stop": breakeven})

    def test_same_day_uses_pullback_low(self):
        self.config.theme_pullback_low = 950.0
        self.assertEqual(
            self.engine._theme_effective_stop(self.config, self._settings(), 1000.0, 1),
            (950.0, "눌림저점"),
        )

    def test_carried_position_uses_breakeven(self):
        self.config.theme_pullback_low = 950.0
        self.assertEqual(
            self.engine._theme_effective_stop(self.config, self._settings(), 1000.0, 2),
            (1000.0, "이월 본전"),
        )

    def test_breakeven_disabled(self):
        self.config.theme_pullback_low = 950.0
        stop, _ = self.engine._theme_effective_stop(
            self.config, self._settings(breakeven=False), 1000.0, 2
        )
        self.assertEqual(stop, 950.0)
