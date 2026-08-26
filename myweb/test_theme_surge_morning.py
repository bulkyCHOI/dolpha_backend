"""급등테마주(theme_surge) 모닝 세션 및 듀얼 모드 종합 단위 테스트."""

import calendar
from datetime import date, datetime, time
from django.test import TestCase

from dolpha.theme_surge.config import (
    ENTRY_MIN_BARS,
    LEADER_MIN_TRADING_VALUE,
    MORNING_BREAKOUT_VOLUME_RATIO_MIN,
    MORNING_LOOKBACK_BARS,
    MORNING_MIN_BARS,
    MORNING_PEAK_MIN_RISE_PCT,
    MORNING_PULLBACK_MAX_BARS,
    MORNING_PULLBACK_MAX_PCT,
    MORNING_PULLBACK_MIN_BARS,
    MORNING_PULLBACK_MIN_PCT,
    MORNING_PULLBACK_VOLUME_RATIO_MAX,
    SURGE_MIN_TRADING_VALUE,
    THEME_PULLBACK_MOMENTUM_TOLERANCE_PCT,
    get_leader_min_trading_value,
    get_theme_min_trading_value,
)
from dolpha.theme_surge.detector import detect_surge_themes
from dolpha.theme_surge.entry import EntryDecision, check_theme_surge_entry
from dolpha.theme_surge.entry_chart import _best_window, _chart_time, _geometry
from dolpha.theme_surge.foreign_flow import ForeignFlow, get_foreign_flow
from dolpha.theme_surge.leader import select_leaders
from dolpha.theme_surge.patterns import (
    Breakout,
    Pullback,
    SwingHigh,
    analyze_breakout,
    analyze_morning_breakout,
    analyze_morning_pullback,
    analyze_pullback,
    find_last_swing_high,
    find_morning_high,
)
from dolpha.theme_surge.toss_client import ThemeRank, ThemeStock


class DynamicTradingValueTests(TestCase):
    """시간대별 동적 최소 거래대금 테스트."""

    def test_theme_min_trading_value(self):
        self.assertEqual(get_theme_min_trading_value(time(9, 5)), 10_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 10)), 10_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 15)), 20_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 20)), 20_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 25)), 35_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 30)), 35_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(9, 35)), 50_000_000_000)
        self.assertEqual(get_theme_min_trading_value(time(14, 0)), 50_000_000_000)
        self.assertEqual(get_theme_min_trading_value(None), 50_000_000_000)

    def test_leader_min_trading_value(self):
        self.assertEqual(get_leader_min_trading_value(time(9, 5)), 1_000_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 10)), 1_000_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 15)), 2_000_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 20)), 2_000_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 25)), 3_500_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 30)), 3_500_000_000)
        self.assertEqual(get_leader_min_trading_value(time(9, 35)), 5_000_000_000)
        self.assertEqual(get_leader_min_trading_value(None), 5_000_000_000)


class MorningPatternTests(TestCase):
    """모닝 세션 패턴 탐색 및 판정 테스트."""

    def test_find_morning_high_success(self):
        bars = [
            {"time": "09:00", "open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"time": "09:01", "open": 10100, "high": 10300, "low": 10050, "close": 10250, "volume": 1200},
            {"time": "09:02", "open": 10250, "high": 10500, "low": 10200, "close": 10450, "volume": 2000},
            {"time": "09:03", "open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 800},
            {"time": "09:04", "open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 700},
            {"time": "09:05", "open": 10300, "high": 10550, "low": 10280, "close": 10520, "volume": 2500},
        ]
        swing = find_morning_high(bars, exclude_last=1)
        self.assertIsNotNone(swing)
        self.assertEqual(swing.index, 2)
        self.assertEqual(swing.price, 10500)

    def test_find_morning_high_insufficient_rise(self):
        bars = [
            {"time": "09:00", "open": 10000, "high": 10100, "low": 9950, "close": 10050, "volume": 1000},
            {"time": "09:01", "open": 10050, "high": 10150, "low": 10000, "close": 10100, "volume": 1200},
            {"time": "09:02", "open": 10100, "high": 10100, "low": 10050, "close": 10080, "volume": 800},
            {"time": "09:03", "open": 10080, "high": 10090, "low": 10040, "close": 10060, "volume": 700},
            {"time": "09:04", "open": 10060, "high": 10200, "low": 10050, "close": 10180, "volume": 1500},
        ]
        swing = find_morning_high(bars, exclude_last=1)
        self.assertIsNone(swing)

    def test_find_morning_high_not_enough_pullback_bars(self):
        bars = [
            {"time": "09:00", "open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"time": "09:01", "open": 10100, "high": 10500, "low": 10050, "close": 10450, "volume": 2000},
            {"time": "09:02", "open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 800},
            {"time": "09:03", "open": 10350, "high": 10550, "low": 10300, "close": 10520, "volume": 2500},
        ]
        swing = find_morning_high(bars, exclude_last=1)
        self.assertIsNone(swing)

    def test_analyze_morning_pullback_success(self):
        bars = [
            {"open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 500},
            {"open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 400},
            {"open": 10300, "high": 10550, "low": 10280, "close": 10520, "volume": 3000},
        ]
        swing = SwingHigh(index=1, price=10500)
        pullback = analyze_morning_pullback(bars, swing)
        self.assertTrue(pullback.is_valid)
        self.assertEqual(pullback.bar_count, 2)
        self.assertAlmostEqual(pullback.low, 10250)
        self.assertGreater(pullback.depth_pct, 1.0)
        self.assertLess(pullback.depth_pct, 4.5)
        self.assertLessEqual(pullback.volume_ratio, 0.85)

    def test_analyze_morning_pullback_open_breach(self):
        bars = [
            {"open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"open": 10450, "high": 10400, "low": 9900, "close": 9950, "volume": 500},
            {"open": 9950, "high": 10050, "low": 9920, "close": 10020, "volume": 400},
            {"open": 10020, "high": 10550, "low": 10000, "close": 10520, "volume": 3000},
        ]
        swing = SwingHigh(index=1, price=10500)
        pullback = analyze_morning_pullback(bars, swing)
        self.assertFalse(pullback.is_valid)
        self.assertIn("시초가 이탈", pullback.reason)

    def test_analyze_morning_pullback_too_deep(self):
        # 10500 -> 9975 (깊이 5.0% > 4.5%)
        bars = [
            {"open": 9900, "high": 10200, "low": 9900, "close": 10100, "volume": 1000},
            {"open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"open": 10450, "high": 10400, "low": 9975, "close": 10000, "volume": 500},
            {"open": 10000, "high": 10050, "low": 9980, "close": 10020, "volume": 400},
            {"open": 10020, "high": 10550, "low": 10000, "close": 10520, "volume": 3000},
        ]
        swing = SwingHigh(index=1, price=10500)
        pullback = analyze_morning_pullback(bars, swing)
        self.assertFalse(pullback.is_valid)
        self.assertIn("모닝 눌림 과다", pullback.reason)

    def test_analyze_morning_breakout_success(self):
        bars = [
            {"open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 500},
            {"open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 400},
            {"open": 10300, "high": 10550, "low": 10280, "close": 10520, "volume": 1000},
        ]
        swing = SwingHigh(index=1, price=10500)
        breakout = analyze_morning_breakout(bars, swing, current_price=10520)
        self.assertTrue(breakout.is_valid)
        self.assertGreater(breakout.volume_ratio, 2.0)

    def test_analyze_morning_breakout_5bar_avg_fallback(self):
        # 눌림 평균 대비는 2.0배 미달이지만 최근 5봉 평균 대비 1.5배 만족 시
        bars = [
            {"open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 300},
            {"open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 400},
            {"open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 500},
            {"open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 500},
            {"open": 10300, "high": 10550, "low": 10280, "close": 10520, "volume": 700}, # 5봉 평균 ~425, 700 >= 425*1.5=637.5
        ]
        swing = SwingHigh(index=1, price=10500)
        breakout = analyze_morning_breakout(bars, swing, current_price=10520)
        self.assertTrue(breakout.is_valid)


class ForeignFlowColdStartTests(TestCase):
    """외국인 수급 콜드 스타트 버그 수정 검증."""

    def test_cold_start_net_buy_positive(self):
        flow = ForeignFlow(
            available=True,
            net_buy_qty=1500,
            is_increasing=True,
            source="member-firm",
            detail="외국계 순매수 +1,500주 (당일 초기 순매수)",
        )
        self.assertTrue(flow.is_buying)
        self.assertTrue(flow.is_increasing)


class ThemeDetectorMomentumPreservationTests(TestCase):
    """테마 탐지기 동적 거래대금 및 눌림목 모멘텀 보존 테스트."""

    def test_dynamic_trading_value_filter(self):
        theme = ThemeRank(
            tics_id=1,
            name="AI반도체",
            rank=1,
            fluctuation_rate=5.0,
            trading_value=15_000_000_000,
            market_cap=500_000_000_000,
            stock_count=5,
            leading_stock_code="005930",
            leading_stock_name="삼성전자",
        )
        verdicts_morning = detect_surge_themes([theme], slot=time(9, 5))
        self.assertTrue(verdicts_morning[0].is_surge)

        verdicts_regular = detect_surge_themes([theme], slot=time(9, 35))
        self.assertFalse(verdicts_regular[0].is_surge)
        self.assertIn("거래대금 미달", verdicts_regular[0].reason)

    def test_momentum_preservation_in_pullback(self):
        theme = ThemeRank(
            tics_id=1,
            name="로봇",
            rank=1,
            fluctuation_rate=6.8,
            trading_value=60_000_000_000,
            market_cap=500_000_000_000,
            stock_count=5,
            leading_stock_code="005930",
            leading_stock_name="삼성전자",
        )
        prev_rates = {1: 6.7}
        today_max_rates = {1: 7.5}

        verdicts = detect_surge_themes(
            [theme],
            prev_rates=prev_rates,
            slot=time(9, 25),
            today_max_rates=today_max_rates,
        )
        self.assertTrue(verdicts[0].is_surge)
        self.assertIn("눌림목 모멘텀 유지", verdicts[0].reason)


class LeaderCandidateSelectionTests(TestCase):
    """주도주 선정 시간대별 동적 거래대금 필터 테스트."""

    def test_select_leaders_dynamic_trading_value(self):
        stocks = [
            ThemeStock(
                code="005930",
                name="삼성전자",
                price=70000,
                change_rate=5.0,
                trading_value=1_500_000_000, # 15억
                volume=20000,
                market_cap=400_000_000_000_000,
            ),
        ]
        # 09:05 슬롯 (기준 10억) -> 통과
        leaders_morning = select_leaders(stocks, slot=time(9, 5))
        self.assertEqual(len(leaders_morning), 1)

        # 09:35 슬롯 (기준 50억) -> 탈락
        leaders_regular = select_leaders(stocks, slot=time(9, 35))
        self.assertEqual(len(leaders_regular), 0)


class EntryDualModeTests(TestCase):
    """진입 판정 듀얼 모드 (모닝 vs 레귤러) 테스트."""

    def test_insufficient_bars_under_6(self):
        bars = [
            {"time": "09:00", "open": 10000, "high": 10100, "low": 9900, "close": 10050, "volume": 1000},
            {"time": "09:01", "open": 10050, "high": 10200, "low": 10000, "close": 10150, "volume": 1000},
            {"time": "09:02", "open": 10150, "high": 10300, "low": 10100, "close": 10250, "volume": 1000},
        ]
        decision = check_theme_surge_entry("005930", current_price=10300, bars=bars, use_foreign_filter=False)
        self.assertFalse(decision.passed)
        self.assertIn("분봉 부족", decision.reason)
        self.assertIn("< 6봉", decision.reason)

    def test_morning_session_entry_success(self):
        bars = [
            {"time": "09:00", "open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"time": "09:01", "open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"time": "09:02", "open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 400},
            {"time": "09:03", "open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 350},
            {"time": "09:04", "open": 10300, "high": 10400, "low": 10280, "close": 10380, "volume": 300},
            {"time": "09:05", "open": 10380, "high": 10550, "low": 10350, "close": 10530, "volume": 1500},
        ]
        decision = check_theme_surge_entry("005930", current_price=10530, bars=bars, use_foreign_filter=False)
        self.assertTrue(decision.passed)
        self.assertTrue(decision.has_pullback)
        self.assertTrue(decision.has_breakout)
        self.assertEqual(decision.prev_high, 10500)
        self.assertEqual(decision.pullback_low, 10250)


class EntryChartGeometryTests(TestCase):
    """차트 기하좌표 복원 모닝 세션 테스트."""

    def test_best_window_and_geometry_morning(self):
        target_date = date(2026, 8, 26)
        def make_epoch(h, m):
            return calendar.timegm((2026, 8, 26, h, m, 0, 0, 0, 0))

        bars = [
            {"time": make_epoch(9, 0), "label": "09:00", "open": 10000, "high": 10200, "low": 9950, "close": 10100, "volume": 1000},
            {"time": make_epoch(9, 1), "label": "09:01", "open": 10100, "high": 10500, "low": 10100, "close": 10450, "volume": 2000},
            {"time": make_epoch(9, 2), "label": "09:02", "open": 10450, "high": 10400, "low": 10300, "close": 10350, "volume": 400},
            {"time": make_epoch(9, 3), "label": "09:03", "open": 10350, "high": 10350, "low": 10250, "close": 10300, "volume": 350},
            {"time": make_epoch(9, 4), "label": "09:04", "open": 10300, "high": 10400, "low": 10280, "close": 10380, "volume": 300},
            {"time": make_epoch(9, 5), "label": "09:05", "open": 10380, "high": 10450, "low": 10350, "close": 10400, "volume": 320},
            {"time": make_epoch(9, 6), "label": "09:06", "open": 10400, "high": 10550, "low": 10380, "close": 10530, "volume": 1500},
        ]
        class MockSignal:
            id = 1
            price = 10530
            prev_high = 10500
            pullback_low = 10250
            pullback_pct = 2.38
            volume_ratio = 4.28
            foreign_net_buy = 1000
            has_pullback = True
            has_breakout = True
            has_foreign_buying = True
            passed = True
            executed = False
            reason = "테스트"

        checked = datetime(2026, 8, 26, 9, 7)
        geom = _geometry(bars, MockSignal(), checked)
        self.assertIsNotNone(geom)
        self.assertTrue(geom["verified"])
        self.assertEqual(geom["swing_high"]["price"], 10500)
        self.assertEqual(geom["pullback_low"]["price"], 10250)
        self.assertIsNotNone(geom["pullback_zone"])
