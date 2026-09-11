"""급등테마주 오버나이트 보유 판정 단위 테스트.

- normalize_conditions: 조건 키 정규화
- evaluate_overnight_signal: 수급 4조건 실시간 판정 (KIS 조회는 mock)
- evaluate_exit: 강제청산 시각의 오버나이트 이월 / 보유기간 만료 분기
"""

from datetime import datetime, time
from unittest.mock import patch

import pytz
from django.test import TestCase

from dolpha.theme_surge import overnight as overnight_mod
from dolpha.theme_surge.exit_rules import ExitSettings, evaluate_exit
from dolpha.theme_surge.overnight import (
    evaluate_overnight_signal,
    normalize_conditions,
)

_MEMBER = "dolpha.kis.investor_flow.GetMemberFirmTrading"
_INVESTOR = "dolpha.kis.investor_flow.GetInvestorToday"
_PROGRAM = "dolpha.kis.investor_flow.GetProgramTradeToday"

_TODAY = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d")


def _investor(orgn_qty):
    return {"rows": [{"date": _TODAY, "orgn_qty": orgn_qty}]}


def _program(ntby_qty):
    return {"rows": [{"time": "091500", "ntby_qty": 0}, {"time": "152000", "ntby_qty": ntby_qty}]}


def _settings(**over):
    base = dict(
        use_own_exit=True,
        max_loss_pct=1.0,
        max_position_pct=20.0,
        stages=((2.0, 50.0),),
        use_trailing=True,
        trailing_start_t=1.5,
        trailing_bar_unit="1m",
        trailing_bar_count=3,
        force_exit_enabled=True,
        force_exit_time=time(15, 20),
        overnight_enabled=False,
        overnight_conditions=("foreign", "institution", "program", "shinhan_top5"),
        overnight_min_count=2,
        overnight_max_days=3,
    )
    base.update(over)
    return ExitSettings(**base)


class NormalizeConditionsTests(TestCase):
    def test_invalid_dropped_and_deduped(self):
        self.assertEqual(
            normalize_conditions(["program", "bogus", "program", "foreign"]),
            ["program", "foreign"],
        )

    def test_none_or_invalid_falls_back_to_all(self):
        allk = list(overnight_mod.OVERNIGHT_CONDITION_KEYS)
        self.assertEqual(normalize_conditions(None), allk)
        self.assertEqual(normalize_conditions("nope"), allk)

    def test_explicit_empty_stays_empty(self):
        self.assertEqual(normalize_conditions([]), [])
        self.assertEqual(normalize_conditions(["bogus"]), [])


class EvaluateOvernightSignalTests(TestCase):
    def test_all_conditions_met(self):
        member = {
            "foreign": {"ntby_qty": 12000},
            "buy": [{"name": "미래에셋증권"}, {"name": "신한투자증권"}],
        }
        with patch(_MEMBER, return_value=member), patch(
            _INVESTOR, return_value=_investor(5000)
        ), patch(_PROGRAM, return_value=_program(4000)):
            sig = evaluate_overnight_signal(
                "000660", ["foreign", "institution", "program", "shinhan_top5"], 3
            )
        self.assertTrue(sig.available)
        self.assertEqual(sig.met_count, 4)
        self.assertTrue(sig.should_hold)

    def test_below_required_count(self):
        member = {"foreign": {"ntby_qty": -500}, "buy": [{"name": "키움증권"}]}
        with patch(_MEMBER, return_value=member), patch(
            _INVESTOR, return_value=_investor(5000)
        ), patch(_PROGRAM, return_value=_program(-100)):
            sig = evaluate_overnight_signal(
                "000660", ["foreign", "institution", "program", "shinhan_top5"], 2
            )
        self.assertTrue(sig.available)
        self.assertEqual(sig.met_count, 1)  # institution 만 충족
        self.assertFalse(sig.should_hold)

    def test_fetch_failure_makes_unavailable(self):
        with patch(_MEMBER, side_effect=RuntimeError("KIS down")), patch(
            _INVESTOR, return_value=_investor(9999)
        ), patch(_PROGRAM, return_value=_program(9999)):
            sig = evaluate_overnight_signal(
                "000660", ["foreign", "institution", "program"], 1
            )
        self.assertFalse(sig.available)
        self.assertFalse(sig.should_hold)

    def test_partial_failure_one_condition_makes_unavailable(self):
        member = {"foreign": {"ntby_qty": 9999}, "buy": [{"name": "신한투자증권"}]}
        with patch(_MEMBER, return_value=member), patch(
            _PROGRAM, side_effect=RuntimeError("boom")
        ), patch(_INVESTOR, return_value=_investor(9999)):
            sig = evaluate_overnight_signal(
                "000660", ["foreign", "program", "institution"], 1
            )
        self.assertFalse(sig.available)
        self.assertFalse(sig.should_hold)

    def test_empty_member_payload_treated_as_error(self):
        with patch(_MEMBER, return_value={}):
            sig = evaluate_overnight_signal("000660", ["foreign", "shinhan_top5"], 1)
        self.assertFalse(sig.available)
        self.assertFalse(sig.should_hold)

    def test_institution_without_today_row_is_unavailable(self):
        with patch(_INVESTOR, return_value={"rows": [{"date": "19990101", "orgn_qty": 9}]}):
            sig = evaluate_overnight_signal("000660", ["institution"], 1)
        self.assertFalse(sig.available)

    def test_program_uses_latest_time_bucket(self):
        # 시간순이 뒤섞여 와도 가장 늦은 시각(152000)의 누적값을 쓴다
        rows = {"rows": [{"time": "152000", "ntby_qty": -50}, {"time": "091500", "ntby_qty": 9999}]}
        with patch(_PROGRAM, return_value=rows):
            sig = evaluate_overnight_signal("000660", ["program"], 1)
        self.assertFalse(sig.met["program"])

    def test_empty_conditions_never_holds(self):
        sig = evaluate_overnight_signal("000660", [], 1)
        self.assertFalse(sig.available)
        self.assertFalse(sig.should_hold)

    def test_only_selected_conditions_queried(self):
        with patch(_PROGRAM, return_value=_program(8000)) as prog, patch(_MEMBER) as mem, patch(
            _INVESTOR
        ) as inv:
            sig = evaluate_overnight_signal("000660", ["program"], 1)
        prog.assert_called_once()
        mem.assert_not_called()
        inv.assert_not_called()
        self.assertTrue(sig.should_hold)

    def test_shinhan_alias_match(self):
        member = {"foreign": {"ntby_qty": 0}, "buy": [{"name": "신한금융투자"}]}
        with patch(_MEMBER, return_value=member):
            sig = evaluate_overnight_signal("000660", ["shinhan_top5"], 1)
        self.assertTrue(sig.met["shinhan_top5"])


class EvaluateExitOvernightTests(TestCase):
    def _base_kwargs(self, **over):
        kw = dict(
            avg_price=1000.0,
            current_price=1005.0,
            stop_price=950.0,
            t_value=20.0,
            peak_price=1010.0,
            completed_stages=[],
            trailing_started=False,
            trailing_line=None,
            now=datetime(2026, 9, 8, 15, 25),
        )
        kw.update(over)
        return kw

    def test_force_exit_unchanged_when_overnight_disabled(self):
        d = evaluate_exit(_settings(overnight_enabled=False), **self._base_kwargs())
        self.assertTrue(d.should_exit)
        self.assertIn("당일 강제청산", d.reason)

    def test_overnight_hold_skips_force_exit(self):
        d = evaluate_exit(
            _settings(overnight_enabled=True),
            **self._base_kwargs(overnight_hold=True, days_held=1),
        )
        self.assertFalse(d.should_exit)

    def test_overnight_not_met_still_force_exits(self):
        d = evaluate_exit(
            _settings(overnight_enabled=True),
            **self._base_kwargs(overnight_hold=False, days_held=1),
        )
        self.assertTrue(d.should_exit)
        self.assertIn("당일 강제청산", d.reason)

    def test_max_days_forces_exit_even_if_conditions_met(self):
        d = evaluate_exit(
            _settings(overnight_enabled=True, overnight_max_days=3),
            **self._base_kwargs(overnight_hold=True, days_held=3),
        )
        self.assertTrue(d.should_exit)
        self.assertIn("보유기간 만료", d.reason)

    def test_stop_loss_still_fires_while_holding_overnight(self):
        d = evaluate_exit(
            _settings(overnight_enabled=True),
            **self._base_kwargs(overnight_hold=True, days_held=2, current_price=940.0),
        )
        self.assertTrue(d.should_exit)
        self.assertIn("손절", d.reason)

    def test_before_force_exit_time_is_noop_for_overnight(self):
        d = evaluate_exit(
            _settings(overnight_enabled=True),
            **self._base_kwargs(
                overnight_hold=False,
                days_held=1,
                now=datetime(2026, 9, 8, 14, 0),
            ),
        )
        self.assertFalse(d.should_exit)


class EngineOvernightWiringTests(TestCase):
    """_theme_days_held / _theme_overnight_hold / 캐시 동작."""

    def setUp(self):
        from django.contrib.auth import get_user_model
        from myweb.models import TradingConfig
        from dolpha.trading_engine import TradingEngine

        User = get_user_model()
        self.user = User.objects.create(username="ov_tester")
        self.config = TradingConfig.objects.create(
            user=self.user, stock_code="000660", stock_name="SK하이닉스",
            trading_mode="manual", strategy_type="theme_surge", is_active=True,
        )
        self.engine = TradingEngine(user=self.user)
        TradingEngine._overnight_cache_date = ""
        TradingEngine._overnight_decision = {}
        TradingEngine._days_held_cache = {}

    def _add_buy(self, filled_at):
        from decimal import Decimal
        from myweb.models import TradeEntry

        return TradeEntry.objects.create(
            user=self.user, trading_config=self.config, stock_code="000660",
            stock_name="SK하이닉스", trade_type="BUY", entry_type="INITIAL",
            order_quantity=10, filled_quantity=10, filled_price=Decimal("1000"),
            status="FILLED", filled_at=filled_at,
        )

    def test_days_held_same_day_entry_is_one(self):
        from django.utils import timezone as tz

        self._add_buy(tz.now())
        self.assertEqual(self.engine._theme_days_held(self.config), 1)

    def test_days_held_counts_trading_days(self):
        from django.utils import timezone as tz

        self._add_buy(tz.now() - tz.timedelta(days=5))
        with patch("dolpha.kis.holiday.is_trading_day", return_value=True):
            days = self.engine._theme_days_held(self.config)
        self.assertEqual(days, 6)  # 진입일 포함 6일

    def test_days_held_cached_per_day(self):
        from django.utils import timezone as tz

        self._add_buy(tz.now() - tz.timedelta(days=3))
        with patch("dolpha.kis.holiday.is_trading_day", return_value=True) as m:
            self.engine._theme_days_held(self.config)
            self.engine._theme_days_held(self.config)
        # 두 번째 호출은 캐시 히트 → is_trading_day 재호출 없음
        first_call_count = m.call_count
        with patch("dolpha.kis.holiday.is_trading_day", return_value=True) as m2:
            self.engine._theme_days_held(self.config)
        self.assertEqual(m2.call_count, 0)
        self.assertGreater(first_call_count, 0)

    def test_overnight_signal_none_when_disabled(self):
        s = _settings(overnight_enabled=False)
        self.assertIsNone(
            self.engine._theme_overnight_signal(self.config, s, datetime(2026, 9, 8, 15, 25), 1)
        )

    def test_overnight_signal_none_before_force_time(self):
        s = _settings(overnight_enabled=True, force_exit_time=time(15, 20))
        self.assertIsNone(
            self.engine._theme_overnight_signal(self.config, s, datetime(2026, 9, 8, 14, 0), 1)
        )

    def test_overnight_signal_at_max_days_skips_lookup(self):
        s = _settings(overnight_enabled=True, overnight_max_days=3)
        with patch(_MEMBER) as m:
            result = self.engine._theme_overnight_signal(
                self.config, s, datetime(2026, 9, 8, 15, 25), 3
            )
        self.assertIsNone(result)
        m.assert_not_called()

    def test_overnight_signal_evaluated_once_and_cached(self):
        s = _settings(
            overnight_enabled=True, overnight_conditions=("program",), overnight_min_count=1
        )
        with patch(_PROGRAM, return_value=_program(5000)) as prog:
            r1 = self.engine._theme_overnight_signal(
                self.config, s, datetime(2026, 9, 8, 15, 25), 1
            )
            r2 = self.engine._theme_overnight_signal(
                self.config, s, datetime(2026, 9, 8, 15, 26), 1
            )
        self.assertTrue(r1.should_hold)
        self.assertTrue(r2.should_hold)
        prog.assert_called_once()  # 하루 1회만 평가

    def test_classify_and_record_exit_signal(self):
        from myweb.models import ThemeExitSignal
        from dolpha.theme_surge.exit_rules import ExitDecision

        s = _settings(
            overnight_enabled=True, overnight_conditions=("program",), overnight_min_count=1
        )
        now = datetime(2026, 9, 8, 15, 25)
        with patch(_PROGRAM, return_value=_program(5000)):
            sig = self.engine._theme_overnight_signal(self.config, s, now, 1)
        self.engine._record_theme_exit_signal(
            self.config, s, now, 1, ExitDecision(False), sig
        )
        row = ThemeExitSignal.objects.get(user=self.user, stock_code="000660")
        self.assertEqual(row.decision, "overnight")
        self.assertTrue(row.overnight_evaluated)
        self.assertEqual(row.overnight_met_count, 1)

        # 이후 손절 체결로 갱신되면 같은 행이 stop_loss 로 바뀐다
        self.engine._record_theme_exit_signal(
            self.config, s, now, 1, ExitDecision(True, 100.0, "손절(눌림저점 이탈)"), sig
        )
        row.refresh_from_db()
        self.assertEqual(row.decision, "stop_loss")
        self.assertEqual(ThemeExitSignal.objects.filter(user=self.user).count(), 1)

    def test_timeline_exposes_exit_markers(self):
        from django.utils import timezone as tz
        from myweb.models import ThemeExitSignal
        from dolpha.theme_surge.timeline import build_timeline

        today = tz.localdate()
        ThemeExitSignal.objects.create(
            user=self.user, date=today, checked_at=tz.now(),
            tics_id=7, theme_name="AI", stock_code="000660", stock_name="SK하이닉스",
            force_exit_time="15:20", days_held=2, decision="overnight",
            reason="수급 3/2 충족", overnight_evaluated=True, overnight_available=True,
            overnight_conditions=["foreign", "program"], overnight_met={"foreign": True, "program": True},
            overnight_met_count=2, overnight_required=2,
        )
        tl = build_timeline(today, user=self.user)
        self.assertEqual(len(tl["exits"]), 1)
        ex = tl["exits"][0]
        self.assertEqual(ex["decision"], "overnight")
        self.assertEqual(ex["slot"], "15:30")  # 마커는 장 마감 슬롯에
        self.assertEqual(ex["force_exit_time"], "15:20")  # 실제 판정 시각은 필드로
        self.assertEqual(ex["stock_code"], "000660")
        self.assertEqual(tl["summary"]["overnight_count"], 1)


class ExitFinalizerTests(TestCase):
    """장 마감 후 오버나이트 확정 (exit_finalizer)."""

    def setUp(self):
        from django.contrib.auth import get_user_model
        from decimal import Decimal
        from django.utils import timezone as tz
        from myweb.models import TradingConfig, TradeEntry, TradingDefaults

        User = get_user_model()
        self.user = User.objects.create(username="fin_tester")
        TradingDefaults.objects.create(
            user=self.user, theme_surge_overnight_enabled=True,
            theme_surge_overnight_conditions=["foreign", "program"],
            theme_surge_overnight_min_count=2,
        )
        self.config = TradingConfig.objects.create(
            user=self.user, stock_code="032820", stock_name="우리기술",
            trading_mode="manual", strategy_type="theme_surge", is_active=True,
        )
        TradeEntry.objects.create(
            user=self.user, trading_config=self.config, stock_code="032820",
            stock_name="우리기술", trade_type="BUY", entry_type="INITIAL",
            order_quantity=10, filled_quantity=10, filled_price=Decimal("100"),
            status="FILLED", filled_at=tz.now(),
        )

    def _snapshot(self, foreign_ntby, program_ntby):
        from django.utils import timezone as tz
        from myweb.models import InvestorFlowSnapshot

        today = tz.localdate()
        InvestorFlowSnapshot.objects.create(
            stock_code="032820", stock_name="우리기술", date=today,
            member_firm={"buy": [{"name": "키움증권"}], "foreign": {"ntby_qty": foreign_ntby}},
            program_trade={"rows": [{"time": "152900", "ntby_qty": program_ntby}]},
            investor_today={"rows": []},
        )

    def test_records_overnight_for_active_position(self):
        from django.utils import timezone as tz
        from myweb.models import ThemeExitSignal
        from dolpha.theme_surge.exit_finalizer import finalize_theme_exit_signals

        self._snapshot(foreign_ntby=5000, program_ntby=9000)
        result = finalize_theme_exit_signals()
        self.assertEqual(result["recorded"], 1)
        row = ThemeExitSignal.objects.get(user=self.user, stock_code="032820")
        self.assertEqual(row.decision, "overnight")
        self.assertTrue(row.overnight_evaluated)
        self.assertEqual(row.overnight_met_count, 2)
        self.assertIn("충족", row.reason)

    def test_records_overnight_but_flags_rule_violation(self):
        from myweb.models import ThemeExitSignal
        from dolpha.theme_surge.exit_finalizer import finalize_theme_exit_signals

        self._snapshot(foreign_ntby=-5000, program_ntby=-9000)  # 0/2 충족
        finalize_theme_exit_signals()
        row = ThemeExitSignal.objects.get(user=self.user, stock_code="032820")
        self.assertEqual(row.decision, "overnight")
        self.assertEqual(row.overnight_met_count, 0)
        self.assertIn("청산 대상", row.reason)

    def test_does_not_overwrite_liquidated_record(self):
        from django.utils import timezone as tz
        from myweb.models import ThemeExitSignal
        from dolpha.theme_surge.exit_finalizer import finalize_theme_exit_signals

        ThemeExitSignal.objects.create(
            user=self.user, date=tz.localdate(), checked_at=tz.now(),
            stock_code="032820", stock_name="우리기술", force_exit_time="15:10",
            days_held=1, decision="stop_loss", reason="손절",
        )
        self._snapshot(foreign_ntby=5000, program_ntby=9000)
        result = finalize_theme_exit_signals()
        self.assertEqual(result["skipped"], 1)
        row = ThemeExitSignal.objects.get(user=self.user, stock_code="032820")
        self.assertEqual(row.decision, "stop_loss")

    def test_count_trading_days(self):
        from datetime import date
        from unittest.mock import patch
        from dolpha.theme_surge.exit_finalizer import count_trading_days

        with patch("dolpha.kis.holiday.is_trading_day", return_value=True):
            self.assertEqual(count_trading_days(date(2026, 9, 8), date(2026, 9, 8)), 1)
            self.assertEqual(count_trading_days(date(2026, 9, 7), date(2026, 9, 9)), 3)
