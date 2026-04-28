"""
Django 통합 자동매매 엔진

autobot/tradingBot/autoTrading_Bot.py 를 Django 환경으로 포팅.

주요 변경 사항:
  - 파일 기반 trade_history.json  →  TradeEntry DB
  - 파일 기반 trading_configs.json →  TradingConfig DB
  - CSV 거래 로그                   →  TradeEntry DB (status/entry_type 필드)
  - CSV 거래 요약                   →  TradingSummary DB
  - KIS Common 모듈                →  dolpha.kis.trade
  - limit 기반 GetOhlcv             →  날짜 범위 기반 stockCommon.GetOhlcv

사용법 (APScheduler cron job 등에서):
    from dolpha.trading_engine import TradingEngine
    engine = TradingEngine(user=some_user)
    engine.run_trading_cycle()
"""

import traceback
from datetime import datetime, timedelta
from decimal import Decimal

from django.utils import timezone as tz

from myweb.models import TradingConfig, TradeEntry, TradingSummary
from dolpha.kis import trade as KIS
from dolpha.stockCommon import GetOhlcv


class TradingEngine:
    """
    사용자 1명의 TradingConfig(is_active=True) 목록을 순회하며
    매매 사이클을 실행합니다.
    """

    def __init__(self, user):
        """
        Args:
            user: myweb.models.User 인스턴스
        """
        self.user = user
        self.trading_configs: list[TradingConfig] = []
        self._load_configs()

    # ──────────────────────────────────────────────
    # 설정 로드
    # ──────────────────────────────────────────────

    def _load_configs(self):
        """활성화된 TradingConfig 목록을 DB에서 로드합니다."""
        self.trading_configs = list(
            TradingConfig.objects.filter(user=self.user, is_active=True)
        )
        print(
            f"[TradingEngine] 활성 설정 {len(self.trading_configs)}개 로드됨"
            f" (유저: {self.user.username})"
        )

    # ──────────────────────────────────────────────
    # DB 기반 매수 이력 조회
    # ──────────────────────────────────────────────

    def _buy_entries(self, stock_code: str):
        """해당 종목의 BUY/FILLED TradeEntry 쿼리셋 반환."""
        return TradeEntry.objects.filter(
            user=self.user,
            stock_code=stock_code,
            trade_type="BUY",
            status="FILLED",
        ).order_by("filled_at")

    def get_entry_count(self, stock_code: str) -> int:
        """현재 보유 매수 횟수 (BUY/FILLED 체결 건수)."""
        return self._buy_entries(stock_code).count()

    def get_last_entry_price(self, stock_code: str) -> float | None:
        """마지막 매수 체결가."""
        entry = self._buy_entries(stock_code).last()
        return float(entry.filled_price) if entry else None

    def get_average_price(self, stock_code: str) -> float | None:
        """가중 평균 매수가."""
        entries = self._buy_entries(stock_code)
        if not entries.exists():
            return None
        total_amount = sum(float(e.filled_price) * e.filled_quantity for e in entries)
        total_qty = sum(e.filled_quantity for e in entries)
        return total_amount / total_qty if total_qty > 0 else None

    # ──────────────────────────────────────────────
    # ATR 계산
    # ──────────────────────────────────────────────

    def get_atr(self, stock_code: str, period: int = 14) -> float | None:
        """
        ATR(Average True Range) 계산.

        Args:
            stock_code: 종목코드
            period: ATR 기간 (기본 14일)
        Returns:
            ATR 값(원) 또는 None
        """
        try:
            # period + 10 거래일 ≈ (period + 10) * 7 / 5 달력일 → 여유 있게 2배
            calendar_days = (period + 10) * 2
            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")

            df = GetOhlcv("KRX", stock_code, start_date=start_date, end_date=end_date)
            if df is None or len(df) < period:
                print(f"[{stock_code}] ATR: 데이터 부족 ({len(df) if df is not None else 0}행)")
                return None

            # True Range 계산
            df = df.copy()
            df["h_l"]  = df["high"] - df["low"]
            df["h_pc"] = (df["high"] - df["close"].shift(1)).abs()
            df["l_pc"] = (df["low"]  - df["close"].shift(1)).abs()
            df["tr"]   = df[["h_l", "h_pc", "l_pc"]].max(axis=1)

            atr = df["tr"].rolling(window=period).mean().iloc[-1]
            return float(atr)

        except Exception as e:
            print(f"[{stock_code}] ATR 계산 오류: {e}")
            return None

    # ──────────────────────────────────────────────
    # 포지션 크기 계산
    # ──────────────────────────────────────────────

    def calculate_position_size(self, config: TradingConfig) -> float:
        """
        매매모드에 따른 총 포지션 크기(원)를 계산합니다.
          - manual : 가용현금 × max_loss% ÷ stop_loss%
          - atr    : (가용현금 × max_loss%) ÷ (ATR × stop_loss배수 / 현재가)
        기준: RemainMoney(가용현금)만 사용 — 보유주식 평가금액 제외
        """
        try:
            balance           = KIS.GetBalance()
            confirmed_capital = float(balance["ConfirmedCapital"])
            mode              = config.trading_mode
            max_loss_pct      = (config.max_loss or 2.0) / 100.0  # 기본 2%

            if mode == "manual":
                stop_pct   = (config.stop_loss or 8.0) / 100.0
                pos_amount = confirmed_capital * max_loss_pct / stop_pct
                print(
                    f"[{config.stock_name}] Manual 포지션:"
                    f" 확정원금={confirmed_capital:,.0f}, 위험={max_loss_pct*100:.1f}%,"
                    f" 손절={stop_pct*100:.1f}%, 포지션={pos_amount:,.0f}원"
                )
                return pos_amount

            elif mode == "atr":
                atr = self.get_atr(config.stock_code)
                if atr:
                    current_price   = float(KIS.GetCurrentPrice(config.stock_code))
                    stop_multiplier = config.stop_loss or 2.0
                    risk_amount     = confirmed_capital * max_loss_pct
                    stop_loss_price = atr * stop_multiplier
                    stop_loss_ratio = stop_loss_price / current_price
                    pos_amount      = risk_amount / stop_loss_ratio
                    print(
                        f"[{config.stock_name}] ATR 포지션:"
                        f" 확정원금={confirmed_capital:,.0f}, ATR={atr:.1f}원,"
                        f" 손절폭={stop_loss_price:,.0f}원({stop_loss_ratio*100:.1f}%),"
                        f" 포지션={pos_amount:,.0f}원"
                    )
                    return pos_amount
                else:
                    # ATR 실패 → manual 방식으로 fallback
                    print(f"[{config.stock_name}] ATR 실패 → Manual fallback")
                    stop_pct   = (config.stop_loss or 2.0) / 100.0
                    pos_amount = confirmed_capital * max_loss_pct / stop_pct
                    return pos_amount

            else:
                # 알 수 없는 모드 → manual 방식
                stop_pct   = (config.stop_loss or 8.0) / 100.0
                pos_amount = confirmed_capital * max_loss_pct / stop_pct
                return pos_amount

        except Exception as e:
            print(f"[{config.stock_name}] 포지션 크기 계산 오류: {e}")
            return 0.0

    # ──────────────────────────────────────────────
    # 피라미딩 금액 계산
    # ──────────────────────────────────────────────

    def calculate_pyramiding_amounts(
        self, config: TradingConfig, total_amount: float
    ) -> list[float]:
        """
        피라미딩 각 차수별 증분 금액 배열을 반환합니다.
        config.positions 에 비율 배열이 있으면 그것을 사용하고,
        없으면 균등 분할합니다.
        """
        pyramid_count = config.pyramiding_count or 0
        total_entries = pyramid_count + 1  # 최초 진입 포함

        positions = config.positions or []
        if not positions:
            # 균등 분할
            return [total_amount / total_entries] * total_entries

        # positions 배열의 앞 total_entries 항목 합계로 정규화
        slice_ = positions[:total_entries]
        total_ratio = sum(slice_) or 1
        return [total_amount * r / total_ratio for r in slice_]

    def get_current_entry_amount(
        self, config: TradingConfig, total_amount: float
    ) -> float:
        """현재 진입 차수에 해당하는 증분 금액을 반환합니다."""
        current_count = self.get_entry_count(config.stock_code)
        amounts = self.calculate_pyramiding_amounts(config, total_amount)
        print(
            f"[{config.stock_name}] 진입 차수={current_count}, 금액 배열={amounts}"
        )
        if current_count < len(amounts):
            return amounts[current_count]
        return 0.0  # 피라미딩 한도 초과

    # ──────────────────────────────────────────────
    # 진입 조건 체크
    # ──────────────────────────────────────────────

    def check_entry_conditions(self, config: TradingConfig) -> bool:
        """
        신규 진입 또는 피라미딩 조건을 체크합니다.
        이미 보유 중이면 피라미딩 조건을 확인합니다.
        """
        try:
            stock_code    = config.stock_code
            current_price = float(KIS.GetCurrentPrice(stock_code))

            # 실제 보유 수량 확인
            holding_qty = 0
            for s in KIS.GetMyStockList():
                if s["StockCode"] == stock_code:
                    holding_qty = int(s["StockAmt"])
                    break

            if holding_qty > 0:
                print(
                    f"[{config.stock_name}] 보유 중({holding_qty}주)"
                    f" → 피라미딩 조건 체크"
                )
                return self.check_pyramiding_conditions(
                    config, current_price, holding_qty
                )

            # 신규 진입: entry_point 이상이면 진입
            entry_point = config.entry_point or 0
            print(
                f"[{config.stock_name}] 신규 진입 체크:"
                f" 현재가={current_price:,.0f}, 진입가={entry_point:,.0f}"
            )
            if entry_point > 0 and current_price >= entry_point:
                print(f"[{config.stock_name}] 신규 진입 조건 충족")
                return True
            print(f"[{config.stock_name}] 신규 진입 조건 불충족")
            return False

        except Exception as e:
            print(f"[{config.stock_name}] 진입 조건 체크 오류: {e}")
            return False

    def check_pyramiding_conditions(
        self, config: TradingConfig, current_price: float, holding_qty: int
    ) -> bool:
        """피라미딩 추가 진입 조건을 체크합니다."""
        try:
            pyramid_count   = config.pyramiding_count or 0
            pyramid_entries = config.pyramiding_entries or []
            current_count   = self.get_entry_count(config.stock_code)

            if pyramid_count <= 0 or not pyramid_entries:
                return False
            if current_count > pyramid_count:
                print(f"[{config.stock_name}] 피라미딩 횟수 초과")
                return False

            # 실제 체결가를 기준가로 사용 (entry_point는 트리거 가격이므로 부적합)
            from myweb.models import TradeEntry as _TradeEntry
            initial_entry = _TradeEntry.objects.filter(
                trading_config=config,
                entry_type="INITIAL",
                status="FILLED",
            ).order_by("-filled_at").first()
            base_price = float(initial_entry.filled_price) if initial_entry else config.entry_point
            if not base_price:
                return False

            next_idx = current_count  # 0-based: 현재 체결 수 = 다음 인덱스
            if next_idx >= len(pyramid_entries):
                return False

            entry_str = str(pyramid_entries[next_idx]).strip()
            if not entry_str:
                return False

            mode = config.trading_mode

            if mode == "manual":
                # % 기반 피라미딩
                threshold_pct = float(entry_str.lstrip("+")) / 100.0
                change_pct    = (current_price - base_price) / base_price
                print(
                    f"[{config.stock_name}] 피라미딩 체크:"
                    f" 상승률={change_pct*100:.2f}%, 목표={threshold_pct*100:.2f}%"
                )
                return change_pct >= threshold_pct

            else:  # atr
                atr = self.get_atr(config.stock_code)
                if atr is None:
                    return False
                multiplier     = float(entry_str)
                threshold_price = base_price + atr * multiplier
                print(
                    f"[{config.stock_name}] ATR 피라미딩:"
                    f" ATR={atr:.2f}, 목표가={threshold_price:,.0f}"
                )
                return current_price >= threshold_price

        except Exception as e:
            print(f"[{config.stock_name}] 피라미딩 조건 체크 오류: {e}")
            return False

    # ──────────────────────────────────────────────
    # 청산 조건 체크
    # ──────────────────────────────────────────────

    def _get_trailing_stop_settings(self, config: TradingConfig) -> tuple[bool, float, float]:
        """
        user의 TradingDefaults에서 트레일링 스탑 설정을 읽어 반환합니다.

        Returns:
            (use_trailing_stop, trailing_stop_trigger, trailing_stop_value)
            trailing_stop_trigger: 트레일링 발동 조건 (manual이면 %, atr이면 ATR 배수)
            trailing_stop_value:   트레일링 스탑 거리 (manual이면 %, atr이면 ATR 배수)
        """
        try:
            defaults = self.user.tradingdefaults
            if config.trading_mode == "manual":
                return (
                    defaults.manual_use_trailing_stop,
                    defaults.manual_trailing_stop_trigger,
                    defaults.manual_trailing_stop_percent,
                )
            else:
                return (
                    defaults.turtle_use_trailing_stop,
                    defaults.turtle_trailing_stop_trigger,
                    defaults.turtle_trailing_stop_percent,
                )
        except Exception:
            return False, 0.0, 0.0

    def _update_peak_price(self, config: TradingConfig, current_price: float):
        """현재가가 기존 고점보다 높으면 peak_price를 갱신합니다."""
        if config.trailing_stop_peak_price is None or current_price > config.trailing_stop_peak_price:
            config.trailing_stop_peak_price = current_price
            config.save(update_fields=["trailing_stop_peak_price"])

    def check_exit_conditions(
        self, config: TradingConfig
    ) -> tuple[bool, str | None]:
        """
        트레일링 스탑 / 고정 손절 / 익절 조건을 체크합니다.

        트레일링 스탑 동작 방식:
          - Phase 1 (수익 < stop_loss): 고정 손절만 동작
          - Phase 2 (수익 >= stop_loss): 트레일링 발동
              stop_line = max(avg_price, peak * (1 - stop_pct))  # manual
              stop_line = max(avg_price, peak - atr * stop_mul)  # atr

        Returns:
            (should_exit, reason_str)
        """
        try:
            stock_code  = config.stock_code
            mode        = config.trading_mode
            stop_loss   = config.stop_loss  or 8.0
            take_profit = config.take_profit or 24.0

            # KIS에서 실제 보유 확인
            holding_qty = 0
            avg_price   = 0.0
            for s in KIS.GetMyStockList():
                if s["StockCode"] == stock_code:
                    holding_qty = int(s["StockAmt"])
                    avg_price   = float(s["StockAvgPrice"])
                    break

            if holding_qty <= 0:
                return False, None

            current_price = float(KIS.GetCurrentPrice(stock_code))

            # 트레일링 스탑 설정 조회
            use_trailing_stop, trailing_stop_trigger, trailing_stop_value = self._get_trailing_stop_settings(config)

            if use_trailing_stop:
                # 고점 업데이트
                self._update_peak_price(config, current_price)
                peak = config.trailing_stop_peak_price

                if mode == "manual":
                    stop_pct = stop_loss / 100.0
                    trailing_activated = (peak >= avg_price * (1 + trailing_stop_trigger / 100.0))
                    if trailing_activated:
                        stop_line = max(avg_price, peak * (1 - trailing_stop_value / 100.0))
                        print(
                            f"[{config.stock_name}] 트레일링 스탑 활성:"
                            f" peak={peak:,.0f}, stop_line={stop_line:,.0f},"
                            f" current={current_price:,.0f}"
                        )
                        if current_price <= stop_line:
                            return True, "트레일링스탑"
                    else:
                        # 트레일링 미발동 → 고정 손절
                        if current_price <= avg_price * (1 - stop_pct):
                            return True, "손절"

                else:  # atr
                    atr = self.get_atr(stock_code)
                    if atr is not None:
                        trailing_activated = (peak >= avg_price + atr * trailing_stop_trigger)
                        if trailing_activated:
                            stop_line = max(avg_price, peak - atr * trailing_stop_value)
                            print(
                                f"[{config.stock_name}] ATR 트레일링 스탑 활성:"
                                f" peak={peak:,.0f}, ATR={atr:.1f},"
                                f" stop_line={stop_line:,.0f}, current={current_price:,.0f}"
                            )
                            if current_price <= stop_line:
                                return True, "트레일링스탑"
                        else:
                            if current_price <= avg_price - atr * stop_loss:
                                return True, "ATR 손절"

            else:
                # 트레일링 스탑 미사용 → 고정 손절만
                if mode == "manual":
                    profit_pct = (current_price - avg_price) / avg_price * 100.0
                    if profit_pct <= -stop_loss:
                        return True, "손절"
                else:
                    atr = self.get_atr(stock_code)
                    if atr is not None:
                        if current_price <= avg_price - atr * stop_loss:
                            return True, "ATR 손절"

            # 익절 체크 (공통)
            if mode == "manual":
                profit_pct = (current_price - avg_price) / avg_price * 100.0
                if profit_pct >= take_profit:
                    return True, "익절"
            else:
                atr = self.get_atr(stock_code)
                if atr is not None and current_price >= avg_price + atr * take_profit:
                    return True, "ATR 익절"

            return False, None

        except Exception as e:
            print(f"[{config.stock_name}] 청산 조건 체크 오류: {e}")
            return False, None

    # ──────────────────────────────────────────────
    # 분할 익절 조건 체크
    # ──────────────────────────────────────────────

    def _get_ohlcv_enough(self, stock_code: str, min_periods: int):
        """분할 익절 지표 계산에 필요한 OHLCV 데이터를 반환합니다."""
        calendar_days = min_periods * 3
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")
        return GetOhlcv("KRX", stock_code, start_date=start_date, end_date=end_date)

    def _mark_staged_exit_stage(self, config: TradingConfig, stage: int):
        """해당 단계를 완료 목록에 기록합니다."""
        completed = list(config.staged_exit_completed_stages or [])
        if stage not in completed:
            completed.append(stage)
            config.staged_exit_completed_stages = completed
            config.save(update_fields=["staged_exit_completed_stages"])

    def _check_ma_staged_exit(
        self, config: TradingConfig, defaults, completed: list, max_stage: int
    ) -> tuple[int | None, float, str]:
        """이동평균선 하회 분할 익절 체크."""
        stages = [
            (1, defaults.ma_stage1_period, defaults.ma_stage1_sell_pct),
            (2, defaults.ma_stage2_period, defaults.ma_stage2_sell_pct),
            (3, defaults.ma_stage3_period, defaults.ma_stage3_sell_pct),
        ]
        max_period = max(p for _, p, _ in stages)
        df = self._get_ohlcv_enough(config.stock_code, max_period)
        if df is None or len(df) < max_period:
            return None, 0.0, ""

        current_price = float(KIS.GetCurrentPrice(config.stock_code))

        for stage_num, period, sell_pct in stages[:max_stage]:
            if stage_num in completed:
                continue
            ma = df["close"].rolling(window=period).mean().iloc[-1]
            if current_price < ma:
                reason = f"MA{period}하회_{stage_num}단계"
                print(f"[{config.stock_name}] {reason}: 현재={current_price:,.0f}, MA={ma:,.0f}")
                return stage_num, sell_pct, reason

        return None, 0.0, ""

    def _check_dc_staged_exit(
        self, config: TradingConfig, defaults, completed: list, max_stage: int
    ) -> tuple[int | None, float, str]:
        """데드크로스 분할 익절 체크 (단기 MA가 장기 MA 아래로 돌파)."""
        stages = [
            (1, defaults.dc_stage1_short, defaults.dc_stage1_long, defaults.dc_stage1_sell_pct),
            (2, defaults.dc_stage2_short, defaults.dc_stage2_long, defaults.dc_stage2_sell_pct),
            (3, defaults.dc_stage3_short, defaults.dc_stage3_long, defaults.dc_stage3_sell_pct),
        ]
        max_period = max(max(s, l) for _, s, l, _ in stages)
        df = self._get_ohlcv_enough(config.stock_code, max_period + 2)
        if df is None or len(df) < max_period + 2:
            return None, 0.0, ""

        for stage_num, short, long, sell_pct in stages[:max_stage]:
            if stage_num in completed:
                continue
            short_ma = df["close"].rolling(window=short).mean()
            long_ma  = df["close"].rolling(window=long).mean()
            # 데드크로스: 전봉 short >= long, 현재봉 short < long
            if (short_ma.iloc[-2] >= long_ma.iloc[-2]) and (short_ma.iloc[-1] < long_ma.iloc[-1]):
                reason = f"데드크로스{short}/{long}_{stage_num}단계"
                print(
                    f"[{config.stock_name}] {reason}:"
                    f" MA{short}={short_ma.iloc[-1]:,.0f}, MA{long}={long_ma.iloc[-1]:,.0f}"
                )
                return stage_num, sell_pct, reason

        return None, 0.0, ""

    def _check_nl_staged_exit(
        self, config: TradingConfig, defaults, completed: list, max_stage: int
    ) -> tuple[int | None, float, str]:
        """N일 신저가 분할 익절 체크."""
        stages = [
            (1, defaults.nl_stage1_days, defaults.nl_stage1_sell_pct),
            (2, defaults.nl_stage2_days, defaults.nl_stage2_sell_pct),
            (3, defaults.nl_stage3_days, defaults.nl_stage3_sell_pct),
        ]
        max_days = max(d for _, d, _ in stages)
        df = self._get_ohlcv_enough(config.stock_code, max_days + 1)
        if df is None or len(df) < max_days + 1:
            return None, 0.0, ""

        current_price = float(KIS.GetCurrentPrice(config.stock_code))

        for stage_num, days, sell_pct in stages[:max_stage]:
            if stage_num in completed:
                continue
            # 직전 days 거래일의 저가 최솟값 (오늘 제외)
            past_low = df["low"].iloc[-(days + 1):-1].min()
            if current_price < past_low:
                reason = f"{days}일신저가_{stage_num}단계"
                print(f"[{config.stock_name}] {reason}: 현재={current_price:,.0f}, {days}일저가={past_low:,.0f}")
                return stage_num, sell_pct, reason

        return None, 0.0, ""

    def check_staged_exit(
        self, config: TradingConfig
    ) -> tuple[int | None, float, str]:
        """
        분할 익절 조건을 체크합니다.

        트레일링 스탑이 ON이면 3단계(100% 매도)는 체크하지 않습니다.
        트레일링 스탑이 최종 청산을 담당합니다.

        Returns:
            (stage_number, sell_pct, reason) 또는 (None, 0.0, "")
        """
        try:
            defaults = self.user.tradingdefaults
        except Exception:
            return None, 0.0, ""

        exit_type = defaults.staged_exit_type
        if exit_type == "none":
            return None, 0.0, ""

        completed = list(config.staged_exit_completed_stages or [])
        use_trailing_stop = self._get_trailing_stop_settings(config)[0]
        # 트레일링 스탑 ON → 3단계(전량) 스킵, 트레일링 스탑이 최종 청산
        max_stage = 2 if use_trailing_stop else 3

        # 모든 단계가 이미 완료된 경우
        if all(s in completed for s in range(1, max_stage + 1)):
            return None, 0.0, ""

        try:
            if exit_type == "ma":
                return self._check_ma_staged_exit(config, defaults, completed, max_stage)
            elif exit_type == "dead_cross":
                return self._check_dc_staged_exit(config, defaults, completed, max_stage)
            elif exit_type == "new_low":
                return self._check_nl_staged_exit(config, defaults, completed, max_stage)
        except Exception as e:
            print(f"[{config.stock_name}] 분할 익절 체크 오류: {e}")

        return None, 0.0, ""

    # ──────────────────────────────────────────────
    # 매수 주문 실행
    # ──────────────────────────────────────────────

    def execute_buy_order(self, config: TradingConfig, amount: float) -> bool:
        """
        시장가 매수를 실행하고 TradeEntry를 DB에 저장합니다.

        Args:
            config: TradingConfig 인스턴스
            amount: 매수 금액(원)
        Returns:
            성공 여부
        """
        stock_code = config.stock_code
        stock_name = config.stock_name

        try:
            current_price = float(KIS.GetCurrentPrice(stock_code))
            buy_qty       = int(amount / current_price)

            if buy_qty <= 0:
                print(f"[{stock_name}] 매수 수량 0 — 투자금액 부족: {amount:,.0f}원")
                return False

            # 현재 진입 차수 결정
            entry_count = self.get_entry_count(stock_code)
            entry_type  = "INITIAL" if entry_count == 0 else "PYRAMIDING"

            # KIS 시장가 매수 주문
            result = KIS.MakeBuyMarketOrder(stock_code, buy_qty)
            if result is None:
                print(f"[{stock_name}] 매수 주문 실패")
                return False

            # DB에 TradeEntry 저장
            atr_val    = self.get_atr(stock_code)
            avg_price  = self.get_average_price(stock_code)  # 이전 평균가

            # 손절 참고가 계산
            if config.trading_mode == "manual":
                stop_pct   = (config.stop_loss or 8.0) / 100.0
                stop_price = Decimal(str(current_price * (1 - stop_pct)))
            elif atr_val:
                stop_mul   = config.stop_loss or 2.0
                stop_price = Decimal(str(current_price - atr_val * stop_mul))
            else:
                stop_price = None

            TradeEntry.objects.create(
                user            = self.user,
                trading_config  = config,
                stock_code      = stock_code,
                stock_name      = stock_name,
                trade_type      = "BUY",
                entry_type      = entry_type,
                order_no        = result["OrderNum2"],
                order_quantity  = buy_qty,
                order_price     = Decimal(str(current_price)),
                filled_quantity = buy_qty,
                filled_price    = Decimal(str(current_price)),
                filled_amount   = Decimal(str(current_price * buy_qty)),
                status          = "FILLED",
                atr_value       = atr_val,
                stop_price      = stop_price,
                ordered_at      = tz.now(),
                filled_at       = tz.now(),
            )

            # 트레일링 스탑 고점 초기화: 신규 진입이면 세팅, 이미 있으면 유지
            if config.trailing_stop_peak_price is None:
                config.trailing_stop_peak_price = current_price
                config.save(update_fields=["trailing_stop_peak_price"])

            print(
                f"[{stock_name}] 매수 완료: {buy_qty}주 @ {current_price:,.0f}원"
                f" (유형={entry_type}, 주문번호={result['OrderNum2']})"
            )
            return True

        except Exception as e:
            print(f"[{stock_name}] 매수 주문 오류: {e}")
            traceback.print_exc()
            return False

    # ──────────────────────────────────────────────
    # 매도 주문 실행
    # ──────────────────────────────────────────────

    def execute_sell_order(self, config: TradingConfig, reason: str = "") -> bool:
        """
        보유 전량을 시장가로 매도하고 TradeEntry를 DB에 저장합니다.
        BUY 체결 기록을 'CANCELLED' 상태로 변경하여 포지션 종료를 표시합니다.

        Args:
            config: TradingConfig 인스턴스
            reason: 매도 사유 (예: "손절", "익절")
        Returns:
            성공 여부
        """
        stock_code = config.stock_code
        stock_name = config.stock_name

        try:
            # KIS에서 보유 수량 확인
            holding_qty = 0
            for s in KIS.GetMyStockList():
                if s["StockCode"] == stock_code:
                    holding_qty = int(s["StockAmt"])
                    break

            if holding_qty <= 0:
                print(f"[{stock_name}] 보유 수량 없음 — 매도 스킵")
                return False

            # 매도 사유 → entry_type 매핑
            if reason == "트레일링스탑":
                entry_type = "TRAILING_STOP"
            elif "손절" in reason:
                entry_type = "STOP_LOSS"
            elif "익절" in reason:
                entry_type = "EXIT_FULL"
            else:
                entry_type = "EXIT_FULL"

            current_price = float(KIS.GetCurrentPrice(stock_code))
            sell_amount   = current_price * holding_qty

            # DB 평균가 조회 (손익 계산)
            avg_price = self.get_average_price(stock_code)

            profit_loss = None
            profit_loss_pct = None
            if avg_price:
                cost         = avg_price * holding_qty
                profit_loss  = sell_amount - cost
                profit_loss_pct = profit_loss / cost * 100.0 if cost > 0 else 0.0

            # KIS 시장가 매도 주문
            result = KIS.MakeSellMarketOrder(stock_code, holding_qty)
            if result is None:
                print(f"[{stock_name}] 매도 주문 실패")
                return False

            now = tz.now()

            # SELL TradeEntry 생성
            TradeEntry.objects.create(
                user            = self.user,
                trading_config  = config,
                stock_code      = stock_code,
                stock_name      = stock_name,
                trade_type      = "SELL",
                entry_type      = entry_type,
                order_no        = result["OrderNum2"],
                order_quantity  = holding_qty,
                order_price     = Decimal(str(current_price)),
                filled_quantity = holding_qty,
                filled_price    = Decimal(str(current_price)),
                filled_amount   = Decimal(str(sell_amount)),
                profit_loss     = Decimal(str(profit_loss)) if profit_loss is not None else None,
                profit_loss_percent = profit_loss_pct,
                status          = "FILLED",
                note            = reason,
                ordered_at      = now,
                filled_at       = now,
            )

            # 기존 BUY/FILLED 엔트리 → 포지션 종료 표시 ('CANCELLED' 재사용)
            self._buy_entries(stock_code).update(status="CANCELLED")

            # 트레일링 스탑 고점 리셋 + 분할 익절 단계 리셋
            config.trailing_stop_peak_price = None
            config.staged_exit_completed_stages = []
            config.save(update_fields=["trailing_stop_peak_price", "staged_exit_completed_stages"])

            # TradingSummary 업데이트
            self._update_trading_summary(config, stock_code, stock_name)

            print(
                f"[{stock_name}] 매도 완료: {holding_qty}주 @ {current_price:,.0f}원"
                f" | 손익={profit_loss:+,.0f}원 ({profit_loss_pct:+.2f}%)"
                if profit_loss is not None
                else f"[{stock_name}] 매도 완료: {holding_qty}주 @ {current_price:,.0f}원"
            )
            return True

        except Exception as e:
            print(f"[{stock_name}] 매도 주문 오류: {e}")
            traceback.print_exc()
            return False

    # ──────────────────────────────────────────────
    # 분할 매도 주문 실행
    # ──────────────────────────────────────────────

    def execute_partial_sell_order(
        self, config: TradingConfig, sell_pct: float, reason: str = ""
    ) -> bool:
        """
        보유 수량의 sell_pct%를 시장가로 매도합니다 (분할 익절용).
        포지션이 완전히 청산되지 않으므로 BUY 엔트리 상태와 트레일링 스탑 고점은 유지됩니다.

        Args:
            config:   TradingConfig 인스턴스
            sell_pct: 현재 보유 수량 대비 매도 비율 (0~100)
            reason:   매도 사유
        Returns:
            성공 여부
        """
        stock_code = config.stock_code
        stock_name = config.stock_name

        try:
            # KIS에서 보유 수량 확인
            holding_qty = 0
            for s in KIS.GetMyStockList():
                if s["StockCode"] == stock_code:
                    holding_qty = int(s["StockAmt"])
                    break

            if holding_qty <= 0:
                print(f"[{stock_name}] 보유 수량 없음 — 분할 매도 스킵")
                return False

            sell_qty = max(1, int(holding_qty * sell_pct / 100.0))

            current_price = float(KIS.GetCurrentPrice(stock_code))
            sell_amount   = current_price * sell_qty
            avg_price     = self.get_average_price(stock_code)

            profit_loss = None
            profit_loss_pct = None
            if avg_price:
                cost            = avg_price * sell_qty
                profit_loss     = sell_amount - cost
                profit_loss_pct = profit_loss / cost * 100.0 if cost > 0 else 0.0

            result = KIS.MakeSellMarketOrder(stock_code, sell_qty)
            if result is None:
                print(f"[{stock_name}] 분할 매도 주문 실패")
                return False

            now = tz.now()
            TradeEntry.objects.create(
                user                = self.user,
                trading_config      = config,
                stock_code          = stock_code,
                stock_name          = stock_name,
                trade_type          = "SELL",
                entry_type          = "EXIT_PARTIAL",
                order_no            = result["OrderNum2"],
                order_quantity      = sell_qty,
                order_price         = Decimal(str(current_price)),
                filled_quantity     = sell_qty,
                filled_price        = Decimal(str(current_price)),
                filled_amount       = Decimal(str(sell_amount)),
                profit_loss         = Decimal(str(profit_loss)) if profit_loss is not None else None,
                profit_loss_percent = profit_loss_pct,
                status              = "FILLED",
                note                = reason,
                ordered_at          = now,
                filled_at           = now,
            )

            print(
                f"[{stock_name}] 분할 매도 완료: {sell_qty}주({sell_pct:.0f}%) @ {current_price:,.0f}원"
                + (
                    f" | 손익={profit_loss:+,.0f}원 ({profit_loss_pct:+.2f}%)"
                    if profit_loss is not None
                    else ""
                )
            )
            return True

        except Exception as e:
            print(f"[{stock_name}] 분할 매도 주문 오류: {e}")
            traceback.print_exc()
            return False

    # ──────────────────────────────────────────────
    # TradingSummary 업데이트
    # ──────────────────────────────────────────────

    def _update_trading_summary(
        self,
        config: TradingConfig,
        stock_code: str,
        stock_name: str,
    ):
        """TradeEntry 집계를 기반으로 TradingSummary를 생성/갱신합니다."""
        try:
            all_entries = TradeEntry.objects.filter(
                user=self.user, stock_code=stock_code
            )
            buy_entries  = all_entries.filter(trade_type="BUY",  status__in=["FILLED", "CANCELLED"])
            sell_entries = all_entries.filter(trade_type="SELL", status="FILLED")

            if not buy_entries.exists():
                return

            total_buy_amount  = int(sum(float(e.filled_amount) for e in buy_entries))
            total_sell_amount = int(sum(float(e.filled_amount) for e in sell_entries))
            total_pl          = total_sell_amount - total_buy_amount
            pl_pct            = total_pl / total_buy_amount * 100.0 if total_buy_amount else 0.0

            entry_count = buy_entries.count()
            exit_count  = sell_entries.count()

            first_entry_date = buy_entries.order_by("filled_at").first()
            first_entry_date = first_entry_date.filled_at if first_entry_date else None

            last_exit_date = sell_entries.order_by("-filled_at").first()
            last_exit_date = last_exit_date.filled_at if last_exit_date else None

            # 보유 일수
            holding_days = 0.0
            if first_entry_date and last_exit_date:
                holding_days = (last_exit_date - first_entry_date).total_seconds() / 86400.0

            # 현재 보유 여부
            active_buys = all_entries.filter(trade_type="BUY", status="FILLED").exists()
            final_status = "HOLDING" if active_buys else "CLOSED"

            # 승률 (수익 매도 건 / 전체 매도 건)
            profitable = sum(
                1 for e in sell_entries
                if e.profit_loss is not None and float(e.profit_loss) > 0
            )
            win_rate = profitable / exit_count * 100.0 if exit_count else 0.0

            # get_or_create: first_entry_date 기준
            # max_profit_percent / max_drawdown은 _update_peak_stats()가 실시간 관리하므로 여기서 덮어쓰지 않음
            summary, _ = TradingSummary.objects.update_or_create(
                user             = self.user,
                stock_code       = stock_code,
                first_entry_date = first_entry_date,
                defaults=dict(
                    stock_name          = stock_name,
                    last_exit_date      = last_exit_date,
                    total_buy_amount    = total_buy_amount,
                    total_sell_amount   = total_sell_amount,
                    total_profit_loss   = total_pl,
                    profit_loss_percent = pl_pct,
                    holding_days        = holding_days,
                    entry_count         = entry_count,
                    exit_count          = exit_count,
                    trading_mode        = "manual" if config.trading_mode == "manual" else "turtle",
                    win_rate            = win_rate,
                    avg_holding_days    = holding_days,
                    final_status        = final_status,
                ),
            )

            # TradeEntry → TradingSummary FK 연결
            all_entries.update(trading_summary=summary)

            print(f"[{stock_name}] TradingSummary 업데이트 완료 (status={final_status})")

        except Exception as e:
            print(f"[{stock_name}] TradingSummary 업데이트 오류: {e}")
            traceback.print_exc()

    def _update_peak_stats(self, config: TradingConfig):
        """
        보유 중인 포지션의 현재가 기반으로 TradingSummary의
        max_profit_percent와 max_drawdown을 실시간 업데이트합니다.

        - max_profit_percent : 매매 중 달성한 최고 평가수익률
        - max_drawdown       : 고점 대비 최대 낙폭 (음수 저장, 예: -8.0 = -8%p)
          낙폭 = 최고수익률 - 현재수익률 (수익률 포인트 기준)
        """
        try:
            stock_code = config.stock_code

            holding_qty = 0
            avg_price   = 0.0
            for s in KIS.GetMyStockList():
                if s["StockCode"] == stock_code:
                    holding_qty = int(s["StockAmt"])
                    avg_price   = float(s["StockAvgPrice"])
                    break

            if holding_qty <= 0 or avg_price <= 0:
                return

            current_price      = float(KIS.GetCurrentPrice(stock_code))
            current_profit_pct = (current_price - avg_price) / avg_price * 100.0

            summary = TradingSummary.objects.filter(
                user        = self.user,
                stock_code  = stock_code,
                final_status = "HOLDING",
            ).order_by("-created_at").first()

            if summary is None:
                return

            update_fields = []

            # 최고 수익률 갱신
            if summary.max_profit_percent is None or current_profit_pct > summary.max_profit_percent:
                summary.max_profit_percent = current_profit_pct
                update_fields.append("max_profit_percent")

            # 고점 대비 낙폭 갱신 (낙폭 = -(고점 - 현재), 음수값이 클수록 낙폭이 큼)
            if summary.max_profit_percent is not None:
                drawdown = current_profit_pct - summary.max_profit_percent  # 음수
                if summary.max_drawdown is None or drawdown < summary.max_drawdown:
                    summary.max_drawdown = drawdown
                    if "max_drawdown" not in update_fields:
                        update_fields.append("max_drawdown")

            if update_fields:
                summary.save(update_fields=update_fields)

        except Exception as e:
            print(f"[{config.stock_name}] 피크 통계 업데이트 오류: {e}")

    # ──────────────────────────────────────────────
    # 시장 운영 시간 체크
    # ──────────────────────────────────────────────

    @staticmethod
    def is_market_open() -> bool:
        """한국 주식시장 운영 시간 여부 (09:00~15:30, 평일)."""
        from pytz import timezone as pytz_tz
        now = datetime.now(pytz_tz("Asia/Seoul"))
        if now.weekday() >= 5:   # 토/일
            return False
        t = now.strftime("%H%M")
        return "0900" <= t <= "1530"

    # ──────────────────────────────────────────────
    # 트레이딩 사이클 메인
    # ──────────────────────────────────────────────

    def run_trading_cycle(self):
        """
        1회 트레이딩 사이클을 실행합니다.
        APScheduler 또는 crontab 에서 분당 1회 호출합니다.
        """
        print(f"\n[TradingEngine] === 사이클 시작 ({datetime.now()}) ===")

        if not self.is_market_open():
            print("[TradingEngine] 장 시간 아님 — 종료")
            return

        try:
            balance = KIS.GetBalance()
            if float(balance["TotalMoney"]) <= 0:
                print("[TradingEngine] 잔고 부족 — 종료")
                return
            print(
                f"[TradingEngine] 잔고: 총={balance['TotalMoney']:,.0f}원,"
                f" 현금={balance['RemainMoney']:,.0f}원"
            )
        except Exception as e:
            print(f"[TradingEngine] 잔고 조회 실패: {e}")
            return

        # 설정 다시 로드 (사이클 시작 시 최신 상태 반영)
        self._load_configs()

        print(f"[TradingEngine] 대상 종목 수: {len(self.trading_configs)}")

        for config in self.trading_configs:
            try:
                print(f"\n[{config.stock_name}] 매매 체크 시작")

                # 0. 보유 중인 포지션의 최고수익률 / 최대낙폭 실시간 업데이트
                self._update_peak_stats(config)

                # 1. 청산 조건 체크 (우선 — 손절/트레일링스탑/전량 익절)
                should_exit, exit_reason = self.check_exit_conditions(config)
                if should_exit:
                    self.execute_sell_order(config, exit_reason)
                    continue

                # 1-b. 분할 익절 체크
                stage, sell_pct, stage_reason = self.check_staged_exit(config)
                if stage is not None:
                    self._mark_staged_exit_stage(config, stage)
                    if sell_pct >= 100:
                        self.execute_sell_order(config, stage_reason)
                    else:
                        self.execute_partial_sell_order(config, sell_pct, stage_reason)
                    continue

                # 2. 진입 조건 체크
                should_enter = self.check_entry_conditions(config)
                if should_enter:
                    position_amount = self.calculate_position_size(config)
                    if position_amount > 0:
                        entry_amount = self.get_current_entry_amount(config, position_amount)
                        if entry_amount > 0:
                            self.execute_buy_order(config, entry_amount)
                        else:
                            print(f"[{config.stock_name}] 피라미딩 한도 초과")

            except Exception as e:
                print(f"[{config.stock_name}] 처리 오류: {e}")
                traceback.print_exc()

        print(f"[TradingEngine] === 사이클 완료 ({datetime.now()}) ===\n")
