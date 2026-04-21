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
          - manual : 계좌잔고 × max_loss% ÷ stop_loss%
          - atr    : (계좌잔고 × max_loss%) ÷ (ATR × stop_loss배수 / 현재가)
        """
        try:
            balance     = KIS.GetBalance()
            total_money = float(balance["TotalMoney"])
            mode        = config.trading_mode
            max_loss_pct = (config.max_loss or 2.0) / 100.0  # 기본 2%

            if mode == "manual":
                stop_pct     = (config.stop_loss or 8.0) / 100.0
                pos_amount   = total_money * max_loss_pct / stop_pct
                print(
                    f"[{config.stock_name}] Manual 포지션:"
                    f" 잔고={total_money:,.0f}, 위험={max_loss_pct*100:.1f}%,"
                    f" 손절={stop_pct*100:.1f}%, 포지션={pos_amount:,.0f}원"
                )
                return pos_amount

            elif mode == "atr":
                atr = self.get_atr(config.stock_code)
                if atr:
                    current_price     = float(KIS.GetCurrentPrice(config.stock_code))
                    stop_multiplier   = config.stop_loss or 2.0
                    risk_amount       = total_money * max_loss_pct
                    stop_loss_price   = atr * stop_multiplier
                    stop_loss_ratio   = stop_loss_price / current_price
                    pos_amount        = risk_amount / stop_loss_ratio
                    print(
                        f"[{config.stock_name}] ATR 포지션:"
                        f" 잔고={total_money:,.0f}, ATR={atr:.1f}원,"
                        f" 손절폭={stop_loss_price:,.0f}원({stop_loss_ratio*100:.1f}%),"
                        f" 포지션={pos_amount:,.0f}원"
                    )
                    return pos_amount
                else:
                    # ATR 실패 → manual 방식으로 fallback
                    print(f"[{config.stock_name}] ATR 실패 → Manual fallback")
                    stop_pct   = (config.stop_loss or 2.0) / 100.0
                    pos_amount = total_money * max_loss_pct / stop_pct
                    return pos_amount

            else:
                # 알 수 없는 모드 → manual 방식
                stop_pct   = (config.stop_loss or 8.0) / 100.0
                pos_amount = total_money * max_loss_pct / stop_pct
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

    def check_exit_conditions(
        self, config: TradingConfig
    ) -> tuple[bool, str | None]:
        """
        손절 / 익절 조건을 체크합니다.

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
            profit_pct    = (current_price - avg_price) / avg_price * 100.0

            if mode == "manual":
                if profit_pct <= -stop_loss:
                    return True, "손절"
                if profit_pct >= take_profit:
                    return True, "익절"

            else:  # atr
                atr = self.get_atr(stock_code)
                if atr is not None:
                    if current_price <= avg_price - atr * stop_loss:
                        return True, "ATR 손절"
                    if current_price >= avg_price + atr * take_profit:
                        return True, "ATR 익절"

            return False, None

        except Exception as e:
            print(f"[{config.stock_name}] 청산 조건 체크 오류: {e}")
            return False, None

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
            if "손절" in reason:
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
            summary, _ = TradingSummary.objects.update_or_create(
                user             = self.user,
                stock_code       = stock_code,
                first_entry_date = first_entry_date,
                defaults=dict(
                    stock_name        = stock_name,
                    last_exit_date    = last_exit_date,
                    total_buy_amount  = total_buy_amount,
                    total_sell_amount = total_sell_amount,
                    total_profit_loss = total_pl,
                    profit_loss_percent = pl_pct,
                    holding_days      = holding_days,
                    entry_count       = entry_count,
                    exit_count        = exit_count,
                    trading_mode      = "manual" if config.trading_mode == "manual" else "turtle",
                    win_rate          = win_rate,
                    avg_holding_days  = holding_days,
                    final_status      = final_status,
                ),
            )
            print(f"[{stock_name}] TradingSummary 업데이트 완료 (status={final_status})")

        except Exception as e:
            print(f"[{stock_name}] TradingSummary 업데이트 오류: {e}")
            traceback.print_exc()

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

                # 1. 청산 조건 체크 (우선)
                should_exit, exit_reason = self.check_exit_conditions(config)
                if should_exit:
                    self.execute_sell_order(config, exit_reason)
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
