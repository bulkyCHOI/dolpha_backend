"""급등테마주(Theme Surge) 1분봉 데이터 기반 고정밀 백테스팅 엔진 및 그리드 탐색.

사용 DB 테이블:
- ThemeEntrySignal: 진입 신호(스윙/모닝 전고점, 눌림목 저점 1T/1R 좌표, 발생 시점)
- StockMinuteOhlcv: 20만여 건의 종목별 1분봉 실데이터 (고가, 저가, 종가, 거래량)
- ThemeSnapshot: 테마 랭킹 실시간 스냅샷 (테마 등락률 검증)

분석 지표:
- 총 거래 수 (Trades)
- 승률 (Win Rate, %)
- 손익비 (Profit/Loss Ratio, RRR)
- 누적 수익률 (Cumulative Return, %)
- 거래당 기대값 (Expectancy, R 및 %)
- 최대 낙폭 (MDD, %)

작성일: 2026-10-16
작성자: 최재성 퀀트 트레이딩 엔진 백테스터
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import django
import pytz

# Django 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dolpha.settings")
django.setup()

from myweb.models import StockMinuteOhlcv, ThemeEntrySignal, ThemeSnapshot  # noqa: E402

KST = pytz.timezone("Asia/Seoul")
DEFAULT_COST_PCT = 0.23  # 왕복 거래비용 (수수료 + 증권거래세 등 %)


@dataclass
class TradeResult:
    stock_code: str
    stock_name: str
    date: dt.date
    entry_time: dt.datetime
    entry_price: float
    exit_time: dt.datetime
    exit_price: float
    exit_reason: str
    pnl_pct: float          # 세전 손익률 (%)
    net_pnl_pct: float      # 비용(0.23%) 차감 후 손익률 (%)
    r_multiple: float       # 1R 손절폭 대비 실현 R배수
    t_value: float          # 1T 가격폭
    stop_price: float       # 초기 손절가
    overnight: bool         # 오버나이트 이월 여부
    stages_hit: list[int]   # 익절 차수 통과 목록


@dataclass
class BacktestSummary:
    name: str
    params: dict
    total_trades: int
    win_trades: int
    loss_trades: int
    even_trades: int
    win_rate: float         # %
    avg_win_pct: float      # %
    avg_loss_pct: float     # %
    profit_loss_ratio: float
    cum_return_pct: float   # %
    avg_net_pnl_pct: float  # %
    avg_r: float            # R
    mdd_pct: float          # %
    overnight_trades: int


def load_market_data():
    """DB에서 ThemeEntrySignal, StockMinuteOhlcv, ThemeSnapshot을 메모리로 고속 적재합니다."""
    print(">> [1/3] DB 데이터 메모리 로딩 시작...")
    
    # 1. ThemeEntrySignal 중 유효한 눌림목 셋업만 추출
    raw_signals = ThemeEntrySignal.objects.all().order_by("checked_at")
    sig_by_stock_date = defaultdict(list)
    for s in raw_signals:
        if s.has_pullback and s.prev_high and s.pullback_low and s.prev_high > s.pullback_low:
            sig_by_stock_date[(s.date, s.stock_code)].append(s)

    # 2. ThemeSnapshot 테마별 등락률 매핑
    snap_dict = defaultdict(list)
    for sn in ThemeSnapshot.objects.all().order_by("slot_time"):
        snap_dict[(sn.date, sn.tics_id)].append(sn)

    # 3. StockMinuteOhlcv 1분봉 데이터 매핑
    # 메모리 효율과 속도를 위해 (date, stock_code) -> list of dicts
    bar_dict = defaultdict(list)
    total_bars = StockMinuteOhlcv.objects.count()
    print(f"   - 총 1분봉 레코드 수: {total_bars:,} 건 로딩 중...")
    
    bars_qs = StockMinuteOhlcv.objects.all().order_by("bar_datetime").values(
        "stock_code", "bar_datetime", "open", "high", "low", "close", "volume"
    )
    for b in bars_qs:
        d = b["bar_datetime"].astimezone(KST).date()
        bar_dict[(d, b["stock_code"])].append(b)

    print(f">> [1/3] 로딩 완료: 분석 가능 (날짜, 종목) 세트 {len(sig_by_stock_date)} 개, 분봉 일자군 {len(bar_dict)} 개")
    return sig_by_stock_date, snap_dict, bar_dict


def simulate_strategy(
    sig_by_stock_date: dict,
    snap_dict: dict,
    bar_dict: dict,
    *,
    theme_min_fluct: float = 3.0,          # 진입 테마 등락률 하한 (%)
    breakout_vol_max: float | None = None,  # 돌파 거래량비 상한 (예: 4.0, None이면 무제한)
    breakout_buffer_pct: float = 0.1,      # 돌파 버퍼 (%)
    breakout_vol_min: float = 1.5,         # 돌파 거래량비 하한
    exit_stages: tuple = ((1.5, 25.0), (3.0, 35.0), (4.5, 20.0)),  # ((T배수, 청산비율%), ...)
    trailing_start_t: float = 1.5,         # 트레일링 스탑 개시 T배수
    trailing_bar_count: int = 3,           # 트레일링 직전 N봉
    entry_cutoff_hour: int = 15,           # 진입 마감 시 (기본 15시, 14시 제한 시 14)
    exclude_lunch: bool = False,           # 점심시간 (11:00 ~ 12:30) 진입 배제 여부
    overnight_mode: str = "liquidate_all", # "liquidate_all"(당일 15:10 강제청산) vs "profit_only"(수익 포지션만 이월)
    force_exit_time_str: str = "15:10",    # 당일 강제청산 시각
    cost_pct: float = DEFAULT_COST_PCT,
) -> list[TradeResult]:
    """주어진 파라미터로 전 구간 1분 단위 백테스트를 실행합니다."""
    force_exit_t = dt.datetime.strptime(force_exit_time_str, "%H:%M").time()
    
    trades: list[TradeResult] = []

    # 전체 거래일 목록
    all_dates = sorted(list(set(d for d, _ in sig_by_stock_date.keys())))
    
    # 일별 연속 시뮬레이션 (오버나이트 이월 포지션 추적)
    active_overnight_positions = []  # list of open position dicts

    for day_idx, current_date in enumerate(all_dates):
        # 1. 전일에서 이월된 포지션이 있다면 당일 분봉에서 우선 추적 청산
        next_day_positions = []
        for pos in active_overnight_positions:
            code = pos["stock_code"]
            bars = bar_dict.get((current_date, code), [])
            if not bars:
                # 당일 분봉이 없으면 전일 종가 또는 현재 보유 유지
                next_day_positions.append(pos)
                continue

            closed = False
            # 이월 2일차 본전 손절 적용 (평단과 초기 손절가 중 큰 것)
            effective_stop = max(pos["stop_price"], pos["entry_price"])
            
            for b_idx, b in enumerate(bars):
                b_time = b["bar_datetime"].astimezone(KST).time()
                high, low, close, open_p = b["high"], b["low"], b["close"], b["open"]

                # 1) 손절 체크
                if low <= effective_stop:
                    exit_price = min(open_p, effective_stop)
                    pnl_pct = (exit_price / pos["entry_price"] - 1.0) * 100.0
                    net_pnl = pnl_pct - cost_pct
                    r_mult = (exit_price - pos["entry_price"]) / pos["1r"] if pos["1r"] > 0 else 0
                    trades.append(TradeResult(
                        stock_code=code, stock_name=pos["stock_name"], date=pos["entry_date"],
                        entry_time=pos["entry_time"], entry_price=pos["entry_price"],
                        exit_time=b["bar_datetime"], exit_price=exit_price,
                        exit_reason="오버나이트 본전/손절", pnl_pct=pnl_pct, net_pnl_pct=net_pnl,
                        r_multiple=r_mult, t_value=pos["t_value"], stop_price=effective_stop,
                        overnight=True, stages_hit=pos["stages_hit"],
                    ))
                    closed = True
                    break

                # 2) 당일 마감 시각 강제청산
                if b_time >= force_exit_t:
                    exit_price = close
                    pnl_pct = (exit_price / pos["entry_price"] - 1.0) * 100.0
                    net_pnl = pnl_pct - cost_pct
                    r_mult = (exit_price - pos["entry_price"]) / pos["1r"] if pos["1r"] > 0 else 0
                    trades.append(TradeResult(
                        stock_code=code, stock_name=pos["stock_name"], date=pos["entry_date"],
                        entry_time=pos["entry_time"], entry_price=pos["entry_price"],
                        exit_time=b["bar_datetime"], exit_price=exit_price,
                        exit_reason=f"오버나이트 만료 강제청산({force_exit_time_str})",
                        pnl_pct=pnl_pct, net_pnl_pct=net_pnl, r_multiple=r_mult,
                        t_value=pos["t_value"], stop_price=effective_stop,
                        overnight=True, stages_hit=pos["stages_hit"],
                    ))
                    closed = True
                    break

            if not closed:
                # 2거래일 이상 초과 보유는 방지 (최대 1박 2일)
                exit_price = bars[-1]["close"]
                pnl_pct = (exit_price / pos["entry_price"] - 1.0) * 100.0
                net_pnl = pnl_pct - cost_pct
                r_mult = (exit_price - pos["entry_price"]) / pos["1r"] if pos["1r"] > 0 else 0
                trades.append(TradeResult(
                    stock_code=code, stock_name=pos["stock_name"], date=pos["entry_date"],
                    entry_time=pos["entry_time"], entry_price=pos["entry_price"],
                    exit_time=bars[-1]["bar_datetime"], exit_price=exit_price,
                    exit_reason="오버나이트 최대 보유일 만료",
                    pnl_pct=pnl_pct, net_pnl_pct=net_pnl, r_multiple=r_mult,
                    t_value=pos["t_value"], stop_price=effective_stop,
                    overnight=True, stages_hit=pos["stages_hit"],
                ))

        active_overnight_positions = []

        # 2. 당일 신규 진입 종목 탐색
        day_stock_codes = [c for d, c in sig_by_stock_date.keys() if d == current_date]

        for code in day_stock_codes:
            sig_list = sig_by_stock_date[(current_date, code)]
            bars = bar_dict.get((current_date, code), [])
            if not bars:
                continue

            # 중복 진입 방지 (종목당 당일 1회 진입)
            day_entered = False

            for s in sig_list:
                if day_entered:
                    break

                prev_high = s.prev_high
                pullback_low = s.pullback_low
                t_val = prev_high - pullback_low
                if t_val <= 0:
                    continue
                one_r = prev_high * (1.0 + breakout_buffer_pct / 100.0) - pullback_low
                breakout_threshold = prev_high * (1.0 + breakout_buffer_pct / 100.0)

                # 진입 가능 분봉 순회
                for b_idx, b in enumerate(bars):
                    if b["bar_datetime"] < s.checked_at:
                        continue

                    b_kst = b["bar_datetime"].astimezone(KST)
                    b_time = b_kst.time()

                    # 시간 필터 체크
                    if b_kst.hour >= entry_cutoff_hour:
                        break  # 진입 시각 마감
                    if exclude_lunch and dt.time(11, 0) <= b_time < dt.time(12, 30):
                        continue

                    # 돌파 가격 체크
                    if b["high"] < breakout_threshold:
                        continue

                    # 거래량비 체크 (최근 20봉 평균)
                    recent_bars = bars[max(0, b_idx - 20) : b_idx]
                    avg_vol = sum(rb["volume"] for rb in recent_bars) / len(recent_bars) if recent_bars else 0
                    v_ratio = b["volume"] / avg_vol if avg_vol > 0 else 0
                    if v_ratio < breakout_vol_min:
                        continue
                    if breakout_vol_max is not None and v_ratio > breakout_vol_max:
                        continue

                    # 테마 등락률 체크
                    th_fluc = s.theme_fluctuation
                    if th_fluc is None:
                        for sn in reversed(snap_dict.get((current_date, s.tics_id), [])):
                            if sn.slot_time <= b_time:
                                th_fluc = sn.fluctuation_rate
                                break
                    if th_fluc is None or th_fluc < theme_min_fluct:
                        continue

                    # === 진입 성립 ===
                    day_entered = True
                    entry_price = max(b["open"], breakout_threshold)
                    entry_time = b["bar_datetime"]
                    stop_price = pullback_low
                    real_1r = entry_price - stop_price
                    if real_1r <= 0:
                        real_1r = t_val

                    # 3. 진입 이후 1분 단위 청산 추적 (Simulation Loop)
                    remaining_pct = 100.0
                    realized_pnl_weighted = 0.0
                    stages_hit = []
                    trailing_started = False
                    peak_price = entry_price
                    pos_closed = False

                    # 진입봉 이후의 봉들로 청산 검증
                    for cur_idx in range(b_idx + 1, len(bars)):
                        cur_b = bars[cur_idx]
                        cur_time = cur_b["bar_datetime"].astimezone(KST).time()
                        high = cur_b["high"]
                        low = cur_b["low"]
                        close = cur_b["close"]
                        open_p = cur_b["open"]

                        peak_price = max(peak_price, high)

                        # (1) 손절 체크: low <= stop_price
                        if low <= stop_price:
                            exit_p = min(open_p, stop_price)
                            pnl = (exit_p / entry_price - 1.0) * 100.0
                            realized_pnl_weighted += (remaining_pct / 100.0) * pnl
                            r_mult = (exit_p - entry_price) / real_1r
                            trades.append(TradeResult(
                                stock_code=code, stock_name=s.stock_name, date=current_date,
                                entry_time=entry_time, entry_price=entry_price,
                                exit_time=cur_b["bar_datetime"], exit_price=exit_p,
                                exit_reason=f"손절(눌림저점 {stop_price:,.0f} 이탈)",
                                pnl_pct=realized_pnl_weighted,
                                net_pnl_pct=realized_pnl_weighted - cost_pct,
                                r_multiple=r_mult, t_value=t_val, stop_price=stop_price,
                                overnight=False, stages_hit=stages_hit,
                            ))
                            pos_closed = True
                            break

                        # (2) 당일 마감 시각 도달 (15:10 강제청산 vs 오버나이트 판정)
                        if cur_time >= force_exit_t:
                            if overnight_mode == "profit_only" and close >= entry_price + t_val * 0.5:
                                # 0.5T(또는 0.5R) 이상 수익 포지션은 익일로 이월!
                                active_overnight_positions.append({
                                    "stock_code": code,
                                    "stock_name": s.stock_name,
                                    "entry_date": current_date,
                                    "entry_time": entry_time,
                                    "entry_price": entry_price,
                                    "stop_price": stop_price,
                                    "t_value": t_val,
                                    "1r": real_1r,
                                    "stages_hit": stages_hit,
                                })
                                pos_closed = True
                                break
                            else:
                                # 전량 강제청산
                                exit_p = close
                                pnl = (exit_p / entry_price - 1.0) * 100.0
                                realized_pnl_weighted += (remaining_pct / 100.0) * pnl
                                r_mult = (exit_p - entry_price) / real_1r
                                trades.append(TradeResult(
                                    stock_code=code, stock_name=s.stock_name, date=current_date,
                                    entry_time=entry_time, entry_price=entry_price,
                                    exit_time=cur_b["bar_datetime"], exit_price=exit_p,
                                    exit_reason=f"당일 강제청산({force_exit_time_str})",
                                    pnl_pct=realized_pnl_weighted,
                                    net_pnl_pct=realized_pnl_weighted - cost_pct,
                                    r_multiple=r_mult, t_value=t_val, stop_price=stop_price,
                                    overnight=False, stages_hit=stages_hit,
                                ))
                                pos_closed = True
                                break

                        # (3) 트레일링 스탑 체크
                        trailing_trigger_price = entry_price + t_val * trailing_start_t
                        if not trailing_started and peak_price >= trailing_trigger_price:
                            trailing_started = True

                        if trailing_started:
                            # 직전 N봉 최저가 계산 (진행 중인 봉 제외)
                            hist_window = bars[max(0, cur_idx - trailing_bar_count) : cur_idx]
                            if hist_window:
                                trail_line = min(hb["low"] for hb in hist_window)
                                if low <= trail_line:
                                    exit_p = min(open_p, trail_line)
                                    pnl = (exit_p / entry_price - 1.0) * 100.0
                                    realized_pnl_weighted += (remaining_pct / 100.0) * pnl
                                    r_mult = (exit_p - entry_price) / real_1r
                                    trades.append(TradeResult(
                                        stock_code=code, stock_name=s.stock_name, date=current_date,
                                        entry_time=entry_time, entry_price=entry_price,
                                        exit_time=cur_b["bar_datetime"], exit_price=exit_p,
                                        exit_reason=f"트레일링스탑({trailing_bar_count}봉 최저 {trail_line:,.0f} 이탈)",
                                        pnl_pct=realized_pnl_weighted,
                                        net_pnl_pct=realized_pnl_weighted - cost_pct,
                                        r_multiple=r_mult, t_value=t_val, stop_price=stop_price,
                                        overnight=False, stages_hit=stages_hit,
                                    ))
                                    pos_closed = True
                                    break

                        # (4) 분할 익절 단계 체크
                        for stg_idx, (t_mult, sell_pct) in enumerate(exit_stages, start=1):
                            if stg_idx not in stages_hit:
                                target_price = entry_price + t_val * t_mult
                                if high >= target_price:
                                    # 분할 익절 도달
                                    actual_sell = min(remaining_pct, sell_pct)
                                    pnl = (target_price / entry_price - 1.0) * 100.0
                                    realized_pnl_weighted += (actual_sell / 100.0) * pnl
                                    remaining_pct -= actual_sell
                                    stages_hit.append(stg_idx)
                                    if remaining_pct <= 0:
                                        r_mult = (target_price - entry_price) / real_1r
                                        trades.append(TradeResult(
                                            stock_code=code, stock_name=s.stock_name, date=current_date,
                                            entry_time=entry_time, entry_price=entry_price,
                                            exit_time=cur_b["bar_datetime"], exit_price=target_price,
                                            exit_reason=f"{stg_idx}차 전량 익절 완료",
                                            pnl_pct=realized_pnl_weighted,
                                            net_pnl_pct=realized_pnl_weighted - cost_pct,
                                            r_multiple=r_mult, t_value=t_val, stop_price=stop_price,
                                            overnight=False, stages_hit=stages_hit,
                                        ))
                                        pos_closed = True
                                        break

                        if pos_closed:
                            break

                    # 장 마감 봉까지도 청산되지 않은 경우 마지막 봉 종가 청산
                    if not pos_closed and bars:
                        last_b = bars[-1]
                        exit_p = last_b["close"]
                        pnl = (exit_p / entry_price - 1.0) * 100.0
                        realized_pnl_weighted += (remaining_pct / 100.0) * pnl
                        r_mult = (exit_p - entry_price) / real_1r
                        trades.append(TradeResult(
                            stock_code=code, stock_name=s.stock_name, date=current_date,
                            entry_time=entry_time, entry_price=entry_price,
                            exit_time=last_b["bar_datetime"], exit_price=exit_p,
                            exit_reason="장마감 잔량 종가 청산",
                            pnl_pct=realized_pnl_weighted,
                            net_pnl_pct=realized_pnl_weighted - cost_pct,
                            r_multiple=r_mult, t_value=t_val, stop_price=stop_price,
                            overnight=False, stages_hit=stages_hit,
                        ))

                    break  # 해당 종목 진입 완료

    return trades


def evaluate_metrics(name: str, params: dict, trades: list[TradeResult]) -> BacktestSummary:
    """거래 결과를 바탕으로 승률, 손익비, 순수익률, MDD 등을 계산합니다."""
    if not trades:
        return BacktestSummary(
            name=name, params=params, total_trades=0, win_trades=0, loss_trades=0,
            even_trades=0, win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            profit_loss_ratio=0.0, cum_return_pct=0.0, avg_net_pnl_pct=0.0,
            avg_r=0.0, mdd_pct=0.0, overnight_trades=0,
        )

    win_trades = [t for t in trades if t.net_pnl_pct > 0]
    loss_trades = [t for t in trades if t.net_pnl_pct < 0]
    even_trades = [t for t in trades if t.net_pnl_pct == 0]
    overnight_cnt = sum(1 for t in trades if t.overnight)

    win_rate = len(win_trades) / len(trades) * 100.0
    avg_win = sum(t.net_pnl_pct for t in win_trades) / len(win_trades) if win_trades else 0.0
    avg_loss = abs(sum(t.net_pnl_pct for t in loss_trades) / len(loss_trades)) if loss_trades else 0.0
    rrr = avg_win / avg_loss if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)

    # 누적 수익률 계산 (단리 합산 기준 및 복리 시뮬레이션)
    cum_return = sum(t.net_pnl_pct for t in trades)
    avg_net_pnl = cum_return / len(trades)
    avg_r = sum(t.r_multiple for t in trades) / len(trades)

    # MDD 계산 (Equity Curve 기반)
    peak = 100.0
    equity = 100.0
    max_dd = 0.0
    for t in trades:
        equity *= (1.0 + t.net_pnl_pct / 100.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    return BacktestSummary(
        name=name,
        params=params,
        total_trades=len(trades),
        win_trades=len(win_trades),
        loss_trades=len(loss_trades),
        even_trades=len(even_trades),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_loss_ratio=rrr,
        cum_return_pct=cum_return,
        avg_net_pnl_pct=avg_net_pnl,
        avg_r=avg_r,
        mdd_pct=max_dd,
        overnight_trades=overnight_cnt,
    )


def run_grid_search(sig_by_stock_date, snap_dict, bar_dict):
    """지정된 파라미터 조합으로 Grid Search를 수행합니다."""
    print("\n" + "=" * 80)
    print(">> [2/3] 급등테마주(theme_surge) 파라미터 격자 탐색(Grid Search) 시작")
    print("=" * 80)

    # 1. Baseline 시뮬레이션
    baseline_params = {
        "theme_min_fluct": 3.0,
        "breakout_vol_max": None,
        "breakout_buffer_pct": 0.1,
        "exit_stages": ((1.5, 25.0), (3.0, 35.0), (4.5, 20.0)),
        "trailing_start_t": 1.5,
        "entry_cutoff_hour": 15,
        "exclude_lunch": False,
        "overnight_mode": "liquidate_all",
    }
    baseline_trades = simulate_strategy(sig_by_stock_date, snap_dict, bar_dict, **baseline_params)
    baseline_res = evaluate_metrics("Baseline (현재 기준)", baseline_params, baseline_trades)

    # 2. 파라미터 탐색 공간 정의
    param_grid = {
        "theme_min_fluct": [3.0, 4.0, 5.0],
        "breakout_vol_max": [None, 4.0],
        "breakout_buffer_pct": [0.1, 0.3, 0.5],
        "exit_stages": [
            ((1.5, 50.0), (3.0, 50.0)),
            ((2.0, 50.0), (4.0, 50.0)),
            ((1.5, 25.0), (2.5, 35.0), (4.0, 20.0)),
            ((1.5, 25.0), (3.0, 35.0), (4.5, 20.0)),
        ],
        "trailing_start_t": [1.5, 2.0],
        "time_filter": [
            ("all", 15, False),
            ("no_late", 14, False),
            ("no_lunch", 15, True),
            ("strict_time", 14, True),
        ],
        "overnight_mode": ["liquidate_all", "profit_only"],
    }

    # 개별 단일 요인 분석 (Factor Analysis)
    print("\n--- [개별 요인별 민감도 분석] ---")
    factor_experiments = [
        # (1) 테마 등락률
        ("테마 등락률 >= 4.0%", {**baseline_params, "theme_min_fluct": 4.0}),
        ("테마 등락률 >= 5.0%", {**baseline_params, "theme_min_fluct": 5.0}),
        # (2) 거래량비 상한
        ("거래량비 상한 4.0배", {**baseline_params, "breakout_vol_max": 4.0}),
        # (3) 돌파 버퍼
        ("돌파 버퍼 0.3%", {**baseline_params, "breakout_buffer_pct": 0.3}),
        ("돌파 버퍼 0.5%", {**baseline_params, "breakout_buffer_pct": 0.5}),
        # (4) 분할 익절 단계
        ("익절 (1.5T 50%, 3.0T 50%)", {**baseline_params, "exit_stages": ((1.5, 50.0), (3.0, 50.0))}),
        ("익절 (2.0T 50%, 4.0T 50%)", {**baseline_params, "exit_stages": ((2.0, 50.0), (4.0, 50.0))}),
        ("익절 (1.5T 25%, 2.5T 35%, 4.0T 20%)", {**baseline_params, "exit_stages": ((1.5, 25.0), (2.5, 35.0), (4.0, 20.0))}),
        # (5) 트레일링 시작
        ("트레일링 2.0T 개시", {**baseline_params, "trailing_start_t": 2.0}),
        # (6) 시간대 제한
        ("14시 이후 진입 금지", {**baseline_params, "entry_cutoff_hour": 14}),
        ("점심시간(11:00~12:30) 제외", {**baseline_params, "exclude_lunch": True}),
        ("14시 이후 + 점심 제외", {**baseline_params, "entry_cutoff_hour": 14, "exclude_lunch": True}),
        # (7) 오버나이트
        ("수익 포지션만 오버나이트 이월", {**baseline_params, "overnight_mode": "profit_only"}),
    ]

    factor_results = []
    for exp_name, params in factor_experiments:
        tds = simulate_strategy(sig_by_stock_date, snap_dict, bar_dict, **params)
        res = evaluate_metrics(exp_name, params, tds)
        factor_results.append(res)

    # 3. 최적 파라미터 조합 탐색 (Full Grid Search 샘플링 및 핵심 조합 30여 개)
    print("\n--- [핵심 복합 파라미터 조합 탐색] ---")
    combination_results = []
    
    # 전략적 베스트 후보군 테스트
    candidate_combos = [
        ("개선안 A (보수적 품질 + 수익 이월)", {
            "theme_min_fluct": 4.0,
            "breakout_vol_max": 4.0,
            "breakout_buffer_pct": 0.3,
            "exit_stages": ((1.5, 25.0), (3.0, 35.0), (4.5, 20.0)),
            "trailing_start_t": 1.5,
            "entry_cutoff_hour": 14,
            "exclude_lunch": False,
            "overnight_mode": "profit_only",
        }),
        ("개선안 B (강한 테마 5% + 와이드 익절)", {
            "theme_min_fluct": 5.0,
            "breakout_vol_max": 4.0,
            "breakout_buffer_pct": 0.3,
            "exit_stages": ((2.0, 50.0), (4.0, 50.0)),
            "trailing_start_t": 2.0,
            "entry_cutoff_hour": 14,
            "exclude_lunch": True,
            "overnight_mode": "profit_only",
        }),
        ("개선안 C (단기 집중: 14시 전 + 빠른 익절)", {
            "theme_min_fluct": 4.0,
            "breakout_vol_max": None,
            "breakout_buffer_pct": 0.1,
            "exit_stages": ((1.5, 50.0), (3.0, 50.0)),
            "trailing_start_t": 1.5,
            "entry_cutoff_hour": 14,
            "exclude_lunch": False,
            "overnight_mode": "liquidate_all",
        }),
        ("개선안 D (적극 모멘텀: 테마 4% + 3단 익절 + 오버나이트)", {
            "theme_min_fluct": 4.0,
            "breakout_vol_max": 4.0,
            "breakout_buffer_pct": 0.1,
            "exit_stages": ((1.5, 25.0), (2.5, 35.0), (4.0, 20.0)),
            "trailing_start_t": 1.5,
            "entry_cutoff_hour": 14,
            "exclude_lunch": True,
            "overnight_mode": "profit_only",
        }),
        ("개선안 E (초강력 필터: 5% + 클라이맥스 배제 + 버퍼 0.5%)", {
            "theme_min_fluct": 5.0,
            "breakout_vol_max": 4.0,
            "breakout_buffer_pct": 0.5,
            "exit_stages": ((1.5, 25.0), (3.0, 35.0), (4.5, 20.0)),
            "trailing_start_t": 1.5,
            "entry_cutoff_hour": 14,
            "exclude_lunch": True,
            "overnight_mode": "profit_only",
        }),
    ]

    for c_name, params in candidate_combos:
        tds = simulate_strategy(sig_by_stock_date, snap_dict, bar_dict, **params)
        res = evaluate_metrics(c_name, params, tds)
        combination_results.append(res)

    return baseline_res, factor_results, combination_results


def print_report(baseline_res: BacktestSummary, factor_results: list[BacktestSummary], combination_results: list[BacktestSummary]):
    """콘솔 및 파일 출력을 위한 깔끔한 테이블 형식 리포트 출력."""
    header = f"{'전략 / 파라미터 조합':<36} | {'거래수':>5} | {'승률':>6} | {'손익비':>6} | {'누적수익률':>9} | {'거래당평균':>8} | {'평균R':>6} | {'MDD':>6}"
    sep = "-" * len(header)

    print("\n" + "=" * 95)
    print("📊 [3/3] 급등테마주(theme_surge) 1분봉 백테스팅 시뮬레이션 결과 리포트")
    print("=" * 95)
    print(header)
    print(sep)

    def print_row(res: BacktestSummary):
        print(
            f"{res.name:<36} | {res.total_trades:>5} | {res.win_rate:>5.1f}% | {res.profit_loss_ratio:>6.2f} | "
            f"{res.cum_return_pct:>+8.2f}% | {res.avg_net_pnl_pct:>+7.2f}% | {res.avg_r:>+5.2f}R | {res.mdd_pct:>5.1f}%"
        )

    print_row(baseline_res)
    print(sep)
    print("[단일 요인별 민감도 분석]")
    for r in factor_results:
        print_row(r)

    print(sep)
    print("[주요 복합 개선안]")
    for r in combination_results:
        print_row(r)
    print("=" * 95 + "\n")


def main():
    sig_by_stock_date, snap_dict, bar_dict = load_market_data()
    baseline_res, factor_results, combination_results = run_grid_search(
        sig_by_stock_date, snap_dict, bar_dict
    )
    print_report(baseline_res, factor_results, combination_results)


if __name__ == "__main__":
    main()
