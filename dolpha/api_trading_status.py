"""
거래 상태 조회 API 엔드포인트 (Django Ninja)

TradeEntry DB + KIS 실계좌 데이터를 합산하여 stock_code 키 딕셔너리로 반환합니다.
avg_price는 KIS 실계좌 기준값을 우선 사용합니다.
"""

from datetime import date as date_type

from ninja import Router
from django.http import JsonResponse

from .api_mypage_ninja import get_authenticated_user
from myweb.models import TradeEntry, TradingConfig, TradingDefaults, StockAnalysis, Company, DailyAccountSnapshot

trading_status_router = Router()


@trading_status_router.get("/account-balance")
def get_account_balance(request):
    """KIS 계좌 잔고 조회 (총 평가금액, 예수금, 주식 평가금액, 평가손익)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        try:
            from dolpha.kis.trade import GetBalance
            balance = GetBalance()
            return JsonResponse({"success": True, "data": balance})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=503)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@trading_status_router.get("/trading-status")
def get_trading_status(request):
    """
    현재 거래 상태 및 포지션 정보를 stock_code 키 딕셔너리로 반환합니다.

    반환 형식:
    {
      "success": true,
      "data": {
        "272210": {
          "actual_entries": 1,
          "total_possible_entries": 2,
          "position_sum": 50.0,
          "total_quantity": 1902,
          "avg_price": 131397.0,   ← KIS 실제 평균단가 (반올림 정수)
          "holding_amount": 249,997,194.0
        }
      }
    }
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)

        # KIS 실계좌 보유 현황 조회 (avg_price 정확도 향상)
        kis_holdings = {}
        try:
            from dolpha.kis.trade import GetMyStockList
            for s in GetMyStockList():
                code = s["StockCode"]
                kis_holdings[code] = {
                    "qty": int(s["StockAmt"]),
                    "avg_price": round(float(s["StockAvgPrice"])),
                    "current_price": round(float(s["StockNowPrice"])),
                    "profit_loss_amount": round(float(s["StockRevenueMoney"])),
                    "profit_loss_rate": float(s["StockRevenueRate"]),
                    "stock_name": s["StockName"],
                }
        except Exception:
            pass  # KIS 조회 실패 시 DB 계산값으로 fallback

        # 유저 TradingDefaults (트레일링 스탑 설정)
        try:
            defaults = TradingDefaults.objects.get(user=user)
        except TradingDefaults.DoesNotExist:
            defaults = None

        # 활성 매수 체결 내역 (FILLED BUY만)
        active_entries = TradeEntry.objects.filter(
            user=user,
            trade_type="BUY",
            status="FILLED",
        ).order_by("filled_at")

        entries_by_code: dict = {}
        for e in active_entries:
            entries_by_code.setdefault(e.stock_code, []).append(e)

        configs = {
            c.stock_code: c
            for c in TradingConfig.objects.filter(user=user, is_active=True)
        }

        # DB 기록이 있는 종목 + KIS에 실제 보유 중인 종목 모두 포함
        all_codes = set(entries_by_code.keys()) | set(kis_holdings.keys())

        # ATR 일괄 조회 (종목코드 → atr)
        atr_by_code: dict = {}
        for code in all_codes:
            try:
                company = Company.objects.filter(code=code).first()
                if company:
                    analysis = StockAnalysis.objects.filter(code=company).order_by("-date").first()
                    if analysis and analysis.atr > 0:
                        atr_by_code[code] = float(analysis.atr)
            except Exception:
                pass

        data = {}
        for code in all_codes:
            entries = entries_by_code.get(code, [])
            config = configs.get(code)
            kis = kis_holdings.get(code, {})

            # KIS 보유 수량/평단가 우선
            if kis and kis["qty"] > 0:
                total_qty = kis["qty"]
                avg_price = kis["avg_price"]
                current_price = kis.get("current_price")
                profit_loss_amount = kis.get("profit_loss_amount")
                profit_loss_rate = kis.get("profit_loss_rate")
                stock_name_kis = kis.get("stock_name")
            elif entries:
                total_qty = sum(e.filled_quantity for e in entries)
                if total_qty <= 0:
                    continue
                total_amount = sum(float(e.filled_price) * e.filled_quantity for e in entries)
                avg_price = round(total_amount / total_qty)
                current_price = None
                profit_loss_amount = None
                profit_loss_rate = None
                stock_name_kis = None
            else:
                continue

            actual_entries = len(entries)
            pyramiding_count = config.pyramiding_count if config else 0
            total_possible_entries = pyramiding_count + 1

            position_sum = 0.0
            if config and config.positions:
                used = config.positions[:actual_entries]
                total_ratio = sum(config.positions[:total_possible_entries]) or 1
                position_sum = sum(used) / total_ratio * 100.0

            atr = atr_by_code.get(code)
            mode = config.trading_mode if config else "manual"

            # ── 손절가 계산 ──────────────────────────────────────────
            stop_price = None
            if config:
                if mode == "manual" and config.stop_loss:
                    stop_price = round(avg_price * (1 - config.stop_loss / 100.0))
                elif mode in ("atr", "turtle") and atr and config.stop_loss:
                    stop_price = round(avg_price - atr * config.stop_loss)

            # ── 트레일링 스탑가 계산 ──────────────────────────────────
            trailing_stop_price = None
            if config and defaults:
                peak = config.trailing_stop_peak_price
                if mode == "manual":
                    use_ts = defaults.manual_use_trailing_stop
                    ts_value = defaults.manual_trailing_stop_percent
                    if use_ts and peak:
                        trailing_stop_price = round(max(avg_price, peak * (1 - ts_value / 100.0)))
                else:  # atr/turtle
                    use_ts = defaults.turtle_use_trailing_stop
                    ts_value = defaults.turtle_trailing_stop_percent
                    if use_ts and peak and atr:
                        trailing_stop_price = round(max(avg_price, peak - atr * ts_value))

            # ── 분할 매수 진입가·비중 (차수별, 실제 가격 환산) ──────────
            # 1차 체결가를 기준가로 사용 (trading_engine과 동일 로직)
            initial_entry = next(
                (e for e in entries if getattr(e, "entry_type", None) == "INITIAL"),
                entries[0] if entries else None,
            )
            base_price = float(initial_entry.filled_price) if initial_entry else config.entry_point if config else None

            entry_slots = []  # [{price, weight, raw_trigger, is_done}]

            if config:
                # 1차
                positions_list = config.positions or []
                total_weight = sum(positions_list[:total_possible_entries]) or 1
                entry_slots.append({
                    "label": "1차",
                    "price": round(config.entry_point) if config.entry_point else None,
                    "weight": round(positions_list[0] / total_weight * 100, 1) if positions_list else None,
                    "is_done": actual_entries >= 1,
                })

                # 2차~
                for idx, raw in enumerate(config.pyramiding_entries or []):
                    price = None
                    try:
                        if raw and str(raw).strip():
                            raw_str = str(raw).strip()
                            if mode == "manual":
                                pct = float(raw_str.lstrip("+")) / 100.0
                                if base_price:
                                    price = round(base_price * (1 + pct))
                            else:  # atr/turtle
                                multiplier = float(raw_str)
                                if base_price and atr:
                                    price = round(base_price + atr * multiplier)
                    except (TypeError, ValueError):
                        pass

                    slot_idx = idx + 1  # 0-based positions index for this pyramiding slot
                    w = positions_list[slot_idx] / total_weight * 100 if len(positions_list) > slot_idx else None
                    entry_slots.append({
                        "label": f"{idx + 2}차",
                        "price": price,
                        "weight": round(w, 1) if w is not None else None,
                        "is_done": actual_entries >= idx + 2,
                    })

            # ── 분할 매도 설정 ────────────────────────────────────────────
            staged_exit_info = None
            if defaults and defaults.staged_exit_type != "none":
                exit_type = defaults.staged_exit_type
                completed_stages = list(config.staged_exit_completed_stages or []) if config else []

                stages = []
                if exit_type == "ma":
                    for i, (period, sell_pct) in enumerate([
                        (defaults.ma_stage1_period, defaults.ma_stage1_sell_pct),
                        (defaults.ma_stage2_period, defaults.ma_stage2_sell_pct),
                        (defaults.ma_stage3_period, defaults.ma_stage3_sell_pct),
                    ], start=1):
                        stages.append({
                            "stage": i,
                            "trigger": f"MA{period} 이탈",
                            "sell_pct": sell_pct,
                            "is_done": i in completed_stages,
                        })
                elif exit_type == "dead_cross":
                    for i, (short, long, sell_pct) in enumerate([
                        (defaults.dc_stage1_short, defaults.dc_stage1_long, defaults.dc_stage1_sell_pct),
                        (defaults.dc_stage2_short, defaults.dc_stage2_long, defaults.dc_stage2_sell_pct),
                        (defaults.dc_stage3_short, defaults.dc_stage3_long, defaults.dc_stage3_sell_pct),
                    ], start=1):
                        stages.append({
                            "stage": i,
                            "trigger": f"MA{short}/MA{long} 데드크로스",
                            "sell_pct": sell_pct,
                            "is_done": i in completed_stages,
                        })
                elif exit_type == "new_low":
                    for i, (days, sell_pct) in enumerate([
                        (defaults.nl_stage1_days, defaults.nl_stage1_sell_pct),
                        (defaults.nl_stage2_days, defaults.nl_stage2_sell_pct),
                        (defaults.nl_stage3_days, defaults.nl_stage3_sell_pct),
                    ], start=1):
                        stages.append({
                            "stage": i,
                            "trigger": f"{days}일 신저가",
                            "sell_pct": sell_pct,
                            "is_done": i in completed_stages,
                        })

                exit_type_label = {
                    "ma": "이동평균선",
                    "dead_cross": "데드크로스",
                    "new_low": "N일 신저가",
                }.get(exit_type, exit_type)

                staged_exit_info = {
                    "type": exit_type,
                    "type_label": exit_type_label,
                    "stages": stages,
                }

            data[code] = {
                "stock_name": config.stock_name if config else (stock_name_kis or code),
                "trading_mode": mode,
                "actual_entries": actual_entries,
                "total_possible_entries": total_possible_entries,
                "position_sum": round(position_sum, 1),
                "total_quantity": total_qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "profit_loss_amount": profit_loss_amount,
                "profit_loss_rate": profit_loss_rate,
                "holding_amount": round(avg_price * total_qty),
                "stop_price": stop_price,
                "trailing_stop_price": trailing_stop_price,
                "atr": atr,
                "entry_slots": entry_slots,
                "staged_exit_info": staged_exit_info,
            }

        return JsonResponse({"success": True, "data": data})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@trading_status_router.get("/account-snapshots")
def get_account_snapshots(request, days: int = 90):
    """일별 계좌 잔고 스냅샷 조회 (최근 N일)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)

        from datetime import timedelta
        cutoff = date_type.today() - timedelta(days=days)
        snapshots = DailyAccountSnapshot.objects.filter(
            user=user, date__gte=cutoff
        ).order_by("date")

        data = [
            {
                "date": s.date.isoformat(),
                "total_money": s.total_money,
                "stock_money": s.stock_money,
                "remain_money": s.remain_money,
                "stock_revenue": s.stock_revenue,
                "confirmed_capital": s.confirmed_capital,
            }
            for s in snapshots
        ]
        return JsonResponse({"success": True, "data": data})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@trading_status_router.get("/daily-realized-pnl")
def get_daily_realized_pnl(request, days: int = 90):
    """일자별 확정 손익 집계 (매도 체결 기준)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)

        from datetime import timedelta, timezone, datetime
        from django.db.models import Sum, DateField
        from django.db.models.functions import Cast

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            TradeEntry.objects
            .filter(
                user=user,
                trade_type="SELL",
                status="FILLED",
                profit_loss__isnull=False,
                filled_at__gte=cutoff,
            )
            .annotate(trade_date=Cast("filled_at", output_field=DateField()))
            .values("trade_date")
            .annotate(daily_pnl=Sum("profit_loss"))
            .order_by("trade_date")
        )

        data = [
            {
                "date": row["trade_date"].isoformat(),
                "daily_pnl": int(row["daily_pnl"]),
            }
            for row in rows
        ]
        return JsonResponse({"success": True, "data": data})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@trading_status_router.post("/account-snapshots/save-today")
def save_today_snapshot(request):
    """오늘 계좌 잔고 스냅샷 저장 (스케줄러 또는 수동 호출)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)

        from dolpha.kis.trade import GetBalance
        balance = GetBalance()

        snapshot, created = DailyAccountSnapshot.objects.update_or_create(
            user=user,
            date=date_type.today(),
            defaults={
                "total_money": int(balance.get("TotalMoney", 0)),
                "stock_money": int(balance.get("StockMoney", 0)),
                "remain_money": int(balance.get("RemainMoney", 0)),
                "stock_revenue": int(balance.get("StockRevenue", 0)),
                "confirmed_capital": int(balance.get("ConfirmedCapital", 0)),
            },
        )
        return JsonResponse({
            "success": True,
            "created": created,
            "date": snapshot.date.isoformat(),
            "balance": balance,
        })

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
