"""
거래 상태 조회 API 엔드포인트 (Django Ninja)

TradeEntry DB + KIS 실계좌 데이터를 합산하여 stock_code 키 딕셔너리로 반환합니다.
avg_price는 KIS 실계좌 기준값을 우선 사용합니다.
"""

from ninja import Router
from django.http import JsonResponse

from .api_mypage_ninja import get_authenticated_user
from myweb.models import TradeEntry, TradingConfig

trading_status_router = Router()


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
                    "avg_price": round(float(s["StockAvgPrice"])),  # 정수 반올림
                }
        except Exception:
            pass  # KIS 조회 실패 시 DB 계산값으로 fallback

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

        data = {}
        for code, entries in entries_by_code.items():
            total_qty = sum(e.filled_quantity for e in entries)
            if total_qty <= 0:
                continue

            # KIS 실계좌 avg_price 우선, 없으면 DB 가중평균
            if code in kis_holdings:
                avg_price = kis_holdings[code]["avg_price"]
            else:
                total_amount = sum(float(e.filled_price) * e.filled_quantity for e in entries)
                avg_price = round(total_amount / total_qty)

            actual_entries = len(entries)
            config = configs.get(code)
            pyramiding_count = config.pyramiding_count if config else 0
            total_possible_entries = pyramiding_count + 1

            position_sum = 0.0
            if config and config.positions:
                used = config.positions[:actual_entries]
                total_ratio = sum(config.positions[:total_possible_entries]) or 1
                position_sum = sum(used) / total_ratio * 100.0

            data[code] = {
                "actual_entries": actual_entries,
                "total_possible_entries": total_possible_entries,
                "position_sum": round(position_sum, 1),
                "total_quantity": total_qty,
                "avg_price": avg_price,
                "holding_amount": round(avg_price * total_qty),
            }

        return JsonResponse({"success": True, "data": data})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
