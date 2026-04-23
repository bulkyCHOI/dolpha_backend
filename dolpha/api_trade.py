"""
수동 매매 API

GET  /api/trade/holdings        보유 주식 목록 조회
POST /api/trade/buy             시장가 매수
POST /api/trade/sell            시장가 매도 (보유 수량의 N%)
"""

from typing import Optional, List

from ninja import Router, Schema
from django.http import JsonResponse
from django.utils import timezone

from .api_mypage_ninja import get_authenticated_user
from dolpha.kis.trade import (
    GetMyStockList,
    GetCurrentPrice,
    MakeBuyMarketOrder,
    MakeSellMarketOrder,
)
from myweb.models import TradeEntry

trade_router = Router()


# ── 스키마 ─────────────────────────────────────────────────────────────────


class SellOrderIn(Schema):
    stock_code: str
    percentage: float   # 매도 비율 1~100 (%)


class BuyOrderIn(Schema):
    stock_code: str
    stock_name: str = ""
    quantity: Optional[int] = None    # 수량 직접 지정
    amount: Optional[int] = None      # 원화 금액 (quantity 없을 때 현재가로 환산)


class OrderOut(Schema):
    success: bool
    stock_code: Optional[str] = None
    stock_name: Optional[str] = None
    quantity: Optional[int] = None
    order_no: Optional[str] = None
    order_time: Optional[str] = None
    error: Optional[str] = None


class HoldingItem(Schema):
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: float
    current_price: int
    eval_amount: float
    revenue_rate: float
    revenue_amount: float


class HoldingsOut(Schema):
    success: bool
    holdings: List[HoldingItem] = []
    error: Optional[str] = None


# ── 엔드포인트 ─────────────────────────────────────────────────────────────


@trade_router.get("/holdings", response=HoldingsOut)
def get_holdings(request):
    """모의/실계좌 보유 주식 목록을 KIS API로 조회합니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    try:
        stock_list = GetMyStockList()
        holdings = [
            {
                "stock_code":    s["StockCode"],
                "stock_name":    s["StockName"],
                "quantity":      int(s["StockAmt"]),
                "avg_price":     float(s["StockAvgPrice"]),
                "current_price": int(s["StockNowPrice"]),
                "eval_amount":   float(s["StockNowMoney"]),
                "revenue_rate":  float(s["StockRevenueRate"]),
                "revenue_amount": float(s["StockRevenueMoney"]),
            }
            for s in stock_list
        ]
        return {"success": True, "holdings": holdings}
    except Exception as e:
        return {"success": False, "holdings": [], "error": str(e)}


@trade_router.post("/sell", response=OrderOut)
def sell_stock(request, data: SellOrderIn):
    """
    보유 주식을 시장가로 매도합니다.

    - percentage: 보유 수량 대비 매도 비율 (1~100)
    - 50 → 보유 수량의 50% (소수점 버림)
    - 100 → 전량 매도
    """
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    if not (1 <= data.percentage <= 100):
        return {"success": False, "error": "percentage는 1~100 사이여야 합니다."}

    try:
        stock_list = GetMyStockList()
        holding = next(
            (s for s in stock_list if s["StockCode"] == data.stock_code), None
        )

        if not holding:
            return {
                "success": False,
                "error": f"보유 종목 없음: {data.stock_code}",
            }

        total_qty = int(holding["StockAmt"])
        sell_qty = int(total_qty * data.percentage / 100)

        if sell_qty <= 0:
            return {
                "success": False,
                "error": f"매도 수량 0주 (보유 {total_qty}주, {data.percentage}%)",
            }

        result = MakeSellMarketOrder(data.stock_code, sell_qty)
        if not result:
            return {"success": False, "error": "KIS API 주문 실패"}

        entry_type = "EXIT_FULL" if data.percentage == 100 else "EXIT_PARTIAL"
        TradeEntry.objects.create(
            user=user,
            stock_code=data.stock_code,
            stock_name=holding["StockName"],
            trade_type="SELL",
            entry_type=entry_type,
            order_no=result["OrderNum2"],
            order_quantity=sell_qty,
            order_price=0,
            status="SUBMITTED",
            ordered_at=timezone.now(),
            note=f"수동매도 {data.percentage}%",
        )

        return {
            "success": True,
            "stock_code": data.stock_code,
            "stock_name": holding["StockName"],
            "quantity": sell_qty,
            "order_no": result["OrderNum2"],
            "order_time": result["OrderTime"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@trade_router.post("/buy", response=OrderOut)
def buy_stock(request, data: BuyOrderIn):
    """
    시장가 매수 주문을 접수합니다.

    - quantity: 매수 수량 직접 지정
    - amount: 원화 금액 (현재가로 나눠 수량 환산) — quantity 없을 때 사용
    - quantity와 amount 중 하나는 반드시 입력해야 합니다.
    """
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    if data.quantity is None and data.amount is None:
        return {
            "success": False,
            "error": "quantity 또는 amount 중 하나를 입력해야 합니다.",
        }

    try:
        if data.quantity is not None:
            buy_qty = data.quantity
        else:
            current_price = GetCurrentPrice(data.stock_code)
            buy_qty = data.amount // current_price

        if buy_qty <= 0:
            return {"success": False, "error": "매수 수량 0주"}

        result = MakeBuyMarketOrder(data.stock_code, buy_qty)
        if not result:
            return {"success": False, "error": "KIS API 주문 실패"}

        TradeEntry.objects.create(
            user=user,
            stock_code=data.stock_code,
            stock_name=data.stock_name,
            trade_type="BUY",
            entry_type="INITIAL",
            order_no=result["OrderNum2"],
            order_quantity=buy_qty,
            order_price=0,
            status="SUBMITTED",
            ordered_at=timezone.now(),
            note="수동매수",
        )

        return {
            "success": True,
            "stock_code": data.stock_code,
            "stock_name": data.stock_name,
            "quantity": buy_qty,
            "order_no": result["OrderNum2"],
            "order_time": result["OrderTime"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
