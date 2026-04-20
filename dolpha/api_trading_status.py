"""
거래 상태 조회 API 엔드포인트 (Django Ninja)

autobot 통합 후: TradeEntry 모델에서 직접 상태를 조회합니다.
(이전: autobot FastAPI 서버에 HTTP 요청 → 현재: Django DB 직접 조회)
"""

from ninja import Router
from django.http import JsonResponse

from .api_mypage_ninja import get_authenticated_user
from myweb.models import TradeEntry, TradingConfig

trading_status_router = Router()


@trading_status_router.get("/trading-status")
def get_trading_status(request):
    """
    현재 거래 상태 및 포지션 정보를 조회합니다.
    TradeEntry DB에서 직접 SUBMITTED/FILLED 상태의 체결 내역을 반환합니다.
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)

        # 활성 포지션 (매수 체결, 아직 전량청산 안 된 것)
        active_entries = TradeEntry.objects.filter(
            user=user,
            trade_type="BUY",
            status="FILLED",
        ).select_related("trading_config").order_by("-filled_at")

        active_configs = TradingConfig.objects.filter(user=user, is_active=True)

        data = {
            "active_positions": active_entries.count(),
            "total_configs": active_configs.count(),
            "positions": [
                {
                    "id": e.id,
                    "stock_code": e.stock_code,
                    "stock_name": e.stock_name,
                    "entry_type": e.entry_type,
                    "filled_quantity": e.filled_quantity,
                    "filled_price": float(e.filled_price),
                    "filled_amount": float(e.filled_amount),
                    "stop_price": float(e.stop_price) if e.stop_price else None,
                    "atr_value": e.atr_value,
                    "filled_at": e.filled_at.isoformat() if e.filled_at else None,
                }
                for e in active_entries
            ],
        }

        return JsonResponse({"success": True, "data": data})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
