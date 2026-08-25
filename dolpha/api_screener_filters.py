"""
종목 화면 재무 필터 기본값 API

MTT · 52주 신고가 등 종목 목록 화면 상단의 재무 필터에 자동으로 채워질
기본값을 계정 단위로 저장/조회한다. 값이 null 이면 해당 조건은 걸지 않는다.
"""
from typing import Optional

from ninja import Router, Schema
from django.http import JsonResponse

from myweb.models import TradingDefaults
from .api_mypage_ninja import get_authenticated_user, ResponseSchema

screener_filters_router = Router()

# 재무 필터 값의 허용 범위(%). 증가율은 역성장을 감안해 음수까지 허용한다.
MIN_PERCENT = -1000.0
MAX_PERCENT = 10000.0


class ScreenerFiltersSchema(Schema):
    """재무 필터 기본값 (모두 '이 값 이상'으로 동작, null 이면 미적용)"""

    min_revenue_growth: Optional[float] = None
    min_op_profit_growth: Optional[float] = None
    min_op_margin: Optional[float] = None


class ScreenerFiltersResponseSchema(Schema):
    success: bool
    data: ScreenerFiltersSchema


def _clean_percent(value) -> Optional[float]:
    """입력값을 퍼센트 값으로 정리한다. 비었거나 숫자가 아니면 None(미적용)."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(min(max(number, MIN_PERCENT), MAX_PERCENT), 2)


def _to_payload(defaults: TradingDefaults) -> dict:
    return {
        "min_revenue_growth": defaults.screener_min_revenue_growth,
        "min_op_profit_growth": defaults.screener_min_op_profit_growth,
        "min_op_margin": defaults.screener_min_op_margin,
    }


@screener_filters_router.get("/screener-filters", response=ScreenerFiltersResponseSchema)
def get_screener_filters(request):
    """로그인 사용자의 재무 필터 기본값 조회"""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    defaults, _ = TradingDefaults.objects.get_or_create(user=user)
    return {"success": True, "data": _to_payload(defaults)}


@screener_filters_router.post("/screener-filters", response=ResponseSchema)
def save_screener_filters(request, data: ScreenerFiltersSchema):
    """재무 필터 기본값 저장. 빈 값은 '조건 미적용'으로 저장한다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    try:
        defaults, _ = TradingDefaults.objects.get_or_create(user=user)
        defaults.screener_min_revenue_growth = _clean_percent(data.min_revenue_growth)
        defaults.screener_min_op_profit_growth = _clean_percent(data.min_op_profit_growth)
        defaults.screener_min_op_margin = _clean_percent(data.min_op_margin)
        defaults.save(
            update_fields=[
                "screener_min_revenue_growth",
                "screener_min_op_profit_growth",
                "screener_min_op_margin",
                "updated_at",
            ]
        )
        return {"success": True, "message": "재무 필터 기본값이 저장되었습니다."}
    except Exception as e:
        return {"success": False, "error": str(e)}
