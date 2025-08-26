"""
매매복기 API 엔드포인트 (Django Ninja)
"""

from datetime import datetime, date
from typing import List, Optional
from ninja import Router, Schema, Query
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.db.models import Q
from django.core.paginator import Paginator

from myweb.models import TradingSummary
from .api_mypage_ninja import get_authenticated_user
from .autobot_service import autobot_service

# 라우터 생성
trading_reviews_router = Router()


# Schema 정의
class TradingSummaryIn(Schema):
    stock_code: str
    stock_name: str
    first_entry_date: Optional[datetime] = None
    last_exit_date: Optional[datetime] = None
    total_buy_amount: int = 0
    total_sell_amount: int = 0
    total_profit_loss: int = 0
    profit_loss_percent: float = 0.0
    max_drawdown: Optional[float] = None
    holding_days: float = 0.0
    entry_count: int = 0
    exit_count: int = 0
    trading_mode: str
    win_rate: float = 0.0
    avg_holding_days: float = 0.0
    max_profit_percent: Optional[float] = None
    final_status: str


class TradingSummaryOut(Schema):
    id: int
    stock_code: str
    stock_name: str
    first_entry_date: Optional[datetime]
    last_exit_date: Optional[datetime]
    total_buy_amount: int
    total_sell_amount: int
    total_profit_loss: int
    profit_loss_percent: float
    max_drawdown: Optional[float]
    holding_days: float
    entry_count: int
    exit_count: int
    trading_mode: str
    win_rate: float
    avg_holding_days: float
    max_profit_percent: Optional[float]
    final_status: str
    memo: str
    created_at: datetime
    updated_at: datetime


class TradingSummaryUpdate(Schema):
    memo: Optional[str] = None


class TradingSummaryFilter(Schema):
    stock_name: Optional[str] = None
    trading_mode: Optional[str] = None
    final_status: Optional[str] = None
    profit_filter: Optional[str] = None  # 'positive', 'negative', 'all'
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    page: int = 1
    page_size: int = 20


# API 엔드포인트
@trading_reviews_router.post("/trading-summary", response=TradingSummaryOut)
def create_trading_summary(request, payload: TradingSummaryIn):
    """
    매매복기 데이터 생성 (Autobot에서 호출)
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        # 중복 체크 (같은 사용자, 종목코드, 첫 매수일)
        existing = TradingSummary.objects.filter(
            user=user,
            stock_code=payload.stock_code,
            first_entry_date=payload.first_entry_date
        ).first()
        
        if existing:
            # 기존 데이터 업데이트
            for attr, value in payload.dict().items():
                if hasattr(existing, attr):
                    setattr(existing, attr, value)
            existing.save()
            return existing
        else:
            # 새 데이터 생성
            trading_summary = TradingSummary.objects.create(
                user=user,
                **payload.dict()
            )
            return trading_summary
            
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/trading-summary", response=List[TradingSummaryOut])
def list_trading_summaries(request, filters: TradingSummaryFilter = Query(...)):
    """
    매매복기 목록 조회 (필터링, 페이징)
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        # 기본 쿼리셋
        queryset = TradingSummary.objects.filter(user=user)
        
        # 필터링 적용
        if filters.stock_name:
            queryset = queryset.filter(
                Q(stock_name__icontains=filters.stock_name) |
                Q(stock_code__icontains=filters.stock_name)
            )
        
        if filters.trading_mode and filters.trading_mode != 'all':
            queryset = queryset.filter(trading_mode=filters.trading_mode)
        
        if filters.final_status and filters.final_status != 'all':
            queryset = queryset.filter(final_status=filters.final_status)
        
        if filters.profit_filter == 'positive':
            queryset = queryset.filter(total_profit_loss__gt=0)
        elif filters.profit_filter == 'negative':
            queryset = queryset.filter(total_profit_loss__lt=0)
        
        if filters.start_date:
            queryset = queryset.filter(first_entry_date__date__gte=filters.start_date)
        
        if filters.end_date:
            queryset = queryset.filter(first_entry_date__date__lte=filters.end_date)
        
        # 페이지네이션
        paginator = Paginator(queryset, filters.page_size)
        page_obj = paginator.get_page(filters.page)
        
        return list(page_obj)
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/trading-summary/{trading_summary_id}", response=TradingSummaryOut)
def get_trading_summary(request, trading_summary_id: int):
    """
    개별 매매복기 상세 조회
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        trading_summary = get_object_or_404(
            TradingSummary, 
            id=trading_summary_id, 
            user=user
        )
        return trading_summary
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.put("/trading-summary/{trading_summary_id}", response=TradingSummaryOut)
def update_trading_summary(request, trading_summary_id: int, payload: TradingSummaryUpdate):
    """
    매매복기 수정 (메모 등)
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        trading_summary = get_object_or_404(
            TradingSummary, 
            id=trading_summary_id, 
            user=user
        )
        
        for attr, value in payload.dict(exclude_unset=True).items():
            setattr(trading_summary, attr, value)
        
        trading_summary.save()
        return trading_summary
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.delete("/trading-summary/{trading_summary_id}")
def delete_trading_summary(request, trading_summary_id: int):
    """
    매매복기 삭제
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        trading_summary = get_object_or_404(
            TradingSummary, 
            id=trading_summary_id, 
            user=user
        )
        
        trading_summary.delete()
        return {"success": True}
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/trading-summary-stats")
def get_trading_summary_stats(request):
    """
    매매복기 통계 조회
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        queryset = TradingSummary.objects.filter(user=user)
        
        total_count = queryset.count()
        closed_count = queryset.filter(final_status='CLOSED').count()
        holding_count = queryset.filter(final_status='HOLDING').count()
        
        total_profit_loss = sum(ts.total_profit_loss for ts in queryset)
        avg_profit_loss = total_profit_loss / total_count if total_count > 0 else 0
        
        profitable_count = queryset.filter(total_profit_loss__gt=0).count()
        win_rate = (profitable_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "total_count": total_count,
            "closed_count": closed_count,
            "holding_count": holding_count,
            "total_profit_loss": total_profit_loss,
            "avg_profit_loss": avg_profit_loss,
            "win_rate": win_rate,
            "profitable_count": profitable_count
        }
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


# Autobot API 연동 엔드포인트들
@trading_reviews_router.get("/autobot/trading-summary")
def get_autobot_trading_summary(request):
    """
    Autobot에서 매매복기 데이터를 조회합니다.
    """
    try:
        if not autobot_service.is_available():
            return JsonResponse({
                "success": False,
                "error": "AUTOBOT_UNAVAILABLE",
                "message": "Autobot 서버에 연결할 수 없습니다."
            }, status=503)
        
        data = autobot_service.get_trading_summary()
        if data is None:
            return JsonResponse({
                "success": False,
                "error": "DATA_FETCH_ERROR",
                "message": "Autobot에서 데이터를 가져올 수 없습니다."
            }, status=502)
        
        return JsonResponse({
            "success": True,
            "data": data
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/autobot/trading-summary/stats")
def get_autobot_trading_summary_stats(request):
    """
    Autobot에서 매매복기 통계를 조회합니다.
    """
    try:
        if not autobot_service.is_available():
            return JsonResponse({
                "success": False,
                "error": "AUTOBOT_UNAVAILABLE",
                "message": "Autobot 서버에 연결할 수 없습니다."
            }, status=503)
        
        data = autobot_service.get_trading_summary_stats()
        if data is None:
            return JsonResponse({
                "success": False,
                "error": "DATA_FETCH_ERROR",
                "message": "Autobot에서 통계를 가져올 수 없습니다."
            }, status=502)
        
        return JsonResponse({
            "success": True,
            "data": data
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/autobot/trading-summary/{stock_code}")
def get_autobot_trading_summary_by_stock(request, stock_code: str):
    """
    Autobot에서 특정 종목의 매매복기 데이터를 조회합니다.
    """
    try:
        if not autobot_service.is_available():
            return JsonResponse({
                "success": False,
                "error": "AUTOBOT_UNAVAILABLE",
                "message": "Autobot 서버에 연결할 수 없습니다."
            }, status=503)
        
        data = autobot_service.get_trading_summary_by_stock(stock_code)
        if data is None:
            return JsonResponse({
                "success": False,
                "error": "DATA_NOT_FOUND",
                "message": f"종목 '{stock_code}'의 데이터를 찾을 수 없습니다."
            }, status=404)
        
        return JsonResponse({
            "success": True,
            "data": data
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.get("/autobot/status")
def get_autobot_status(request):
    """
    Autobot 서버 상태를 확인합니다.
    """
    try:
        is_available = autobot_service.is_available()
        file_status = autobot_service.check_trading_summary_file()
        
        return JsonResponse({
            "success": True,
            "data": {
                "is_available": is_available,
                "api_url": autobot_service.base_url,
                "file_status": file_status
            }
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)


@trading_reviews_router.post("/autobot/trading-summary")
def autobot_create_trading_summary(request, payload: TradingSummaryIn):
    """
    Autobot에서 매매복기 데이터를 저장합니다. (인증 불필요)
    """
    try:
        # 기본 사용자 ID (Autobot용)
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # 기본 사용자 가져오기 또는 생성
        user, created = User.objects.get_or_create(
            id=1,
            defaults={
                "username": "autobot",
                "email": "autobot@dolpha.com"
            }
        )
        
        # 중복 체크 (같은 사용자, 종목코드, 첫 매수일)
        existing = TradingSummary.objects.filter(
            user=user,
            stock_code=payload.stock_code,
            first_entry_date=payload.first_entry_date
        ).first()
        
        if existing:
            # 기존 데이터 업데이트
            for attr, value in payload.dict().items():
                if hasattr(existing, attr):
                    setattr(existing, attr, value)
            existing.save()
            return JsonResponse({
                "success": True,
                "message": "데이터 업데이트 완료",
                "data": {
                    "id": existing.id,
                    "stock_code": existing.stock_code,
                    "stock_name": existing.stock_name
                }
            })
        else:
            # 새 데이터 생성
            trading_summary = TradingSummary.objects.create(
                user=user,
                **payload.dict()
            )
            return JsonResponse({
                "success": True,
                "message": "데이터 생성 완료",
                "data": {
                    "id": trading_summary.id,
                    "stock_code": trading_summary.stock_code,
                    "stock_name": trading_summary.stock_name
                }
            })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)



@trading_reviews_router.get("/autobot/trading-summary-data")
def get_autobot_trading_summary_data(request):
    """
    Autobot 사용자의 매매복기 데이터를 조회합니다. (인증 불필요)
    """
    try:
        # 기본 사용자 ID (Autobot용)
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # 기본 사용자 가져오기
        user = User.objects.filter(id=1).first()
        if not user:
            return JsonResponse({
                "success": False,
                "error": "USER_NOT_FOUND",
                "message": "Autobot 사용자를 찾을 수 없습니다."
            }, status=404)
        
        # 매매복기 데이터 조회
        queryset = TradingSummary.objects.filter(user=user).order_by("-updated_at")
        
        # 데이터 직렬화
        data = []
        for ts in queryset:
            data.append({
                "id": ts.id,
                "stock_code": ts.stock_code,
                "stock_name": ts.stock_name,
                "first_entry_date": ts.first_entry_date.isoformat() if ts.first_entry_date else None,
                "last_exit_date": ts.last_exit_date.isoformat() if ts.last_exit_date else None,
                "total_buy_amount": ts.total_buy_amount,
                "total_sell_amount": ts.total_sell_amount,
                "total_profit_loss": ts.total_profit_loss,
                "profit_loss_percent": ts.profit_loss_percent,
                "max_drawdown": ts.max_drawdown,
                "holding_days": ts.holding_days,
                "entry_count": ts.entry_count,
                "exit_count": ts.exit_count,
                "trading_mode": ts.trading_mode,
                "win_rate": ts.win_rate,
                "avg_holding_days": ts.avg_holding_days,
                "max_profit_percent": ts.max_profit_percent,
                "final_status": ts.final_status,
                "memo": ts.memo,
                "created_at": ts.created_at.isoformat(),
                "updated_at": ts.updated_at.isoformat()
            })
        
        return JsonResponse({
            "success": True,
            "data": data,
            "total": len(data)
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)

