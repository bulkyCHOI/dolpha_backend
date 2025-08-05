"""
종목 검색 관련 API 엔드포인트 (Django Ninja)
- 전체 종목 검색 (Company 테이블)
- 즐겨찾기 상태 포함 검색 결과
"""

from ninja import Router, Schema
from django.http import JsonResponse
from django.db.models import Q
from typing import Optional, List
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from myweb.models import Company, FavoriteStock, StockAnalysis, StockOHLCV

search_router = Router()

# JWT 인증을 위한 헬퍼 함수
def get_authenticated_user(request):
    """
    JWT 토큰을 사용하여 사용자 인증
    """
    try:
        # Authorization 헤더 확인
        auth_header = request.headers.get('Authorization')
        
        jwt_auth = JWTAuthentication()
        auth_result = jwt_auth.authenticate(request)
        
        if auth_result is None:
            return None
            
        user, token = auth_result
        
        if user and user.is_authenticated:
            return user
        else:
            return None
            
    except (InvalidToken, TokenError) as e:
        return None
    except Exception as e:
        return None

# 종목 검색 관련 스키마
class StockSearchResponseSchema(Schema):
    code: str
    name: str
    market: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    rsRank: Optional[float] = None
    is_favorite: bool = False

class StockSearchResultSchema(Schema):
    success: bool
    stocks: List[StockSearchResponseSchema]
    total: int
    message: Optional[str] = None

@search_router.get("/stocks", response=StockSearchResultSchema)
def search_stocks(request, q: str = "", limit: int = 20):
    """
    전체 종목 검색 (Company 테이블에서)
    - 종목코드 또는 종목명으로 검색
    - 즐겨찾기 상태 포함하여 반환
    """
    try:
        # 검색어가 없으면 빈 결과 반환
        if not q or not q.strip():
            return {
                'success': True,
                'stocks': [],
                'total': 0,
                'message': '검색어를 입력해주세요.'
            }
        
        search_query = q.strip()
        
        # Company 테이블에서 검색 (종목코드 또는 종목명)
        companies = Company.objects.filter(
            Q(code__icontains=search_query) | Q(name__icontains=search_query)
        ).order_by('code')[:limit]
        
        # 사용자 인증 확인 (즐겨찾기 상태 조회용)
        user = get_authenticated_user(request)
        user_favorites = set()
        
        if user:
            # 사용자의 즐겨찾기 목록 조회
            favorites = FavoriteStock.objects.filter(user=user).values_list('stock_code', flat=True)
            user_favorites = set(favorites)
        
        # 검색 결과 구성
        result_stocks = []
        for company in companies:
            is_favorite = company.code in user_favorites
            
            # 최신 분석 데이터 조회
            latest_analysis = StockAnalysis.objects.filter(
                code=company
            ).order_by('-date').first()
            
            # 최신 OHLCV 데이터 조회 (현재가)
            latest_ohlcv = StockOHLCV.objects.filter(
                code=company
            ).order_by('-date').first()
            
            current_price = latest_ohlcv.close if latest_ohlcv else None
            change_percent = latest_ohlcv.change if latest_ohlcv else None
            rsRank = latest_analysis.rsRank if latest_analysis else None
            
            result_stocks.append({
                'code': company.code,
                'name': company.name,
                'market': company.market,
                'industry': company.industry,
                'current_price': current_price,
                'change_percent': change_percent,
                'rsRank': rsRank,
                'is_favorite': is_favorite
            })
        
        return {
            'success': True,
            'stocks': result_stocks,
            'total': len(result_stocks),
            'message': f'{len(result_stocks)}개의 종목을 찾았습니다.' if result_stocks else '검색 결과가 없습니다.'
        }
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'stocks': [],
            'total': 0,
            'error': str(e)
        }, status=500)