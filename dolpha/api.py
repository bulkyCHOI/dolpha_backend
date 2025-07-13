from ninja import NinjaAPI, Router
from .api_data import data_router
from .api_query import query_router
from .api_simple_auth import simple_auth_router

api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}

# 데이터 수집/저장/계산 관련 API 라우터 추가
api.add_router("/", data_router)

# 데이터 조회 관련 API 라우터 추가  
api.add_router("/", query_router)

# 사용자 인증 관련 API 라우터 추가 (간단한 버전)
api.add_router("/auth", simple_auth_router)

# Google OAuth 인증 관련 API 라우터 추가
from .api_auth import auth_router
api.add_router("/auth", auth_router)

# 마이페이지 관련 API 라우터 추가
from .api_mypage_ninja import mypage_router
api.add_router("/mypage", mypage_router)