from ninja import NinjaAPI, Router
from .api_data import data_router
from .api_query import query_router

api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}

# 데이터 수집/저장/계산 관련 API 라우터 추가
api.add_router("/", data_router)

# 데이터 조회 관련 API 라우터 추가  
api.add_router("/", query_router)