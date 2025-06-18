from ninja import NinjaAPI

from . import stockCommon as Common
from myweb.models import StockOHLCV  # Import the StockOHLCV model


api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}


@api.get("/stock/{code}")
def get_stock_data(request, code: str):
    """
    주식 코드에 해당하는 OHLCV 데이터를 반환합니다.
    """
    # 데이터 가져오기
    df = Common.GetOhlcv("KR", code, limit=1, adj_ok="1")
    
    if df is None or len(df) == 0:
        return {"error": "No data found for the given stock code."}
    
    # 데이터 프레임을 JSON으로 변환
    data = df.to_dict(orient='records')
    
    # 데이터베이스에 저장
    # for row in data:
    #     # 중복 데이터 확인 및 처리
    #     obj, created = StockOHLCV.objects.update_or_create(
    #         ticker=code,
    #         date=row.get("date"),
    #         defaults={
    #             "name": row.get("name"),
    #             "market": row.get("market"),
    #             "open": row.get("Open"),
    #             "high": row.get("High"),
    #             "low": row.get("Low"),
    #             "close": row.get("Close"),
    #             "volume": row.get("Volume"),
    #         }
    #     )
    
    return {"stock_code": code, "data": data}

@api.get("/stockData")
def get_all_stocks_ohlcv(request):
    """
    모든 주식의 OHLCV 데이터를 반환합니다.
    """
    df_kospi = Common.GetStockList(area="KOSPI")
    df_kospi.head()
    
    # df_kosdaq = Common.GetStockList(area="KOSDAQ")
    # df_kosdaq.head()
    
    


    # 데이터를 JSON으로 변환
    data = df_kospi.to_dict(orient='records')
    
    return {"stocks": data}