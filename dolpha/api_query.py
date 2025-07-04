from ninja import NinjaAPI, Router
from django.db import transaction
from django.http import HttpResponse

from . import stockCommon as Common
from myweb.models import *  # Import the StockOHLCV model
from .schemas import *

from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from tqdm import tqdm

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font  # Add this import
from io import BytesIO

import OpenDartReader

import traceback

# 데이터 조회 관련 API 라우터
query_router = Router()

def growth_rate(current, previous):
    if abs(previous) == 0:
        return 0.0
    return ((current - previous) / abs(previous)) * 100

# 미너비니 트렌드 템플릿에 해당하는 종목을 조회합니다.
@query_router.get("/find_stock_inMTT", response={200: SuccessResponseStockAnalysis, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_inMTT(request, date: str = None, format: str = "json"):
    """
    주식 분석 데이터를 조회하여 미너비니 트렌드 템플릿에 해당하는 종목을 반환합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - date (str, optional): 조회할 날짜 (YYYY-MM-DD 형식). 기본값: None (최신 날짜 조회).\n
       - format (str, optional): 응답 형식 ("json" 또는 "excel"). 기본값: "json".\n
    ㅇ Returns\n
       - SuccessResponseStockAnalysis: 성공 시 미너비니 트렌드 템플릿에 해당하는 종목 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # Validate format parameter
        if format.lower() not in ["json", "excel"]:
            return 400, ErrorResponse(status="error", message="Invalid format. Use 'json' or 'excel'")

        # Convert date string to date object if provided
        query_date = None
        if date:
            try:
                query_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                return 400, ErrorResponse(status="error", message="Invalid date format. Use YYYY-MM-DD")
        else:
            # If no date is provided, use the latest date from StockAnalysis
            query_date = StockAnalysis.objects.latest('date').date

        # Build query
        queryset = StockAnalysis.objects.filter(is_minervini_trend=True).order_by('-rsRank')
        
        queryset = queryset.filter(date=query_date)
        
        # Check if any records exist
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No stocks found matching Minervini Trend Template")

        # Combine with Company data using select_related
        results = []
        for analysis in queryset.select_related('code'):
            finance = StockFinancialStatement.objects.filter(code=analysis.code).order_by('-year', '-quarter')
            매출 = finance.filter(account_name="매출액").values_list('amount', flat=True).distinct()
            영업이익 = finance.filter(account_name="영업이익").values_list('amount', flat=True).distinct()
            매출증가율 = growth_rate(매출[0], 매출[1]) if len(매출) > 1 else 0.0
            영업이익증가율 = growth_rate(영업이익[0], 영업이익[1]) if len(영업이익) > 1 else 0.0


            combined_data = {
                # Company fields
                'code': analysis.code.code,
                'name': analysis.code.name,
                'market': analysis.code.market,
                'sector': analysis.code.sector,
                'industry': analysis.code.industry,
                # StockAnalysis fields
                'date': str(analysis.date),
                'ma50': analysis.ma50,
                'ma150': analysis.ma150,
                'ma200': analysis.ma200,
                'rsScore': analysis.rsScore,
                'rsScore1m': analysis.rsScore1m,
                'rsScore3m': analysis.rsScore3m,
                'rsScore6m': analysis.rsScore6m,
                'rsScore12m': analysis.rsScore12m,
                'rsRank': analysis.rsRank,
                'rsRank1m': analysis.rsRank1m,
                'rsRank3m': analysis.rsRank3m,
                'rsRank6m': analysis.rsRank6m,
                'rsRank12m': analysis.rsRank12m,
                'max_52w': analysis.max_52w,
                'min_52w': analysis.min_52w,
                'max_52w_date': str(analysis.max_52w_date) if analysis.max_52w_date else None,
                'min_52w_date': str(analysis.min_52w_date) if analysis.min_52w_date else None,
                'atr': analysis.atr,
                'is_minervini_trend': analysis.is_minervini_trend,
                '매출증가율': 매출증가율,
                '영업이익증가율': 영업이익증가율,
                '전전기매출' : 매출[2] if len(매출) > 2 else 0,
                '전기매출': 매출[1] if len(매출) > 1 else 0,
                '당기매출': 매출[0] if 매출 else 0,
                '전전기영업이익': 영업이익[2] if len(영업이익) > 2 else 0,
                '전기영업이익': 영업이익[1] if len(영업이익) > 1 else 0,
                '당기영업이익': 영업이익[0] if 영업이익 else 0,
                # StockFinancialStatement fields 
            }
            results.append(combined_data)
        
        if format.lower() == "excel":
            # BytesIO 버퍼 생성
            output = BytesIO()
            df = pd.DataFrame(results)
            filename = f"mtt_stocks_{date or 'latest'}.xlsx"
            
            # DataFrame을 BytesIO 버퍼에 쓰기
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            
            # 버퍼 포인터를 처음으로 되돌리기
            output.seek(0)
            
            # Excel 파일을 HttpResponse로 반환
            response = HttpResponse(
                content=output.getvalue(),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        
        elif format.lower() == "json":
            # Return JSON response
            return 200, SuccessResponseStockAnalysis(
                status="OK",
                data=results
            )
        else:
            return 400, ErrorResponse(status="error", message="Invalid format. Use 'json' or 'excel'")

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))
    
# 주식 OHLCV 데이터를 조회합니다.
@query_router.get("/find_stock_ohlcv", response={200: SuccessResponseStockOhlcvSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_ohlcv(request, code: str = "005930", limit: int = 21):
    """
    주식 OHLCV 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).\n
    ㅇ Returns\n
       - SuccessResponseStockOHLCV: 성공 시 OHLCV 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 종목코드가 주어지지 않은 경우 404 에러 반환
        if code is not None:
            queryset = StockOHLCV.objects.filter(code=code).order_by('-date')[:limit]
        else:
            return 404, ErrorResponse(status="error", message="종목코드가 필요합니다.")

        # 데이터가 없으면 404 에러 반환
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No OHLCV data found for the given code")

        # 결과를 리스트로 변환
        results = []
        for record in queryset:
            results.append({
                'code': record.code.code,
                'date': str(record.date),
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume,
                'change': record.change,
            })

        return 200, SuccessResponseStockOhlcvSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))

# 주식 분석 데이터를 조회합니다.
@query_router.get("/find_stock_analysis", response={200: SuccessResponseStockAnalysisSchema, 404: ErrorResponse, 500: ErrorResponse})    
def find_stock_analysis(request, code: str = "005930", limit: int = 21):
    """
    주식 분석 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).\n
    ㅇ Returns\n
       - SuccessResponseStockAnalysis: 성공 시 주식 분석 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 종목코드가 주어지지 않은 경우 404 에러 반환
        if code is not None:
            queryset = StockAnalysis.objects.filter(code__code=code).order_by('-date')[:limit]
        else:
            return 404, ErrorResponse(status="error", message="종목코드가 필요합니다.")

        # 데이터가 없으면 404 에러 반환
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No analysis data found for the given code")

        # 결과를 리스트로 변환
        results = []
        for record in queryset:
            results.append({
                'code': record.code.code,
                'name': record.code.name,
                'market': record.code.market,
                'sector': record.code.sector,
                'industry': record.code.industry,
                'date': str(record.date),
                'ma50': record.ma50,
                'ma150': record.ma150,
                'ma200': record.ma200,
                'rsScore': record.rsScore,
                'rsScore1m': record.rsScore1m,
                'rsScore3m': record.rsScore3m,
                'rsScore6m': record.rsScore6m,
                'rsScore12m': record.rsScore12m,
                'rsRank': record.rsRank,
                'rsRank1m': record.rsRank1m,
                'rsRank3m': record.rsRank3m,
                'rsRank6m': record.rsRank6m,
                'rsRank12m': record.rsRank12m,
                'max_52w': record.max_52w,
                'min_52w': record.min_52w,
                'max_52w_date': str(record.max_52w_date) if record.max_52w_date else None,
                'min_52w_date': str(record.min_52w_date) if record.min_52w_date else None,
                'atr': record.atr,
                'is_minervini_trend': record.is_minervini_trend,
                '매출증가율': record.매출증가율 if hasattr(record, '매출증가율') else None,
                '영업이익증가율': record.영업이익증가율 if hasattr(record, '영업이익증가율') else None,
            })
        return 200, SuccessResponseStockAnalysisSchema(
            status="OK",
            data=results
        )
    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))

# 주식 재무제표 데이터를 조회합니다.
@query_router.get("/find_stock_financial", response={200: SuccessResponseStockFinancialSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_financial(request, code: str = "005930"):
    """
    주식 재무제표 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).\n
    ㅇ Returns\n
       - SuccessResponseStockFinancial: 성공 시 재무제표 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n 
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 종목코드가 주어지지 않은 경우 404 에러 반환
        if code is not None:
            queryset = StockFinancialStatement.objects.filter(code=code).order_by('-year', '-quarter')
        else:
            return 404, ErrorResponse(status="error", message="종목코드가 필요합니다.")

        # 데이터가 없으면 404 에러 반환
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No financial data found for the given code")

        # 결과를 리스트로 변환
        results = []
        for record in queryset:
            results.append({
                'code': record.code.code,
                'year': record.year,
                'quarter': record.quarter,
                'statement_type': record.statement_type,
                'account_name': record.account_name,
                'amount': record.amount,
            })

        return 200, SuccessResponseStockFinancialSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))
    
# 주식 종목과 관련된 인덱스 데이터를 조회합니다.
@query_router.get("/find_stock_index", response={200: SuccessResponseStockIndexSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_index(request, code: str = "005930", limit: int = 21):
    """
    주식 종목과 관련된 인덱스의 OHLCV 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 인덱스 코드 (예: '005930').\n
    ㅇ Returns\n
       - SuccessResponseStockIndexOHLCV: 성공 시 인덱스 OHLCV 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n 
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 종목코드를 통해 인덱스 코드를 찾자
        indexList = []

        if code is not None:
            try:
                company = Company.objects.get(code=code)
                indexList = company.indices.all()  # 인덱스가 있는지 확인
            except Company.DoesNotExist:
                return 404, ErrorResponse(status="error", message=f"No Company found with code: {code}")
        else:
            return 404, ErrorResponse(status="error", message="종목 코드가 필요합니다.")

        print("indexList", indexList)

        return 200, SuccessResponseStockIndexSchema(
            status="OK",
            data=indexList, 
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))

# index OHLCV 데이터를 조회합니다.
@query_router.get("/find_index_ohlcv", response={200: SuccessResponseIndexOhlcvSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_index_ohlcv(request, code: str = 1011, limit: int = 21):
    """
    인덱스 OHLCV 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 인덱스 코드 (예: 'KOSPI').\n
       - limit (int): 조회할 데이터의 개수. 기본값: 21.\n
    ㅇ Returns\n
       - SuccessResponseIndexOHLCV: 성공 시 인덱스 OHLCV 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n 
    ㅇ Raises\n
       - ValueError: 잘못된 날짜 형식이 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 인덱스 코드를 통해 OHLCV 데이터를 가져옴
        queryset = IndexOHLCV.objects.filter(code=code).order_by('-date')[:limit]

        # 데이터가 없으면 404 에러 반환
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No index OHLCV data found for the given code")

        # 결과를 리스트로 변환
        results = []
        for record in queryset:
            results.append({
                'code': record.code.code,
                'date': str(record.date),
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume,
                'change': record.change,
            })

        return 200, SuccessResponseIndexOhlcvSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=str(e))
