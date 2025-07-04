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

        # Build optimized query with select_related for Company data
        queryset = (StockAnalysis.objects
                   .filter(is_minervini_trend=True, date=query_date)
                   .select_related('code')
                   .order_by('-rsRank'))
        
        # Check if any records exist
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No stocks found matching Minervini Trend Template")

        # 모든 종목의 재무 데이터를 한 번에 가져와서 메모리에 캐싱 (N+1 쿼리 문제 해결)
        company_codes = [analysis.code for analysis in queryset]
        
        # 매출액과 영업이익 데이터를 벌크로 조회
        finance_data = (StockFinancialStatement.objects
                       .filter(code__in=company_codes, 
                              account_name__in=["매출액", "영업이익"])
                       .order_by('code', 'account_name', '-year', '-quarter')
                       .values('code', 'account_name', 'amount'))
        
        # 재무 데이터를 딕셔너리로 그룹화하여 빠른 조회 가능하게 함
        finance_dict = {}
        for item in finance_data:
            code_id = item['code']
            account = item['account_name']
            amount = item['amount']
            
            if code_id not in finance_dict:
                finance_dict[code_id] = {'매출액': [], '영업이익': []}
            
            if account in finance_dict[code_id]:
                finance_dict[code_id][account].append(amount)

        # 결과 데이터 구성
        results = []
        for analysis in queryset:
            # 재무 데이터 가져오기 (이미 메모리에 캐싱됨)
            code = analysis.code
            매출 = finance_dict.get(code, {}).get('매출액', [])
            영업이익 = finance_dict.get(code, {}).get('영업이익', [])
            
            # 성장률 계산
            매출증가율 = growth_rate(매출[0], 매출[1]) if len(매출) > 1 else 0.0
            영업이익증가율 = growth_rate(영업이익[0], 영업이익[1]) if len(영업이익) > 1 else 0.0

            combined_data = {
                # Company fields (select_related로 이미 로드됨)
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
                # 재무 데이터 (캐싱된 데이터에서 조회)
                '매출증가율': 매출증가율,
                '영업이익증가율': 영업이익증가율,
                '전전기매출': 매출[2] if len(매출) > 2 else 0,
                '전기매출': 매출[1] if len(매출) > 1 else 0,
                '당기매출': 매출[0] if 매출 else 0,
                '전전기영업이익': 영업이익[2] if len(영업이익) > 2 else 0,
                '전기영업이익': 영업이익[1] if len(영업이익) > 1 else 0,
                '당기영업이익': 영업이익[0] if 영업이익 else 0,
            }
            results.append(combined_data)
        
        # 응답 형식에 따른 처리
        if format.lower() == "excel":
            # pandas DataFrame으로 변환하여 Excel 생성 최적화
            df = pd.DataFrame(results)
            filename = f"mtt_stocks_{date or 'latest'}.xlsx"
            
            # BytesIO 버퍼 생성 및 Excel 파일 작성
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl', options={'remove_timezone': True}) as writer:
                df.to_excel(writer, index=False, sheet_name='MTT_Stocks')
                
                # 워크시트 스타일링 (선택사항)
                worksheet = writer.sheets['MTT_Stocks']
                # 헤더 스타일 적용
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
            
            output.seek(0)
            
            # Excel 파일을 HttpResponse로 반환
            response = HttpResponse(
                content=output.getvalue(),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        
        else:  # format.lower() == "json"
            # JSON 응답 반환
            return 200, SuccessResponseStockAnalysis(
                status="OK",
                data=results
            )

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
       - code (str): 종목코드 (예: '005930'). 필수 파라미터.\n
       - limit (int): 조회할 데이터 개수. 기본값: 21.\n
    ㅇ Returns\n
       - SuccessResponseStockOHLCV: 성공 시 OHLCV 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 파라미터가 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 입력 파라미터 검증
        if not code or not code.strip():
            return 400, ErrorResponse(status="error", message="종목코드가 필요합니다.")
        
        if limit <= 0 or limit > 1000:  # 최대 1000개까지 제한
            return 400, ErrorResponse(status="error", message="limit은 1-1000 사이의 값이어야 합니다.")

        # Company 객체 조회 및 캐싱 (한 번만 조회)
        try:
            company = Company.objects.get(code=code.strip())
        except Company.DoesNotExist:
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'를 찾을 수 없습니다.")

        # 최적화된 쿼리: select_related 제거하고 values()로 필요한 필드만 조회
        queryset = (StockOHLCV.objects
                   .filter(code=company)
                   .order_by('-date')
                   .values('date', 'open', 'high', 'low', 'close', 'volume', 'change')
                   [:limit])

        # QuerySet을 리스트로 변환하여 데이터 존재 여부 확인
        results_list = list(queryset)
        
        if not results_list:
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'에 대한 OHLCV 데이터를 찾을 수 없습니다.")

        # 결과 데이터 구성 (이미 values()로 필요한 형태로 조회됨)
        results = []
        for record in results_list:
            results.append({
                'code': code,  # 이미 알고 있는 값 사용
                'date': str(record['date']),
                'open': record['open'],
                'high': record['high'],
                'low': record['low'],
                'close': record['close'],
                'volume': record['volume'],
                'change': record['change'],
            })

        return 200, SuccessResponseStockOhlcvSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"서버 오류가 발생했습니다: {str(e)}")

# 주식 분석 데이터를 조회합니다.
@query_router.get("/find_stock_analysis", response={200: SuccessResponseStockAnalysisSchema, 404: ErrorResponse, 500: ErrorResponse})    
def find_stock_analysis(request, code: str = "005930", limit: int = 21):
    """
    주식 분석 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 종목코드 (예: '005930'). 필수 파라미터.\n
       - limit (int): 조회할 데이터 개수. 기본값: 21.\n
    ㅇ Returns\n
       - SuccessResponseStockAnalysis: 성공 시 주식 분석 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 파라미터가 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 입력 파라미터 검증
        if not code or not code.strip():
            return 400, ErrorResponse(status="error", message="종목코드가 필요합니다.")
        
        if limit <= 0 or limit > 1000:  # 최대 1000개까지 제한
            return 400, ErrorResponse(status="error", message="limit은 1-1000 사이의 값이어야 합니다.")

        # Company 객체 조회 (존재 여부 확인)
        try:
            company = Company.objects.get(code=code.strip())
        except Company.DoesNotExist:
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'를 찾을 수 없습니다.")

        # 최적화된 쿼리: values()로 필요한 필드만 조회하여 메모리 사용량 최소화
        queryset = (StockAnalysis.objects
                   .filter(code=company)
                   .order_by('-date')
                   .values(
                       'date', 'ma50', 'ma150', 'ma200',
                       'rsScore', 'rsScore1m', 'rsScore3m', 'rsScore6m', 'rsScore12m',
                       'rsRank', 'rsRank1m', 'rsRank3m', 'rsRank6m', 'rsRank12m',
                       'max_52w', 'min_52w', 'max_52w_date', 'min_52w_date',
                       'atr', 'is_minervini_trend'
                   )
                   [:limit])

        # QuerySet을 리스트로 변환하여 데이터 존재 여부 확인
        results_list = list(queryset)
        
        if not results_list:
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'에 대한 분석 데이터를 찾을 수 없습니다.")

        # 재무 데이터 조회 (매출액, 영업이익만 필요한 경우에만)
        # 최신 3개 분기 데이터만 조회하여 성장률 계산
        finance_data = (StockFinancialStatement.objects
                       .filter(code=company, account_name__in=["매출액", "영업이익"])
                       .order_by('account_name', '-year', '-quarter')
                       .values('account_name', 'amount')[:6])  # 각 계정당 최대 3개씩
        
        # 재무 데이터를 딕셔너리로 그룹화
        finance_dict = {'매출액': [], '영업이익': []}
        for item in finance_data:
            account = item['account_name']
            amount = item['amount']
            if account in finance_dict and len(finance_dict[account]) < 3:
                finance_dict[account].append(amount)

        # 성장률 계산
        매출 = finance_dict.get('매출액', [])
        영업이익 = finance_dict.get('영업이익', [])
        매출증가율 = growth_rate(매출[0], 매출[1]) if len(매출) > 1 else 0.0
        영업이익증가율 = growth_rate(영업이익[0], 영업이익[1]) if len(영업이익) > 1 else 0.0

        # 결과 데이터 구성 (Company 정보는 한 번만 사용)
        company_info = {
            'code': company.code,
            'name': company.name,
            'market': company.market,
            'sector': company.sector,
            'industry': company.industry,
        }

        results = []
        for record in results_list:
            result_data = {
                # Company fields (미리 조회한 정보 재사용)
                **company_info,
                # StockAnalysis fields (이미 values()로 조회됨)
                'date': str(record['date']),
                'ma50': record['ma50'],
                'ma150': record['ma150'],
                'ma200': record['ma200'],
                'rsScore': record['rsScore'],
                'rsScore1m': record['rsScore1m'],
                'rsScore3m': record['rsScore3m'],
                'rsScore6m': record['rsScore6m'],
                'rsScore12m': record['rsScore12m'],
                'rsRank': record['rsRank'],
                'rsRank1m': record['rsRank1m'],
                'rsRank3m': record['rsRank3m'],
                'rsRank6m': record['rsRank6m'],
                'rsRank12m': record['rsRank12m'],
                'max_52w': record['max_52w'],
                'min_52w': record['min_52w'],
                'max_52w_date': str(record['max_52w_date']) if record['max_52w_date'] else None,
                'min_52w_date': str(record['min_52w_date']) if record['min_52w_date'] else None,
                'atr': record['atr'],
                'is_minervini_trend': record['is_minervini_trend'],
                # 재무 데이터 (한 번만 계산된 값 재사용)
                '매출증가율': 매출증가율,
                '영업이익증가율': 영업이익증가율,
            }
            results.append(result_data)

        return 200, SuccessResponseStockAnalysisSchema(
            status="OK",
            data=results
        )
        
    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"서버 오류가 발생했습니다: {str(e)}")

# 주식 재무제표 데이터를 조회합니다.
@query_router.get("/find_stock_financial", response={200: SuccessResponseStockFinancialSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_financial(request, code: str = "005930", limit: int = 50, account_name: str = None, statement_type: str = None):
    """
    주식 재무제표 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 종목코드 (예: '005930'). 필수 파라미터.\n
       - limit (int): 조회할 데이터 개수. 기본값: 50.\n
       - account_name (str, optional): 특정 계정명으로 필터링 (예: '매출액', '영업이익').\n
       - statement_type (str, optional): 재무제표 유형으로 필터링 (예: '손익계산서', '재무상태표').\n
    ㅇ Returns\n
       - SuccessResponseStockFinancial: 성공 시 재무제표 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n
    ㅇ Raises\n
       - ValueError: 잘못된 파라미터가 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 입력 파라미터 검증
        if not code or not code.strip():
            return 400, ErrorResponse(status="error", message="종목코드가 필요합니다.")
        
        if limit <= 0 or limit > 5000:  # 재무데이터는 더 많을 수 있으므로 5000개까지 허용
            return 400, ErrorResponse(status="error", message="limit은 1-5000 사이의 값이어야 합니다.")

        # Company 객체 존재 여부 확인 (가장 빠른 존재 체크)
        if not Company.objects.filter(code=code.strip()).exists():
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'를 찾을 수 없습니다.")
        
        # Company ID만 필요하므로 values_list로 더 빠르게 조회
        companyCode = Company.objects.filter(code=code.strip()).values_list('code', flat=True).first()

        # 최적화된 쿼리 구성
        queryset = StockFinancialStatement.objects.filter(code=companyCode)
        
        # 선택적 필터링 (인덱스 활용 최적화)
        if statement_type:
            queryset = queryset.filter(statement_type=statement_type.strip())
        
        if account_name:
            queryset = queryset.filter(account_name__icontains=account_name.strip())
        
        # values()로 필요한 필드만 조회하여 메모리 사용량 최소화
        queryset = (queryset
                   .order_by('-year', '-quarter', 'statement_type', 'account_name')
                   .values('year', 'quarter', 'statement_type', 'account_name', 'amount')
                   [:limit])

        # QuerySet을 리스트로 변환하여 데이터 존재 여부 확인
        results_list = list(queryset)
        
        if not results_list:
            error_msg = f"종목코드 '{code}'에 대한 재무제표 데이터를 찾을 수 없습니다."
            if statement_type:
                error_msg += f" (재무제표 유형: {statement_type})"
            if account_name:
                error_msg += f" (계정명: {account_name})"
            return 404, ErrorResponse(status="error", message=error_msg)

        # 결과 데이터 구성 (이미 values()로 필요한 형태로 조회됨)
        results = []
        for record in results_list:
            results.append({
                'code': code,  # 이미 알고 있는 값 재사용
                'year': record['year'],
                'quarter': record['quarter'],
                'statement_type': record['statement_type'],
                'account_name': record['account_name'],
                'amount': record['amount'],
            })

        return 200, SuccessResponseStockFinancialSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"서버 오류가 발생했습니다: {str(e)}")
    
# 주식 종목과 관련된 인덱스 데이터를 조회합니다.
@query_router.get("/find_stock_index", response={200: SuccessResponseStockIndexSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_index(request, code: str = "005930", limit: int = 21):
    """
    주식 종목과 관련된 인덱스의 OHLCV 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 종목 코드 (예: '005930').\n
       - limit (int): 조회할 데이터 개수. 기본값: 21.\n
    ㅇ Returns\n
       - SuccessResponseStockIndex: 성공 시 인덱스 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n 
    ㅇ Raises\n
       - ValueError: 잘못된 파라미터가 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 입력 파라미터 검증
        if not code or not code.strip():
            return 400, ErrorResponse(status="error", message="종목코드가 필요합니다.")
        
        if limit <= 0 or limit > 1000:  # 최대 1000개까지 제한
            return 400, ErrorResponse(status="error", message="limit은 1-1000 사이의 값이어야 합니다.")

        # Company 객체 존재 여부 확인 (가장 빠른 존재 체크)
        if not Company.objects.filter(code=code.strip()).exists():
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'를 찾을 수 없습니다.")

        # 최적화된 쿼리: Company와 관련된 인덱스 데이터를 values()로 조회
        # ManyToMany 관계를 통해 인덱스 정보 조회
        indices_queryset = (StockIndex.objects
                           .filter(companies__code=code.strip())
                           .values('code', 'name', 'market')
                           [:limit])

        # QuerySet을 리스트로 변환하여 데이터 존재 여부 확인
        results_list = list(indices_queryset)
        
        if not results_list:
            return 404, ErrorResponse(status="error", message=f"종목코드 '{code}'에 대한 인덱스 데이터를 찾을 수 없습니다.")

        # 결과 데이터 구성 (이미 values()로 필요한 형태로 조회됨)
        results = []
        for record in results_list:
            results.append({
                'code': record['code'],
                'name': record['name'],
                'market': record['market'],
            })

        return 200, SuccessResponseStockIndexSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"서버 오류가 발생했습니다: {str(e)}")

# index OHLCV 데이터를 조회합니다.
@query_router.get("/find_index_ohlcv", response={200: SuccessResponseIndexOhlcvSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_index_ohlcv(request, code: str = "1011", limit: int = 21):
    """
    인덱스 OHLCV 데이터를 조회합니다.\n
    ㅇ Args\n
       - request: Ninja API 요청 객체.\n
       - code (str): 인덱스 코드 (예: '1011', 'KOSPI').\n
       - limit (int): 조회할 데이터의 개수. 기본값: 21.\n
    ㅇ Returns\n
       - SuccessResponseIndexOHLCV: 성공 시 인덱스 OHLCV 데이터.\n
       - ErrorResponse: 에러 발생 시 에러 메시지.\n 
    ㅇ Raises\n
       - ValueError: 잘못된 파라미터가 입력된 경우.\n
       - Exception: 기타 예상치 못한 오류 발생 시.\n
    """
    try:
        # 입력 파라미터 검증
        if not code or not str(code).strip():
            return 400, ErrorResponse(status="error", message="인덱스 코드가 필요합니다.")
        
        if limit <= 0 or limit > 1000:  # 최대 1000개까지 제한
            return 400, ErrorResponse(status="error", message="limit은 1-1000 사이의 값이어야 합니다.")

        # 코드 정규화 (문자열로 변환 후 공백 제거)
        normalized_code = str(code).strip()

        # StockIndex 존재 여부 확인 (가장 빠른 존재 체크)
        if not StockIndex.objects.filter(code=normalized_code).exists():
            return 404, ErrorResponse(status="error", message=f"인덱스 코드 '{code}'를 찾을 수 없습니다.")

        # StockIndex 객체 조회 (한 번만 조회하여 재사용)
        try:
            index_obj = StockIndex.objects.get(code=normalized_code)
        except StockIndex.DoesNotExist:
            return 404, ErrorResponse(status="error", message=f"인덱스 코드 '{code}'를 찾을 수 없습니다.")

        # 최적화된 쿼리: values()로 필요한 필드만 조회하여 메모리 사용량 최소화
        queryset = (IndexOHLCV.objects
                   .filter(code=index_obj)
                   .order_by('-date')
                   .values('date', 'open', 'high', 'low', 'close', 'volume', 'change')
                   [:limit])

        # QuerySet을 리스트로 변환하여 데이터 존재 여부 확인
        results_list = list(queryset)
        
        if not results_list:
            return 404, ErrorResponse(status="error", message=f"인덱스 코드 '{code}'에 대한 OHLCV 데이터를 찾을 수 없습니다.")

        # 결과 데이터 구성 (이미 values()로 필요한 형태로 조회됨)
        results = []
        for record in results_list:
            results.append({
                'code': normalized_code,  # 이미 알고 있는 값 재사용
                'date': str(record['date']),
                'open': record['open'],
                'high': record['high'],
                'low': record['low'],
                'close': record['close'],
                'volume': record['volume'],
                'change': record['change'],
            })

        return 200, SuccessResponseIndexOhlcvSchema(
            status="OK",
            data=results
        )

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"서버 오류가 발생했습니다: {str(e)}")
