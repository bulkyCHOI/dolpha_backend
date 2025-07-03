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


api = NinjaAPI()

@api.get("/hello")
def hello(request):
    # 왜안되니
    return {"message": "Hello, Django Ninja!"}

# KRX에서 모든 주식의 종목 코드를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
@api.post("/getAndSave_index_list", response={200: IndexListResponse, 400: ErrorResponse, 500: ErrorResponse})
def getAndSave_index_list(request):
    """
    KRX에서 모든 주식의 종목 코드를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
    
    Returns:
        IndexListResponse: 처리 결과, 저장된 종목 코드 리스트
    """
    try:
        # 종목정보 조회
        df_krx = Common.GetSnapDataReader()
        print(df_krx.head(), len(df_krx))

        # 컬럼명 매핑
        column_mapping = {
            'Code': 'code',
            'Name': 'name',
            'Market': 'market',
        }
        df_krx = df_krx.rename(columns=column_mapping)

        # 예상 컬럼 확인
        expected_columns = list(column_mapping.values())
        if not all(col in df_krx.columns for col in expected_columns):
            return 400, {
                "status": "ERROR",
                "message": "Required columns are missing in the KRX data"
            }

        # 데이터 전처리
        df_krx['code'] = df_krx['code'].astype(str)  # 코드 문자열로 변환

        # 벌크 데이터 준비
        stockIndex_to_create = []
        stockIndex_to_update = []
        existing_codes = set(StockIndex.objects.values_list('code', flat=True))

        for index, row in df_krx.iterrows():
            try:
                stockIndex_data = {
                    'code': str(row['code']),
                    'name': str(row['name']),
                    'market': str(row['market']),
                }

                # 생성 또는 업데이트 분류
                if row['code'] not in existing_codes:
                    stockIndex_to_create.append(StockIndex(**stockIndex_data))
                else:
                    stockIndex = StockIndex.objects.get(code=row['code'])
                    stockIndex.name = stockIndex_data['name']
                    stockIndex.market = stockIndex_data['market']
                    stockIndex_to_update.append(StockIndex(**stockIndex_data))
                    
            except Exception as e:
                traceback.print_exc()
                return 500, {
                    "status": "ERROR",
                    "message": f"Failed to process row {index}: {str(e)}"
                }

        # 데이터베이스 트랜잭션
        with transaction.atomic():
            if stockIndex_to_create:
                StockIndex.objects.bulk_create(stockIndex_to_create)
            elif stockIndex_to_update:
                StockIndex.objects.bulk_update(stockIndex_to_update, ['name', 'market'])
        
        krx_codes = df_krx['code'].tolist()  # KRX 종목 코드 리스트 code: 소문자로 변경했으므로 사용가능
        for code in krx_codes:
            df_stocks = Common.GetSnapDataReader_IndexCode(code)    # 1002 코스피 대형주에 속한 종목에 대해 조회가능
            print(df_stocks.head(), len(df_stocks))
            
            stockList = df_stocks['Code'].tolist()  # 종목 코드 리스트
            index = StockIndex.objects.get(code=code)  # 인덱스 객체 가져오기
            index.companies.add(*stockList)  # 인덱스에 종목 추가
            
            print(index.companies.all())  # 인덱스에 추가된 종목 확인
            # print(company.indexes.all())  # 종목에 속한 인덱스 확인
        
        # 응답 구성
        response = {
            "status": "OK",
            "count_total": len(df_krx),
            "count_created": len(stockIndex_to_create),
            "count_updated": len(stockIndex_to_update),
            "count_failed": 0,  # 실패한 레코드는 없음
        }
        return response

    except Exception as e:
        traceback.print_exc()
        return

# 모든 주식의 설명 데이터를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
@api.post("/getAndSave_stock_description", response={200: StockDescriptionResponse, 400: ErrorResponse, 500: ErrorResponse})
def getAndSave_stock_description(request):
    """
    모든 주식의 설명 데이터를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
    
    Returns:
        StockDescriptionResponse: 처리 결과, 저장된 레코드 수, 실패한 레코드 수 및 오류 메시지
    """
    try:
        # 종목정보 조회
        df_krx_desc = Common.GetStockList("KRX-DESC")
        print(df_krx_desc.head())

        # 컬럼명 매핑 (이전 데이터: Code, Name 등)
        column_mapping = {
            'Code': 'code',
            'Name': 'name',
            'Market': 'market',
            'Sector': 'sector',
            'Industry': 'industry',
            # 'ListingDate': 'listing_date',
            # 'SettleMonth': 'settle_month',
            # 'Representative': 'representative',
            # 'HomePage': 'homepage',
            # 'Region': 'region'
        }
        df_krx_desc = df_krx_desc.rename(columns=column_mapping)
        
        # 예상 컬럼 확인
        expected_columns = list(column_mapping.values())
        if not all(col in df_krx_desc.columns for col in expected_columns):
            return 400, {
                "status": "ERROR",
                "message": "Required columns are missing in the KRX-DESC data"
            }

        # 데이터 전처리
        # df_krx_desc['listing_date'] = pd.to_datetime(df_krx_desc['listing_date'], errors='coerce')
        # df_krx_desc = df_krx_desc.dropna(subset=['code'])  # 필수 필드 누락 제거

        # 벌크 데이터 준비
        companies_to_create = []
        companies_to_update = []
        failed_records = []
        existing_codes = set(Company.objects.values_list('code', flat=True))

        for index, row in df_krx_desc.iterrows():
            try:
                # 유효성 검사
                # print(row['listing_date'], type(row['listing_date']))
                # if pd.isna(row['listing_date']):
                #     row['listing_date'] = None
                # else:
                #     row['listing_date'] = row['listing_date'].date()
                # if not isinstance(row['homepage'], str) or not row['homepage'].startswith('http'):
                #     row['homepage'] = 'https://example.com'  # 기본값

                # if row['market'] in ['KONEX']:
                #     continue  # KONEX 시장은 제외
                
                company_data = {
                    'code': str(row['code']),
                    'name': str(row['name']),
                    'market': str(row['market']),
                    'sector': str(row['sector']),
                    'industry': str(row['industry']),
                    # 'listing_date': row['listing_date'].date()
                    # 'settle_month': str(row['settle_month']),
                    # 'representative': str(row['representative']),
                    # 'homepage': str(row['homepage']),
                    # 'region': str(row['region'])
                }

                # 생성 또는 업데이트 분류
                if row['code'] in existing_codes:
                    companies_to_update.append(company_data)
                else:
                    companies_to_create.append(Company(**company_data))

            except Exception as e:
                traceback.print_exc()
                failed_records.append({
                    'index': index,
                    'code': row.get('code', 'N/A'),
                    'error': str(e)
                })

        # 데이터베이스 트랜잭션
        with transaction.atomic():
            # 벌크 생성
            if companies_to_create:
                Company.objects.bulk_create(companies_to_create)
            
            # 벌크 업데이트
            for company_data in companies_to_update:
                Company.objects.update_or_create(
                    code=company_data['code'],
                    defaults={
                        'name': company_data['name'],
                        'market': company_data['market'],
                        'sector': company_data['sector'],
                        'industry': company_data['industry'],
                        # 'listing_date': company_data['listing_date'],
                        # 'settle_month': company_data['settle_month'],
                        # 'representative': company_data['representative'],
                        # 'homepage': company_data['homepage'],
                        # 'region': company_data['region']
                    }
                )

        # 응답 구성
        response = {
            "status": "OK",
            "count_total": len(df_krx_desc),
            "count_created": len(companies_to_create),
            "count_updated": len(companies_to_update),
            "count_failed": len(failed_records),
            "failed_records": failed_records if failed_records else None
        }
        return response

    except Exception as e:
        traceback.print_exc()
        return 500, {
            "status": "ERROR",
            "message": f"Failed to process stock data: {str(e)}"
        }

# Index 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
@api.post("/getAndSave_index_data", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
def getAndSave_index_data(request, code: str=None, limit: int=1):
    """
    Index 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
    코드를 입력하지 않으면, 모든 인덱스의 OHLCV 데이터를 가져옵니다.
    
    Args:
        code (str): 인덱스 코드
        limit (int): 가져올 데이터의 개수 (기본값: 1)
    
    Returns:
        SuccessResponse: 데이터 저장 성공 시 메시지와 저장된 레코드 수
        ErrorResponse: 에러 발생 시 에러 메시지
    """
    indices = []
    
    if code is not None:
        # code에 맞는 인덱스의 OHLCV 데이터를 가져옴
        try:
            index = StockIndex.objects.get(code=code)
            indices = [index]  # 단일 인덱스 객체를 리스트로 감싸서 처리
        except StockIndex.DoesNotExist:
            return 404, {"error": f"No index found with code: {code}"}
    else:
        # code가 주어지지 않은 경우 모든 인덱스의 OHLCV 데이터를 가져옴
        indices = StockIndex.objects.all()
        
    print(f"Total indices: {len(indices)}")
    
    if len(indices) == 0:
        return 404, {"error": "No indices found in the database."}
    
    for stockIndex in tqdm(indices, desc="Processing indices..."):
        # print(index.code, index.name)
        
        df = Common.GetOhlcv("KR", f"KRX-INDEX:{stockIndex.code}", limit=limit, adj_ok="1")
        print(df.head())

        if df is None or len(df) == 0:
            return 400, {"error": "No OHLCV data found for the given index code."}
        
        # 컬럼명 검증
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'change']
        if not all(col in df.columns for col in expected_columns):
            return 400, {"error": "Required OHLCV columns are missing in the data."}

        # 데이터 전처리
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')    # 인덱스를 Timestamp로 변환
            if df.index.isna().any():   # 변환 실패 시 NaT가 있는지 확인
                return 400, {"error": "Invalid date format in OHLCV data index."}
        except Exception as e:
            return 400, {"error": f"Failed to process date column: {str(e)}"}
        # 데이터베이스 저장
        try:
            with transaction.atomic():
                # 벌크 데이터 준비
                index_ohlcv_list = []
            
                # DataFrame에서 NaN 값을 기본값으로 대체
                df = df.fillna({
                    'open': 0.0,
                    'high': 0.0,
                    'low': 0.0,
                    'close': 0.0,
                    'volume': 0,
                    'change': 0.0
                })
                for index, row in df.iterrows():
                    index_ohlcv = IndexOHLCV(
                        code=stockIndex,
                        date=index.date(),  # 인덱스(Timestamp)에서 date 추출
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        change=float(row['change'] if 'change' in row else 0.0)  # 변화율이 없을 경우 기본값 0.0 사용
                    )
                    index_ohlcv_list.append(index_ohlcv)
                
                # 벌크 삽입
                if index_ohlcv_list:
                    IndexOHLCV.objects.bulk_create(index_ohlcv_list, ignore_conflicts=True)
        except Exception as e:
            traceback.print_exc()
            return 500, {"error": f"Failed to save index data: {str(e)}"}
        
    return {
                "status": "OK",
                "message": f"Index data {len(indices)} saved successfully.",
                "count_saved": limit,
            }

# 주식 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
@api.post("/getAndSave_stock_data", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
def getAndSave_stock_data(request, code: str=None, limit: int=1):
    """
    주식 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
    코드를 입력하지 않으면, 모든 회사의 OHLCV 데이터를 가져옵니다.
    
    Args:
        code (str): 주식 코드
        limit (int): 가져올 데이터의 개수 (기본값: 1)
    
    Returns:
        SuccessResponse: 데이터 저장 성공 시 메시지와 저장된 레코드 수
        ErrorResponse: 에러 발생 시 에러 메시지
    """
    companies = []
    
    if code is not None:
        # code에 맞는 회사의 OHLCV 데이터를 가져옴
        try:
            company = Company.objects.get(code=code)
            companies = [company]  # 단일 회사 객체를 리스트로 감싸서 처리
        except Company.DoesNotExist:
            return 404, {"error": f"No company found with code: {code}"}
    else:
        # code가 주어지지 않은 경우 모든 회사의 OHLCV 데이터를 가져옴
        companies = Company.objects.all()
        
    print(f"Total companies: {len(companies)}")
    
    if len(companies) == 0:
        return 404, {"error": "No companies found in the database."}
    
    for company in tqdm(companies, desc="Processing companies..."):
        # print(company.code, company.name)
        
        df = Common.GetOhlcv("KR", company.code, limit=limit, adj_ok="1")
        print(df.head())

        if df is None or len(df) == 0:
            return 400, {"error": "No OHLCV data found for the given stock code."}
        
        # 컬럼명 검증
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'change']
        if not all(col in df.columns for col in expected_columns):
            return 400, {"error": "Required OHLCV columns are missing in the data."}

        # 데이터 전처리
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')    # 인덱스를 Timestamp로 변환
            if df.index.isna().any():   # 변환 실패 시 NaT가 있는지 확인
                return 400, {"error": "Invalid date format in OHLCV data index."}
        except Exception as e:
            return 400, {"error": f"Failed to process date column: {str(e)}"}

        # 데이터베이스 저장
        try:
            with transaction.atomic():
                # 벌크 데이터 준비
                stock_ohlcv_list = []
            
                # DataFrame에서 NaN 값을 기본값으로 대체
                df = df.fillna({
                    'open': 0.0,
                    'high': 0.0,
                    'low': 0.0,
                    'close': 0.0,
                    'volume': 0,
                    'change': 0.0
                })
                for index, row in df.iterrows():
                    stock_ohlcv = StockOHLCV(
                        code=company,
                        date=index.date(),  # 인덱스(Timestamp)에서 date 추출
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        change=float(row['change'] if 'change' in row else 0.0)  # 변화율이 없을 경우 기본값 0.0 사용
                    )
                    stock_ohlcv_list.append(stock_ohlcv)
                
                # 벌크 삽입
                if stock_ohlcv_list:
                    StockOHLCV.objects.bulk_create(stock_ohlcv_list, ignore_conflicts=True)
        except Exception as e:
            traceback.print_exc()
            return 500, {"error": f"Failed to save stock data: {str(e)}"}    
    return {
                "status": "OK",
                "message": f"Stock data {len(companies)} saved successfully.",
                "count_saved": limit,
            }

# 주어진 기간에 대해 이동평균(MA)을 계산합니다. 1개월 전 MA200도 계산.
def calculate_moving_averages(data, target_date, periods=[50, 150, 200], past_ma200_days=21):
    """
    주어진 기간에 대해 이동평균(MA)을 계산합니다. 1개월 전 MA200도 계산.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: MA를 계산할 목표 날짜.
    periods: 계산할 MA 기간 리스트(예: [50, 150, 200]).
    past_ma200_days: 1개월 전 MA200 계산을 위한 기간(기본값: 21일).
    """
    try:
        data = list(data.order_by('date').filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break
        
        if target_idx is None:
            return {f'ma{period}': 0.0 for period in periods} | {'ma200_past': 0.0}
        
        mas = {}
        for period in periods:
            if target_idx + 1 >= period:
                closes = [data[target_idx - i].close for i in range(period)]
                mas[f'ma{period}'] = np.mean(closes)
            else:
                mas[f'ma{period}'] = 0.0
        
        # 1개월 전 MA200 계산
        past_idx = target_idx - past_ma200_days
        if past_idx + 1 >= 200:
            past_closes = [data[past_idx - i].close for i in range(200)]
            mas['ma200_past'] = np.mean(past_closes)
        else:
            mas['ma200_past'] = 0.0
        
        return mas

    except Exception as e:
        print(f"MA 계산 오류: {e}")
        return {f'ma{period}': 0.0 for period in periods} | {'ma200_past': 0.0}
# 52주 신고가와 신저가 및 발생 날짜를 계산합니다.
def calculate_52w_high_low(data, target_date, period_days=252):
    """
    52주 신고가와 신저가 및 발생 날짜를 계산합니다.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: 계산할 목표 날짜.
    period_days: 계산 기간 (기본값: 252 거래일 = 1년).
    """
    try:
        data = list(data.order_by('date').filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break
        
        if target_idx is None or target_idx + 1 < period_days:
            return {'max_52w': 0.0, 'min_52w': 0.0, 'max_52w_date': None, 'min_52w_date': None}
        
        # 지난 252일 데이터 추출
        period_data = data[target_idx - period_days + 1:target_idx + 1]
        
        # high와 low가 0이 아닌 데이터만 필터링
        highs = [(record.high, record.date) for record in period_data if record.high > 0]
        lows = [(record.low, record.date) for record in period_data if record.low > 0]
        
        # 유효한 데이터가 없으면 기본값 반환
        if not highs or not lows:
            return {'max_52w': 0.0, 'min_52w': 0.0, 'max_52w_date': None, 'min_52w_date': None}
        
        max_high, max_date = max(highs, key=lambda x: x[0])
        min_low, min_date = min(lows, key=lambda x: x[0])
        
        return {
            'max_52w': float(max_high),
            'min_52w': float(min_low),
            'max_52w_date': max_date,
            'min_52w_date': min_date
        }

    except Exception as e:
        print(f"52주 신고가/신저가 계산 오류: {e}")
        return {'max_52w': 0.0, 'min_52w': 0.0, 'max_52w_date': None, 'min_52w_date': None}
# 주어진 기간(예: 1년=252일, 1개월=21일)에 대해 RS 점수를 계산합니다.
def calculate_rs_score(data, target_date, period_days):
    """
    주어진 기간(예: 1년=252일, 1개월=21일)에 대해 RS 점수를 계산합니다.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: RS 점수를 계산할 목표 날짜.
    period_days: 기간(거래일 수, 예: 252일).
    """
    try:
        # Queryset을 리스트로 변환하여 인덱싱
        data = list(data.order_by('date'))
        
        # 목표 날짜의 인덱스 찾기
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break
        
        if target_idx is None or target_idx < period_days:
            return -1, [-1, -1, -1, -1]  # 데이터 부족

        # 각 구간에 대한 점수 계산
        scores = []
        step = period_days // 4  # 기간을 4등분 (예: 1년이면 63일)
        
        for i in range(4):
            current_idx = target_idx - (i * step)
            previous_idx = target_idx - ((i + 1) * step)
            
            if previous_idx < 0:
                return -1, [-1, -1, -1, -1]  # 데이터 부족
            
            current_close = data[current_idx].close
            previous_close = data[previous_idx].close
            score = current_close / previous_close
            scores.append(score)
        
        # RS 점수 계산: (score_1 * 2) + score_2 + score_3 + score_4
        total_score = (scores[0] * 2) + scores[1] + scores[2] + scores[3]
        return total_score, scores  # 전체 점수와 각 기간별 점수 반환

    except Exception as e:
        print(f"{data[0].code}의 RS 계산 오류: {e}")
        return -1, [-1, -1, -1, -1]
# ATR(Average True Range)을 계산하는 함수
def calculate_atr(data, target_date, period=20):
    """
    ATR(Average True Range)을 계산하는 함수
    df: OHLC(시가, 고가, 저가, 종가) 데이터가 포함된 DataFrame
    target_date: 기준 날짜 (문자열, 예: '2025-07-02')
    period: ATR 계산 기간 (기본값: 14)
    """
    try:
        data = list(data.order_by('date').filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break
        
        if target_idx is None or target_idx < period:
            return -1
        
        # 현재 날짜의 ATR 계산
        if target_idx + 1 >= period:
            # True Range 계산
            tr_values = []
            for i in range(target_idx, max(-1, target_idx - period), -1):
                current = data[i]
                prev = data[i - 1] if i > 0 else None
                high_low = current.high - current.low
                high_close = abs(current.high - prev.close) if prev else 0.0
                low_close = abs(current.low - prev.close) if prev else 0.0
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
            return np.mean(tr_values)
        else:
            return -1
        
    except Exception as e:
        print(f"ATR 계산 오류: {e}")
        traceback.print_exc()
        return -1


    
    # True Range는 위 세 값 중 최대값
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    # ATR은 TR의 이동평균
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    return df
# 주식 분석 데이터를 계산하여 StockAnalysis 테이블에 저장합니다.
@api.post("/calculate_stock_analysis", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
def calculate_stock_analysis(request, offset: int=0, limit: int=0):
    """
    주식 분석 데이터를 계산하여 StockAnalysis 테이블에 저장합니다.
    최근 거래일부터 지정된 `limit`만큼의 거래일에 대해 모든 회사의 이동평균, 52주 신고가/신저가, RS 점수,
    미너비니 트렌드 조건을 계산합니다. 휴일(예: 주말)은 StockOHLCV 데이터가 없으므로 자동으로 제외됩니다.

    Args:
        request: Ninja API 요청 객체.
        offset (int, optional): 처리할 데이터의 시작 위치. 기본값: 0.
        limit (int, optional): 처리할 거래일 수. 0이면 offset 거래일만 처리. 기본값: 0.
        즉 offset ~ limit 범위의 거래일을 처리합니다.\n
        0, 0: 오늘 거래일만 처리합니다.\n
        0, 50: 오늘부터 50일 전까지의 거래일을 처리합니다.\n
        50, 100: 50일 전부터 150(50+100)일 전까지의 거래일을 처리합니다.\n

    Returns:
        dict: 처리 결과를 포함하는 응답.
            - message (str): 처리 결과 메시지.
            - count_saved (int): 저장된 StockAnalysis 레코드 수.
            - dates_processed (list): 처리된 날짜 목록 (YYYY-MM-DD 형식).
        tuple: 에러 발생 시 (HTTP 상태 코드, 에러 메시지 딕셔너리).

    Raises:
        DatabaseError: 데이터베이스 저장 중 오류 발생 시.
        Exception: 기타 예상치 못한 오류 발생 시.
    """
    
    
    # 모든 회사의 데이터를 가져옴
    companies = Company.objects.all()
        
    print(f"Total companies: {len(companies)}")
    
    if len(companies) == 0:
        return 404, {"error": "No companies found in the database."}
    
    # 기간 정의 (거래일 기준)
    periods = {
        '12month': 252,  # 1년
        '6month': 126,  # 6개월
        '3month': 63,   # 3개월
        '1month': 21    # 1개월
    }
    
    # StockOHLCV의 고유 날짜 목록 가져오기 (최근 순, limit 적용)
    date_list = StockOHLCV.objects.values('date').distinct().order_by('-date')
    if limit > 0:
        date_list = date_list[offset:offset+limit]  # 최근 limit개의 거래일만 선택
    else:
        date_list = date_list[offset:offset+1]  # limit=0이면 최신 날짜만
        
    if not date_list:
        print("StockOHLCV 데이터가 없습니다.")
        return 404, {"error": "No StockOHLCV data found."}

    total_saved = 0
    print(f"Start date: {date_list[0]['date']}, End date: {date_list[len(date_list)-1]['date']}") 
    print(f"Processing {len(date_list)} dates for {len(companies)} companies...")
    # 각 날짜에 대해 처리
    for date_entry in tqdm(date_list, desc=f"Processing..."):
        target_date = date_entry['date']
        
        rs_data_all = []  # 모든 날짜, 회사에 대한 RS 데이터
        analysis_objects = []  # 모든 StockAnalysis 객체
        
        # 회사별로 처리
        for company in tqdm(companies, desc=f"Date: {date_entry['date']}", leave=False):

            # 회사별 OHLCV 데이터 가져오기
            ohlcv_data = StockOHLCV.objects.filter(code=company).order_by('date')
        
            if not ohlcv_data.exists():
                print(f"{company.code}에 대한 OHLCV 데이터 없음")
                continue
            
            # 해당 날짜의 종가 가져오기
            latest_ohlcv = ohlcv_data.filter(date=target_date).first()
            latest_close = latest_ohlcv.close if latest_ohlcv else 0.0
            
            # 이동평균 계산
            mas = calculate_moving_averages(ohlcv_data, target_date)
            
            # 52주 신고가/신저가 및 날짜 계산
            high_low = calculate_52w_high_low(ohlcv_data, target_date)
            
            # # 각 기간별 RS 점수 계산
            # rs_scores = {}
            # for period_name, period_days in periods.items():
            #     rs_score = calculate_rs_score(ohlcv_data, target_date, period_days)
            #     rs_scores[period_name] = rs_score
            
            # # 가중평균 RS 점수 계산
            # weighted_score = -1
            # if all(rs_scores[p] != -1 for p in periods):
            #     weighted_score = (rs_scores['1month'] * 4 + rs_scores['3month'] * 3 + rs_scores['6month'] * 2 + rs_scores['12month'] * 1) / 10
            
            # 위에서는 1개월, 3개월, 6개월, 12개월 4번을 구했지만 12개월 1번만 구하면 된다.
            rs_score, rsScores = calculate_rs_score(ohlcv_data, target_date, periods['12month'])
            
            # ATR(Average True Range) 계산
            atr = calculate_atr(ohlcv_data, target_date, period=20)
            # print(f"ATR for {company.code} on {target_date}: {atr}")

            rs_data_all.append({
                'date': target_date,
                'code': company.code,
                'name': company.name,
                'market': company.market,
                'rsScore': rs_score,
                'rsScore1m': rsScores[0],
                'rsScore3m': rsScores[1],
                'rsScore6m': rsScores[1],
                'rsScore12m': rsScores[3],
            })
            
            # 미너비니 트렌드 템플릿 조건 확인
            is_minervini_trend = (
                latest_close > mas['ma50'] and
                latest_close > mas['ma150'] and
                latest_close > mas['ma200'] and
                mas['ma50'] > mas['ma150'] > mas['ma200'] and
                mas['ma200_past'] > 0 and mas['ma200_past'] < mas['ma200'] and
                latest_close > high_low['min_52w'] * 1.3 and
                latest_close <= high_low['max_52w'] * 0.75
            )  # rsRank는 랭킹 계산 후 확인
            
            # StockAnalysis 객체 준비
            analysis_objects.append(StockAnalysis(
                code=company,
                date=target_date,
                ma50=mas['ma50'],
                ma150=mas['ma150'],
                ma200=mas['ma200'],
                rsScore=rs_score,
                rsScore1m=rsScores[0],
                rsScore3m=rsScores[1],
                rsScore6m=rsScores[2],
                rsScore12m=rsScores[3],
                rsRank=0.0,
                rsRank1m=0.0,
                rsRank3m=0.0,
                rsRank6m=0.0,
                rsRank12m=0.0,
                max_52w=high_low['max_52w'],
                min_52w=high_low['min_52w'],
                max_52w_date=high_low['max_52w_date'],
                min_52w_date=high_low['min_52w_date'],
                atr=atr,
                is_minervini_trend=is_minervini_trend
            ))
    
        # 날짜별로 랭킹 계산
        rs_df = pd.DataFrame(rs_data_all)
        for date in tqdm([entry['date'] for entry in date_list], desc=f"Calculating rankings...", leave=False):
            date_df = rs_df[rs_df['date'] == date]
            for market in date_df['market'].unique():
                market_df = date_df[date_df['market'] == market]
                if market_df.empty:
                    continue
                for period in ['rsScore1m', 'rsScore3m', 'rsScore6m', 'rsScore12m', 'rsScore']:
                    rank_values = market_df[period].rank(ascending=True, na_option='bottom')
                    rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
                    rs_df.loc[market_df.index, f'{period}_Rank'] = rank_values
                    rs_df.loc[market_df.index, f'{period}_RS'] = rs_values
                rank_values = market_df['rsScore'].rank(ascending=True, na_option='bottom')
                rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
                rs_df.loc[market_df.index, f'{'rsScore'}_Rank'] = rank_values
                rs_df.loc[market_df.index, f'{'rsScore'}_RS'] = rs_values

        # StockAnalysis 객체에 랭킹 반영
        for obj in tqdm(analysis_objects, desc="Updating rankings and MTT", leave=False):
            row = rs_df[(rs_df['code'] == obj.code.code) & (rs_df['date'] == obj.date)].iloc[0]
            obj.rsRank = row['rsScore_RS'] if row['rsScore'] != -1 else 0.0
            obj.rsRank1m = row['rsScore1m_RS'] if row['rsScore1m'] != -1 else 0.0
            obj.rsRank3m = row['rsScore3m_RS'] if row['rsScore3m'] != -1 else 0.0
            obj.rsRank6m = row['rsScore6m_RS'] if row['rsScore6m'] != -1 else 0.0
            obj.rsRank12m = row['rsScore12m_RS'] if row['rsScore12m'] != -1 else 0.0
            if obj.is_minervini_trend:
                obj.is_minervini_trend = obj.is_minervini_trend and obj.rsRank >= 70
                    
        # 기존 StockAnalysis 레코드 삭제
        StockAnalysis.objects.filter(date=target_date).delete()
    
        # Bulk create
        try:
            with transaction.atomic():
                StockAnalysis.objects.bulk_create(analysis_objects)
            total_saved += len(analysis_objects)
        except Exception as e:
            traceback.print_exc()
            return 500, {"error": f"주식 분석 데이터 저장 실패: {str(e)}"}
    
    return {
        "status": "OK",
        "message": "주식 분석 데이터가 성공적으로 저장되었습니다.\n"+f"{date_list[0]['date']}, Last date: {date_list[len(date_list)-1]['date']}",
        "count_saved": total_saved,
    }

# OpenDART에서 재무제표 데이터를 가져와 4분기 값을 조정하고 피벗 테이블로 반환합니다.
@api.post("/getAndSave_stock_dartData", response={200: SuccessResponse, 404: ErrorResponse, 500: ErrorResponse})
def getAndSave_stock_dartData(request, code: str = None):
    """
    OpenDART에서 재무제표 데이터를 가져와 4분기 값을 조정하고 피벗 테이블로 반환합니다.
    
    Args:
        code (str): 종목코드 (예: '005930')
        
    Returns:
        SuccessResponseStockDart: 성공 시 피벗 테이블 데이터
        ErrorResponse: 에러 발생 시 에러 메시지
    """
    try:
        companies = []
        
        # OpenDART API 초기화
        dart = OpenDartReader("b6533c6c78ba430e7c63ef16db7bb893ae440d43")
    
        if code is not None:
            # code에 맞는 회사의 OHLCV 데이터를 가져옴
            try:
                company = Company.objects.get(code=code)
                companies = [company]  # 단일 회사 객체를 리스트로 감싸서 처리
            except Company.DoesNotExist:
                return 404, {"error": f"No company found with code: {code}"}
        else:
            # code가 주어지지 않은 경우 모든 회사의 OHLCV 데이터를 가져옴
            companies = Company.objects.all()
            
        print(f"Total companies: {len(companies)}")
        
        if len(companies) == 0:
            return 404, {"error": "No companies found in the database."}

        for company in tqdm(companies, desc="Processing companies..."):
            # 보고서 코드 리스트: 1분기, 2분기, 3분기, 4분기(사업보고서)
            reprt_list = ['11013', '11012', '11014', '11011']
            # 현재 년도와 전년도로 설정
            year = datetime.now().year
            years = [year, year - 1]  # 현재 년도와 전년도
            all_dfs = pd.DataFrame()
            
            code = company.code  # Company 객체에서 종목코드 추출

            for year in years:
                year_data = []
                for reprt in reprt_list:
                    try:
                        # 재무제표 데이터 조회 (연결재무제표)
                        df = dart.finstate(code, year, reprt_code=reprt)
                        
                        if df is None or df.empty:
                            # print(f"{code}, {year}년, 보고서 {reprt}에 데이터가 없습니다.")
                            continue
                        
                        if sum(df['fs_nm'] == '연결재무제표') > 0:
                            df = df.loc[df['fs_nm'] == '연결재무제표']
                        else:
                            df = df.loc[df['fs_nm'] == '재무제표']
                        
                        # thstrm_amount의 쉼표 제거 및 정수 변환
                        if 'thstrm_amount' in df.columns:
                            df['thstrm_amount'] = df['thstrm_amount'].astype(str).str.replace(',', '').replace('-', '0').astype(int)
                        else:
                            # print(f"{code}, {year}년, 보고서 {reprt}에 thstrm_amount 열이 없습니다.")
                            continue

                        # '년도'와 '분기' 필드 추가
                        df['year'] = df['bsns_year'].astype(str)
                        def get_quarter(reprt_code):
                            mapping = {
                                '11011': '4Q',  # 사업보고서
                                '11012': '2Q',
                                '11013': '1Q',
                                '11014': '3Q'
                            }
                            return mapping.get(reprt_code, 'Unknown')
                        df['quarter'] = df['reprt_code'].apply(get_quarter)

                        # print(df[['year', 'quarter', 'fs_nm', 'sj_nm', 'account_nm', 'thstrm_amount']])
                        year_data.append(df)
                    
                    except Exception as e:
                        traceback.print_exc()
                        # print(f"보고서 {reprt} 데이터 조회 중 오류: {str(e)}")
                        continue  # 개별 보고서 오류는 무시하고 다음으로 진행

                if not year_data:
                    continue

                # 연도별 모든 분기 데이터를 하나로 합침
                df_year = pd.concat(year_data, ignore_index=True)

                # 4분기 값 조정: 1Q+2Q+3Q 합계를 4Q에서 뺌
                if '4Q' in df_year['quarter'].values:
                    # 1Q, 2Q, 3Q 데이터 합계 계산
                    q123 = df_year[df_year['quarter'].isin(['1Q', '2Q', '3Q'])][['sj_nm', 'account_nm', 'thstrm_amount']]
                    q123_sum = q123.groupby(['sj_nm', 'account_nm'])['thstrm_amount'].sum().reset_index()
                    q123_sum.rename(columns={'thstrm_amount': 'q123_total'}, inplace=True)
                    # print(q123_sum[['sj_nm', 'account_nm', 'q123_total']])

                    # 4Q 데이터와 합계 병합
                    q4 = df_year[df_year['quarter'] == '4Q'][['sj_nm', 'account_nm', 'thstrm_amount']]
                    q4 = q4.merge(q123_sum, on=['sj_nm', 'account_nm'], how='left')
                    q4['q123_total'] = q4['q123_total'].fillna(0).astype(int)
                    
                    # 재무상태표와 손익계산서 분리 > 1Q+2Q+3Q 합산 데이터가 아님 >> 그대로 유지
                    q4_jm = q4.loc[q4['sj_nm'] == '재무상태표', ['sj_nm', 'account_nm', 'thstrm_amount']]
                    
                    # 손익계산서에서 1Q+2Q+3Q 합계를 뺌
                    q4_si = q4.loc[q4['sj_nm'] == '손익계산서', ['sj_nm', 'account_nm', 'thstrm_amount', 'q123_total']]
                    q4_si['thstrm_amount'] = q4_si['thstrm_amount'] - q4_si['q123_total']
                    q4_si = q4_si.drop(columns=['q123_total'])
                    
                    # 재무상태표와 손익계산서 데이터를 합침
                    q4_merged = pd.concat([q4_jm, q4_si], ignore_index=True)
                    q4_merged['year'] = str(year)  # 연도 설정
                    q4_merged['quarter'] = '4Q'  # 4Q로 분기 설정

                    # 'year', 'quarter', 'sj_nm', 'account_nm', 'thstrm_amount' 필드만 남기기
                    df_year = df_year[['year', 'quarter', 'sj_nm', 'account_nm', 'thstrm_amount']]
                    # 4Q 데이터는 지우고
                    df_year = df_year[df_year['quarter'] != '4Q']  # 4Q 데이터 제거
                    # 계산된 4Q 데이터를 업데이트
                    df_year = pd.concat([df_year, q4_merged], ignore_index=True)
                else:
                    df_year = df_year[['year', 'quarter', 'sj_nm', 'account_nm', 'thstrm_amount']]
                    
                # 데이터 합치기
                all_dfs = pd.concat([all_dfs, df_year], ignore_index=True)
                # print(all_dfs)
                
            # 모든 연도 데이터 합침
            # df = pd.concat(all_dfs, ignore_index=True)
            # print(all_dfs)

            # StockFinancialStatement 모델에 저장
            for _, row in all_dfs.iterrows():
                try:
                    # 이미 존재하는 데이터는 업데이트
                    StockFinancialStatement.objects.update_or_create(
                        code=company,
                        year=row['year'],
                        quarter=row['quarter'],
                        statement_type=row['sj_nm'],
                        account_name=row['account_nm'],
                        amount=row['thstrm_amount']
                    )
                except Exception as e:
                    traceback.print_exc()
                    print(f"데이터 저장 오류: {str(e)}")
                    
        # 성공 응답
        return {
            "status": "OK",
            "message": "success",
            "count_saved": len(all_dfs),
        }
    
    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"DART 데이터 조회 실패: {str(e)}")

def growth_rate(current, previous):
    if abs(previous) == 0:
        return 0.0
    return ((current - previous) / abs(previous)) * 100
# 미너비니 트렌드 템플릿에 해당하는 종목을 조회합니다.
@api.get("/find_stock_inMTT", response={200: SuccessResponseStockAnalysis, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_inMTT(request, date: str = None, format: str = "json"):
    """
    주식 분석 데이터를 조회하여 미너비니 트렌드 템플릿에 해당하는 종목을 반환합니다.
    Args:
        request: Ninja API 요청 객체.
        date (str, optional): 조회할 날짜 (YYYY-MM-DD 형식). 기본값: None (최신 날짜 조회).
        format (str, optional): 응답 형식 ("json" 또는 "excel"). 기본값: "json".
    Returns:
        SuccessResponseStockAnalysis: 성공 시 미너비니 트렌드 템플릿에 해당하는 종목 데이터.
        ErrorResponse: 에러 발생 시 에러 메시지.
    Raises:
        ValueError: 잘못된 날짜 형식이 입력된 경우.
        Exception: 기타 예상치 못한 오류 발생 시.
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
@api.get("/find_stock_ohlcv", response={200: SuccessResponseStockOhlcvSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_ohlcv(request, code: str = "005930", limit: int = 21):
    """
    주식 OHLCV 데이터를 조회합니다.
    Args:
        request: Ninja API 요청 객체.
        code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).
    Returns:
        SuccessResponseStockOHLCV: 성공 시 OHLCV 데이터.
        ErrorResponse: 에러 발생 시 에러 메시지.
    Raises:
        ValueError: 잘못된 날짜 형식이 입력된 경우.
        Exception: 기타 예상치 못한 오류 발생 시.
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
@api.get("/find_stock_analysis", response={200: SuccessResponseStockAnalysisSchema, 404: ErrorResponse, 500: ErrorResponse})    
def find_stock_analysis(request, code: str = "005930", limit: int = 21):
    """
    주식 분석 데이터를 조회합니다.
    Args:
        request: Ninja API 요청 객체.
        code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).
    Returns:
        SuccessResponseStockAnalysis: 성공 시 주식 분석 데이터.
        ErrorResponse: 에러 발생 시 에러 메시지.
    Raises:
        ValueError: 잘못된 날짜 형식이 입력된 경우.
        Exception: 기타 예상치 못한 오류 발생 시.
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
@api.get("/find_stock_financial", response={200: SuccessResponseStockFinancialSchema, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_financial(request, code: str = "005930"):
    """
    주식 재무제표 데이터를 조회합니다.
    Args:
        request: Ninja API 요청 객체.
        code (str, optional): 종목코드 (예: '005930'). 기본값: None (모든 종목 조회).
    Returns:
        SuccessResponseStockFinancial: 성공 시 재무제표 데이터.
        ErrorResponse: 에러 발생 시 에러 메시지.
    Raises:
        ValueError: 잘못된 날짜 형식이 입력된 경우.
        Exception: 기타 예상치 못한 오류 발생 시.
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