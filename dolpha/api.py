from ninja import NinjaAPI, Router
from django.db import transaction
from django.http import HttpResponse
from django.db.models import Max

from . import stockCommon as Common
from myweb.models import *  # Import the StockOHLCV model
from .schemas import *

from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font  # Add this import
from io import BytesIO
import json
import pprint

import OpenDartReader

import traceback


api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}


@api.post("/get_stock_description", response={200: StockDescriptionResponse, 400: Dict, 500: Dict})
def get_all_stock_description(request):
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
                "result": "ERROR",
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
            "result": "OK",
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
            "result": "ERROR",
            "message": f"Failed to process stock data: {str(e)}"
        }
        

    
@api.post("/get_stock_data", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
def get_stock_data(request, code: str=None, limit: int=1):
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
        # print(df.head())

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
                "message": "Stock data saved successfully.",
                "count_saved_stocks": len(companies),
                "count_saved": limit,
            }

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
            return -1  # 데이터 부족

        # 각 구간에 대한 점수 계산
        scores = []
        step = period_days // 4  # 기간을 4등분 (예: 1년이면 63일)
        
        for i in range(4):
            current_idx = target_idx - (i * step)
            previous_idx = target_idx - ((i + 1) * step)
            
            if previous_idx < 0:
                return -1  # 데이터 부족
            
            current_close = data[current_idx].close
            previous_close = data[previous_idx].close
            score = current_close / previous_close
            scores.append(score)
        
        # RS 점수 계산: (score_1 * 2) + score_2 + score_3 + score_4
        total_score = (scores[0] * 2) + scores[1] + scores[2] + scores[3]
        return total_score

    except Exception as e:
        print(f"{data[0].code}의 RS 계산 오류: {e}")
        return -1

@api.post("/update_stock_analysis", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
def update_stock_analysis(request, offset: int=0, limit: int=0):
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
            
            # 각 기간별 RS 점수 계산
            rs_scores = {}
            for period_name, period_days in periods.items():
                rs_score = calculate_rs_score(ohlcv_data, target_date, period_days)
                rs_scores[period_name] = rs_score
            
            # 가중평균 RS 점수 계산
            weighted_score = -1
            if all(rs_scores[p] != -1 for p in periods):
                weighted_score = (rs_scores['1month'] * 4 + rs_scores['3month'] * 3 + rs_scores['6month'] * 2 + rs_scores['12month'] * 1) / 10
            
            rs_data_all.append({
                'date': target_date,
                'code': company.code,
                'name': company.name,
                'market': company.market,
                'rsScore1m': rs_scores['1month'],
                'rsScore3m': rs_scores['3month'],
                'rsScore6m': rs_scores['6month'],
                'rsScore12m': rs_scores['12month'],
                'rsScore': weighted_score
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
                rsScore=weighted_score,
                rsScore1m=rs_scores['1month'],
                rsScore3m=rs_scores['3month'],
                rsScore6m=rs_scores['6month'],
                rsScore12m=rs_scores['12month'],
                rsRank=0.0,
                rsRank1m=0.0,
                rsRank3m=0.0,
                rsRank6m=0.0,
                rsRank12m=0.0,
                max_52w=high_low['max_52w'],
                min_52w=high_low['min_52w'],
                max_52w_date=high_low['max_52w_date'],
                min_52w_date=high_low['min_52w_date'],
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
        "message": "주식 분석 데이터가 성공적으로 저장되었습니다.",
        "count_saved": total_saved,
        "dates_processed": f"{date_list[0]['date']}, Last date: {date_list[len(date_list)-1]['date']}"
    }

@api.get("/find_stock_inMTT", response={200: SuccessResponseStockAnalysis, 404: ErrorResponse, 500: ErrorResponse})
def find_stock_inMTT(request, date: str = None, format: str = "json"):
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

        # Build query
        queryset = StockAnalysis.objects.filter(is_minervini_trend=True).order_by('-rsRank')
        
        # If date is provided, filter by specific date
        # If not, get the latest date
        if query_date:
            queryset = queryset.filter(date=query_date)
        else:
            latest_date = StockAnalysis.objects.latest('date').date
            queryset = queryset.filter(date=latest_date)

        # Check if any records exist
        if not queryset.exists():
            return 404, ErrorResponse(status="error", message="No stocks found matching Minervini Trend Template")

        # Combine with Company data using select_related
        results = []
        for analysis in queryset.select_related('code'):
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
                'is_minervini_trend': analysis.is_minervini_trend
                # StockFinancialStatement fields 
            }
            results.append(combined_data)
        # Handle Excel output
        if format.lower() == "excel":
            wb = Workbook()
            ws = wb.active
            ws.title = "Stock Analysis"

            # Define headers
            headers = [
                'Code', 'Name', 'Market', 'Sector', 'Industry', 'Date',
                'MA50', 'MA150', 'MA200', 'RS Score', 'RS Score 1M', 'RS Score 3M',
                'RS Score 6M', 'RS Score 12M', 'RS Rank', 'RS Rank 1M', 'RS Rank 3M',
                'RS Rank 6M', 'RS Rank 12M', '52W Max', '52W Min', '52W Max Date',
                '52W Min Date', 'Minervini Trend'
            ]
            
            # Write headers
            for col_num, header in enumerate(headers, 1):
                col_letter = get_column_letter(col_num)
                ws[f"{col_letter}1"] = header
                ws[f"{col_letter}1"].font = Font(bold=True)

            # Write data
            for row_num, data in enumerate(results, 2):
                ws[f"A{row_num}"] = data['code']
                ws[f"B{row_num}"] = data['name']
                ws[f"C{row_num}"] = data['market']
                ws[f"D{row_num}"] = data['sector']
                ws[f"E{row_num}"] = data['industry']
                ws[f"F{row_num}"] = data['date']
                ws[f"G{row_num}"] = data['ma50']
                ws[f"H{row_num}"] = data['ma150']
                ws[f"I{row_num}"] = data['ma200']
                ws[f"J{row_num}"] = data['rsScore']
                ws[f"K{row_num}"] = data['rsScore1m']
                ws[f"L{row_num}"] = data['rsScore3m']
                ws[f"M{row_num}"] = data['rsScore6m']
                ws[f"N{row_num}"] = data['rsScore12m']
                ws[f"O{row_num}"] = data['rsRank']
                ws[f"P{row_num}"] = data['rsRank1m']
                ws[f"Q{row_num}"] = data['rsRank3m']
                ws[f"R{row_num}"] = data['rsRank6m']
                ws[f"S{row_num}"] = data['rsRank12m']
                ws[f"T{row_num}"] = data['max_52w']
                ws[f"U{row_num}"] = data['min_52w']
                ws[f"V{row_num}"] = data['max_52w_date']
                ws[f"W{row_num}"] = data['min_52w_date']
                ws[f"X{row_num}"] = data['is_minervini_trend']

            # Auto-adjust column widths
            for col in ws.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                ws.column_dimensions[column].width = adjusted_width

            # Save to BytesIO
            output = BytesIO()
            wb.save(output)
            output.seek(0)

            # Return Excel file
            filename = f"stock_analysis_{query_date or latest_date}.xlsx"
            response = HttpResponse(
                content=output.read(),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        
        return 200, SuccessResponseStockAnalysis(
            status="success",
            data=results
        )

    except Exception as e:
        return 500, ErrorResponse(status="error", message=str(e))

# @api.get("/get_stock_dartData", response={200: SuccessResponseStockDart, 404: ErrorResponse, 500: ErrorResponse})
# def get_stock_dartData(request, code: str = None):
#     """
#     OpenDART에서 재무제표 데이터를 가져와 데이터베이스에 저장합니다.
    
#     Args:
#         code (str): 종목코드 (예: '005930')
        
#     Returns:
#         SuccessResponse: 성공 시 저장된 데이터
#         ErrorResponse: 에러 발생 시 에러 메시지
#     """
#     try:
#         dart = OpenDartReader("b6533c6c78ba430e7c63ef16db7bb893ae440d43")
        
#         # 보고서 코드 리스트
#         reprt_list = ['11013', '11012', '11014', '11011']  # 1분기, 반기, 3분기, 사업보고서

#         years = [2024, 2025]
#         filtered_data = []
        
#         # code = "064350"
#         code = "005930"
        
#         for year in years:
#             for reprt in reprt_list:
#                 try:
#                     # 재무제표 데이터 조회
#                     df = dart.finstate(code, year, reprt_code=reprt)
                    
#                     if df is None or df.empty:
#                         print(f"No data found for {code}, year {year}, report {reprt}")
#                         continue
                    
#                     # Add 'year' and 'quarter' columns
#                     df['year'] = df['bsns_year']
#                     def get_quarter(reprt_code):
#                         mapping = {
#                             '11011': '4Q',
#                             '11012': '2Q',
#                             '11013': '1Q',
#                             '11014': '3Q'
#                         }
#                         return mapping.get(reprt_code, 'Unknown')
#                     df['quarter'] = df['reprt_code'].apply(get_quarter)
        
#                     # Create pivot table
#                     df_pivot = pd.pivot_table(
#                         df,
#                         values='thstrm_amount',
#                         # index=['year', 'quarter'],
#                         columns=['year', 'quarter', 'sj_nm', 'account_nm'],
#                         aggfunc='first',
#                         fill_value=0
#                     )
                    
#                     # JSON으로 변환
#                     result = json.loads(df_pivot.to_json(orient='records'))
#                     filtered_data.extend(result)
                    
#                 except Exception as e:
#                     print(f"Error fetching data for report {reprt}: {str(e)}")
#                     continue  # 개별 보고서 에러는 무시하고 다음 보고서로 진행
        
#         if not filtered_data:
#             return 404, {"error": f"No data available for {code} in {year}"}
        
#         # 성공 응답
#         return {
#             "status": "success",
#             "data": filtered_data
#         }
    
#     except Exception as e:
#         traceback.print_exc()
#         return 500, {"error": f"Failed to fetch DART data: {str(e)}"}
    
@api.post("/get_stock_dartData", response={200: SuccessResponse, 404: ErrorResponse, 500: ErrorResponse})
def get_stock_dartData(request, code: str = None):
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
            "message": "success",
            "count_saved": 0
        }
    
    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(status="error", message=f"DART 데이터 조회 실패: {str(e)}")
    