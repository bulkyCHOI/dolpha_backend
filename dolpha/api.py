from ninja import NinjaAPI, Router, Schema
from django.db import transaction
from django.http import HttpResponse
from django.db.models import Max

from . import stockCommon as Common
from myweb.models import *  # Import the StockOHLCV model

from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

import traceback


api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}





# # 오늘 날짜의 데이터만 가져오기
# df_kospi = Common.GetStockList("KOSPI")
# print(df_kospi.head())
# print(len(df_kospi))

# df_kospi = Common.GetOhlcv("KRX", "005930", limit=1, adj_ok="1")
# print(df_kospi.head())
# print(len(df_kospi))

class FailedRecord(Schema):
    index: int
    code: str
    error: str

class StockDescriptionResponse(Schema):
    result: str
    count_total: int
    count_created: int
    count_updated: int
    count_failed: int
    failed_records: Optional[List[FailedRecord]] = None

    
@api.get("/stock_description", response={200: StockDescriptionResponse, 400: Dict, 500: Dict})
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
        
class ErrorResponse(Schema):
    error: str

class SuccessResponse(Schema):
    message: str
    count_saved: int
    
@api.get("/stock", response={200: SuccessResponse, 400: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse})
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
    
    for company in companies:
        print(company.code, company.name)
        
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
        highs = [(record.high, record.date) for record in period_data]
        lows = [(record.low, record.date) for record in period_data]
        
        max_high, max_date = max(highs, key=lambda x: x[0]) if highs else (0.0, None)
        min_low, min_date = min(lows, key=lambda x: x[0]) if lows else (0.0, None)
        
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
def update_stock_analysis(request):
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
    
    # StockOHLCV의 최신 날짜 가져오기
    latest_date = StockOHLCV.objects.aggregate(Max('date'))['date__max']
    if not latest_date:
        print("StockOHLCV 데이터가 없습니다.")
        return

     # RS 점수 저장용 데이터프레임
    rs_data = []
    analysis_objects = []

    
    for company in companies:
        print(f"처리 중: {company.code} - {company.name}")
        
        # 회사별 OHLCV 데이터 가져오기
        ohlcv_data = StockOHLCV.objects.filter(code=company)
        
        if not ohlcv_data.exists():
            print(f"{company.code}에 대한 OHLCV 데이터 없음")
            continue
        
        # 최신 종가 가져오기
        latest_ohlcv = ohlcv_data.filter(date=latest_date).first()
        latest_close = latest_ohlcv.close if latest_ohlcv else 0.0
        
        # 이동평균 계산
        mas = calculate_moving_averages(ohlcv_data, latest_date)
        
        # 52주 신고가/신저가 및 날짜 계산
        high_low = calculate_52w_high_low(ohlcv_data, latest_date)
        
        # 각 기간별 RS 점수 계산
        rs_scores = {}
        for period_name, period_days in periods.items():
            rs_score = calculate_rs_score(ohlcv_data, latest_date, period_days)
            rs_scores[period_name] = rs_score
        
        # 가중평균 RS 점수 계산
        weighted_score = -1
        if all(rs_scores[p] != -1 for p in periods):
            weighted_score = (rs_scores['1month'] * 4 + rs_scores['3month'] * 3 + rs_scores['6month'] * 2 + rs_scores['12month'] * 1) / 10
        
        rs_data.append({
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
            date=latest_date,
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
        
    # 랭킹 계산 (market별, 1등=99, 꼴지=1)
    rs_df = pd.DataFrame(rs_data)
    for market in rs_df['market'].unique():
        market_df = rs_df[rs_df['market'] == market]
        if market_df.empty:
            continue
        for period in ['rsScore1m', 'rsScore3m', 'rsScore6m', 'rsScore12m', 'rsScore']:
            # 랭킹 계산
            rank_values = market_df[period].rank(ascending=True, na_option='bottom')
            rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
            
            # rs_df에 값 반영
            rs_df.loc[market_df.index, f'{period}_Rank'] = rank_values
            rs_df.loc[market_df.index, f'{period}_RS'] = rs_values

    # StockAnalysis 객체에 랭킹 반영
    for i, obj in enumerate(analysis_objects):
        row = rs_df[rs_df['code'] == obj.code.code].iloc[0]
        obj.rsRank = row['rsScore_RS'] if row['rsScore'] != -1 else 0.0
        obj.rsRank1m = row['rsScore1m_RS'] if row['rsScore1m'] != -1 else 0.0
        obj.rsRank3m = row['rsScore3m_RS'] if row['rsScore3m'] != -1 else 0.0
        obj.rsRank6m = row['rsScore6m_RS'] if row['rsScore6m'] != -1 else 0.0
        obj.rsRank12m = row['rsScore12m_RS'] if row['rsScore12m'] != -1 else 0.0
        # rsRank >= 70 조건 적용
        if obj.is_minervini_trend:
            obj.is_minervini_trend = obj.is_minervini_trend and obj.rsRank >= 70
            
    # 중복된 StockAnalysis 레코드 삭제
    StockAnalysis.objects.filter(date=latest_date).delete()
    
    # Bulk create
    try:
        with transaction.atomic():
            StockAnalysis.objects.bulk_create(analysis_objects)
        print(f"{len(analysis_objects)}개의 StockAnalysis 레코드 생성 완료: {latest_date}")
        return {
            "message": "Stock analysis data saved successfully.",
            "count_saved": len(analysis_objects),
        }
    except Exception as e:
        traceback.print_exc()
        return 500, {"error": f"Failed to save stock analysis data: {str(e)}"}
    