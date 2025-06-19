from ninja import NinjaAPI, Router, Schema
from django.db import transaction
from django.http import HttpResponse

from . import stockCommon as Common
from myweb.models import *  # Import the StockOHLCV model

from typing import Dict
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

import traceback


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