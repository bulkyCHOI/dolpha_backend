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
import time
import logging

# Phase 3: 성능 모니터링 및 로깅 설정
logger = logging.getLogger(__name__)

# HTF Pattern Analysis Utility Functions


def calculate_htf_pattern(
    ohlcv_data,
    target_date,
    min_gain_percent=100.0,
    max_pullback_percent=30.0,
    analysis_period_days=56,
):
    """
    특정 날짜의 HTF 패턴 계산 (유틸리티 함수)

    Args:
        ohlcv_data: OHLCV QuerySet (ordered by date)
        target_date: 분석 대상 날짜
        min_gain_percent: 최소 상승률 (기본 100%)
        max_pullback_percent: 최대 조정폭 (기본 25%)
        analysis_period_days: 분석 기간 (기본 56일 = 8주)

    Returns:
        HTF 패턴 분석 결과 딕셔너리
    """
    try:
        # DataFrame으로 변환
        ohlcv_list = list(
            ohlcv_data.filter(date__lte=target_date)
            .order_by("date")
            .values("date", "open", "high", "low", "close", "volume")
        )

        if len(ohlcv_list) < analysis_period_days:
            return {
                "detected": False,
                "gain_percent": 0.0,
                "pullback_percent": 0.0,
                "start_date": None,
                "peak_date": None,
                "status": "none",
            }

        df = pd.DataFrame(ohlcv_list)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 최근 analysis_period_days 데이터 선택
        window_data = df.tail(analysis_period_days).copy()

        if len(window_data) < analysis_period_days:
            return {
                "detected": False,
                "gain_percent": 0.0,
                "pullback_percent": 0.0,
                "start_date": None,
                "peak_date": None,
                "status": "none",
            }

        current_price = window_data.iloc[-1]["close"]

        # HTF 패턴 분석
        return _check_htf_pattern(
            window_data, current_price, min_gain_percent, max_pullback_percent
        )

    except Exception as e:
        logger.error(f"HTF 패턴 계산 중 오류: {str(e)}")
        return {
            "detected": False,
            "gain_percent": 0.0,
            "pullback_percent": 0.0,
            "start_date": None,
            "peak_date": None,
            "status": "none",
        }


def _check_htf_pattern(
    window_data, current_price, min_gain_percent, max_pullback_percent
):
    """
    HTF 패턴 조건 확인 (내부 함수)

    Args:
        window_data: 8주 윈도우 데이터
        current_price: 현재 가격
        min_gain_percent: 최소 상승률
        max_pullback_percent: 최대 조정폭

    Returns:
        패턴 분석 결과
    """
    try:
        # 1. 8주간 최저점 찾기
        min_idx = window_data["low"].idxmin()
        min_price = window_data.loc[min_idx, "low"]
        min_date = window_data.loc[min_idx, "date"].date()

        # 2. 최저점 이후 최고점 찾기
        after_min = window_data[window_data.index > min_idx]
        if after_min.empty:
            return _default_htf_result()

        max_idx = after_min["high"].idxmax()
        max_price = after_min.loc[max_idx, "high"]
        max_date = after_min.loc[max_idx, "date"].date()

        # 3. 상승률 계산
        if min_price == 0:
            gain_percent = 0.0
        else:
            gain_percent = ((max_price - min_price) / min_price) * 100

        # 4. 최소 상승률 조건 확인
        # 넘지 않더라도 DB에는 기록해주자
        # if gain_percent < min_gain_percent:
        #     return _default_htf_result()

        # 5. 최고점 이후 조정폭 계산
        after_max = after_min[after_min.index > max_idx]
        pullback_percent = 0.0
        current_status = "rising"

        if not after_max.empty:
            # 최고점 이후 최저점
            pullback_min_price = after_max["low"].min()
            pullback_percent = ((max_price - pullback_min_price) / max_price) * 100

            # 현재 상태 판단
            if pullback_percent > 0:
                if current_price < max_price:
                    current_status = "pullback"
                elif current_price > max_price:  # 신고가 돌파
                    current_status = "breakout"
                else:
                    current_status = "pullback"

        # 6. 조정폭 조건 확인
        # 조정폭이 넘어도 일단 기록은 하자
        # if pullback_percent > max_pullback_percent:
        #     return _default_htf_result()

        # 7. HTF 패턴 확인
        pattern_detected = (
            gain_percent >= min_gain_percent
            and pullback_percent <= max_pullback_percent
        )

        return {
            "detected": pattern_detected,
            "gain_percent": round(gain_percent, 2),
            "pullback_percent": round(pullback_percent, 2),
            "start_date": min_date,
            "peak_date": max_date,
            "status": current_status,
        }

    except Exception as e:
        logger.error(f"HTF 패턴 확인 중 오류: {str(e)}")
        return _default_htf_result()


def _default_htf_result():
    """기본 HTF 결과 반환"""
    return {
        "detected": False,
        "gain_percent": 0.0,
        "pullback_percent": 0.0,
        "start_date": None,
        "peak_date": None,
        "status": "none",
    }


# 성능 모니터링 데코레이터
from functools import wraps


def performance_monitor(func_name):
    """API 성능 모니터링 데코레이터"""

    def decorator(func):
        @wraps(func)  # 원본 함수의 메타데이터 보존
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{func_name} 완료 - 소요시간: {duration:.2f}초")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func_name} 실패 - 소요시간: {duration:.2f}초, 오류: {str(e)}"
                )
                raise

        return wrapper

    return decorator


# Phase 3: 표준화된 에러 처리 함수들
def handle_api_error(operation_name: str, error: Exception, error_code: int = 500):
    """표준화된 API 에러 처리"""
    error_msg = f"{operation_name} 실패: {str(error)}"
    logger.error(error_msg)
    traceback.print_exc()
    return error_code, {"status": "ERROR", "message": error_msg}


def log_batch_results(
    operation_name: str, total: int, created: int, updated: int, failed: int
):
    """배치 작업 결과 로깅"""
    success_rate = ((created + updated) / total * 100) if total > 0 else 0
    logger.info(
        f"{operation_name} 결과 - 총:{total}, 생성:{created}, 수정:{updated}, 실패:{failed}, 성공률:{success_rate:.1f}%"
    )


def validate_required_data(data, required_fields: list, operation_name: str):
    """필수 데이터 유효성 검사"""
    if data is None or (hasattr(data, "empty") and data.empty):
        raise ValueError(f"{operation_name}: 데이터가 없습니다")

    if hasattr(data, "columns"):  # DataFrame인 경우
        missing_fields = [
            field for field in required_fields if field not in data.columns
        ]
        if missing_fields:
            raise ValueError(f"{operation_name}: 필수 필드 누락 - {missing_fields}")


# 데이터 수집/저장/계산 관련 API 라우터
data_router = Router()


# 모든 주식의 설명 데이터를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
@data_router.post(
    "/getAndSave_stock_description",
    response={200: StockDescriptionResponse, 400: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("Stock Description Sync")
def getAndSave_stock_description(request, stock: str = "KRX-DESC"):
    """
    모든 주식의 설명 데이터를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.\n
        Args:\n
            stock (str): 조회할 주식의 종류 (기본값: "KRX-DESC")\n
                        NASDAQ, NYSE 등 다양한 주식 시장 코드 지원\n
        Returns:\n
            StockDescriptionResponse: 처리 결과, 저장된 레코드 수, 실패한 레코드 수 및 오류 메시지\n
    """
    try:
        # 종목정보 조회
        df_stocks = Common.GetStockList(stock)
        print(f"조회된 {stock} 주식 설명 데이터: {len(df_stocks)}개")

        # 컬럼명 매핑
        if stock == "KRX-DESC":
            column_mapping = {
                "Code": "code",
                "Name": "name",
                "Market": "market",
                "Sector": "sector",
                "Industry": "industry",
                # 'ListingDate': 'listing_date',
                # 'SettleMonth': 'settle_month',
                # 'Representative': 'representative',
                # 'HomePage': 'homepage',
                # 'Region': 'region'
            }
        # 미장 컬럼
        elif stock in ["NASDAQ", "NYSE"]:
            column_mapping = {
                "Symbol": "code",
                "Name": "name",
                "IndustryCode": "industrycode",
                "Industry": "industry",
            }
        else:
            return 400, {
                "status": "ERROR",
                "message": f"Unsupported stock type: {stock}",
            }

        df_stocks = df_stocks.rename(columns=column_mapping)
        print(df_stocks.head())

        # 예상 컬럼 확인
        expected_columns = list(column_mapping.values())
        if not all(col in df_stocks.columns for col in expected_columns):
            return 400, {
                "status": "ERROR",
                "message": f"Required columns are missing in the {stock} data",
            }

        # 데이터 전처리
        df_stocks["code"] = df_stocks["code"].astype(str)  # 코드 문자열로 변환

        if "sector" not in df_stocks.columns:
            df_stocks["sector"] = None
        if "market" not in df_stocks.columns:
            df_stocks["market"] = stock

        # NaN 값을 None으로 변환 (중요: "기타"로 변환하지 않음)
        df_stocks["sector"] = df_stocks["sector"].where(
            pd.notna(df_stocks["sector"]), None
        )
        df_stocks["industry"] = df_stocks["industry"].where(
            pd.notna(df_stocks["industry"]), None
        )

        # 기존 데이터를 values()로 효율적 조회 - 메모리 최적화
        existing_companies = {
            obj["code"]: obj
            for obj in Company.objects.values(
                "code", "name", "market", "sector", "industry"
            )
        }
        existing_codes = set(existing_companies.keys())

        # 벌크 데이터 준비
        companies_to_create = []
        companies_to_update = []
        failed_records = []
        processed_codes = set()  # 처리된 코드를 추적하여 중복 방지

        # iterrows() 대신 itertuples() 사용 (더 빠름)
        for row in tqdm(
            df_stocks.itertuples(index=False),
            total=len(df_stocks),
            desc="데이터 처리 중",
        ):
            try:
                code = str(row.code)
                name = str(row.name)
                market = str(row.market)
                sector = row.sector if row.sector is not None else None
                industry = row.industry if row.industry is not None else None

                # KONEX 시장 제외 (선택사항)
                # if market == 'KONEX':
                #     continue

                # 이미 처리된 코드인 경우 건너뛰기 (중복 방지)
                if code in processed_codes:
                    continue
                processed_codes.add(code)

                # 생성 또는 업데이트 분류
                if code not in existing_codes:
                    companies_to_create.append(
                        Company(
                            code=code,
                            name=name,
                            market=market,
                            sector=row.sector if row.sector is not None else None,
                            industry=row.industry if row.industry is not None else None,
                        )
                    )
                else:
                    # 변경사항이 있는 경우만 업데이트 대상에 추가
                    existing_data = existing_companies[code]

                    # None이 아닌 값만 업데이트 - 딕셔너리 기반 비교로 성능 향상
                    should_update = False
                    updates = {}

                    if existing_data["name"] != name:
                        updates["name"] = name
                        should_update = True
                    if existing_data["market"] != market:
                        updates["market"] = market
                        should_update = True
                    if sector is not None and existing_data["sector"] != sector:
                        updates["sector"] = sector
                        should_update = True
                    if industry is not None and existing_data["industry"] != industry:
                        updates["industry"] = industry
                        should_update = True

                    if should_update:
                        # 실제 객체는 업데이트 시에만 조회
                        update_obj = Company.objects.get(id=existing_data["id"])
                        for field, value in updates.items():
                            setattr(update_obj, field, value)
                        companies_to_update.append(update_obj)

            except Exception as e:
                traceback.print_exc()
                failed_records.append(
                    {"code": getattr(row, "code", "N/A"), "error": str(e)}
                )

        # 데이터베이스 트랜잭션 - 모든 작업을 한 번에 수행
        with transaction.atomic():
            # 벌크 생성
            if companies_to_create:
                print(f"신규 회사 {len(companies_to_create)}개 생성 완료")
                Company.objects.bulk_create(companies_to_create, batch_size=500)

            # 벌크 업데이트 - update_or_create 대신 bulk_update 사용
            if companies_to_update:
                print(f"기존 회사 {len(companies_to_update)}개 업데이트 완료")
                Company.objects.bulk_update(
                    companies_to_update,
                    ["name", "market", "sector", "industry"],
                    batch_size=500,  # 메모리 효율성을 위해 배치 크기 감소
                )

        # 응답 구성
        response = {
            "status": "OK",
            "count_total": len(df_stocks),
            "count_created": len(companies_to_create),
            "count_updated": len(companies_to_update),
            "count_failed": len(failed_records),
            "failed_records": failed_records if failed_records else None,
        }
        return response

    except Exception as e:
        traceback.print_exc()
        return 500, {
            "status": "ERROR",
            "message": f"Failed to process stock data: {str(e)}",
        }


# KRX에서 모든 주식의 종목 코드를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.
@data_router.post(
    "/getAndSave_index_list",
    response={200: IndexListResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def getAndSave_index_list(request):
    """
    KRX에서 모든 주식의 종목 코드를 조회하고 Django ORM을 사용해 데이터베이스에 저장합니다.

    Returns:
        IndexListResponse: 처리 결과, 저장된 종목 코드 리스트
    """
    try:
        # 종목정보 조회
        df_krx = Common.GetSnapDataReader()
        print(f"조회된 KRX 인덱스 데이터: {len(df_krx)}개")

        # 입력 데이터 검증
        if df_krx is None or df_krx.empty:
            return 400, {
                "status": "ERROR",
                "message": "KRX 데이터를 가져올 수 없습니다.",
            }

        # 컬럼명 매핑 및 검증
        column_mapping = {
            "Code": "code",
            "Name": "name",
            "Market": "market",
        }
        df_krx = df_krx.rename(columns=column_mapping)

        expected_columns = list(column_mapping.values())
        if not all(col in df_krx.columns for col in expected_columns):
            return 400, {
                "status": "ERROR",
                "message": "KRX 데이터에 필수 컬럼이 누락되었습니다.",
            }

        # 데이터 전처리 최적화
        df_krx["code"] = df_krx["code"].astype(str).str.strip()  # 공백 제거
        df_krx = df_krx.dropna(subset=["code", "name"])  # 필수 값이 없는 행 제거
        df_krx = df_krx.drop_duplicates(subset=["code"])  # 중복 제거

        # 기존 데이터를 values()로 효율적 조회
        existing_indices = {
            obj["code"]: obj
            for obj in StockIndex.objects.values("code", "name", "market")
        }
        existing_codes = set(existing_indices.keys())

        # 벌크 데이터 준비 최적화
        stockIndex_to_create = []
        stockIndex_to_update = []
        failed_records = []

        # DataFrame을 딕셔너리로 변환하여 더 빠른 처리
        df_records = df_krx.to_dict("records")

        for record in df_records:
            try:
                code = str(record["code"])
                name = str(record["name"])
                market = str(record["market"])

                # 생성 또는 업데이트 분류
                if code not in existing_codes:
                    stockIndex_to_create.append(
                        StockIndex(code=code, name=name, market=market)
                    )
                else:
                    # 변경사항 확인 (딕셔너리 기반으로 더 빠름)
                    existing_obj = existing_indices[code]
                    if existing_obj["name"] != name or existing_obj["market"] != market:
                        # update용 객체는 실제 모델 인스턴스가 필요
                        update_obj = StockIndex.objects.get(id=existing_obj["id"])
                        update_obj.name = name
                        update_obj.market = market
                        stockIndex_to_update.append(update_obj)

            except Exception as e:
                failed_records.append(
                    {"code": record.get("code", "N/A"), "error": str(e)}
                )

        print(
            f"처리 대상: 생성 {len(stockIndex_to_create)}개, 업데이트 {len(stockIndex_to_update)}개"
        )

        # 데이터베이스 트랜잭션 최적화
        with transaction.atomic():
            # 벌크 생성 (배치 크기 최적화)
            if stockIndex_to_create:
                StockIndex.objects.bulk_create(stockIndex_to_create, batch_size=500)
                print(f"인덱스 {len(stockIndex_to_create)}개 생성 완료")

            # 벌크 업데이트
            if stockIndex_to_update:
                StockIndex.objects.bulk_update(
                    stockIndex_to_update, ["name", "market"], batch_size=500
                )
                print(f"인덱스 {len(stockIndex_to_update)}개 업데이트 완료")

            # 인덱스-종목 관계 설정 최적화
            print("인덱스별 종목 관계 설정 시작...")

            # 모든 인덱스와 회사 데이터를 한 번에 메모리로 로드
            all_indices = {obj.code: obj for obj in StockIndex.objects.all()}
            all_companies = {obj.code: obj for obj in Company.objects.all()}

            # 배치로 관계 데이터 수집
            relationship_data = []  # [(index_obj, [company_objs])]

            for code in tqdm(df_krx["code"], desc="인덱스별 종목 관계 데이터 수집"):
                try:
                    # 인덱스에 속한 종목 조회
                    df_stocks = Common.GetSnapDataReader_IndexCode(code)

                    if df_stocks is None or df_stocks.empty:
                        continue

                    # 종목 코드 리스트 추출 및 정리
                    stock_codes = df_stocks["Code"].astype(str).str.strip().tolist()

                    # 인덱스 객체 가져오기
                    if code in all_indices:
                        index_obj = all_indices[code]

                        # 유효한 종목만 필터링 (메모리에서 처리)
                        valid_companies = [
                            all_companies[stock_code]
                            for stock_code in stock_codes
                            if stock_code in all_companies
                        ]

                        if valid_companies:
                            relationship_data.append((index_obj, valid_companies))

                except Exception as e:
                    print(f"인덱스 {code} 데이터 수집 중 오류: {str(e)}")
                    failed_records.append(
                        {"code": code, "error": f"관계 데이터 수집 오류: {str(e)}"}
                    )
                    continue

            # 관계 설정을 배치로 처리 (DB 액세스 최소화)
            print(f"관계 설정 시작: {len(relationship_data)}개 인덱스")
            for index_obj, companies in tqdm(relationship_data, desc="관계 설정 처리"):
                try:
                    # 기존 관계를 모두 지우고 새로 설정
                    index_obj.companies.clear()

                    # 배치 크기를 제한하여 메모리 효율성 확보
                    batch_size = 1000
                    for i in range(0, len(companies), batch_size):
                        batch_companies = companies[i : i + batch_size]
                        index_obj.companies.add(*batch_companies)

                except Exception as e:
                    print(f"인덱스 {index_obj.code} 관계 설정 중 오류: {str(e)}")
                    failed_records.append(
                        {"code": index_obj.code, "error": f"관계 설정 오류: {str(e)}"}
                    )

        print("인덱스별 종목 관계 설정 완료")

        # 응답 구성
        response = {
            "status": "OK",
            "count_total": len(df_krx),
            "count_created": len(stockIndex_to_create),
            "count_updated": len(stockIndex_to_update),
            "count_failed": len(failed_records),
            "failed_records": failed_records if failed_records else None,
        }
        return response

    except Exception as e:
        traceback.print_exc()
        return 500, {"status": "ERROR", "message": f"인덱스 데이터 처리 실패: {str(e)}"}


# Index 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
@data_router.post(
    "/getAndSave_index_data",
    response={
        200: SuccessResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
def getAndSave_index_data(request, code: str = None, limit: int = 1):
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

        df = Common.GetOhlcv(
            "KR", f"KRX-INDEX:{stockIndex.code}", limit=limit, adj_ok="1"
        )
        # print(df.head())

        if df is None or len(df) == 0:
            # return 400, {"error": "No OHLCV data found for the given index code."}
            continue  # 인덱스가 없으면 다음 인덱스로 넘어감

        # 컬럼명 검증
        expected_columns = ["open", "high", "low", "close", "volume", "change"]
        if not all(col in df.columns for col in expected_columns):
            # return 400, {"error": "Required OHLCV columns are missing in the data."}
            continue  # 컬럼이 누락된 경우 다음 인덱스로 넘어감

        # 데이터 전처리
        try:
            df.index = pd.to_datetime(
                df.index, errors="coerce"
            )  # 인덱스를 Timestamp로 변환
            if df.index.isna().any():  # 변환 실패 시 NaT가 있는지 확인
                # return 400, {"error": "Invalid date format in OHLCV data index."}
                continue  # 날짜 형식이 잘못된 경우 다음 인덱스로 넘어감
        except Exception as e:
            # return 400, {"error": f"Failed to process date column: {str(e)}"}
            continue  # 날짜 처리 실패 시 다음 인덱스로 넘어감
        # 데이터베이스 저장
        try:
            with transaction.atomic():
                # 벌크 데이터 준비
                index_ohlcv_list = []

                # DataFrame에서 NaN 값을 기본값으로 대체
                df = df.fillna(
                    {
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "close": 0.0,
                        "volume": 0,
                        "change": 0.0,
                    }
                )

                # DataFrame을 날짜 기준으로 정렬 (오래된 날짜부터)
                df_sorted = df.sort_index()

                for i, (index, row) in enumerate(df_sorted.iterrows()):
                    # 전일종가 대비 당일 종가 변동률 계산
                    if (
                        i > 0
                    ):  # 첫 번째 데이터가 아닌 경우 (DataFrame 내에서 전일 데이터 사용)
                        prev_close = df_sorted.iloc[i - 1]["close"]
                        current_close = row["close"]
                        if prev_close > 0:
                            change_rate = (current_close - prev_close) / prev_close
                        else:
                            change_rate = 0.0
                    else:
                        # 첫 번째 데이터는 API에서 제공한 change 값 사용 (성능 최적화)
                        # N+1 쿼리 문제 해결: 개별 DB 조회 대신 API 데이터 사용
                        change_rate = float(row["change"] if "change" in row else 0.0)

                    index_ohlcv = IndexOHLCV(
                        code=stockIndex,
                        date=index.date(),  # 인덱스(Timestamp)에서 date 추출
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                        change=change_rate,  # 전일종가 대비 변동률 사용
                    )
                    index_ohlcv_list.append(index_ohlcv)

                # 기존 데이터 삭제 후 새로 삽입 (change 값 업데이트를 위해)
                if index_ohlcv_list:
                    # 해당 날짜들의 기존 데이터 삭제
                    dates_to_update = [obj.date for obj in index_ohlcv_list]
                    IndexOHLCV.objects.filter(
                        code=stockIndex, date__in=dates_to_update
                    ).delete()
                    # 새 데이터 삽입
                    IndexOHLCV.objects.bulk_create(index_ohlcv_list)
        except Exception as e:
            traceback.print_exc()
            # return 500, {"error": f"Failed to save index data: {str(e)}"}
            continue  # 저장 실패 시 다음 인덱스로 넘어감

    return {
        "status": "OK",
        "message": f"Index data {len(indices)} saved successfully.",
        "count_saved": limit,
    }


# 주식 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
@data_router.post(
    "/getAndSave_stock_data",
    response={
        200: SuccessResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
def getAndSave_stock_data(request, code: str = None, area: str = "KR", limit: int = 1):
    """
    주식 코드에 해당하는 OHLCV 데이터를 데이터베이스에 저장합니다.
    코드를 입력하지 않으면, 모든 회사의 OHLCV 데이터를 가져옵니다.

    Args:
        code (str): 주식 코드
        area (str): 주식 시장 지역 (기본값: "KR" - 한국, "US" - 미국)
        limit (int): 가져올 데이터의 개수 (기본값: 1)

    Returns:
        SuccessResponse: 데이터 저장 성공 시 메시지와 저장된 레코드 수
        ErrorResponse: 에러 발생 시 에러 메시지
    """
    companies = []

    if code is not None:
        # code에 맞는 회사의 OHLCV 데이터를 가져옴
        try:
            if area == "KR":
                company = Company.objects.get(
                    code=code, market__in=["KOSDAQ", "KONEX", "KOSPI"]
                )
            elif area == "US":
                company = Company.objects.get(
                    code=code, market__in=["NASDAQ", "NYSE", "S&P500"]
                )
            companies = [company]  # 단일 회사 객체를 리스트로 감싸서 처리
        except Company.DoesNotExist:
            return 404, {"error": f"No company found with code: {code}"}
    else:
        # code가 주어지지 않은 경우 모든 회사의 OHLCV 데이터를 가져옴
        if area == "KR":
            companies = Company.objects.filter(market__in=["KOSDAQ", "KONEX", "KOSPI"])
        elif area == "US":
            companies = Company.objects.filter(market__in=["NASDAQ", "NYSE", "S&P500"])

    print(f"Total companies: {len(companies)}")

    if len(companies) == 0:
        return 404, {"error": "No companies found in the database."}

    for company in tqdm(companies, desc="Processing companies..."):
        # print(company.code, company.name)

        if company.market in ["KOSDAQ", "KONEX", "KOSPI"]:
            df = Common.GetOhlcv("KR", company.code, limit=limit, adj_ok="1")
        elif company.market in ["NASDAQ", "NYSE", "S&P500"]:
            df = Common.GetOhlcv("US", company.code, limit=limit, adj_ok="1")
        else:
            return 400, {"error": f"Unsupported market type: {company.market}"}
            continue
        # print(df.head())

        if df is None or len(df) == 0:
            # return 400, {"error": "No OHLCV data found for the given stock code."}
            continue  # 회사가 없으면 다음 회사로 넘어감

        # 컬럼명 검증
        expected_columns = ["open", "high", "low", "close", "volume", "change"]
        if not all(col in df.columns for col in expected_columns):
            # return 400, {"error": "Required OHLCV columns are missing in the data."}
            continue  # 컬럼이 누락된 경우 다음 회사로 넘어감

        # 데이터 전처리
        try:
            df.index = pd.to_datetime(
                df.index, errors="coerce"
            )  # 인덱스를 Timestamp로 변환
            if df.index.isna().any():  # 변환 실패 시 NaT가 있는지 확인
                # return 400, {"error": "Invalid date format in OHLCV data index."}
                continue  # 날짜 형식이 잘못된 경우 다음 회사로 넘어감
        except Exception as e:
            # return 400, {"error": f"Failed to process date column: {str(e)}"}
            continue  # 날짜 처리 실패 시 다음 회사로 넘어감

        # 데이터베이스 저장
        try:
            with transaction.atomic():
                # 벌크 데이터 준비
                stock_ohlcv_list = []

                # DataFrame에서 NaN 값을 기본값으로 대체
                df = df.fillna(
                    {
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "close": 0.0,
                        "volume": 0,
                        "change": 0.0,
                    }
                )

                # DataFrame을 날짜 기준으로 정렬 (오래된 날짜부터)
                df_sorted = df.sort_index()

                # Feature flag를 사용한 변동률 계산 방식 선택
                if USE_NEW_CHANGE_CALCULATION:
                    # 새로운 방식: DB 기반 정확한 계산
                    dates_in_df = [idx.date() for idx in df_sorted.index]
                    prev_close_lookup = batch_get_previous_closes(
                        [company], dates_in_df
                    )

                for i, (index, row) in enumerate(df_sorted.iterrows()):
                    current_date = index.date()
                    current_close = float(row["close"])

                    # Feature flag에 따른 변동률 계산
                    if USE_NEW_CHANGE_CALCULATION:
                        # 새로운 방식: 정확한 변동률 계산
                        if i > 0:
                            # DataFrame 내에서 이전 데이터 사용 (연속 데이터인 경우)
                            prev_close = float(df_sorted.iloc[i - 1]["close"])
                            if prev_close > 0:
                                change_rate = (current_close - prev_close) / prev_close
                            else:
                                change_rate = 0.0
                        else:
                            # 첫 번째 데이터: DB에서 실제 이전 거래일 종가 사용
                            change_rate = calculate_accurate_change_rate(
                                company.code, current_close, prev_close_lookup
                            )

                        # API 제공 change 값과 비교 검증 (개발 시에만)
                        api_change = float(row["change"] if "change" in row else 0.0)
                        if abs(change_rate - api_change) > 0.001:  # 0.1% 이상 차이
                            logger.info(
                                f"변동률 차이: {company.code} {current_date} "
                                f"DB계산={change_rate:.4f} API값={api_change:.4f}"
                            )
                    else:
                        # 기존 방식: 롤백용
                        change_rate = calculate_change_rate_legacy(df_sorted, i, row)

                    stock_ohlcv = StockOHLCV(
                        code=company,
                        date=index.date(),  # 인덱스(Timestamp)에서 date 추출
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                        change=change_rate,  # 전일종가 대비 변동률 사용
                    )
                    stock_ohlcv_list.append(stock_ohlcv)

                # 기존 데이터 삭제 후 새로 삽입 (change 값 업데이트를 위해)
                if stock_ohlcv_list:
                    # 해당 날짜들의 기존 데이터 삭제
                    dates_to_update = [obj.date for obj in stock_ohlcv_list]
                    StockOHLCV.objects.filter(
                        code=company, date__in=dates_to_update
                    ).delete()
                    # 새 데이터 삽입
                    StockOHLCV.objects.bulk_create(stock_ohlcv_list)
        except Exception as e:
            traceback.print_exc()
            # return 500, {"error": f"Failed to save stock data: {str(e)}"}
            continue
    return {
        "status": "OK",
        "message": f"Stock data {len(companies)} saved successfully.",
        "count_saved": limit,
    }


# ============================================================================
# 주식 분석 계산 유틸리티 함수들 (Phase 2: 함수 분해 및 최적화)
# ============================================================================


def calculate_moving_averages(
    data, target_date, periods=[50, 150, 200], past_ma200_days=21
):
    """
    주어진 기간에 대해 이동평균(MA)을 계산합니다. 1개월 전 MA200도 계산.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: MA를 계산할 목표 날짜.
    periods: 계산할 MA 기간 리스트(예: [50, 150, 200]).
    past_ma200_days: 1개월 전 MA200 계산을 위한 기간(기본값: 21일).
    """
    try:
        data = list(data.order_by("date").filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break

        if target_idx is None:
            return {f"ma{period}": 0.0 for period in periods} | {"ma200_past": 0.0}

        mas = {}
        for period in periods:
            if target_idx + 1 >= period:
                closes = [data[target_idx - i].close for i in range(period)]
                mas[f"ma{period}"] = np.mean(closes)
            else:
                mas[f"ma{period}"] = 0.0

        # 1개월 전 MA200 계산
        past_idx = target_idx - past_ma200_days
        if past_idx + 1 >= 200:
            past_closes = [data[past_idx - i].close for i in range(200)]
            mas["ma200_past"] = np.mean(past_closes)
        else:
            mas["ma200_past"] = 0.0

        return mas

    except Exception as e:
        print(f"MA 계산 오류: {e}")
        return {f"ma{period}": 0.0 for period in periods} | {"ma200_past": 0.0}


# 50일 신고가와 신저가 및 발생 날짜를 계산합니다.
def calculate_50d_high_low(data, target_date, period_days=50):
    """
    50일 신고가와 신저가 및 발생 날짜를 계산합니다.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: 계산할 목표 날짜.
    period_days: 계산 기간 (기본값: 50 거래일).
    """
    try:
        data = list(data.order_by("date").filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break

        if target_idx is None or target_idx + 1 < period_days:
            return {
                "max_50d": 0.0,
                "min_50d": 0.0,
                "max_50d_date": None,
                "min_50d_date": None,
            }

        # 지난 50일 데이터 추출
        period_data = data[target_idx - period_days + 1 : target_idx + 1]

        # high와 low가 0이 아닌 데이터만 필터링
        highs = [
            (record.high, record.date) for record in period_data if record.high > 0
        ]
        lows = [(record.low, record.date) for record in period_data if record.low > 0]

        # 유효한 데이터가 없으면 기본값 반환
        if not highs or not lows:
            return {
                "max_50d": 0.0,
                "min_50d": 0.0,
                "max_50d_date": None,
                "min_50d_date": None,
            }

        max_high, max_date = max(highs, key=lambda x: x[0])
        min_low, min_date = min(lows, key=lambda x: x[0])

        return {
            "max_50d": float(max_high),
            "min_50d": float(min_low),
            "max_50d_date": max_date,
            "min_50d_date": min_date,
        }

    except Exception as e:
        print(f"50일 신고가/신저가 계산 오류: {e}")
        return {
            "max_50d": 0.0,
            "min_50d": 0.0,
            "max_50d_date": None,
            "min_50d_date": None,
        }


# 52주 신고가와 신저가 및 발생 날짜를 계산합니다.
def calculate_52w_high_low(data, target_date, period_days=252):
    """
    52주 신고가와 신저가 및 발생 날짜를 계산합니다.
    data: StockOHLCV 객체의 Queryset, 날짜 기준 오름차순 정렬.
    target_date: 계산할 목표 날짜.
    period_days: 계산 기간 (기본값: 252 거래일 = 1년).
    """
    try:
        data = list(data.order_by("date").filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break

        if target_idx is None or target_idx + 1 < period_days:
            return {
                "max_52w": 0.0,
                "min_52w": 0.0,
                "max_52w_date": None,
                "min_52w_date": None,
            }

        # 지난 252일 데이터 추출
        period_data = data[target_idx - period_days + 1 : target_idx + 1]

        # high와 low가 0이 아닌 데이터만 필터링
        highs = [
            (record.high, record.date) for record in period_data if record.high > 0
        ]
        lows = [(record.low, record.date) for record in period_data if record.low > 0]

        # 유효한 데이터가 없으면 기본값 반환
        if not highs or not lows:
            return {
                "max_52w": 0.0,
                "min_52w": 0.0,
                "max_52w_date": None,
                "min_52w_date": None,
            }

        max_high, max_date = max(highs, key=lambda x: x[0])
        min_low, min_date = min(lows, key=lambda x: x[0])

        return {
            "max_52w": float(max_high),
            "min_52w": float(min_low),
            "max_52w_date": max_date,
            "min_52w_date": min_date,
        }

    except Exception as e:
        print(f"52주 신고가/신저가 계산 오류: {e}")
        return {
            "max_52w": 0.0,
            "min_52w": 0.0,
            "max_52w_date": None,
            "min_52w_date": None,
        }


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
        data = list(data.order_by("date"))

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
        data = list(data.order_by("date").filter(date__lte=target_date))
        target_idx = None
        for i, record in enumerate(data):
            if record.date >= target_date:
                target_idx = i
                break

        if target_idx is None or target_idx < period:
            return -1, 0.0  # 데이터 부족

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
            atr = np.mean(tr_values)
            atrRatio = (
                atr / data[target_idx].close if data[target_idx].close != 0 else 0.0
            )
            return atr, atrRatio
        else:
            return -1, 0.0  # 데이터 부족

    except Exception as e:
        print(f"ATR 계산 오류: {e}")
        traceback.print_exc()
        return -1, 0.0  # 오류 발생 시 -1과 0.0 반환


# ============================================================================
# Phase 2: 벌크 처리 최적화 함수들
# ============================================================================


def process_companies_in_batches(companies, batch_size=50):
    """
    회사 데이터를 배치 단위로 처리하여 메모리 사용량 최적화

    Args:
        companies: Company 객체 리스트
        batch_size: 배치 크기 (기본값: 50)

    Yields:
        배치 단위의 회사 리스트
    """
    for i in range(0, len(companies), batch_size):
        yield companies[i : i + batch_size]


def bulk_calculate_rs_rankings(rs_data_all, date_list):
    """
    RS 랭킹을 벌크로 계산하여 성능 최적화

    Args:
        rs_data_all: 모든 RS 데이터 리스트
        date_list: 처리할 날짜 리스트

    Returns:
        pandas.DataFrame: 랭킹이 계산된 RS 데이터
    """
    rs_df = pd.DataFrame(rs_data_all)

    # 날짜별, 시장별 랭킹 계산 최적화
    for date_entry in date_list:
        target_date = date_entry["date"]
        date_df = rs_df[rs_df["date"] == target_date]

        for market in date_df["market"].unique():
            market_df = date_df[date_df["market"] == market]
            if market_df.empty:
                continue

            # 벡터화된 랭킹 계산 (개별 루프 대신)
            for period in [
                "rsScore1m",
                "rsScore3m",
                "rsScore6m",
                "rsScore12m",
                "rsScore",
            ]:
                if period in market_df.columns:
                    rank_values = market_df[period].rank(
                        ascending=True, na_option="bottom"
                    )
                    rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
                    rs_df.loc[market_df.index, f"{period}_Rank"] = rank_values
                    rs_df.loc[market_df.index, f"{period}_RS"] = rs_values

    return rs_df


def optimize_ohlcv_data_loading(area, target_date):
    """
    OHLCV 데이터를 메모리 효율적으로 로드

    Args:
        companies: Company 객체 리스트
        target_date: 대상 날짜

    Returns:
        dict: 회사 코드별 OHLCV 데이터 딕셔너리
    """
    try:
        print(f"OHLCV 데이터 로드 시작: 지역={area}, 날짜={target_date}")
        if area == "KR":
            markets = ["KOSDAQ", "KONEX", "KOSPI"]
        elif area == "US":
            markets = ["NASDAQ", "NYSE"]
        else:
            raise ValueError("지원하지 않는 지역입니다. 'KR' 또는 'US'만 가능합니다.")
        # 벌크로 OHLCV 데이터 조회 (메모리 최적화)
        ohlcv_queryset = (
            # StockOHLCV.objects.select_related("code")
            # .filter(code__market__in=markets, date__lte=target_date)
            # .order_by("code__code", "date")
            StockOHLCV.objects.filter(code__market__in=markets, date__lte=target_date)
            .order_by("code__code", "date")
            .only(
                "code__code", "date", "open", "high", "low", "close", "volume", "change"
            )
        )
        print(f"OHLCV 데이터 로드: {len(ohlcv_queryset)}개 레코드")

        # 회사별 데이터 그룹화
        ohlcv_data_dict = {}
        current_code = None
        current_data = []

        for ohlcv in tqdm(ohlcv_queryset, desc="Grouping OHLCV data..."):
            if current_code != ohlcv.code.code:
                if current_code is not None:
                    ohlcv_data_dict[current_code] = current_data
                current_code = ohlcv.code.code
                current_data = [ohlcv]
            else:
                current_data.append(ohlcv)

        # 마지막 그룹 추가
        if current_code is not None:
            ohlcv_data_dict[current_code] = current_data

        return ohlcv_data_dict
    except Exception as e:
        logger.error(f"OHLCV 데이터 로딩 오류: {str(e)}")
        print(f"OHLCV 데이터 로딩 오류: {str(e)}")
        return {}


# ============================================================================
# 주식 가격 변동률 계산 유틸리티 함수들 (Phase 3: Change 계산 정확성 개선)
# ============================================================================


def get_previous_trading_day_close(company, current_date):
    """
    특정 회사의 이전 거래일 종가를 조회합니다.

    Args:
        company: Company 객체
        current_date: 현재 날짜 (date 객체)

    Returns:
        float: 이전 거래일의 종가, 없으면 None
    """
    try:
        previous_ohlcv = (
            StockOHLCV.objects.filter(code=company, date__lt=current_date)
            .order_by("-date")
            .first()
        )

        return previous_ohlcv.close if previous_ohlcv else None
    except Exception as e:
        logger.error(f"이전 거래일 종가 조회 오류 ({company.code}): {str(e)}")
        return None


def batch_get_previous_closes(companies, target_dates):
    """
    여러 회사의 이전 거래일 종가를 배치로 조회하여 N+1 쿼리 방지

    Args:
        companies: Company 객체 리스트
        target_dates: 처리할 날짜 리스트

    Returns:
        dict: {company_code: previous_close_price} 형태의 딕셔너리
    """
    try:
        if not companies or not target_dates:
            return {}

        # 가장 이른 날짜 기준으로 이전 데이터 조회
        earliest_date = min(target_dates)
        company_codes = [company.code for company in companies]

        # 모든 회사의 이전 거래일 종가를 한 번에 조회
        # distinct('code')를 사용하여 각 회사별로 가장 최근 데이터만 가져옴
        previous_data = (
            StockOHLCV.objects.filter(
                code__code__in=company_codes, date__lt=earliest_date
            )
            .select_related("code")
            .order_by("code__code", "-date")
        )

        # 회사별 가장 최근 종가만 저장
        prev_close_lookup = {}
        for item in previous_data:
            if item.code.code not in prev_close_lookup:
                prev_close_lookup[item.code.code] = item.close

        return prev_close_lookup

    except Exception as e:
        logger.error(f"배치 이전 거래일 종가 조회 오류: {str(e)}")
        return {}


def calculate_accurate_change_rate(company_code, current_close, prev_close_lookup):
    """
    정확한 주식 가격 변동률을 계산합니다.

    Args:
        company_code: 회사 코드 (문자열)
        current_close: 현재 종가 (float)
        prev_close_lookup: 이전 종가 딕셔너리 {company_code: prev_close}

    Returns:
        float: 변동률 (소수점 형태, 예: 0.05 = 5% 상승)
    """
    try:
        prev_close = prev_close_lookup.get(company_code)

        if prev_close and prev_close > 0 and current_close > 0:
            change_rate = (current_close - prev_close) / prev_close
            return change_rate
        else:
            # 이전 데이터가 없거나 유효하지 않은 경우
            return 0.0

    except Exception as e:
        logger.error(f"변동률 계산 오류 ({company_code}): {str(e)}")
        return 0.0


def validate_change_calculation(company, date, old_change, new_change):
    """
    변동률 계산 결과를 검증하고 로깅합니다.

    Args:
        company: Company 객체
        date: 날짜
        old_change: 기존 변동률
        new_change: 새로 계산한 변동률

    Returns:
        bool: 유의미한 차이가 있는지 여부
    """
    try:
        diff = abs(new_change - old_change)
        threshold = 0.001  # 0.1% 이상 차이를 유의미하다고 판단

        if diff > threshold:
            logger.warning(
                f"변동률 차이 발견: {company.code} {date} "
                f"기존={old_change:.4f} 신규={new_change:.4f} 차이={diff:.4f}"
            )
            return True
        return False

    except Exception as e:
        logger.error(f"변동률 검증 오류 ({company.code}): {str(e)}")
        return False


# ============================================================================
# 변동률 계산 테스트 및 검증 API
# ============================================================================


@data_router.post(
    "/test_change_calculation",
    response={200: SuccessResponse, 400: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("Change Calculation Test")
def test_change_calculation(request, code: str = None, days: int = 30):
    """
    변동률 계산 정확성을 테스트하고 검증합니다.

    Args:
        code (str): 테스트할 종목 코드 (없으면 전체)
        days (int): 테스트할 일수 (기본값: 30일)

    Returns:
        SuccessResponse: 테스트 결과 및 통계
    """
    try:
        # 테스트 대상 회사 선택
        companies = []
        if code:
            try:
                company = Company.objects.get(code=code)
                companies = [company]
            except Company.DoesNotExist:
                return 404, {"error": f"No company found with code: {code}"}
        else:
            # 전체 회사 중 무작위로 10개 선택 (테스트 용도)
            companies = list(
                Company.objects.filter(market__in=["KOSPI", "KOSDAQ"]).order_by("?")[
                    :10
                ]
            )

        if not companies:
            return 404, {"error": "No companies found for testing"}

        # 최근 days일간의 데이터로 테스트
        from datetime import date, timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        test_results = []
        total_tests = 0
        significant_diffs = 0

        for company in companies:
            # 해당 기간의 OHLCV 데이터 조회
            ohlcv_data = StockOHLCV.objects.filter(
                code=company, date__gte=start_date, date__lte=end_date
            ).order_by("date")

            for i, current_data in enumerate(ohlcv_data):
                if i == 0:  # 첫 번째 데이터는 건너뛰기
                    continue

                # 이전 거래일 데이터 조회
                prev_data = get_previous_trading_day_close(company, current_data.date)

                if prev_data and prev_data > 0:
                    # 새로운 방식으로 계산
                    new_change = (current_data.close - prev_data) / prev_data

                    # 기존 저장된 값과 비교
                    old_change = current_data.change
                    diff = abs(new_change - old_change)

                    total_tests += 1

                    if diff > 0.001:  # 0.1% 이상 차이
                        significant_diffs += 1
                        test_results.append(
                            {
                                "company_code": company.code,
                                "company_name": company.name,
                                "date": current_data.date.strftime("%Y-%m-%d"),
                                "old_change": round(old_change, 4),
                                "new_change": round(new_change, 4),
                                "difference": round(diff, 4),
                                "prev_close": prev_data,
                                "current_close": current_data.close,
                            }
                        )

        # 통계 계산
        accuracy_rate = (
            ((total_tests - significant_diffs) / total_tests * 100)
            if total_tests > 0
            else 0
        )

        return {
            "status": "OK",
            "message": "Change calculation test completed",
            "statistics": {
                "total_tests": total_tests,
                "significant_differences": significant_diffs,
                "accuracy_rate": round(accuracy_rate, 2),
                "test_period_days": days,
                "companies_tested": len(companies),
            },
            "differences_found": test_results[:20],  # 상위 20개만 반환
        }

    except Exception as e:
        return handle_api_error("Change Calculation Test", e)


def calculate_change_rate_legacy(df_sorted, i, row):
    """
    기존 변동률 계산 방식 (롤백용 백업)

    Args:
        df_sorted: 정렬된 DataFrame
        i: 현재 인덱스
        row: 현재 행 데이터

    Returns:
        float: 변동률
    """
    if i > 0:
        prev_close = df_sorted.iloc[i - 1]["close"]
        current_close = row["close"]
        if prev_close > 0:
            return (current_close - prev_close) / prev_close
        else:
            return 0.0
    else:
        # API 제공 change 값 사용
        return float(row["change"] if "change" in row else 0.0)


# Feature flag for change calculation method
USE_NEW_CHANGE_CALCULATION = True  # True: 새로운 방식, False: 기존 방식


@data_router.post(
    "/toggle_change_calculation_method",
    response={200: SuccessResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def toggle_change_calculation_method(request, use_new: bool = True):
    """
    변동률 계산 방식을 전환합니다 (롤백/복구용)

    Args:
        use_new (bool): True=새로운 방식, False=기존 방식

    Returns:
        SuccessResponse: 현재 설정 상태
    """
    global USE_NEW_CHANGE_CALCULATION

    old_method = "새로운 방식" if USE_NEW_CHANGE_CALCULATION else "기존 방식"
    USE_NEW_CHANGE_CALCULATION = use_new
    new_method = "새로운 방식" if USE_NEW_CHANGE_CALCULATION else "기존 방식"

    logger.info(f"변동률 계산 방식 변경: {old_method} → {new_method}")

    return {
        "status": "OK",
        "message": f"변동률 계산 방식이 '{new_method}'으로 변경되었습니다.",
        "previous_method": old_method,
        "current_method": new_method,
        "use_new_calculation": USE_NEW_CHANGE_CALCULATION,
    }


# 주식 분석 데이터를 계산하여 StockAnalysis 테이블에 저장합니다.
@data_router.post(
    "/calculate_stock_analysis",
    response={
        200: SuccessResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
# @performance_monitor("Stock Analysis Calculation")
# def calculate_stock_analysis(
#     request, area: str = "KR", offset: int = 0, limit: int = 0
# ):
#     """
#     주식 분석 데이터를 계산하여 StockAnalysis 테이블에 저장합니다.
#     최근 거래일부터 지정된 `limit`만큼의 거래일에 대해 모든 회사의 이동평균, 52주 신고가/신저가, RS 점수,
#     미너비니 트렌드 조건을 계산합니다. 휴일(예: 주말)은 StockOHLCV 데이터가 없으므로 자동으로 제외됩니다.

#     Args:
#         request: Ninja API 요청 객체.
#         area (str): 주식 시장 지역 ("KR" - 한국, "US" - 미국). 기본값: "KR".
#         offset (int, optional): 처리할 데이터의 시작 위치. 기본값: 0.
#         limit (int, optional): 처리할 거래일 수. 0이면 offset 거래일만 처리. 기본값: 0.
#         즉 offset ~ limit 범위의 거래일을 처리합니다.\n
#         0, 0: 오늘 거래일만 처리합니다.\n
#         0, 50: 오늘부터 50일 전까지의 거래일을 처리합니다.\n
#         50, 100: 50일 전부터 150(50+100)일 전까지의 거래일을 처리합니다.\n

#     Returns:
#         dict: 처리 결과를 포함하는 응답.
#             - message (str): 처리 결과 메시지.
#             - count_saved (int): 저장된 StockAnalysis 레코드 수.
#             - dates_processed (list): 처리된 날짜 목록 (YYYY-MM-DD 형식).
#         tuple: 에러 발생 시 (HTTP 상태 코드, 에러 메시지 딕셔너리).

#     Raises:
#         DatabaseError: 데이터베이스 저장 중 오류 발생 시.
#         Exception: 기타 예상치 못한 오류 발생 시.
#     """

#     # 메모리 최적화: select_related 사용하여 N+1 쿼리 방지
#     if area == "KR":
#         companies = (
#             Company.objects.filter(market__in=["KOSPI", "KOSDAQ"])
#             .select_related()
#             .all()
#         )
#     elif area == "US":
#         companies = (
#             Company.objects.filter(market__in=["NASDAQ", "NYSE"]).select_related().all()
#         )
#     else:
#         return 400, {"error": "Invalid area specified. Use 'KR' or 'US'."}

#     print(f"Total companies: {len(companies)}")

#     if len(companies) == 0:
#         return 404, {"error": "No companies found in the database."}

#     # 기간 정의 (거래일 기준)
#     periods = {
#         "12month": 252,  # 1년
#         "6month": 126,  # 6개월
#         "3month": 63,  # 3개월
#         "1month": 21,  # 1개월
#     }

#     # StockOHLCV의 고유 날짜 목록 가져오기 (최근 순, limit 적용)
#     date_list = StockOHLCV.objects.values("date").distinct().order_by("-date")
#     if limit > 0:
#         date_list = date_list[offset : offset + limit]  # 최근 limit개의 거래일만 선택
#     else:
#         date_list = date_list[offset : offset + 1]  # limit=0이면 최신 날짜만

#     if not date_list:
#         print("StockOHLCV 데이터가 없습니다.")
#         return 404, {"error": "No StockOHLCV data found."}

#     total_saved = 0
#     print(
#         f"Start date: {date_list[0]['date']}, End date: {date_list[len(date_list)-1]['date']}"
#     )
#     print(f"Processing {len(date_list)} dates for {len(companies)} companies...")
#     # 각 날짜에 대해 처리
#     for date_entry in tqdm(date_list, desc=f"Processing..."):
#         target_date = date_entry["date"]

#         rs_data_all = []  # 모든 날짜, 회사에 대한 RS 데이터
#         analysis_objects = []  # 모든 StockAnalysis 객체

#         # Phase 2 최적화: 배치 처리로 메모리 사용량 감소
#         print(f"날짜 {target_date} 처리 중... (총 {len(companies)}개 회사)")

#         # OHLCV 데이터를 벌크로 로드하여 N+1 쿼리 방지
#         ohlcv_data_dict = optimize_ohlcv_data_loading(area, target_date)
#         print(f"OHLCV 데이터 로드 완료: {len(ohlcv_data_dict)}개 회사")

#         # 회사별로 처리 (배치 방식으로 메모리 최적화)
#         for company_batch in process_companies_in_batches(area, batch_size=50):
#             for company in company_batch:
#                 # 메모리에서 OHLCV 데이터 조회 (DB 쿼리 대신)
#                 ohlcv_data = ohlcv_data_dict.get(company.code, [])

#                 if not ohlcv_data:
#                     print(f"{company.code}에 대한 OHLCV 데이터 없음")
#                     continue

#                 # 해당 날짜의 종가 가져오기 (리스트에서 검색)
#                 latest_ohlcv = None
#                 for ohlcv in ohlcv_data:
#                     if ohlcv.date == target_date:
#                         latest_ohlcv = ohlcv
#                         break
#                 latest_close = latest_ohlcv.close if latest_ohlcv else 0.0

#                 # 이동평균 계산 (QuerySet 대신 리스트 사용을 위해 수정 필요)
#                 # 임시로 기존 방식 유지하되 성능 개선
#                 ohlcv_queryset = StockOHLCV.objects.filter(code=company).order_by(
#                     "date"
#                 )
#                 mas = calculate_moving_averages(ohlcv_queryset, target_date)

#                 # 52주 신고가/신저가 및 날짜 계산
#                 high_low = calculate_52w_high_low(ohlcv_queryset, target_date)

#                 # 50일 신고가/신저가 및 날짜 계산
#                 high_low_50d = calculate_50d_high_low(ohlcv_queryset, target_date)

#                 # RS 점수 계산 (12개월 기준)
#                 rs_score, rsScores = calculate_rs_score(
#                     ohlcv_queryset, target_date, periods["12month"]
#                 )

#                 # ATR(Average True Range) 계산
#                 atr, atrRatio = calculate_atr(ohlcv_queryset, target_date, period=20)

#                 rs_data_all.append(
#                     {
#                         "date": target_date,
#                         "code": company.code,
#                         "name": company.name,
#                         "market": company.market,
#                         "rsScore": rs_score,
#                         "rsScore1m": rsScores[0],
#                         "rsScore3m": rsScores[1],
#                         "rsScore6m": rsScores[2],  # 수정: rsScores[1] → rsScores[2]
#                         "rsScore12m": rsScores[3],
#                     }
#                 )

#                 # 미너비니 트렌드 템플릿 조건 확인
#                 is_minervini_trend = (
#                     latest_close > mas["ma50"]
#                     and latest_close > mas["ma150"]
#                     and latest_close > mas["ma200"]
#                     and mas["ma50"] > mas["ma150"] > mas["ma200"]
#                     and mas["ma200_past"] > 0
#                     and mas["ma200_past"] < mas["ma200"]
#                     and latest_close > high_low["min_52w"] * 1.3
#                     and latest_close <= high_low["max_52w"] * 0.75
#                 )  # rsRank는 랭킹 계산 후 확인

#                 # StockAnalysis 객체 준비
#                 analysis_objects.append(
#                     StockAnalysis(
#                         code=company,
#                         date=target_date,
#                         ma50=mas["ma50"],
#                         ma150=mas["ma150"],
#                         ma200=mas["ma200"],
#                         rsScore=rs_score,
#                         rsScore1m=rsScores[0],
#                         rsScore3m=rsScores[1],
#                         rsScore6m=rsScores[2],
#                         rsScore12m=rsScores[3],
#                         rsRank=0.0,
#                         rsRank1m=0.0,
#                         rsRank3m=0.0,
#                         rsRank6m=0.0,
#                         rsRank12m=0.0,
#                         max_52w=high_low["max_52w"],
#                         min_52w=high_low["min_52w"],
#                         max_52w_date=high_low["max_52w_date"],
#                         min_52w_date=high_low["min_52w_date"],
#                         max_50d=high_low_50d["max_50d"],
#                         min_50d=high_low_50d["min_50d"],
#                         max_50d_date=high_low_50d["max_50d_date"],
#                         min_50d_date=high_low_50d["min_50d_date"],
#                         atr=atr,
#                         atrRatio=atrRatio,
#                         is_minervini_trend=is_minervini_trend,
#                     )
#                 )

#         # Phase 2 최적화: 벌크 랭킹 계산으로 성능 향상
#         print("RS 랭킹 계산 중...")
#         rs_df = bulk_calculate_rs_rankings(rs_data_all, date_list)

#         # StockAnalysis 객체에 랭킹 반영
#         for obj in tqdm(
#             analysis_objects, desc="Updating rankings and MTT", leave=False
#         ):
#             row = rs_df[
#                 (rs_df["code"] == obj.code.code) & (rs_df["date"] == obj.date)
#             ].iloc[0]
#             obj.rsRank = row["rsScore_RS"] if row["rsScore"] != -1 else 0.0
#             obj.rsRank1m = row["rsScore1m_RS"] if row["rsScore1m"] != -1 else 0.0
#             obj.rsRank3m = row["rsScore3m_RS"] if row["rsScore3m"] != -1 else 0.0
#             obj.rsRank6m = row["rsScore6m_RS"] if row["rsScore6m"] != -1 else 0.0
#             obj.rsRank12m = row["rsScore12m_RS"] if row["rsScore12m"] != -1 else 0.0
#             if obj.is_minervini_trend:
#                 obj.is_minervini_trend = obj.is_minervini_trend and obj.rsRank >= 70

#         # 기존 StockAnalysis 레코드 삭제
#         StockAnalysis.objects.filter(date=target_date).delete()

#         # Bulk create
#         try:
#             with transaction.atomic():
#                 StockAnalysis.objects.bulk_create(analysis_objects)
#             total_saved += len(analysis_objects)
#         except Exception as e:
#             traceback.print_exc()
#             return 500, {"error": f"주식 분석 데이터 저장 실패: {str(e)}"}

#     return {
#         "status": "OK",
#         "message": "주식 분석 데이터가 성공적으로 저장되었습니다.\n"
#         + f"{date_list[0]['date']}, Last date: {date_list[len(date_list)-1]['date']}",
#         "count_saved": total_saved,
#     }


def calculate_stock_analysis(
    request, area: str = "KR", offset: int = 0, limit: int = 0
):
    """
    주식 분석 데이터를 계산하여 StockAnalysis 테이블에 저장합니다.
    최근 거래일부터 지정된 `limit`만큼의 거래일에 대해 모든 회사의 이동평균, 52주 신고가/신저가, RS 점수,
    미너비니 트렌드 조건을 계산합니다. 휴일(예: 주말)은 StockOHLCV 데이터가 없으므로 자동으로 제외됩니다.

    Args:
        request: Ninja API 요청 객체.
        area (str): 주식 시장 지역 ("KR" - 한국, "US" - 미국). 기본값: "KR".
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
    if area == "KR":
        companies = Company.objects.filter(market__in=["KOSPI", "KOSDAQ"])
    elif area == "US":
        companies = Company.objects.filter(market__in=["NASDAQ", "NYSE"])
    else:
        return 400, {"error": "Invalid area specified. Use 'KR' or 'US'."}

    print(f"Total companies: {len(companies)}")

    if len(companies) == 0:
        return 404, {"error": "No companies found in the database."}

    # 기간 정의 (거래일 기준)
    periods = {
        "12month": 252,  # 1년
        "6month": 126,  # 6개월
        "3month": 63,  # 3개월
        "1month": 21,  # 1개월
    }

    # StockOHLCV의 고유 날짜 목록 가져오기 (최근 순, limit 적용)
    date_list = StockOHLCV.objects.values("date").distinct().order_by("-date")
    if limit > 0:
        date_list = date_list[offset : offset + limit]  # 최근 limit개의 거래일만 선택
    else:
        date_list = date_list[offset : offset + 1]  # limit=0이면 최신 날짜만

    if not date_list:
        print("StockOHLCV 데이터가 없습니다.")
        return 404, {"error": "No StockOHLCV data found."}

    total_saved = 0
    print(
        f"Start date: {date_list[0]['date']}, End date: {date_list[len(date_list)-1]['date']}"
    )
    print(f"Processing {len(date_list)} dates for {len(companies)} companies...")
    # 각 날짜에 대해 처리
    for date_entry in tqdm(date_list, desc=f"Processing..."):
        target_date = date_entry["date"]

        rs_data_all = []  # 모든 날짜, 회사에 대한 RS 데이터
        analysis_objects = []  # 모든 StockAnalysis 객체

        # 회사별로 처리
        for company in tqdm(companies, desc=f"Date: {date_entry['date']}", leave=False):

            # 회사별 OHLCV 데이터 가져오기
            ohlcv_data = StockOHLCV.objects.filter(code=company).order_by("date")

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

            # 50일 신고가/신저가 및 날짜 계산
            high_low_50d = calculate_50d_high_low(ohlcv_data, target_date)

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
            rs_score, rsScores = calculate_rs_score(
                ohlcv_data, target_date, periods["12month"]
            )

            # ATR(Average True Range) 계산
            atr, atrRatio = calculate_atr(ohlcv_data, target_date, period=20)
            # print(f"ATR for {company.code} on {target_date}: {atr}")

            # HTF(High Tight Flag) 패턴 계산
            htf_result = calculate_htf_pattern(ohlcv_data, target_date)
            htf_8week_gain = htf_result["gain_percent"]
            htf_max_pullback = htf_result["pullback_percent"]
            htf_pattern_detected = htf_result["detected"]
            htf_pattern_start_date = htf_result["start_date"]
            htf_pattern_peak_date = htf_result["peak_date"]
            htf_current_status = htf_result["status"]

            rs_data_all.append(
                {
                    "date": target_date,
                    "code": company.code,
                    "name": company.name,
                    "market": company.market,
                    "rsScore": rs_score,
                    "rsScore1m": rsScores[0],
                    "rsScore3m": rsScores[1],
                    "rsScore6m": rsScores[1],
                    "rsScore12m": rsScores[3],
                }
            )

            # 미너비니 트렌드 템플릿 조건 확인
            is_minervini_trend = (
                latest_close > mas["ma50"]
                and latest_close > mas["ma150"]
                and latest_close > mas["ma200"]
                and mas["ma50"] > mas["ma150"] > mas["ma200"]
                and mas["ma200_past"] > 0
                and mas["ma200_past"] < mas["ma200"]
                and latest_close > high_low["min_52w"] * 1.3
                and latest_close <= high_low["max_52w"] * 0.75
            )  # rsRank는 랭킹 계산 후 확인

            # StockAnalysis 객체 준비 (HTF 데이터 포함)
            analysis_objects.append(
                StockAnalysis(
                    code=company,
                    date=target_date,
                    ma50=mas["ma50"],
                    ma150=mas["ma150"],
                    ma200=mas["ma200"],
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
                    max_52w=high_low["max_52w"],
                    min_52w=high_low["min_52w"],
                    max_52w_date=high_low["max_52w_date"],
                    min_52w_date=high_low["min_52w_date"],
                    max_50d=high_low_50d["max_50d"],
                    min_50d=high_low_50d["min_50d"],
                    max_50d_date=high_low_50d["max_50d_date"],
                    min_50d_date=high_low_50d["min_50d_date"],
                    atr=atr,
                    atrRatio=atrRatio,
                    is_minervini_trend=is_minervini_trend,
                    # HTF 패턴 데이터 추가
                    htf_8week_gain=htf_8week_gain,
                    htf_max_pullback=htf_max_pullback,
                    htf_pattern_detected=htf_pattern_detected,
                    htf_pattern_start_date=htf_pattern_start_date,
                    htf_pattern_peak_date=htf_pattern_peak_date,
                    htf_current_status=htf_current_status,
                )
            )

        # 날짜별로 랭킹 계산
        rs_df = pd.DataFrame(rs_data_all)
        for date in tqdm(
            [entry["date"] for entry in date_list],
            desc=f"Calculating rankings...",
            leave=False,
        ):
            date_df = rs_df[rs_df["date"] == date]
            for market in date_df["market"].unique():
                market_df = date_df[date_df["market"] == market]
                if market_df.empty:
                    continue
                for period in [
                    "rsScore1m",
                    "rsScore3m",
                    "rsScore6m",
                    "rsScore12m",
                    "rsScore",
                ]:
                    rank_values = market_df[period].rank(
                        ascending=True, na_option="bottom"
                    )
                    rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
                    rs_df.loc[market_df.index, f"{period}_Rank"] = rank_values
                    rs_df.loc[market_df.index, f"{period}_RS"] = rs_values
                rank_values = market_df["rsScore"].rank(
                    ascending=True, na_option="bottom"
                )
                rs_values = (rank_values * 98 / len(market_df)).apply(np.int64) + 1
                rs_df.loc[market_df.index, f"rsScore_Rank"] = rank_values
                rs_df.loc[market_df.index, f"rsScore_RS"] = rs_values

        # StockAnalysis 객체에 랭킹 반영
        for obj in tqdm(
            analysis_objects, desc="Updating rankings and MTT", leave=False
        ):
            row = rs_df[
                (rs_df["code"] == obj.code.code) & (rs_df["date"] == obj.date)
            ].iloc[0]
            obj.rsRank = row["rsScore_RS"] if row["rsScore"] != -1 else 0.0
            obj.rsRank1m = row["rsScore1m_RS"] if row["rsScore1m"] != -1 else 0.0
            obj.rsRank3m = row["rsScore3m_RS"] if row["rsScore3m"] != -1 else 0.0
            obj.rsRank6m = row["rsScore6m_RS"] if row["rsScore6m"] != -1 else 0.0
            obj.rsRank12m = row["rsScore12m_RS"] if row["rsScore12m"] != -1 else 0.0
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
        "message": "주식 분석 데이터가 성공적으로 저장되었습니다.\n"
        + f"{date_list[0]['date']}, Last date: {date_list[len(date_list)-1]['date']}",
        "count_saved": total_saved,
    }


# OpenDART에서 재무제표 데이터를 가져와 4분기 값을 조정하고 피벗 테이블로 반환합니다.
@data_router.post(
    "/getAndSave_stock_dartData",
    response={200: SuccessResponse, 404: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("DART Financial Data Sync")
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
                company = Company.objects.get(
                    code=code, market__in=["KOSDAQ", "KONEX", "KOSPI"]
                )
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
            reprt_list = ["11013", "11012", "11014", "11011"]
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

                        if sum(df["fs_nm"] == "연결재무제표") > 0:
                            df = df.loc[df["fs_nm"] == "연결재무제표"]
                        else:
                            df = df.loc[df["fs_nm"] == "재무제표"]

                        # thstrm_amount의 쉼표 제거 및 정수 변환
                        if "thstrm_amount" in df.columns:
                            df["thstrm_amount"] = (
                                df["thstrm_amount"]
                                .astype(str)
                                .str.replace(",", "")
                                .replace("-", "0")
                                .astype(int)
                            )
                        else:
                            # print(f"{code}, {year}년, 보고서 {reprt}에 thstrm_amount 열이 없습니다.")
                            continue

                        # '년도'와 '분기' 필드 추가
                        df["year"] = df["bsns_year"].astype(str)

                        def get_quarter(reprt_code):
                            mapping = {
                                "11011": "4Q",  # 사업보고서
                                "11012": "2Q",
                                "11013": "1Q",
                                "11014": "3Q",
                            }
                            return mapping.get(reprt_code, "Unknown")

                        df["quarter"] = df["reprt_code"].apply(get_quarter)

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
                if "4Q" in df_year["quarter"].values:
                    # 1Q, 2Q, 3Q 데이터 합계 계산
                    q123 = df_year[df_year["quarter"].isin(["1Q", "2Q", "3Q"])][
                        ["sj_nm", "account_nm", "thstrm_amount"]
                    ]
                    q123_sum = (
                        q123.groupby(["sj_nm", "account_nm"])["thstrm_amount"]
                        .sum()
                        .reset_index()
                    )
                    q123_sum.rename(
                        columns={"thstrm_amount": "q123_total"}, inplace=True
                    )
                    # print(q123_sum[['sj_nm', 'account_nm', 'q123_total']])

                    # 4Q 데이터와 합계 병합
                    q4 = df_year[df_year["quarter"] == "4Q"][
                        ["sj_nm", "account_nm", "thstrm_amount"]
                    ]
                    q4 = q4.merge(q123_sum, on=["sj_nm", "account_nm"], how="left")
                    q4["q123_total"] = q4["q123_total"].fillna(0).astype(int)

                    # 재무상태표와 손익계산서 분리 > 1Q+2Q+3Q 합산 데이터가 아님 >> 그대로 유지
                    q4_jm = q4.loc[
                        q4["sj_nm"] == "재무상태표",
                        ["sj_nm", "account_nm", "thstrm_amount"],
                    ]

                    # 손익계산서에서 1Q+2Q+3Q 합계를 뺌
                    q4_si = q4.loc[
                        q4["sj_nm"] == "손익계산서",
                        ["sj_nm", "account_nm", "thstrm_amount", "q123_total"],
                    ]
                    q4_si["thstrm_amount"] = (
                        q4_si["thstrm_amount"] - q4_si["q123_total"]
                    )
                    q4_si = q4_si.drop(columns=["q123_total"])

                    # 재무상태표와 손익계산서 데이터를 합침
                    q4_merged = pd.concat([q4_jm, q4_si], ignore_index=True)
                    q4_merged["year"] = str(year)  # 연도 설정
                    q4_merged["quarter"] = "4Q"  # 4Q로 분기 설정

                    # 'year', 'quarter', 'sj_nm', 'account_nm', 'thstrm_amount' 필드만 남기기
                    df_year = df_year[
                        ["year", "quarter", "sj_nm", "account_nm", "thstrm_amount"]
                    ]
                    # 4Q 데이터는 지우고
                    df_year = df_year[df_year["quarter"] != "4Q"]  # 4Q 데이터 제거
                    # 계산된 4Q 데이터를 업데이트
                    df_year = pd.concat([df_year, q4_merged], ignore_index=True)
                else:
                    df_year = df_year[
                        ["year", "quarter", "sj_nm", "account_nm", "thstrm_amount"]
                    ]

                # 데이터 합치기
                all_dfs = pd.concat([all_dfs, df_year], ignore_index=True)
                # print(all_dfs)

            # 모든 연도 데이터 합침
            # df = pd.concat(all_dfs, ignore_index=True)
            # print(all_dfs)

            # StockFinancialStatement 모델에 벌크 저장 (성능 최적화)
            if not all_dfs.empty:
                try:
                    with transaction.atomic():
                        # 기존 데이터 삭제 (회사별)
                        StockFinancialStatement.objects.filter(code=company).delete()

                        # 벌크 생성을 위한 객체 리스트 준비
                        financial_objects = []
                        for _, row in all_dfs.iterrows():
                            financial_objects.append(
                                StockFinancialStatement(
                                    code=company,
                                    year=row["year"],
                                    quarter=row["quarter"],
                                    statement_type=row["sj_nm"],
                                    account_name=row["account_nm"],
                                    amount=row["thstrm_amount"],
                                )
                            )

                        # 벌크 생성 (update_or_create 대신 bulk_create 사용)
                        StockFinancialStatement.objects.bulk_create(
                            financial_objects, batch_size=500
                        )
                        print(
                            f"{company.code}: {len(financial_objects)}개 재무 데이터 저장 완료"
                        )

                except Exception as e:
                    traceback.print_exc()
                    print(f"{company.code} 데이터 저장 오류: {str(e)}")

        # 성공 응답
        return {
            "status": "OK",
            "message": "success",
            "count_saved": len(all_dfs),
        }

    except Exception as e:
        traceback.print_exc()
        return 500, ErrorResponse(
            status="error", message=f"DART 데이터 조회 실패: {str(e)}"
        )


# =====================================================================
# HTF (High Tight Flag) 패턴 관련 API 엔드포인트

from django.db.models import Max

# HTF 관련 유틸리티 함수들 (통합된 StockAnalysis 데이터 사용)


def get_htf_stocks_from_analysis(
    area="KR", min_gain=100.0, max_pullback=25.0, limit=100
):
    """
    StockAnalysis 테이블에서 HTF 조건을 만족하는 종목 조회 (통합 버전)

    Args:
        area: 시장 지역 ("KR" 또는 "US")
        min_gain: 최소 상승률
        max_pullback: 최대 조정폭
        limit: 결과 제한 수

    Returns:
        HTF 조건을 만족하는 종목 리스트
    """
    try:
        if area == "KR":
            markets = ["KOSPI", "KOSDAQ"]
        elif area == "US":
            markets = ["NASDAQ", "NYSE"]
        else:
            raise ValueError("지원하지 않는 시장입니다. 'KR' 또는 'US'를 선택하세요.")

        # StockAnalysis에서 HTF 패턴 종목 조회 (최신 데이터만)
        latest_date_subquery = StockAnalysis.objects.aggregate(max_date=Max("date"))[
            "max_date"
        ]

        if not latest_date_subquery:
            return []

        htf_stocks = (
            StockAnalysis.objects.filter(
                date=latest_date_subquery,
                code__market__in=markets,
                htf_pattern_detected=True,
                htf_8week_gain__gte=min_gain,
                htf_max_pullback__lte=max_pullback,
            )
            .select_related("code")
            .order_by("-htf_8week_gain")[:limit]
        )

        result = []
        for analysis in htf_stocks:
            result.append(
                {
                    "code": analysis.code.code,
                    "name": analysis.code.name,
                    "market": analysis.code.market,
                    "sector": analysis.code.sector,
                    "industry": analysis.code.industry,
                    "analysis_date": (
                        analysis.date.isoformat() if analysis.date else None
                    ),
                    "htf_8week_gain": analysis.htf_8week_gain,
                    "htf_max_pullback": analysis.htf_max_pullback,
                    "htf_pattern_start_date": (
                        analysis.htf_pattern_start_date.isoformat()
                        if analysis.htf_pattern_start_date
                        else None
                    ),
                    "htf_pattern_peak_date": (
                        analysis.htf_pattern_peak_date.isoformat()
                        if analysis.htf_pattern_peak_date
                        else None
                    ),
                    "htf_current_status": analysis.htf_current_status,
                    "rs_rank": analysis.rsRank,
                    "is_minervini_trend": analysis.is_minervini_trend,
                }
            )

        return result

    except Exception as e:
        logger.error(f"HTF 종목 조회 중 오류: {str(e)}")
        return []


def get_htf_analysis_from_analysis(stock_code):
    """
    StockAnalysis 테이블에서 특정 종목의 HTF 분석 상세 정보 조회 (통합 버전)

    Args:
        stock_code: 종목 코드

    Returns:
        HTF 상세 분석 정보
    """
    try:
        # 최신 분석 데이터 조회
        latest_analysis = (
            StockAnalysis.objects.filter(
                code__code=stock_code, htf_pattern_detected=True
            )
            .select_related("code")
            .order_by("-date")
            .first()
        )

        if not latest_analysis:
            return {"error": f"종목 {stock_code}의 HTF 분석 데이터가 없습니다"}

        # HTF 패턴 기간 OHLCV 데이터 조회
        pattern_data = []
        if latest_analysis.htf_pattern_start_date:
            ohlcv_data = (
                StockOHLCV.objects.filter(
                    code__code=stock_code,
                    date__gte=latest_analysis.htf_pattern_start_date,
                    date__lte=latest_analysis.date,
                )
                .order_by("date")
                .values("date", "open", "high", "low", "close", "volume")
            )

            pattern_data = list(ohlcv_data)

        return {
            "stock_info": {
                "code": latest_analysis.code.code,
                "name": latest_analysis.code.name,
                "market": latest_analysis.code.market,
                "sector": latest_analysis.code.sector,
                "industry": latest_analysis.code.industry,
            },
            "htf_analysis": {
                "analysis_date": latest_analysis.date,
                "htf_8week_gain": latest_analysis.htf_8week_gain,
                "htf_max_pullback": latest_analysis.htf_max_pullback,
                "htf_pattern_start_date": latest_analysis.htf_pattern_start_date,
                "htf_pattern_peak_date": latest_analysis.htf_pattern_peak_date,
                "htf_current_status": latest_analysis.htf_current_status,
                "rs_rank": latest_analysis.rsRank,
                "is_minervini_trend": latest_analysis.is_minervini_trend,
            },
            "pattern_data": pattern_data,
        }

    except Exception as e:
        logger.error(f"HTF 상세 분석 조회 중 오류 ({stock_code}): {str(e)}")
        return {"error": str(e)}


from .schemas import HTFStocksResponse, HTFAnalysisResponse, HTFCalculationResponse


@data_router.get(
    "/htf-stocks/",
    response={200: HTFStocksResponse, 400: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("HTF Stocks List")
def get_htf_stocks_api(
    request,
    area: str = "KR",
    min_gain: float = 100.0,
    max_pullback: float = 25.0,
    limit: int = 100,
):
    """
    High Tight Flag 패턴 조건을 만족하는 종목 리스트 조회

    Args:
        min_gain: 최소 상승률 (기본값: 100%)
        max_pullback: 최대 조정폭 (기본값: 25%)
        limit: 결과 제한 수 (기본값: 100)

    Returns:
        HTF 패턴 종목 리스트
    """
    try:
        # 파라미터 유효성 검사
        if min_gain < 0 or max_pullback < 0:
            return 400, ErrorResponse(
                status="ERROR", message="상승률과 조정폭은 0 이상이어야 합니다"
            )

        if limit < 1 or limit > 1000:
            return 400, ErrorResponse(
                status="ERROR", message="limit은 1~1000 범위여야 합니다"
            )

        # HTF 종목 조회 (통합된 데이터 사용)
        htf_stocks = get_htf_stocks_from_analysis(area, min_gain, max_pullback, limit)

        return {"status": "OK", "data": htf_stocks, "total_count": len(htf_stocks)}

    except Exception as e:
        return handle_api_error("HTF 종목 조회", e)


@data_router.get(
    "/htf-analysis/{stock_code}/",
    response={200: HTFAnalysisResponse, 400: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("HTF Analysis Detail")
def get_htf_analysis_api(request, stock_code: str):
    """
    특정 종목의 HTF 패턴 상세 분석 정보 조회

    Args:
        stock_code: 종목 코드

    Returns:
        HTF 패턴 상세 분석 정보
    """
    try:
        # 종목 코드 유효성 검사
        if not stock_code or len(stock_code) < 3:
            return 400, ErrorResponse(
                status="ERROR", message="올바른 종목 코드를 입력해주세요"
            )

        # HTF 상세 분석 조회 (통합된 데이터 사용)
        analysis_detail = get_htf_analysis_from_analysis(stock_code.upper())

        if "error" in analysis_detail:
            return 400, ErrorResponse(status="ERROR", message=analysis_detail["error"])

        return {"status": "OK", "data": analysis_detail}

    except Exception as e:
        return handle_api_error("HTF 상세 분석", e)


@data_router.post(
    "/calculate-htf-patterns/",
    response={200: HTFCalculationResponse, 400: ErrorResponse, 500: ErrorResponse},
)
@performance_monitor("HTF Pattern Calculation")
def calculate_htf_patterns_api(
    request,
    area: str = "KR",
    stock_codes: Optional[List[str]] = None,
    min_gain_percent: float = 100.0,
    max_pullback_percent: float = 25.0,
    batch_size: int = 50,
):
    """
    HTF 패턴 계산 및 데이터베이스 업데이트 (배치 처리)

    Args:
        stock_codes: 계산할 종목 코드 리스트 (None이면 전체)
        min_gain_percent: 최소 상승률 (기본값: 100%)
        max_pullback_percent: 최대 조정폭 (기본값: 25%)
        batch_size: 배치 크기 (기본값: 50)

    Returns:
        배치 처리 결과
    """
    try:
        # 파라미터 유효성 검사
        if min_gain_percent < 0 or max_pullback_percent < 0:
            return 400, ErrorResponse(
                status="ERROR", message="상승률과 조정폭은 0 이상이어야 합니다"
            )

        if batch_size < 1 or batch_size > 500:
            return 400, ErrorResponse(
                status="ERROR", message="batch_size는 1~500 범위여야 합니다"
            )

        # 통합된 주식 분석 API 호출 (HTF 패턴 포함)
        # 최신 날짜만 계산하여 HTF 데이터 업데이트
        analysis_result = calculate_stock_analysis(
            request=request, area=area, offset=0, limit=0  # 최신 날짜만 계산
        )

        if isinstance(analysis_result, tuple):  # 오류 발생 시
            status_code, error_dict = analysis_result
            return status_code, ErrorResponse(
                status="ERROR",
                message=error_dict.get("error", "HTF 패턴 계산 중 오류가 발생했습니다"),
            )

        # 처리된 종목 수 계산
        result = {
            "total": analysis_result.get("count_saved", 0),
            "success": analysis_result.get("count_saved", 0),
            "failed": 0,
            "success_rate": 100.0,
            "failed_stocks": [],
            "message": f"HTF 패턴을 포함한 주식 분석이 완료되었습니다. {analysis_result.get('count_saved', 0)}개 종목 처리됨",
        }

        if "error" in result:
            return 500, ErrorResponse(status="ERROR", message=result["error"])

        return {"status": "OK", "data": result}

    except Exception as e:
        return handle_api_error("HTF 패턴 계산", e)
