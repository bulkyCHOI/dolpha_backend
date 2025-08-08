"""
High Tight Flag (HTF) 패턴 분석 모듈
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.db import transaction
from django.db.models import Q, F
from myweb.models import StockOHLCV, StockAnalysis, Company
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


class HTFPatternAnalyzer:
    """High Tight Flag 패턴 분석기"""

    def __init__(
        self,
        min_gain_percent: float = 60.0,
        max_pullback_percent: float = 25.0,
        analysis_period_days: int = 56,
    ):
        """
        HTF 패턴 분석기 초기화

        Args:
            min_gain_percent: 최소 상승률 (기본 100%)
            max_pullback_percent: 최대 조정폭 (기본 25%)
            analysis_period_days: 분석 기간 (기본 56일 = 8주)
        """
        self.min_gain_percent = min_gain_percent
        self.max_pullback_percent = max_pullback_percent
        self.analysis_period_days = analysis_period_days

    def calculate_htf_pattern(self, stock_code: str) -> Dict:
        """
        특정 종목의 HTF 패턴 계산

        Args:
            stock_code: 종목 코드

        Returns:
            HTF 패턴 분석 결과 딕셔너리
        """
        try:
            # OHLCV 데이터 조회 (최근 1년)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)

            ohlcv_data = (
                StockOHLCV.objects.filter(
                    code__code=stock_code, date__gte=start_date, date__lte=end_date
                )
                .order_by("date")
                .values("date", "open", "high", "low", "close", "volume")
            )

            if not ohlcv_data:
                return {"error": f"종목 {stock_code}의 데이터가 없습니다"}

            # DataFrame으로 변환
            df = pd.DataFrame(ohlcv_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            if len(df) < self.analysis_period_days:
                return {
                    "error": f"데이터가 부족합니다. 필요: {self.analysis_period_days}일, 보유: {len(df)}일"
                }

            # HTF 패턴 분석
            htf_result = self._analyze_htf_pattern(df)

            # 최신 날짜 기준 결과 반환
            latest_date = df["date"].max().date()
            latest_result = htf_result.get(latest_date)

            if not latest_result:
                return {"pattern_detected": False}

            return {
                "stock_code": stock_code,
                "analysis_date": latest_date,
                "pattern_detected": latest_result["pattern_detected"],
                "htf_8week_gain": latest_result["htf_8week_gain"],
                "htf_max_pullback": latest_result["htf_max_pullback"],
                "htf_pattern_start_date": latest_result["htf_pattern_start_date"],
                "htf_pattern_peak_date": latest_result["htf_pattern_peak_date"],
                "htf_current_status": latest_result["htf_current_status"],
            }

        except Exception as e:
            logger.error(f"HTF 패턴 계산 중 오류 ({stock_code}): {str(e)}")
            return {"error": str(e)}

    def _analyze_htf_pattern(self, df: pd.DataFrame) -> Dict:
        """
        HTF 패턴 분석 핵심 로직

        Args:
            df: OHLCV 데이터프레임

        Returns:
            각 날짜별 HTF 분석 결과
        """
        results = {}

        # 슬라이딩 윈도우로 분석
        for i in range(self.analysis_period_days, len(df)):
            current_date = df.iloc[i]["date"].date()
            current_price = df.iloc[i]["close"]

            # 과거 8주 데이터
            window_data = df.iloc[i - self.analysis_period_days : i + 1]

            # HTF 패턴 분석
            pattern_result = self._check_htf_pattern(window_data, current_price)

            results[current_date] = {
                "pattern_detected": pattern_result["detected"],
                "htf_8week_gain": pattern_result["gain_percent"],
                "htf_max_pullback": pattern_result["pullback_percent"],
                "htf_pattern_start_date": pattern_result["start_date"],
                "htf_pattern_peak_date": pattern_result["peak_date"],
                "htf_current_status": pattern_result["status"],
            }

        return results

    def _check_htf_pattern(
        self, window_data: pd.DataFrame, current_price: float
    ) -> Dict:
        """
        HTF 패턴 조건 확인

        Args:
            window_data: 8주 윈도우 데이터
            current_price: 현재 가격

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
                return self._default_result()

            max_idx = after_min["high"].idxmax()
            max_price = after_min.loc[max_idx, "high"]
            max_date = after_min.loc[max_idx, "date"].date()

            # 3. 상승률 계산
            if min_price == 0:
                return self._default_result()  # 0으로 나누기 방지
            gain_percent = ((max_price - min_price) / min_price) * 100

            # 4. 100% 이상 상승 조건 확인
            if gain_percent < self.min_gain_percent:
                return self._default_result()

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
                    if current_price < max_price * 0.98:  # 2% 이상 하락
                        current_status = "pullback"
                    elif current_price > max_price:  # 신고가 돌파
                        current_status = "breakout"
                    else:
                        current_status = "pullback"

            # 6. 조정폭 조건 확인 (25% 이내)
            if pullback_percent > self.max_pullback_percent:
                return self._default_result()

            # 7. HTF 패턴 확인
            pattern_detected = (
                gain_percent >= self.min_gain_percent
                and pullback_percent <= self.max_pullback_percent
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
            return self._default_result()

    def _default_result(self) -> Dict:
        """기본 결과 반환"""
        return {
            "detected": False,
            "gain_percent": 0.0,
            "pullback_percent": 0.0,
            "start_date": None,
            "peak_date": None,
            "status": "none",
        }

    def update_stock_analysis(self, stock_code: str, htf_data: Dict) -> bool:
        """
        StockAnalysis 테이블에 HTF 데이터 업데이트

        Args:
            stock_code: 종목 코드
            htf_data: HTF 분석 데이터

        Returns:
            업데이트 성공 여부
        """
        try:
            analysis_date = htf_data["analysis_date"]

            # 기존 분석 데이터 조회 또는 생성
            stock_code = Company.objects.get(code=stock_code)
            analysis, created = StockAnalysis.objects.get_or_create(
                code=stock_code,
                date=analysis_date,
                defaults={
                    "htf_8week_gain": htf_data["htf_8week_gain"],
                    "htf_max_pullback": htf_data["htf_max_pullback"],
                    "htf_pattern_detected": htf_data["pattern_detected"],
                    "htf_pattern_start_date": htf_data["htf_pattern_start_date"],
                    "htf_pattern_peak_date": htf_data["htf_pattern_peak_date"],
                    "htf_current_status": htf_data["htf_current_status"],
                },
            )

            if not created:
                # 기존 데이터 업데이트
                analysis.htf_8week_gain = htf_data["htf_8week_gain"]
                analysis.htf_max_pullback = htf_data["htf_max_pullback"]
                analysis.htf_pattern_detected = htf_data["pattern_detected"]
                analysis.htf_pattern_start_date = htf_data["htf_pattern_start_date"]
                analysis.htf_pattern_peak_date = htf_data["htf_pattern_peak_date"]
                analysis.htf_current_status = htf_data["htf_current_status"]
                print("analysis:", analysis)  # 디버깅용 출력
                analysis.save()

            return True

        except Exception as e:
            traceback.print_exc()
            logger.error(f"StockAnalysis 업데이트 중 오류 ({stock_code}): {str(e)}")
            return False

    def batch_calculate_htf_patterns(
        self, area: str = "KR", stock_codes: List[str] = None, batch_size: int = 100
    ) -> Dict:
        """
        여러 종목의 HTF 패턴 배치 계산

        Args:
            stock_codes: 계산할 종목 코드 리스트 (None이면 전체)
            batch_size: 배치 크기

        Returns:
            배치 처리 결과
        """
        try:
            # 종목 코드 조회
            if area == "KR":
                markets = ["KOSPI", "KOSDAQ"]
            elif area == "US":
                markets = ["NASDAQ", "NYSE"]
            else:
                raise ValueError(
                    "지원하지 않는 시장입니다. 'KR' 또는 'US'를 선택하세요."
                )

            if stock_codes is None:
                # 전체 종목 조회(우선 한국주식만)
                stock_codes = list(
                    Company.objects.filter(market__in=markets).values_list(
                        "code", flat=True
                    )
                )
            else:
                # 입력된 종목 코드 필터링
                stock_codes = [
                    code
                    for code in stock_codes
                    if Company.objects.filter(code=code).exists()
                ]

            total_stocks = len(stock_codes)
            success_count = 0
            failed_count = 0
            failed_stocks = []

            logger.info(f"HTF 패턴 배치 계산 시작: {total_stocks}개 종목")

            # 배치 단위로 처리
            for i in tqdm(range(0, total_stocks, batch_size), desc="HTF 배치 처리"):
                batch_codes = stock_codes[i : i + batch_size]

                with transaction.atomic():
                    for code in batch_codes:
                        try:
                            # HTF 패턴 계산
                            htf_result = self.calculate_htf_pattern(code)
                            print(htf_result)  # 디버깅용 출력

                            if "error" in htf_result:
                                failed_count += 1
                                failed_stocks.append(code)
                                continue

                            # 데이터베이스 업데이트
                            if self.update_stock_analysis(code, htf_result):
                                success_count += 1
                            else:
                                failed_count += 1
                                failed_stocks.append(code)

                        except Exception as e:
                            logger.error(f"종목 {code} 처리 중 오류: {str(e)}")
                            failed_count += 1
                            failed_stocks.append(code)

            result = {
                "total": total_stocks,
                "success": success_count,
                "failed": failed_count,
                "success_rate": round((success_count / total_stocks) * 100, 2),
                "failed_stocks": failed_stocks[:10],  # 실패한 종목 최대 10개만
            }

            logger.info(f"HTF 패턴 배치 계산 완료: {result}")
            return result

        except Exception as e:
            logger.error(f"HTF 패턴 배치 계산 중 오류: {str(e)}")
            return {"error": str(e)}


def get_htf_stocks(
    area: str = "KR",
    min_gain: float = 100.0,
    max_pullback: float = 25.0,
    limit: int = 100,
) -> List[Dict]:
    """
    HTF 조건을 만족하는 종목 리스트 조회

    Args:
        min_gain: 최소 상승률
        max_pullback: 최대 조정폭
        limit: 결과 제한 수

    Returns:
        HTF 종목 리스트
    """
    try:
        if area == "KR":
            markets = ["KOSPI", "KOSDAQ"]
        elif area == "US":
            markets = ["NASDAQ", "NYSE"]
        else:
            raise ValueError("지원하지 않는 시장입니다. 'KR' 또는 'US'를 선택하세요.")

        # StockAnalysis에서 HTF 패턴 종목 조회
        htf_stocks = (
            StockAnalysis.objects.filter(
                code__market__in=markets,
                htf_pattern_detected=True,
                htf_8week_gain__gte=min_gain,
                htf_max_pullback__lte=max_pullback,
            )
            .select_related("code")
            .order_by("-htf_8week_gain")[:limit]
        )
        print(htf_stocks)  # 디버깅용 출력

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


def get_htf_analysis_detail(stock_code: str) -> Dict:
    """
    특정 종목의 HTF 분석 상세 정보 조회

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
