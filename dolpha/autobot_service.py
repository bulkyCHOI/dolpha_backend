"""
Autobot API 연동 서비스
"""

import requests
from typing import List, Dict, Optional
from django.conf import settings


class AutobotService:
    """Autobot FastAPI 서버와 통신하는 서비스"""
    
    def __init__(self):
        self.base_url = getattr(settings, 'AUTOBOT_API_URL', 'http://localhost:8080')
        self.timeout = 10

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """HTTP 요청을 수행합니다."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Autobot API 요청 실패: {e}")
            return None

    def get_trading_summary(self) -> Optional[List[Dict]]:
        """모든 매매복기 데이터를 조회합니다."""
        return self._make_request('GET', '/trading-summary')

    def get_trading_summary_by_stock(self, stock_code: str) -> Optional[Dict]:
        """특정 종목의 매매복기 데이터를 조회합니다."""
        return self._make_request('GET', f'/trading-summary/{stock_code}')

    def get_trading_summary_stats(self) -> Optional[Dict]:
        """매매복기 통계를 조회합니다."""
        return self._make_request('GET', '/trading-summary/stats')

    def check_trading_summary_file(self) -> Optional[Dict]:
        """매매복기 CSV 파일 존재 여부를 확인합니다."""
        return self._make_request('GET', '/trading-summary-file-exists')

    def is_available(self) -> bool:
        """Autobot 서버가 사용 가능한지 확인합니다."""
        try:
            response = self._make_request('GET', '/health')
            return response is not None
        except Exception:
            return False


# 싱글톤 인스턴스
autobot_service = AutobotService()