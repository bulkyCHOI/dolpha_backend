"""급등테마주 자동매매 전략 패키지.

구성:
  config.py       — 전략 파라미터 상수
  toss_client.py  — 토스증권 '지금 뜨는 산업' 크롤러
  detector.py     — 급등 테마 판정
  leader.py       — 테마 내 1등 종목(주도주) 선정
  entry.py        — 1분봉 눌림목 → 전고점 돌파 → 외국인 매수세 진입 판정
  entry_chart.py  — 판정 이력을 1분봉 차트 좌표(전고점·눌림 구간)로 재구성
  scanner.py      — 5분 주기 스캔 오케스트레이션 (스냅샷 저장 + 후보 등록)
  timeline.py     — 09:00~15:30 급등 테마 타임라인 조립
  positions.py    — 자동매매 현황(보유 포지션 + 대기 후보) 조립
"""

from .detector import detect_surge_themes
from .entry import EntryDecision, check_theme_surge_entry
from .entry_chart import EntryChartError, build_entry_chart
from .leader import select_leaders
from .positions import build_positions
from .scanner import cleanup_stale_candidates, purge_candidate_date, run_theme_scan
from .timeline import build_timeline
from .toss_client import TossThemeError, fetch_theme_ranking, fetch_theme_stocks

__all__ = [
    "EntryChartError",
    "EntryDecision",
    "TossThemeError",
    "build_entry_chart",
    "build_positions",
    "build_timeline",
    "check_theme_surge_entry",
    "cleanup_stale_candidates",
    "purge_candidate_date",
    "detect_surge_themes",
    "fetch_theme_ranking",
    "fetch_theme_stocks",
    "run_theme_scan",
    "select_leaders",
]
