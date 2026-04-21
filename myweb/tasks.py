import os
import sys
import traceback
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django.utils import timezone
from django.http import HttpRequest
from django.test.client import RequestFactory

# API 함수들을 분리된 모듈에서 임포트
from dolpha.api_data import (
    getAndSave_index_list,
    getAndSave_stock_description,
    getAndSave_stock_data,
    calculate_stock_analysis,
    getAndSave_index_data,
    getAndSave_shares_outstanding,
)
from dolpha.dart_parallel import run_parallel

SCHEDULER_LOG = "/tmp/scheduler.log"


def _slog(msg: str):
    """스케줄러 전용 로그 파일에 타임스탬프와 함께 기록 + stdout 출력"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(SCHEDULER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# 스케줄링할 작업 정의
# def my_scheduled_task():
#     print(f"Scheduled task executed at {timezone.now()}!")


def execute_api_task(api_function, api_name, endpoint_url):
    """
    공통 API 호출 로직을 처리하는 헬퍼 함수

    Args:
        api_function: 호출할 API 함수
        api_name: API 이름 (로깅용)
        endpoint_url: 엔드포인트 URL
    """
    _slog(f"[데이터수집] {api_name} 시작")
    try:
        request_factory = RequestFactory()
        request = request_factory.post(endpoint_url)
        response = api_function(request)

        if isinstance(response, tuple):
            status_code, data = response
            if status_code == 200 and data.get("status") == "OK":
                _slog(f"[데이터수집] {api_name} 완료")
            else:
                _slog(f"[데이터수집] {api_name} 실패: Status {status_code}, Data: {data}")
        elif hasattr(response, "status_code"):
            if response.status_code == 200:
                _slog(f"[데이터수집] {api_name} 완료")
            else:
                _slog(f"[데이터수집] {api_name} 실패: Status {response.status_code}")
        elif isinstance(response, dict) and response.get("status") == "OK":
            _slog(f"[데이터수집] {api_name} 완료")
        else:
            _slog(f"[데이터수집] {api_name} 완료 (응답: {response})")

    except Exception as e:
        _slog(f"[데이터수집] {api_name} 오류: {str(e)}")
        traceback.print_exc()


def my_cron_task_getAndSave_stock_description():
    execute_api_task(
        getAndSave_stock_description,
        "getAndSave_stock_description",
        "/getAndSave_stock_description",
    )


def my_cron_task_getAndSave_stock_data():
    execute_api_task(
        getAndSave_stock_data, "getAndSave_stock_data", "/getAndSave_stock_data"
    )


def my_cron_task_calculate_stock_analysis():
    execute_api_task(
        calculate_stock_analysis,
        "calculate_stock_analysis",
        "/calculate_stock_analysis",
    )
    # 분석 완료 후 MTT 캐시 무효화
    try:
        from dolpha.api_query import invalidate_mtt_cache
        invalidate_mtt_cache()
    except Exception:
        pass


def my_cron_task_getAndSave_stock_dartData():
    _slog("[데이터수집] DART 재무제표 수집 시작")
    try:
        result = run_parallel(workers=10)
        _slog(f"[데이터수집] DART 재무제표 수집 완료: {result['message']}")
    except Exception as e:
        _slog(f"[데이터수집] DART 재무제표 수집 오류: {e}")
        traceback.print_exc()


def my_cron_task_getAndSave_index_list():
    execute_api_task(
        getAndSave_index_list, "getAndSave_index_list", "/getAndSave_index_list"
    )


# ──────────────────────────────────────────────────────────────
# 자동매매 주기적 실행 (KIS_APP_KEY 환경변수가 있을 때만 활성화)
# ──────────────────────────────────────────────────────────────

def run_all_trading_cycles():
    """
    활성 TradingConfig가 있는 모든 유저의 트레이딩 사이클을 실행합니다.
    KIS_APP_KEY 환경변수가 없으면 즉시 반환합니다.
    """
    kis_mode = os.environ.get("KIS_MODE", "VIRTUAL")
    key_var = "KIS_REAL_APP_KEY" if kis_mode == "REAL" else "KIS_VIRTUAL_APP_KEY"
    if not os.environ.get(key_var, ""):
        return  # KIS 미설정 → 자동매매 비활성

    try:
        from myweb.models import TradingConfig
        from dolpha.trading_engine import TradingEngine

        users = (
            TradingConfig.objects
            .filter(is_active=True)
            .select_related("user")
            .values_list("user", flat=True)
            .distinct()
        )

        if not users:
            return

        _slog(f"[자동매매] 사이클 시작 (활성 유저 {len(users)}명)")
        from myweb.models import User
        for user_id in users:
            try:
                user = User.objects.get(pk=user_id)
                engine = TradingEngine(user=user)
                engine.run_trading_cycle()
                _slog(f"[자동매매] 유저 {user.email} 사이클 완료")
            except Exception as e:
                _slog(f"[자동매매] 유저 {user_id} 사이클 오류: {e}")
                traceback.print_exc()

    except Exception as e:
        _slog(f"[자동매매] run_all_trading_cycles 오류: {e}")
        traceback.print_exc()


def my_cron_task_getAndSave_index_data():
    execute_api_task(
        getAndSave_index_data, "getAndSave_index_data", "/getAndSave_index_data"
    )


def my_cron_task_getAndSave_shares_outstanding():
    execute_api_task(
        getAndSave_shares_outstanding,
        "getAndSave_shares_outstanding",
        "/getAndSave_shares_outstanding",
    )


def start():
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")  # 시간대 설정
    scheduler.add_jobstore(
        DjangoJobStore(), "default"
    )  # Django ORM을 작업 저장소로 사용

    def add_cron_job(func, hour, minute, job_id, description=""):
        """
        cron job 추가를 위한 헬퍼 함수

        Args:
            func: 실행할 함수
            hour: 실행 시간 (시)
            minute: 실행 시간 (분)
            job_id: 작업 고유 ID
            description: 작업 설명 (선택사항)
        """
        scheduler.add_job(
            func,
            trigger="cron",
            hour=hour,
            minute=minute,
            id=job_id,
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=3600,  # 서버 재시작 시 1시간 이내 missed job 즉시 실행
        )
        print(f"Scheduled job '{job_id}' at {hour:02d}:{minute:02d} - {description}")

    # 작업 스케줄 정의 (시간 순서대로 정렬)
    job_schedule = [
        # (my_cron_task_getAndSave_index_list, 00, 35, "my_cron_task_getAndSave_index_list", "인덱스 리스트 수집 (약 1분)"),
        # (my_cron_task_getAndSave_stock_description, 00, 40, "my_cron_task_getAndSave_stock_description", "주식 설명 수집 (약 5분)"),
        # (my_cron_task_getAndSave_index_data, 00, 45, "my_cron_task_getAndSave_index_data", "인덱스 데이터 수집 (약 10분)"),
        # (my_cron_task_getAndSave_stock_data, 00, 55, "my_cron_task_getAndSave_stock_data", "주식 OHLCV 데이터 수집 (약 20분)"),
        # (my_cron_task_calculate_stock_analysis, 1, 15, "my_cron_task_calculate_stock_analysis", "주식 기술적 분석 계산 (약 10분)"),
        # (my_cron_task_getAndSave_stock_dartData, 1, 30, "my_cron_task_getAndSave_stock_dartData", "DART 데이터 수집 (약 1시간 30분)"),
        (
            my_cron_task_getAndSave_index_list,
            15,
            35,
            "my_cron_task_getAndSave_index_list_1",
            "인덱스 리스트 수집 (약 1분) - 17:05",
        ),
        (
            my_cron_task_getAndSave_stock_description,
            15,
            40,
            "my_cron_task_getAndSave_stock_description_1",
            "주식 설명 수집 (약 5초) - 17:10",
        ),
        (
            my_cron_task_getAndSave_index_data,
            15,
            45,
            "my_cron_task_getAndSave_index_data_1",
            "인덱스 데이터 수집 (약 1분) - 17:15",
        ),
        (
            my_cron_task_getAndSave_stock_data,
            15,
            50,
            "my_cron_task_getAndSave_stock_data_1",
            "주식 OHLCV 데이터 수집 (약 20분) - 17:20",
        ),
        (
            my_cron_task_calculate_stock_analysis,
            16,
            20,
            "my_cron_task_calculate_stock_analysis_1",
            "주식 기술적 분석 계산 (약 2분) - 17:50",
        ),
        (
            my_cron_task_getAndSave_stock_dartData,
            16,
            30,
            "my_cron_task_getAndSave_stock_dartData_1",
            "DART 데이터 수집 (약 1시간) - 18:00",
        ),
        # NXT마켓 때문에 한번 더 수행
        (
            my_cron_task_getAndSave_index_list,
            20,
            5,
            "my_cron_task_getAndSave_index_list_2",
            "인덱스 리스트 수집 (약 1분) - 20:05",
        ),
        (
            my_cron_task_getAndSave_stock_description,
            20,
            10,
            "my_cron_task_getAndSave_stock_description_2",
            "주식 설명 수집 (약 5초) - 20:10",
        ),
        (
            my_cron_task_getAndSave_index_data,
            20,
            15,
            "my_cron_task_getAndSave_index_data_2",
            "인덱스 데이터 수집 (약 1분) - 20:15",
        ),
        (
            my_cron_task_getAndSave_stock_data,
            20,
            20,
            "my_cron_task_getAndSave_stock_data_2",
            "주식 OHLCV 데이터 수집 (약 20분) - 20:20",
        ),
        (
            my_cron_task_calculate_stock_analysis,
            20,
            50,
            "my_cron_task_calculate_stock_analysis_2",
            "주식 기술적 분석 계산 (약 2분) - 20:50",
        ),
        (
            my_cron_task_getAndSave_stock_dartData,
            21,
            00,
            "my_cron_task_getAndSave_stock_dartData_2",
            "DART 데이터 수집 (약 1시간) - 21:00",
        ),
    ]

    # 모든 작업을 스케줄에 추가
    for func, hour, minute, job_id, description in job_schedule:
        add_cron_job(func, hour, minute, job_id, description)

    # ── 상장주식수 주간 갱신 (월요일 08:00) ──────────────────────────
    scheduler.add_job(
        my_cron_task_getAndSave_shares_outstanding,
        trigger="cron",
        day_of_week="mon",
        hour=8,
        minute=0,
        id="shares_outstanding_weekly",
        max_instances=1,
        replace_existing=True,
    )
    print("Scheduled job 'shares_outstanding_weekly' at Monday 08:00 - 상장주식수 갱신 (약 8분)")

    # ── 자동매매 사이클 (KIS 설정 시에만 등록) ──────────────────────
    _kis_mode = os.environ.get("KIS_MODE", "VIRTUAL")
    _key_var = "KIS_REAL_APP_KEY" if _kis_mode == "REAL" else "KIS_VIRTUAL_APP_KEY"
    if os.environ.get(_key_var, ""):
        scheduler.add_job(
            run_all_trading_cycles,
            trigger="cron",
            day_of_week="mon-fri",   # 평일만
            hour="9-15",             # 9시~15시 (is_market_open이 15:30 이후 차단)
            minute="*",              # 매 분
            id="auto_trading_cycle",
            max_instances=1,         # 중복 실행 방지
            replace_existing=True,
        )
        print("Scheduled job 'auto_trading_cycle' at every minute (09:00-15:59 Mon-Fri)")

    register_events(scheduler)  # Django 관리자 인터페이스와 통합
    scheduler.start()
    print("Scheduler started!", file=sys.stdout)
