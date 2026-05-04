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


PIPELINE_STEPS = [
    my_cron_task_getAndSave_index_list,
    my_cron_task_getAndSave_stock_description,
    my_cron_task_getAndSave_index_data,
    my_cron_task_getAndSave_stock_data,
    my_cron_task_getAndSave_shares_outstanding,
    my_cron_task_calculate_stock_analysis,
    my_cron_task_getAndSave_stock_dartData,
]


def run_data_collection_pipeline(label: str = ""):
    """데이터 수집 파이프라인 - 각 단계가 완료된 후 다음 단계를 순차 실행"""
    tag = f"[파이프라인{':' + label if label else ''}]"
    _slog(f"{tag} 시작")
    for step in PIPELINE_STEPS:
        step()
    _slog(f"{tag} 완료")


def _start_sleep_watchdog(get_scheduler_fn, restart_fn):
    """
    macOS 슬립/웨이크 감지 워치독.
    time.sleep(30) 후 실제 경과 시간이 90초 이상이면 슬립에서 깨어난 것으로 판단하고
    스케줄러를 재시작한다.
    """
    import threading
    import time

    def _watchdog():
        while True:
            t0 = time.time()
            time.sleep(30)
            elapsed = time.time() - t0
            if elapsed > 90:  # 슬립에서 깨어난 경우
                _slog(f"[워치독] 슬립 감지 (경과 {elapsed:.0f}s) → 스케줄러 재시작")
                try:
                    scheduler = get_scheduler_fn()
                    if scheduler and scheduler.running:
                        scheduler.shutdown(wait=False)
                except Exception as e:
                    _slog(f"[워치독] 기존 스케줄러 종료 실패: {e}")
                try:
                    # DB 커넥션 초기화 (슬립 후 stale 커넥션 방지)
                    from django.db import connections
                    for conn in connections.all():
                        conn.close_if_unusable_or_obsolete()
                    restart_fn()
                    _slog("[워치독] 스케줄러 재시작 완료")
                except Exception as e:
                    _slog(f"[워치독] 스케줄러 재시작 실패: {e}")

    t = threading.Thread(target=_watchdog, daemon=True, name="scheduler-watchdog")
    t.start()


_current_scheduler = None
_watchdog_started = False


def _run_pipeline_1535():
    run_data_collection_pipeline("15시")


def _run_pipeline_2005():
    run_data_collection_pipeline("20시")


def start():
    global _current_scheduler, _watchdog_started
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

    # 파이프라인 1: 장 마감 후 (15:35 시작, 순차 실행)
    add_cron_job(
        _run_pipeline_1535,
        15, 35,
        "pipeline_1535",
        "데이터 수집 파이프라인 (장 마감 후)",
    )
    # 파이프라인 2: NXT마켓 반영 (20:05 시작, 순차 실행)
    add_cron_job(
        _run_pipeline_2005,
        20, 5,
        "pipeline_2005",
        "데이터 수집 파이프라인 (NXT마켓 반영)",
    )

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
    _current_scheduler = scheduler
    print("Scheduler started!", file=sys.stdout)

    if not _watchdog_started:
        _watchdog_started = True
        _start_sleep_watchdog(
            get_scheduler_fn=lambda: _current_scheduler,
            restart_fn=start,
        )
