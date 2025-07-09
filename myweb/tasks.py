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
    getAndSave_stock_dartData,
    getAndSave_index_data
)
import sys
import traceback  # 누락된 import 추가

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
    print(f"[Cron task] {api_name} executed at {timezone.now()}!")
    try:
        # Request 객체 생성
        request_factory = RequestFactory()
        request = request_factory.post(endpoint_url)
        
        # API 뷰 호출
        response = api_function(request)
        
        # 응답 처리
        if isinstance(response, tuple):
            # Ninja API는 (status_code, data) 튜플을 반환할 수 있음
            status_code, data = response
            if status_code == 200 and data.get('status') == "OK":
                print(f"[Cron task] {api_name} successfully completed.")
            else:
                print(f"[Cron task] Failed to execute {api_name}: Status {status_code}, Data: {data}")
        elif hasattr(response, 'status_code'):
            # HttpResponse 객체인 경우
            if response.status_code == 200:
                print(f"[Cron task] {api_name} successfully completed.")
            else:
                print(f"[Cron task] Failed to execute {api_name}: Status {response.status_code}")
        elif isinstance(response, dict) and response.get('status') == "OK":
            # 딕셔너리 응답인 경우
            print(f"[Cron task] {api_name} successfully completed.")
        else:
            print(f"[Cron task] {api_name} completed with response: {response}")
            
    except Exception as e:
        traceback.print_exc()  # 에러 추적 정보 출력
        print(f"[Cron task] Error in {api_name}: {str(e)}")

def my_cron_task_getAndSave_stock_description():
    execute_api_task(getAndSave_stock_description, "getAndSave_stock_description", "/getAndSave_stock_description")

def my_cron_task_getAndSave_stock_data():
    execute_api_task(getAndSave_stock_data, "getAndSave_stock_data", "/getAndSave_stock_data")

def my_cron_task_calculate_stock_analysis():
    execute_api_task(calculate_stock_analysis, "calculate_stock_analysis", "/calculate_stock_analysis")

def my_cron_task_getAndSave_stock_dartData():
    execute_api_task(getAndSave_stock_dartData, "getAndSave_stock_dartData", "/getAndSave_stock_dartData")

def my_cron_task_getAndSave_index_list():
    execute_api_task(getAndSave_index_list, "getAndSave_index_list", "/getAndSave_index_list")

def my_cron_task_getAndSave_index_data():
    execute_api_task(getAndSave_index_data, "getAndSave_index_data", "/getAndSave_index_data")

def start():
    scheduler = BackgroundScheduler(timezone='Asia/Seoul')  # 시간대 설정
    scheduler.add_jobstore(DjangoJobStore(), "default")  # Django ORM을 작업 저장소로 사용

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
            trigger='cron',
            hour=hour,
            minute=minute,
            id=job_id,
            max_instances=1,
            replace_existing=True,
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
        (my_cron_task_getAndSave_index_list, 17, 5, "my_cron_task_getAndSave_index_list", "인덱스 리스트 수집 (약 1분)"),
        (my_cron_task_getAndSave_stock_description, 17, 10, "my_cron_task_getAndSave_stock_description", "주식 설명 수집 (약 5초)"),
        (my_cron_task_getAndSave_index_data, 17, 15, "my_cron_task_getAndSave_index_data", "인덱스 데이터 수집 (약 1분)"),
        (my_cron_task_getAndSave_stock_data, 17, 20, "my_cron_task_getAndSave_stock_data", "주식 OHLCV 데이터 수집 (약 20분)"),
        (my_cron_task_calculate_stock_analysis, 17, 50, "my_cron_task_calculate_stock_analysis", "주식 기술적 분석 계산 (약 2분)"),
        (my_cron_task_getAndSave_stock_dartData, 18, 00, "my_cron_task_getAndSave_stock_dartData", "DART 데이터 수집 (약 1시간)"),
    ]

    # 모든 작업을 스케줄에 추가
    for func, hour, minute, job_id, description in job_schedule:
        add_cron_job(func, hour, minute, job_id, description)

    register_events(scheduler)  # Django 관리자 인터페이스와 통합
    scheduler.start()
    print("Scheduler started!", file=sys.stdout)