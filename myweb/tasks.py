from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django.utils import timezone
from django.http import HttpRequest
from django.test.client import RequestFactory
from dolpha.api import *  # api.py에서 함수 임포트
import sys

# 스케줄링할 작업 정의
# def my_scheduled_task():
#     print(f"Scheduled task executed at {timezone.now()}!")

def my_cron_task_getAndSave_stock_description():
    print(f"[Cron task] getAndSave_stock_description executed at {timezone.now()}!")
    try:
        # Request 객체 생성
        request_factory = RequestFactory()
        request = request_factory.post('/getAndSave_stock_description')
        
        # API 뷰 호출
        response = getAndSave_stock_description(request)
        print(f"[Cron task] Response: {response}")  # 디버깅용 출력
        # 응답 처리
        if response['status'] == "OK":
            print("[Cron task] Stock description successfully retrieved and saved.")
        else:
            print(f"[Cron task] Failed to execute API: Status {response.status_code}")
    except Exception as e:
        traceback.print_exc()  # 에러 추적 정보 출력
        print(f"[Cron task] Error in cron task: {str(e)}")

def my_cron_task_getAndSave_stock_data():
    print(f"[Cron task] getAndSave_stock_data executed at {timezone.now()}!")
    try:
        # Request 객체 생성
        request_factory = RequestFactory()
        request = request_factory.post('/getAndSave_stock_data')
        
        # API 뷰 호출
        response = getAndSave_stock_data(request)
        
        # 응답 처리
        if response['status'] == "OK":
            print("[Cron task] Stock description successfully retrieved and saved.")
        else:
            print(f"[Cron task] Failed to execute API: Status {response.status_code}")
    except Exception as e:
        traceback.print_exc()  # 에러 추적 정보 출력
        print(f"[Cron task] Error in cron task: {str(e)}")

def my_cron_task_getAndSave_stock_dartData():
    print(f"[Cron task] getAndSave_stock_dartData executed at {timezone.now()}!")
    try:
        # Request 객체 생성
        request_factory = RequestFactory()
        request = request_factory.post('/getAndSave_stock_dartData')
        
        # API 뷰 호출
        response = getAndSave_stock_dartData(request)
        
        # 응답 처리
        if response['status'] == "OK":
            print("[Cron task] Stock DART data successfully retrieved and saved.")
        else:
            print(f"[Cron task] Failed to execute API: Status {response.status_code}")
    except Exception as e:
        traceback.print_exc()  # 에러 추적 정보 출력
        print(f"[Cron task] Error in cron task: {str(e)}")

def start():
    scheduler = BackgroundScheduler(timezone='Asia/Seoul')  # 시간대 설정
    scheduler.add_jobstore(DjangoJobStore(), "default")  # Django ORM을 작업 저장소로 사용

    # 작업 추가 (예: 30분마다 실행)
    # scheduler.add_job(
    #     my_scheduled_task,
    #     trigger='interval',
    #     minutes=1,
    #     id='my_scheduled_task',  # 고유 ID
    #     max_instances=1,
    #     replace_existing=True,
    # )
    
    # cron 방식 예시 (매일 12:00에 실행)
    # 전체 종목 설명을 가져오고 저장하는 작업
    scheduler.add_job(
        my_cron_task_getAndSave_stock_description,
        trigger='cron',
        hour=16,
        minute=0,
        id='my_cron_task_getAndSave_stock_description',
        max_instances=1,
        replace_existing=True,
    )
    
    # 전체 종목의 OHLCV 데이터를 가져오고 저장하는 작업
    scheduler.add_job(
        my_cron_task_getAndSave_stock_data,
        trigger='cron',
        hour=16,
        minute=10,
        id='my_cron_task_getAndSave_stock_data',
        max_instances=1,
        replace_existing=True,
    )
    
    # 전체 종목의 DART 데이터를 가져오고 저장하는 작업
    scheduler.add_job(
        my_cron_task_getAndSave_stock_dartData,
        trigger='cron',
        hour=16,
        minute=30,
        id='my_cron_task_getAndSave_stock_dartData',
        max_instances=1,
        replace_existing=True,
    )

    register_events(scheduler)  # Django 관리자 인터페이스와 통합
    scheduler.start()
    print("Scheduler started!", file=sys.stdout)