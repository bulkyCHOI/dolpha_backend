"""
데이터 수집 트리거 스크립트 (수동 실행용)

Usage:
  # 일간 전체 파이프라인 (권장)
  python data_trigger.py --task daily
  python data_trigger.py --task daily --limit 3
  python data_trigger.py --task daily --skip-dart

  # 개별 단계 실행
  python data_trigger.py --task ohlcv   --start 2026-04-01 --end 2026-04-15
  python data_trigger.py --task index   --start 2026-04-01 --end 2026-04-15
  python data_trigger.py --task analysis --start 2026-04-01 --end 2026-04-15
  python data_trigger.py --task financial
"""

import os
import sys
import django
import argparse
from datetime import datetime, date

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dolpha.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from django.test.client import RequestFactory
from dolpha.api_data import (
    run_daily_pipeline,
    getAndSave_stock_data,
    getAndSave_index_data,
    calculate_stock_analysis,
    rebuild_company_indices_endpoint,
)
from dolpha.dart_parallel import run_parallel

factory = RequestFactory()


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def calc_limit(start: date, end: date = None) -> int:
    """start ~ end(또는 오늘) 범위를 limit(일수)로 변환"""
    today = date.today()
    end = end or today
    return max(1, (end - start).days + 1)


def run_ohlcv(start: date, end: date):
    print(f"[OHLCV] start={start}, end={end}")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    req = factory.get("/")
    result = getAndSave_stock_data(req, area="KR", start_date=str(start), end_date=str(end))
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {result}")


def run_index(start: date, end: date):
    print(f"[인덱스] start={start}, end={end}")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    req = factory.get("/")
    result = getAndSave_index_data(req, start_date=str(start), end_date=str(end))
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {result}")


def run_analysis(start: date, end: date):
    print(f"[기술적 분석] start={start}, end={end}")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    req = factory.get("/")
    result = calculate_stock_analysis(req, area="KR", start_date=str(start), end_date=str(end))
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {result}")


def run_company_indices():
    print("[Company-Index 관계 재구성] KOSPI/KOSDAQ 전 종목 업종 조회 → M2M 테이블 재구성")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    req = factory.get("/")
    ret = rebuild_company_indices_endpoint(req)
    result = ret[1] if isinstance(ret, tuple) else ret
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {result}")


def run_financial(workers: int = 10):
    print(f"[재무제표] 전체 종목 병렬 수집 (workers={workers})")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result = run_parallel(workers=workers)
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {result['message']}")


def run_daily(start: date = None, end: date = None, skip_dart: bool = False):
    today = date.today()
    start = start or today
    end = end or today
    date_info = f"{start} ~ {end}" if start != end else str(start)
    print(f"[일간 전체 파이프라인] {date_info}, skip_dart={skip_dart}")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    req = factory.get("/")
    status_code, result = run_daily_pipeline(req, start_date=str(start), end_date=str(end), skip_dart=skip_dart)
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n전체 상태: {result['status']} ({result['total_elapsed_sec']}초)")
    print(f"요약: {result['message']}\n")
    for key, step in result["steps"].items():
        icon = {"ok": "✓", "error": "✗", "skipped": "-"}.get(step["status"], "?")
        elapsed = f"{step['elapsed_sec']}초"
        count = f"  ({step['count_saved']}건)" if step.get("count_saved") is not None else ""
        msg = f"  — {step['message']}" if step.get("message") and step["status"] == "error" else ""
        print(f"  {icon} [{elapsed:>6}] {step['label']}{count}{msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["daily", "ohlcv", "index", "analysis", "financial", "company-indices"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--skip-dart", action="store_true", default=False)
    args = parser.parse_args()

    today = date.today()
    start = parse_date(args.start) if args.start else today
    end = parse_date(args.end) if args.end else today

    if args.task == "daily":
        run_daily(
            start=start if args.start else None,
            end=end if args.end else None,
            skip_dart=args.skip_dart,
        )
    elif args.task == "ohlcv":
        run_ohlcv(start, end)
    elif args.task == "index":
        run_index(start, end)
    elif args.task == "analysis":
        run_analysis(start, end)
    elif args.task == "financial":
        run_financial()
    elif args.task == "company-indices":
        run_company_indices()


if __name__ == "__main__":
    main()
