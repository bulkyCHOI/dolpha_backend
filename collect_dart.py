"""
DART 재무제표 데이터 병렬 수집 스크립트

Usage:
  python collect_dart.py              # 전체 종목 수집
  python collect_dart.py --resume     # 이미 수집된 종목 건너뛰기
  python collect_dart.py --workers 20 # 동시 처리 수 지정 (기본: 10)
"""

import os
import sys
import django
import argparse
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dolpha.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from dolpha.dart_parallel import run_parallel
from myweb.models import StockFinancialStatement


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="이미 수집된 종목 건너뛰기")
    parser.add_argument("--workers", type=int, default=10, help="동시 처리 스레드 수 (기본: 10)")
    args = parser.parse_args()

    print(f"\n시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 동시 처리: {args.workers}개")
    print("=" * 60)

    result = run_parallel(workers=args.workers, resume=args.resume)

    total_records = StockFinancialStatement.objects.count()
    print("\n" + "=" * 60)
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(result["message"])
    print(f"총 재무제표 레코드: {total_records}개")
    if result["errors"]:
        print("\n주요 오류 (최대 10건):")
        print("\n".join(result["errors"]))


if __name__ == "__main__":
    run()
