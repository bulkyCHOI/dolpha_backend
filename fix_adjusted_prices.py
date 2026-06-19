"""
수정주가 소급 적용 스크립트

두 가지 기준으로 기업이벤트(무상감자/주식병합 등) 종목을 감지하고 수정주가로 재수집한다.
  1. change > CHANGE_THRESHOLD: 재개 당일 이상 변동률 (ex. 30:1 감자 → +3000%)
  2. 연속 거래정지(volume=0) N일 이상: 변동률이 임계값 이하여도 거래정지 이력으로 감지

Usage:
  python fix_adjusted_prices.py            # 실제 실행
  python fix_adjusted_prices.py --dry-run  # 대상 종목만 출력 (수정 없음)
  python fix_adjusted_prices.py --code 049630  # 특정 종목만 실행
"""

import os
import sys
import argparse
import django
from collections import defaultdict

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dolpha.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from tqdm import tqdm

from myweb.models import Company, StockOHLCV
from dolpha.data_quality import CORPORATE_ACTION_CHANGE_THRESHOLD, HALT_CONSECUTIVE_DAYS, refetch_adjusted_history

# 연속 거래정지 감지 기준 (data_quality와 동일)
HALT_MIN_DAYS = HALT_CONSECUTIVE_DAYS  # 3일


def find_by_change_rate(code: str = None) -> dict:
    """기준 1: change > 임계값(500%) 종목."""
    qs = StockOHLCV.objects.filter(
        code__market__in=["KOSDAQ", "KONEX", "KOSPI"],
        change__gt=CORPORATE_ACTION_CHANGE_THRESHOLD,
    )
    if code:
        qs = qs.filter(code__code=code)

    result = {}
    for stock_code, name, event_date, change_val in qs.values_list(
        "code__code", "code__name", "date", "change"
    ).order_by("code__code", "date"):
        if stock_code not in result:
            result[stock_code] = {"name": name, "date": event_date, "change": change_val, "reason": "change이상"}
    return result


def find_by_trading_halt(code: str = None) -> dict:
    """기준 2: volume=0 연속 N일 이상인 종목."""
    qs = StockOHLCV.objects.filter(
        code__market__in=["KOSDAQ", "KONEX", "KOSPI"],
        volume=0,
    ).values_list("code__code", "code__name", "date").order_by("code__code", "date")
    if code:
        qs = qs.filter(code__code=code)

    # 종목별 volume=0 날짜 묶기 → 연속 구간 길이 계산
    by_company = defaultdict(list)
    for stock_code, name, dt in qs:
        by_company[(stock_code, name)].append(dt)

    result = {}
    for (stock_code, name), dates in by_company.items():
        max_streak = _max_consecutive(dates)
        if max_streak >= HALT_MIN_DAYS:
            result[stock_code] = {
                "name": name,
                "date": dates[0],
                "change": None,
                "reason": f"연속거래정지 {max_streak}일",
            }
    return result


def _max_consecutive(sorted_dates: list) -> int:
    """날짜 리스트에서 최대 연속일수 계산 (캘린더 기준, 주말 제외 근사)."""
    from datetime import timedelta
    if not sorted_dates:
        return 0
    max_streak = streak = 1
    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        if gap <= 4:  # 주말 포함 최대 4일 간격까지 연속으로 간주
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    return max_streak


def find_affected_companies(code: str = None) -> dict:
    """두 기준을 합산해 중복 없이 반환."""
    by_change = find_by_change_rate(code)
    by_halt = find_by_trading_halt(code)

    merged = {**by_halt, **by_change}  # change가 있으면 change 기준 우선
    return dict(sorted(merged.items()))


def run(dry_run: bool = False, code: str = None):
    print(f"[fix_adjusted_prices] 감지 기준 1: change > {CORPORATE_ACTION_CHANGE_THRESHOLD * 100:.0f}%")
    print(f"[fix_adjusted_prices] 감지 기준 2: volume=0 연속 {HALT_MIN_DAYS}일 이상")
    print(f"[fix_adjusted_prices] 대상 시장: KOSPI / KOSDAQ / KONEX")
    if code:
        print(f"[fix_adjusted_prices] 지정 종목: {code}")
    print()

    affected = find_affected_companies(code)

    if not affected:
        print("감지된 종목 없음 — DB가 이미 정상 상태입니다.")
        return

    print(f"대상 종목: {len(affected)}개")
    print("-" * 70)
    for stock_code, info in affected.items():
        change_str = f"change={info['change'] * 100:+.0f}%" if info["change"] is not None else ""
        print(f"  {stock_code:8s}  {info['name']:20s}  {info['date']}  {info['reason']:20s}  {change_str}")
    print("-" * 70)

    if dry_run:
        print("\n[dry-run] 실제 수정 없음. --dry-run 없이 실행하면 위 종목을 재수집합니다.")
        return

    print("\n수정주가 재수집 시작...\n")

    success, fail = 0, 0
    companies = Company.objects.filter(
        code__in=list(affected.keys()),
        market__in=["KOSDAQ", "KONEX", "KOSPI"],
    )

    for company in tqdm(companies, desc="재수집 진행"):
        count = refetch_adjusted_history(company, area="KR")
        if count > 0:
            success += 1
        else:
            fail += 1
            print(f"  [실패] {company.code} ({company.name})")

    print(f"\n완료: 성공 {success}개 / 실패 {fail}개")
    if success > 0:
        print("RS 재계산을 위해 calculate_stock_analysis를 실행하세요.")
        print("  python data_trigger.py --task analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수정주가 소급 적용")
    parser.add_argument("--dry-run", action="store_true", help="대상 종목만 출력 (수정 없음)")
    parser.add_argument("--code", type=str, default=None, help="특정 종목 코드만 실행")
    args = parser.parse_args()

    run(dry_run=args.dry_run, code=args.code)
