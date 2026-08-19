"""비개장일에 잘못 수집된 급등테마주 스냅샷을 제거한다.

휴장일(공휴일·대체공휴일)에는 토스 랭킹이 직전 거래일 종가를 그대로 반환하므로,
스캔이 돌면 같은 값이 78개 슬롯에 복사되어 타임라인이 오염된다.
개장일 게이트(dolpha.kis.holiday)를 도입하기 전에 쌓인 데이터를 정리한다.

사용법:
    python manage.py purge_nontrading_snapshots            # 삭제 대상만 출력 (dry-run)
    python manage.py purge_nontrading_snapshots --apply    # 실제 삭제
"""

from django.core.management.base import BaseCommand
from django.db import transaction

from dolpha.kis.holiday import is_trading_day
from myweb.models import ThemeLeaderCandidate, ThemeSnapshot


class Command(BaseCommand):
    help = "비개장일에 수집된 ThemeSnapshot / ThemeLeaderCandidate 를 삭제합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            help="실제로 삭제합니다 (미지정 시 대상만 출력)",
        )

    def handle(self, *args, **options):
        apply_changes = options["apply"]

        dates = sorted(set(ThemeSnapshot.objects.values_list("date", flat=True)))
        if not dates:
            self.stdout.write("스냅샷이 없습니다.")
            return

        non_trading = [d for d in dates if not is_trading_day(d)]

        self.stdout.write(f"수집된 날짜 {len(dates)}일 중 비개장일 {len(non_trading)}일")
        if not non_trading:
            self.stdout.write(self.style.SUCCESS("삭제할 데이터가 없습니다."))
            return

        total_snapshots = 0
        total_candidates = 0
        for day in non_trading:
            snapshots = ThemeSnapshot.objects.filter(date=day).count()
            candidates = ThemeLeaderCandidate.objects.filter(date=day).count()
            total_snapshots += snapshots
            total_candidates += candidates
            self.stdout.write(
                f"  {day} ({day.strftime('%a')}) — 스냅샷 {snapshots}행, 후보 {candidates}행"
            )

        if not apply_changes:
            self.stdout.write(
                self.style.WARNING(
                    f"\n[dry-run] 스냅샷 {total_snapshots}행, 후보 {total_candidates}행이 삭제 대상입니다."
                    "\n실제로 삭제하려면 --apply 를 붙여 다시 실행하세요."
                )
            )
            return

        with transaction.atomic():
            # 후보는 snapshot FK CASCADE 로도 지워지지만, date 로 직접 지워
            # 스냅샷이 먼저 사라진 고아 행까지 확실히 정리한다
            ThemeLeaderCandidate.objects.filter(date__in=non_trading).delete()
            ThemeSnapshot.objects.filter(date__in=non_trading).delete()

        self.stdout.write(
            self.style.SUCCESS(
                f"\n삭제 완료 — 스냅샷 {total_snapshots}행, 후보 {total_candidates}행"
            )
        )
