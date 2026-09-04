"""급등테마주 자동매매 파이프라인 병목 진단 리포트 (P0 — 코드 변경 없이 현황 측정).

4단계 필터(테마 선정 → 주도주 선정 → 후보 등록 → 진입 신호)에서
후보가 어디서 얼마나 걸러지는지 최근 N일치 로그를 집계한다.

사용법:
    python manage.py theme_surge_diagnose               # 최근 14일
    python manage.py theme_surge_diagnose --days 30
    python manage.py theme_surge_diagnose --days 30 --user <username>
"""

from __future__ import annotations

import datetime as dt
from collections import Counter

from django.core.management.base import BaseCommand
from django.db.models import Avg, Count, Q

from dolpha.theme_surge import config as cfg
from myweb.models import (
    ThemeEntrySignal,
    ThemeLeaderCandidate,
    ThemeSnapshot,
    User,
)

# "거의 통과할 뻔한" 테마를 세기 위한 여유폭(%p)
NEAR_MISS_MARGIN_PCT = 0.5


class Command(BaseCommand):
    help = "급등테마주 파이프라인 단계별 필터 통과율을 집계합니다."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--days", type=int, default=14, help="집계 기간(일)")
        parser.add_argument("--user", type=str, default=None, help="진입 신호를 볼 사용자명")

    def handle(self, *args, **options) -> None:
        days: int = options["days"]
        username: str | None = options["user"]
        since = dt.date.today() - dt.timedelta(days=days)

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"\n=== 급등테마주 파이프라인 진단 ({since} ~ 오늘, {days}일) ===\n"
            )
        )

        self._section_themes(since)
        self._section_leaders(since)
        self._section_entry_signals(since, username)
        self._section_params()

    # ── 1단계: 테마 선정 ────────────────────────────────────────────
    def _section_themes(self, since: dt.date) -> None:
        qs = ThemeSnapshot.objects.filter(date__gte=since)
        total = qs.count()
        if not total:
            self.stdout.write("① 테마 선정: 스냅샷 없음\n")
            return

        surge = qs.filter(is_surge=True).count()
        trading_days = qs.values("date").distinct().count()
        thr = cfg.SURGE_MIN_FLUCTUATION_PCT
        near_miss = qs.filter(
            is_surge=False,
            fluctuation_rate__gte=thr - NEAR_MISS_MARGIN_PCT,
            fluctuation_rate__lt=thr,
        ).count()
        avg_flux = qs.aggregate(v=Avg("fluctuation_rate"))["v"] or 0.0

        self.stdout.write(self.style.HTTP_INFO("① 테마 선정 (ThemeSnapshot)"))
        self.stdout.write(f"   집계 거래일          : {trading_days}일")
        self.stdout.write(f"   테마-슬롯 행 수       : {total:,}")
        self.stdout.write(
            f"   급등 판정            : {surge:,} ({surge / total * 100:.1f}%)"
            f"  · 거래일당 평균 {surge / max(trading_days, 1):.1f}건"
        )
        self.stdout.write(
            f"   판정 직전 탈락       : {near_miss:,}건 "
            f"(등락률 {thr - NEAR_MISS_MARGIN_PCT:.1f}~{thr:.1f}%, 임계값만 낮추면 살아남음)"
        )
        self.stdout.write(f"   평균 테마 등락률     : {avg_flux:+.2f}%\n")

    # ── 2단계: 주도주 선정 ──────────────────────────────────────────
    def _section_leaders(self, since: dt.date) -> None:
        qs = ThemeLeaderCandidate.objects.filter(date__gte=since)
        total = qs.count()
        if not total:
            self.stdout.write("② 주도주 선정: 후보 없음\n")
            return

        selected = qs.filter(is_selected=True).count()
        surge_snap = ThemeSnapshot.objects.filter(
            date__gte=since, is_surge=True
        ).count()
        themes_with_leader = (
            qs.values("date", "tics_id", "slot_time").distinct().count()
        )

        self.stdout.write(self.style.HTTP_INFO("② 주도주 선정 (ThemeLeaderCandidate)"))
        self.stdout.write(f"   후보 종목 행 수       : {total:,}")
        self.stdout.write(
            f"   후보 산출된 급등테마  : {themes_with_leader:,} / 급등판정 {surge_snap:,} "
            f"({themes_with_leader / max(surge_snap, 1) * 100:.1f}%)"
        )
        self.stdout.write(
            f"   자동매매 후보 등록    : {selected:,} ({selected / total * 100:.1f}%)\n"
        )

    # ── 3~4단계: 후보 등록 + 진입 신호 ─────────────────────────────
    def _section_entry_signals(self, since: dt.date, username: str | None) -> None:
        qs = ThemeEntrySignal.objects.filter(date__gte=since)
        if username:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"사용자 '{username}' 없음\n"))
                return
            qs = qs.filter(user=user)

        total = qs.count()
        if not total:
            self.stdout.write(
                "③④ 진입 신호: 로그 없음 "
                "(--user 를 지정했는지, 엔진이 신호를 남기는지 확인)\n"
            )
            return

        agg = qs.aggregate(
            passed=Count("id", filter=Q(passed=True)),
            executed=Count("id", filter=Q(executed=True)),
            pullback=Count("id", filter=Q(has_pullback=True)),
            breakout=Count("id", filter=Q(has_breakout=True)),
            foreign=Count("id", filter=Q(has_foreign_buying=True)),
        )
        distinct_names = qs.values("stock_code").distinct().count()
        trading_days = qs.values("date").distinct().count()

        self.stdout.write(self.style.HTTP_INFO("③④ 진입 신호 판정 (ThemeEntrySignal)"))
        self.stdout.write(
            f"   판정 횟수            : {total:,}  · 대상 종목 {distinct_names}개"
            f"  · {trading_days}거래일"
        )
        self.stdout.write(
            f"   눌림목 성립          : {agg['pullback']:,} ({agg['pullback'] / total * 100:.1f}%)"
        )
        self.stdout.write(
            f"   전고점 돌파          : {agg['breakout']:,} ({agg['breakout'] / total * 100:.1f}%)"
        )
        self.stdout.write(
            f"   외국인 매수세        : {agg['foreign']:,} ({agg['foreign'] / total * 100:.1f}%)"
        )
        self.stdout.write(
            f"   3단 조건 모두 충족   : {agg['passed']:,} ({agg['passed'] / total * 100:.2f}%)"
        )
        self.stdout.write(
            f"   실제 매수 실행       : {agg['executed']:,}"
            f"  · 거래일당 {agg['executed'] / max(trading_days, 1):.2f}건\n"
        )

        # 미진입 사유 Top 10 (reason 앞부분으로 그룹화)
        reasons: Counter[str] = Counter()
        for r in qs.filter(passed=False).values_list("reason", flat=True):
            key = (r or "(사유 없음)").split(" — ")[0].split(" (")[0].strip()
            reasons[key] += 1

        self.stdout.write("   미진입 사유 Top 10:")
        for reason, cnt in reasons.most_common(10):
            self.stdout.write(f"     {cnt:>6,}  {reason}")

        # 눌림 깊이 분포 (완화 대상 파라미터 근거)
        pb = list(
            qs.filter(pullback_pct__isnull=False).values_list("pullback_pct", flat=True)
        )
        if pb:
            lo, hi = cfg.PULLBACK_MIN_PCT, cfg.PULLBACK_MAX_PCT
            below = sum(1 for x in pb if x < lo)
            above = sum(1 for x in pb if x > hi)
            self.stdout.write(
                f"\n   눌림 깊이 관측 {len(pb):,}건 중 "
                f"현재 범위({lo}~{hi}%) 밖: 얕음 {below:,} / 깊음 {above:,}"
            )
        self.stdout.write("")

    # ── 현재 파라미터 요약 ────────────────────────────────────────
    def _section_params(self) -> None:
        self.stdout.write(self.style.HTTP_INFO("현재 핵심 파라미터"))
        rows = [
            ("SURGE_MIN_FLUCTUATION_PCT", cfg.SURGE_MIN_FLUCTUATION_PCT, "테마 등락률 하한%"),
            ("ENTRY_MIN_BARS", cfg.ENTRY_MIN_BARS, "진입 판정 최소 분봉"),
            ("PULLBACK_MIN_PCT", cfg.PULLBACK_MIN_PCT, "눌림 최소 깊이%"),
            ("PULLBACK_MAX_PCT", cfg.PULLBACK_MAX_PCT, "눌림 최대 깊이%"),
            ("CANDIDATE_REGISTER_UNTIL", cfg.CANDIDATE_REGISTER_UNTIL, "신규 후보 마감"),
            ("CANDIDATE_MAX_DEFAULT", cfg.CANDIDATE_MAX_DEFAULT, "동시 추적 수"),
            ("DEFAULT_TRAILING_START_T", cfg.DEFAULT_TRAILING_START_T, "트레일링 시작 T배수"),
            ("DEFAULT_TRAILING_BAR_UNIT", cfg.DEFAULT_TRAILING_BAR_UNIT, "트레일링 봉 단위"),
            ("DEFAULT_FORCE_EXIT_TIME", cfg.DEFAULT_FORCE_EXIT_TIME, "강제 청산 시각"),
            ("DEFAULT_EXIT_STAGES", cfg.DEFAULT_EXIT_STAGES, "분할 익절 단계"),
        ]
        for name, val, desc in rows:
            self.stdout.write(f"   {name:<28} = {str(val):<28} {desc}")
        self.stdout.write("")
