"""과거 부분체결 BUY TradeEntry의 최종 체결값 보정 유틸리티.

16초 재시도 안에 완전 체결을 확인 못한 매수 주문들이 filled_quantity에
부분체결 수량으로 영구 고정돼 실계좌와 DB 수량이 어긋나는 현상을 사후 복구합니다.

사용법:
    python manage.py backfill_fill_confirmation --dry-run          # 현황 리포트만 출력
    python manage.py backfill_fill_confirmation --apply             # 실제 재조회·보정 수행
    python manage.py backfill_fill_confirmation --apply --days 30   # 최근 30일 대상
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand
from django.utils import timezone as tz

from dolpha.trading_engine import TradingEngine
from myweb.models import TradeEntry, User

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser


class Command(BaseCommand):
    help = "부분체결로 고정된 BUY TradeEntry를 재조회해 최종 체결값으로 보정합니다."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--days",
            type=int,
            default=7,
            help="조회 대상 기간(일) — 기본값 7",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="실제로 재조회·보정을 수행 (없으면 dry-run)",
        )
        parser.add_argument(
            "--user",
            type=str,
            default=None,
            help="특정 사용자만 처리 (사용자명)",
        )

    def handle(self, *args, **options) -> None:
        days: int = options["days"]
        apply: bool = options["apply"]
        username: str | None = options["user"]

        mode = "적용" if apply else "검사(dry-run)"
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"\n=== 부분체결 BUY 백필 ({mode}, 최근 {days}일) ===\n"
            )
        )

        now = tz.now()
        cutoff_time = now - dt.timedelta(days=days)
        min_age = dt.timedelta(minutes=1)
        min_time = now - min_age

        # 조건: trade_type="BUY", status="FILLED",
        #       filled_quantity < order_quantity,
        #       created_at >= cutoff_time, created_at < min_time
        qs = TradeEntry.objects.filter(
            trade_type="BUY",
            status="FILLED",
            created_at__gte=cutoff_time,
            created_at__lt=min_time,
        ).order_by("-created_at")

        if username:
            try:
                user = User.objects.get(username=username)
                qs = qs.filter(user=user)
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"사용자 '{username}' 없음"))
                return

        # 파이썬에서 filled_quantity < order_quantity 필터링
        candidates = [e for e in qs if e.filled_quantity < e.order_quantity]

        if not candidates:
            self.stdout.write("대상 항목 없음\n")
            return

        self.stdout.write(f"대상 항목: {len(candidates)}건\n")
        self._print_candidates(candidates)

        if not apply:
            self.stdout.write(
                self.style.WARNING(
                    "\n[dry-run] 변경사항 없음. --apply 옵션으로 실제 보정을 수행하세요.\n"
                )
            )
            return

        # 실제 적용: 사용자별로 TradingEngine 인스턴스 생성해 재조회
        self._apply_refetch(candidates)

    def _print_candidates(self, entries: list[TradeEntry]) -> None:
        """대상 항목 리스트를 보기 좋게 출력합니다."""
        self.stdout.write("\n[대상 항목]\n")
        self.stdout.write("-" * 90)
        self.stdout.write(
            f"{'사용자':<15} {'종목':<10} {'수량':<20} "
            f"{'체결가':<12} {'주문시각':<20}\n"
        )
        self.stdout.write("-" * 90)

        for entry in entries[:20]:  # 최대 20개만 표시
            filled = entry.filled_quantity
            ordered = entry.order_quantity
            created = entry.created_at or "-"
            ordered_at = entry.ordered_at or "-"

            self.stdout.write(
                f"{entry.user.username:<15} "
                f"{entry.stock_name:<10} "
                f"{filled}/{ordered}주{' ' * (13 - len(f'{filled}/{ordered}주'))} "
                f"{float(entry.filled_price) if entry.filled_price else 0:>10,.0f}원 "
                f"{ordered_at}\n"
            )

        if len(entries) > 20:
            self.stdout.write(f"\n... 외 {len(entries) - 20}건 생략\n")
        self.stdout.write("-" * 90 + "\n")

    def _apply_refetch(self, entries: list[TradeEntry]) -> None:
        """실제 재조회·보정을 수행합니다."""
        by_user: dict[int, list[TradeEntry]] = {}
        for entry in entries:
            if entry.user.id not in by_user:
                by_user[entry.user.id] = []
            by_user[entry.user.id].append(entry)

        total_updated = 0
        total_failed = 0

        for user_id, user_entries in by_user.items():
            user = User.objects.get(id=user_id)
            engine = TradingEngine(user=user)

            self.stdout.write(f"\n사용자 {user.username}:")

            for entry in user_entries:
                if entry.trading_config is None or not entry.trading_config.is_active:
                    self.stdout.write(
                        f"  {entry.stock_name}: [스킵] 설정 비활성 또는 없음"
                    )
                    total_failed += 1
                    continue

                try:
                    updated = engine._refetch_and_apply_fill(entry.trading_config, entry)
                    if updated:
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"  {entry.stock_name}: ✓ 보정 완료 "
                                f"({entry.filled_quantity}주)"
                            )
                        )
                        total_updated += 1
                    else:
                        # 재조회 결과가 없으면 기간 만료 표시
                        if "[재조회 포기" not in entry.note:
                            entry.note = (f"{entry.note} " if entry.note else "") + (
                                "[재조회 포기: 기간 만료]"
                            )
                            entry.save(update_fields=["note"])
                        self.stdout.write(
                            f"  {entry.stock_name}: - 증가분 없음 (기간 만료 표시)"
                        )
                        total_failed += 1

                except Exception as e:  # noqa: BLE001
                    self.stdout.write(
                        self.style.ERROR(
                            f"  {entry.stock_name}: ✗ 오류 - {e}"
                        )
                    )
                    total_failed += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"\n=== 완료 ===\n보정: {total_updated}건, 실패/스킵: {total_failed}건\n"
            )
        )
