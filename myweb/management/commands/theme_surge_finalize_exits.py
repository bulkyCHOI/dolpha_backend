"""장 마감 시점에 보유 중인 급등테마주 포지션을 '오버나이트'로 확정 기록한다.

스케줄러(15:31)가 자동 실행하지만, 배포 직후 당일분을 소급 기록할 때 수동 실행한다.

    python manage.py theme_surge_finalize_exits
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "급등테마주 장 마감 시점 보유 포지션을 ThemeExitSignal(오버나이트)로 기록"

    def handle(self, *args, **options):
        from dolpha.theme_surge.exit_finalizer import finalize_theme_exit_signals

        result = finalize_theme_exit_signals()
        self.stdout.write(self.style.SUCCESS(f"완료: {result}"))
