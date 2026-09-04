"""P2 청산 튜닝 — 기본값을 손대지 않은 TradingDefaults 행만 새 기본값으로 갱신한다.

사용자가 프론트에서 직접 바꾼 값은 건드리지 않고, 이전 기본값
(2T 50% 일괄 익절 / 트레일링 2.0T / 5분봉 / 후보 3개) 그대로인 행만 이전한다.
"""

from django.db import migrations

OLD_EXIT_STAGES = [{"t": 2.0, "sell_pct": 50.0}]
NEW_EXIT_STAGES = [
    {"t": 1.5, "sell_pct": 25.0},
    {"t": 3.0, "sell_pct": 35.0},
    {"t": 4.5, "sell_pct": 20.0},
]


def forwards(apps, schema_editor):
    TradingDefaults = apps.get_model("myweb", "TradingDefaults")
    for row in TradingDefaults.objects.all():
        changed = False
        if row.theme_surge_exit_stages == OLD_EXIT_STAGES:
            row.theme_surge_exit_stages = NEW_EXIT_STAGES
            changed = True
        if row.theme_surge_trailing_start_t == 2.0:
            row.theme_surge_trailing_start_t = 1.5
            changed = True
        if row.theme_surge_trailing_bar_unit == "5m":
            row.theme_surge_trailing_bar_unit = "1m"
            changed = True
        if row.theme_surge_max_candidates == 3:
            row.theme_surge_max_candidates = 5
            changed = True
        if changed:
            row.save(update_fields=[
                "theme_surge_exit_stages",
                "theme_surge_trailing_start_t",
                "theme_surge_trailing_bar_unit",
                "theme_surge_max_candidates",
            ])


def backwards(apps, schema_editor):
    TradingDefaults = apps.get_model("myweb", "TradingDefaults")
    for row in TradingDefaults.objects.all():
        changed = False
        if row.theme_surge_exit_stages == NEW_EXIT_STAGES:
            row.theme_surge_exit_stages = OLD_EXIT_STAGES
            changed = True
        if row.theme_surge_trailing_start_t == 1.5:
            row.theme_surge_trailing_start_t = 2.0
            changed = True
        if row.theme_surge_trailing_bar_unit == "1m":
            row.theme_surge_trailing_bar_unit = "5m"
            changed = True
        if row.theme_surge_max_candidates == 5:
            row.theme_surge_max_candidates = 3
            changed = True
        if changed:
            row.save(update_fields=[
                "theme_surge_exit_stages",
                "theme_surge_trailing_start_t",
                "theme_surge_trailing_bar_unit",
                "theme_surge_max_candidates",
            ])


class Migration(migrations.Migration):

    dependencies = [
        ("myweb", "0041_alter_tradingdefaults_theme_surge_max_candidates_and_more"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
