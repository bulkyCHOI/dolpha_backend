"""급등테마주 전략 — 테마 스냅샷/주도주 후보/진입 시그널 모델 및 유저 설정 추가."""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("myweb", "0032_investorflowsnapshot"),
    ]

    operations = [
        # ── TradingConfig 전략 타입에 급등테마주 추가 ──────────────
        migrations.AlterField(
            model_name="tradingconfig",
            name="strategy_type",
            field=models.CharField(
                choices=[
                    ("mtt", "MTT (Minervini Trend Template)"),
                    ("weekly_high", "52주 신고가"),
                    ("fifty_day_high", "50일 신고가"),
                    ("daily_top50", "일일 Top50"),
                    ("htf", "High Tight Flag"),
                    ("theme_surge", "급등테마주"),
                ],
                default="mtt",
                max_length=20,
            ),
        ),
        # ── TradingDefaults 급등테마주 설정 ────────────────────────
        migrations.AddField(
            model_name="tradingdefaults",
            name="theme_surge_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="theme_surge_max_candidates",
            field=models.IntegerField(default=3),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="theme_surge_min_fluctuation",
            field=models.FloatField(default=3.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="theme_surge_min_trading_value",
            field=models.BigIntegerField(default=50000000000),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="theme_surge_use_foreign_filter",
            field=models.BooleanField(default=True),
        ),
        # ── ThemeSnapshot ─────────────────────────────────────────
        migrations.CreateModel(
            name="ThemeSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField(db_index=True)),
                ("slot_time", models.TimeField()),
                ("tics_id", models.IntegerField()),
                ("theme_name", models.CharField(max_length=100)),
                ("rank", models.IntegerField()),
                ("fluctuation_rate", models.FloatField(default=0.0)),
                ("trading_value", models.BigIntegerField(default=0)),
                ("market_cap", models.BigIntegerField(default=0)),
                ("stock_count", models.IntegerField(default=0)),
                ("is_surge", models.BooleanField(default=False)),
                ("surge_reason", models.CharField(blank=True, max_length=200)),
                ("momentum", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "theme_snapshot",
                "ordering": ["-date", "slot_time", "rank"],
            },
        ),
        migrations.AddIndex(
            model_name="themesnapshot",
            index=models.Index(fields=["date", "slot_time"], name="idx_ts_date_slot"),
        ),
        migrations.AddIndex(
            model_name="themesnapshot",
            index=models.Index(fields=["date", "is_surge"], name="idx_ts_date_surge"),
        ),
        migrations.AlterUniqueTogether(
            name="themesnapshot",
            unique_together={("date", "slot_time", "tics_id")},
        ),
        # ── ThemeLeaderCandidate ──────────────────────────────────
        migrations.CreateModel(
            name="ThemeLeaderCandidate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField(db_index=True)),
                ("slot_time", models.TimeField()),
                ("tics_id", models.IntegerField()),
                ("theme_name", models.CharField(max_length=100)),
                ("stock_code", models.CharField(max_length=10)),
                ("stock_name", models.CharField(max_length=100)),
                ("price", models.FloatField(default=0.0)),
                ("change_rate", models.FloatField(default=0.0)),
                ("trading_value", models.BigIntegerField(default=0)),
                ("market_cap", models.BigIntegerField(default=0)),
                ("score", models.FloatField(default=0.0)),
                ("rank_in_theme", models.IntegerField(default=0)),
                ("is_selected", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "snapshot",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="leaders",
                        to="myweb.themesnapshot",
                    ),
                ),
            ],
            options={
                "db_table": "theme_leader_candidate",
                "ordering": ["-date", "slot_time", "rank_in_theme"],
            },
        ),
        migrations.AddIndex(
            model_name="themeleadercandidate",
            index=models.Index(fields=["date", "stock_code"], name="idx_tlc_date_code"),
        ),
        migrations.AddIndex(
            model_name="themeleadercandidate",
            index=models.Index(fields=["date", "is_selected"], name="idx_tlc_date_sel"),
        ),
        migrations.AlterUniqueTogether(
            name="themeleadercandidate",
            unique_together={("snapshot", "stock_code")},
        ),
        # ── ThemeEntrySignal ──────────────────────────────────────
        migrations.CreateModel(
            name="ThemeEntrySignal",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField(db_index=True)),
                ("checked_at", models.DateTimeField()),
                ("tics_id", models.IntegerField(default=0)),
                ("theme_name", models.CharField(blank=True, max_length=100)),
                ("stock_code", models.CharField(max_length=10)),
                ("stock_name", models.CharField(max_length=100)),
                ("price", models.FloatField(default=0.0)),
                ("prev_high", models.FloatField(blank=True, null=True)),
                ("pullback_low", models.FloatField(blank=True, null=True)),
                ("pullback_pct", models.FloatField(blank=True, null=True)),
                ("volume_ratio", models.FloatField(blank=True, null=True)),
                ("foreign_net_buy", models.BigIntegerField(blank=True, null=True)),
                ("has_pullback", models.BooleanField(default=False)),
                ("has_breakout", models.BooleanField(default=False)),
                ("has_foreign_buying", models.BooleanField(default=False)),
                ("passed", models.BooleanField(default=False)),
                ("executed", models.BooleanField(default=False)),
                ("reason", models.CharField(blank=True, max_length=300)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="theme_entry_signals",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "theme_entry_signal",
                "ordering": ["-checked_at"],
            },
        ),
        migrations.AddIndex(
            model_name="themeentrysignal",
            index=models.Index(fields=["user", "date"], name="idx_tes_user_date"),
        ),
        migrations.AddIndex(
            model_name="themeentrysignal",
            index=models.Index(fields=["date", "stock_code"], name="idx_tes_date_code"),
        ),
        migrations.AddIndex(
            model_name="themeentrysignal",
            index=models.Index(fields=["user", "date", "passed"], name="idx_tes_passed"),
        ),
    ]
