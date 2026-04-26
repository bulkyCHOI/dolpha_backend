from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("myweb", "0028_add_trailing_stop_trigger"),
    ]

    operations = [
        # TradingConfig: 완료된 분할 익절 단계 추적
        migrations.AddField(
            model_name="tradingconfig",
            name="staged_exit_completed_stages",
            field=models.JSONField(blank=True, default=list),
        ),
        # TradingDefaults: 분할 익절 타입
        migrations.AddField(
            model_name="tradingdefaults",
            name="staged_exit_type",
            field=models.CharField(
                choices=[
                    ("none", "미사용"),
                    ("ma", "이동평균선"),
                    ("dead_cross", "데드크로스"),
                    ("new_low", "N일 신저가"),
                ],
                default="none",
                max_length=20,
            ),
        ),
        # 이동평균선 분할 익절
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage1_period",
            field=models.IntegerField(default=5),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage1_sell_pct",
            field=models.FloatField(default=30.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage2_period",
            field=models.IntegerField(default=20),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage2_sell_pct",
            field=models.FloatField(default=50.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage3_period",
            field=models.IntegerField(default=60),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="ma_stage3_sell_pct",
            field=models.FloatField(default=100.0),
        ),
        # 데드크로스 분할 익절
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage1_short",
            field=models.IntegerField(default=5),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage1_long",
            field=models.IntegerField(default=10),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage1_sell_pct",
            field=models.FloatField(default=30.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage2_short",
            field=models.IntegerField(default=10),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage2_long",
            field=models.IntegerField(default=30),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage2_sell_pct",
            field=models.FloatField(default=50.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage3_short",
            field=models.IntegerField(default=30),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage3_long",
            field=models.IntegerField(default=60),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="dc_stage3_sell_pct",
            field=models.FloatField(default=100.0),
        ),
        # N일 신저가 분할 익절
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage1_days",
            field=models.IntegerField(default=5),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage1_sell_pct",
            field=models.FloatField(default=30.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage2_days",
            field=models.IntegerField(default=10),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage2_sell_pct",
            field=models.FloatField(default=50.0),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage3_days",
            field=models.IntegerField(default=20),
        ),
        migrations.AddField(
            model_name="tradingdefaults",
            name="nl_stage3_sell_pct",
            field=models.FloatField(default=100.0),
        ),
    ]
