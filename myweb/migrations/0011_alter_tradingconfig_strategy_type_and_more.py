# Generated by Django 5.2.3 on 2025-07-28 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0010_tradingdefaults'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tradingconfig',
            name='strategy_type',
            field=models.CharField(choices=[('mtt', 'MTT (Minervini Trend Template)'), ('weekly_high', '52주 신고가'), ('fifty_day_high', '50일 신고가'), ('daily_top50', '일일 Top50')], default='mtt', max_length=20),
        ),
        migrations.AlterField(
            model_name='tradingconfig',
            name='trading_mode',
            field=models.CharField(choices=[('manual', 'Manual'), ('atr', 'ATR')], max_length=20),
        ),
        migrations.AlterField(
            model_name='tradingdefaults',
            name='default_strategy_type',
            field=models.CharField(choices=[('mtt', 'MTT (Minervini Trend Template)'), ('weekly_high', '52주 신고가'), ('fifty_day_high', '50일 신고가'), ('daily_top50', '일일 Top50')], default='mtt', max_length=20),
        ),
        migrations.AlterField(
            model_name='tradingdefaults',
            name='default_trading_mode',
            field=models.CharField(choices=[('manual', 'Manual'), ('atr', 'ATR')], default='atr', max_length=20),
        ),
    ]
