# Generated by Django 5.2.3 on 2025-07-28 06:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0011_alter_tradingconfig_strategy_type_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tradingconfig',
            name='trading_mode',
            field=models.CharField(choices=[('manual', 'Manual'), ('atr', 'Turtle(ATR)')], max_length=20),
        ),
        migrations.AlterField(
            model_name='tradingdefaults',
            name='default_trading_mode',
            field=models.CharField(choices=[('manual', 'Manual'), ('atr', 'Turtle(ATR)')], default='atr', max_length=20),
        ),
    ]
