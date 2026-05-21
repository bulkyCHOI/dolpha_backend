from django.db.models.signals import post_delete
from django.dispatch import receiver


@receiver(post_delete, sender="myweb.TradingConfig")
def delete_minute_ohlcv_on_config_delete(sender, instance, **kwargs):
    """TradingConfig 삭제 시 해당 종목의 분봉 데이터를 모두 삭제."""
    from myweb.models import StockMinuteOhlcv
    StockMinuteOhlcv.objects.filter(stock_code=instance.stock_code).delete()
