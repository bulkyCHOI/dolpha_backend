from django.db import models

# Create your models here.
class StockOHLCV(models.Model):
    ticker = models.CharField(max_length=20)  # 주식 코드 (예: 'AAPL')
    name = models.CharField(max_length=100)  # 종목명
    market = models.CharField(max_length=50)  # 시장 (예: 'NASDAQ')
    date = models.DateField()  # 날짜
    open = models.IntegerField()  # 시가
    high = models.IntegerField()  # 고가
    low = models.IntegerField()  # 저가
    close = models.IntegerField()  # 종가
    volume = models.IntegerField()  # 거래량

    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = 'stock_ohlcv'
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=['ticker', 'date'], name='unique_ticker_date')
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ['ticker', 'date']

    def __str__(self):
        return f"{self.ticker} ({self.date})"