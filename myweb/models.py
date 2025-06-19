from django.db import models

# Create your models here.

class Company(models.Model):
    code = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=100)
    market = models.CharField(max_length=50)
    sector = models.CharField(max_length=100)
    industry = models.CharField(max_length=200)
    # listing_date = models.DateField()
    # settle_month = models.CharField(max_length=10)
    # representative = models.CharField(max_length=100)
    # homepage = models.URLField(max_length=200)
    # region = models.CharField(max_length=50)

    def __str__(self):
        return self.name
class StockOHLCV(models.Model):
    code = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='ohlcv')  # 종목 코드 (Company 모델과 연결)
    # name = models.CharField(max_length=100)  # 종목명
    # market = models.CharField(max_length=50)  # 시장 (예: 'NASDAQ')
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
            models.UniqueConstraint(fields=['code', 'date'], name='unique_code_date')
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ['code', 'date']

    def __str__(self):
        return f"{self.code} ({self.date})"