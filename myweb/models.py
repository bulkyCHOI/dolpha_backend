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
    change = models.FloatField(default=0.0)  # 전일 대비 변화율

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
    

class StockAnalysis(models.Model):
    code = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='analysis')  # 종목 코드 (Company 모델과 연결)
    # name = models.CharField(max_length=100)  # 종목명
    # market = models.CharField(max_length=50)  # 시장 (예: 'NASDAQ')
    date = models.DateField()  # 날짜
    ma50 = models.FloatField(default=0.0)  # 50일 이동평균
    ma150 = models.FloatField(default=0.0)  # 240일 이동평균
    ma200 = models.FloatField(default=0.0)  # 200일 이동평균
    rsScore = models.FloatField(default=0.0)  # 상대강도지수
    rsScore1m = models.FloatField(default=0.0)  # 1개월 상대강도지수
    rsScore3m = models.FloatField(default=0.0)  # 3개월 상대강도지수
    rsScore6m = models.FloatField(default=0.0)  # 6개월 상대강도지수
    rsScore12m = models.FloatField(default=0.0)  # 12개월 상대강도지수
    rsRank = models.FloatField(default=0.0)  # 상대강도랭킹
    rsRank1m = models.FloatField(default=0.0)  # 1개월 상대강도랭킹
    rsRank3m = models.FloatField(default=0.0)  # 3개월 상대강도랭킹
    rsRank6m = models.FloatField(default=0.0)  # 6개월 상대강도랭킹
    rsRank12m = models.FloatField(default=0.0)  # 12개월 상대강도랭킹
    max_52w = models.FloatField(default=0.0)  # 52주 최고가
    min_52w = models.FloatField(default=0.0)  # 52주 최저가
    max_52w_date = models.DateField(null=True, blank=True)  # 52주 최고가 날짜
    min_52w_date = models.DateField(null=True, blank=True)  # 52주 최저가 날짜
    is_minervini_trend = models.BooleanField(default=False)  # 미너비니 트렌드 템플릿 조건 충족 여부


    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = 'stock_analysis'
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=['code', 'date'], name='uniqueAnalysis_code_date')
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ['code', 'date']

    def __str__(self):
        return f"{self.code} ({self.date})"
    
    
class StockFinancialStatement(models.Model):
    code = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='financial')  # 종목 코드 (Company 모델과 연결)
    # name = models.CharField(max_length=100)  # 종목명
    # market = models.CharField(max_length=50)  # 시장 (예: 'NASDAQ')
    year = models.CharField(max_length=4)  # 연도 (예: '2023', '2022')
    quarter = models.CharField(max_length=3) # 분기 (예: 'Q1', 'Q2', 'Q3', 'Q4')
    statement_type = models.CharField(max_length=10) # sj_nm, 재무제표 종류 (예: 재무상태표 또는 손익계산서)
    account_name = models.CharField(max_length=20) # account_nm, 계정명 (예: 유동자산, 매출액 등)
    amount = models.BigIntegerField()  # thstrm_amount, 금액

    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = 'stock_financial'
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=['code', 'year', 'quarter', 'statement_type', 'account_name'], name='uniqueFinancial_code_year_quarter')
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ['year', 'quarter']

    def __str__(self):
        return f"{self.code} ({self.year}, {self.quarter})"