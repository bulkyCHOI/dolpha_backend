from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.
class StockIndex(models.Model):
    code = models.CharField(max_length=10, primary_key=True)  # 지수 코드
    name = models.CharField(max_length=100)  # 지수명
    market = models.CharField(max_length=50)  # 시장 (예: 'KOSPI', 'NASDAQ')

    def __str__(self):
        return self.name


class IndexOHLCV(models.Model):
    code = models.ForeignKey(
        StockIndex, on_delete=models.CASCADE, related_name="index_ohlcv"
    )  # 지수 코드 (StockIndex 모델과 연결)
    date = models.DateField()  # 날짜
    open = models.IntegerField()  # 시가
    high = models.IntegerField()  # 고가
    low = models.IntegerField()  # 저가
    close = models.IntegerField()  # 종가
    volume = models.IntegerField()  # 거래량
    change = models.FloatField(default=0.0)  # 전일 대비 변화율

    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = "index_ohlcv"
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=["code", "date"], name="unique_index_date")
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ["code", "date"]

    def __str__(self):
        return f"{self.code} ({self.date})"


class Company(models.Model):
    code = models.CharField(max_length=10, primary_key=True)
    indices = models.ManyToManyField(
        StockIndex, related_name="companies"
    )  # 다대다 관계
    name = models.CharField(max_length=100)
    market = models.CharField(max_length=50)
    sector = models.CharField(
        max_length=100, null=True, blank=True
    )  # 섹터 (예: 'Technology')
    industry = models.CharField(
        max_length=200, null=True, blank=True
    )  # 업종 (예: 'Software & Services')
    # listing_date = models.DateField()
    # settle_month = models.CharField(max_length=10)
    # representative = models.CharField(max_length=100)
    # homepage = models.URLField(max_length=200)
    # region = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class StockOHLCV(models.Model):
    code = models.ForeignKey(
        Company, on_delete=models.CASCADE, related_name="ohlcv"
    )  # 종목 코드 (Company 모델과 연결)
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
        db_table = "stock_ohlcv"
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=["code", "date"], name="unique_code_date")
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ["code", "date"]

    def __str__(self):
        return f"{self.code} ({self.date})"


class StockAnalysis(models.Model):
    code = models.ForeignKey(
        Company, on_delete=models.CASCADE, related_name="analysis"
    )  # 종목 코드 (Company 모델과 연결)
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
    atr = models.FloatField(default=0.0)  # 평균 진폭 (Average True Range)
    atrRatio = models.FloatField(default=0.0)  # 평균 진폭 비율 (ATR Rate)
    is_minervini_trend = models.BooleanField(
        default=False
    )  # 미너비니 트렌드 템플릿 조건 충족 여부

    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = "stock_analysis"
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(
                fields=["code", "date"], name="uniqueAnalysis_code_date"
            )
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ["code", "date"]

    def __str__(self):
        return f"{self.code} ({self.date})"


class StockFinancialStatement(models.Model):
    code = models.ForeignKey(
        Company, on_delete=models.CASCADE, related_name="financial"
    )  # 종목 코드 (Company 모델과 연결)
    # name = models.CharField(max_length=100)  # 종목명
    # market = models.CharField(max_length=50)  # 시장 (예: 'NASDAQ')
    year = models.CharField(max_length=4)  # 연도 (예: '2023', '2022')
    quarter = models.CharField(max_length=3)  # 분기 (예: 'Q1', 'Q2', 'Q3', 'Q4')
    statement_type = models.CharField(
        max_length=10
    )  # sj_nm, 재무제표 종류 (예: 재무상태표 또는 손익계산서)
    account_name = models.CharField(
        max_length=20
    )  # account_nm, 계정명 (예: 유동자산, 매출액 등)
    amount = models.BigIntegerField()  # thstrm_amount, 금액

    class Meta:
        # 테이블 이름 지정 (기존 테이블과 매핑)
        db_table = "stock_financial"
        # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(
                fields=["code", "year", "quarter", "statement_type", "account_name"],
                name="uniqueFinancial_code_year_quarter",
            )
        ]
        # 관리자 패널에서 보기 좋게 정렬
        ordering = ["year", "quarter"]

    def __str__(self):
        return f"{self.code} ({self.year}, {self.quarter})"


# 사용자 인증 관련 모델들
class User(AbstractUser):
    google_id = models.CharField(max_length=100, unique=True, null=True, blank=True)
    profile_picture = models.URLField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "auth_user"

    def __str__(self):
        return self.username


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    trading_server_ip = models.GenericIPAddressField(
        null=True, blank=True
    )  # 기존 필드 (호환성 유지)
    autobot_server_ip = models.GenericIPAddressField(
        null=True, blank=True
    )  # autobot 서버 IP
    autobot_server_port = models.IntegerField(default=8080)  # autobot 서버 포트
    server_status = models.CharField(
        max_length=20,
        default="offline",
        choices=[
            ("online", "온라인"),
            ("offline", "오프라인"),
            ("error", "오류"),
        ],
    )  # 서버 상태
    last_connection = models.DateTimeField(null=True, blank=True)  # 마지막 연결 시간
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}의 프로필"


class FavoriteStock(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="favorite_stocks"
    )
    stock_code = models.CharField(max_length=10)
    stock_name = models.CharField(max_length=100)
    memo = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["user", "stock_code"]

    def __str__(self):
        return f"{self.user.username} - {self.stock_name}"


class TradingConfig(models.Model):
    TRADING_MODES = [
        ("manual", "Manual"),
        ("turtle", "Turtle(ATR)"),
    ]
    
    STRATEGY_TYPES = [
        ("mtt", "MTT (Minervini Trend Template)"),
        ("weekly_high", "52주 신고가"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="trading_configs"
    )
    stock_code = models.CharField(max_length=10)  # 종목 코드
    stock_name = models.CharField(max_length=100)  # 종목명
    trading_mode = models.CharField(max_length=20, choices=TRADING_MODES)  # 매매 모드
    strategy_type = models.CharField(max_length=20, choices=STRATEGY_TYPES, default="mtt")  # 전략 타입
    max_loss = models.FloatField(null=True, blank=True)  # 최대손실(%)
    stop_loss = models.FloatField(null=True, blank=True)  # 손절가(%)
    take_profit = models.FloatField(null=True, blank=True)  # 익절가(%)
    pyramiding_count = models.IntegerField(default=0)  # 피라미딩 횟수
    entry_point = models.FloatField(null=True, blank=True)  # 1차 진입시점 가격
    pyramiding_entries = models.JSONField(default=list, blank=True)  # 2차, 3차... 진입시점 배열
    positions = models.JSONField(default=list, blank=True)  # 1차, 2차, 3차... 포지션 비율 배열
    is_active = models.BooleanField(default=True)  # 활성화 여부
    autobot_config_id = models.IntegerField(
        null=True, blank=True
    )  # autobot 서버의 설정 ID
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [
            "user",
            "stock_code",
            "strategy_type",
            "is_active",
        ]  # 사용자별 종목별 전략타입별 하나의 활성 설정만 허용
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.user.username} - {self.stock_name} ({self.strategy_type}/{self.trading_mode})"


class TradingResult(models.Model):
    TRADE_TYPES = [
        ("BUY", "매수"),
        ("SELL", "매도"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="trading_results"
    )
    trading_config = models.ForeignKey(
        TradingConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="results",
    )  # 관련 자동매매 설정
    stock_code = models.CharField(max_length=10)
    stock_name = models.CharField(max_length=100)
    trade_type = models.CharField(max_length=10, choices=TRADE_TYPES)
    quantity = models.IntegerField()
    price = models.DecimalField(max_digits=12, decimal_places=2)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2)
    profit_loss = models.DecimalField(
        max_digits=15, decimal_places=2, null=True, blank=True
    )
    trade_date = models.DateTimeField()
    review = models.TextField(blank=True)  # 매매복기
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-trade_date"]

    def __str__(self):
        return f"{self.user.username} - {self.stock_name} {self.trade_type}"
