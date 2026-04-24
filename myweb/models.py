from django.db import models
from django.contrib.auth.models import AbstractUser


# Default functions for JSONField
def default_manual_positions():
    return [100]


def default_turtle_positions():
    return [25, 25, 25, 25]


def default_turtle_pyramiding_entries():
    return ["", "", ""]


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
    open = models.FloatField(default=0.0)  # 시가
    high = models.FloatField(default=0.0)  # 고가
    low = models.FloatField(default=0.0)  # 저가
    close = models.FloatField(default=0.0)  # 종가
    volume = models.FloatField(default=0.0)  # 거래량
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
    shares_outstanding = models.BigIntegerField(null=True, blank=True)  # 상장주식수
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
    open = models.FloatField(default=0.0)  # 시가
    high = models.FloatField(default=0.0)  # 고가
    low = models.FloatField(default=0.0)  # 저가
    close = models.FloatField(default=0.0)  # 종가
    volume = models.FloatField(default=0.0)  # 거래량
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
    max_50d = models.FloatField(default=0.0)  # 50일 최고가
    min_50d = models.FloatField(default=0.0)  # 50일 최저가
    max_50d_date = models.DateField(null=True, blank=True)  # 50일 최고가 날짜
    min_50d_date = models.DateField(null=True, blank=True)  # 50일 최저가 날짜
    atr = models.FloatField(default=0.0)  # 평균 진폭 (Average True Range)
    atrRatio = models.FloatField(default=0.0)  # 평균 진폭 비율 (ATR Rate)
    is_minervini_trend = models.BooleanField(
        default=False
    )  # 미너비니 트렌드 템플릿 조건 충족 여부

    market_cap = models.BigIntegerField(null=True, blank=True)  # 시가총액 (원)

    # High Tight Flag (HTF) 패턴 관련 필드
    htf_8week_gain = models.FloatField(default=0.0)  # 8주간 최대 상승률 (%)
    htf_max_pullback = models.FloatField(default=0.0)  # 최대 조정폭 (%)
    htf_pattern_detected = models.BooleanField(default=False)  # HTF 패턴 인식 여부
    htf_pattern_start_date = models.DateField(null=True, blank=True)  # 패턴 시작일 (최저점)
    htf_pattern_peak_date = models.DateField(null=True, blank=True)  # 고점 날짜
    htf_current_status = models.CharField(
        max_length=20, 
        default='none',
        choices=[
            ('none', '해당 없음'),
            ('rising', '상승중'),
            ('pullback', '조정중'),
            ('breakout', '돌파'),
        ]
    )  # 현재 HTF 패턴 상태

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
        indexes = [
            # latest("date") 쿼리 및 날짜별 필터 최적화
            models.Index(fields=["date"], name="idx_stockanalysis_date"),
            # MTT 필터 조건 최적화 (is_minervini_trend=True, date=X)
            models.Index(
                fields=["date", "is_minervini_trend"],
                name="idx_stockanalysis_date_mtt",
            ),
            # MTT 연속 유지일 히스토리 쿼리 최적화 (code_id, date DESC)
            models.Index(
                fields=["code", "-date"],
                name="idx_sa_code_date_desc",
            ),
        ]

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
    # autobot 통합 후 서버 IP/포트/상태 필드 제거 (2026-04-17)
    # autobot_server_ip, autobot_server_port, server_status, last_connection 삭제됨
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
        ("atr", "Turtle(ATR)"),
    ]

    STRATEGY_TYPES = [
        ("mtt", "MTT (Minervini Trend Template)"),
        ("weekly_high", "52주 신고가"),
        ("fifty_day_high", "50일 신고가"),
        ("daily_top50", "일일 Top50"),
        ("htf", "High Tight Flag"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="trading_configs"
    )
    stock_code = models.CharField(max_length=10)  # 종목 코드
    stock_name = models.CharField(max_length=100)  # 종목명
    trading_mode = models.CharField(max_length=20, choices=TRADING_MODES)  # 매매 모드
    strategy_type = models.CharField(
        max_length=20, choices=STRATEGY_TYPES, default="mtt"
    )  # 전략 타입
    max_loss = models.FloatField(null=True, blank=True)  # 최대손실(%)
    stop_loss = models.FloatField(null=True, blank=True)  # 손절가(%)
    take_profit = models.FloatField(null=True, blank=True)  # 익절가(%)
    pyramiding_count = models.IntegerField(default=0)  # 피라미딩 횟수
    entry_point = models.FloatField(null=True, blank=True)  # 1차 진입시점 가격
    pyramiding_entries = models.JSONField(
        default=list, blank=True
    )  # 2차, 3차... 진입시점 배열
    positions = models.JSONField(
        default=list, blank=True
    )  # 1차, 2차, 3차... 포지션 비율 배열
    is_active = models.BooleanField(default=True)  # 활성화 여부
    trailing_stop_peak_price = models.FloatField(null=True, blank=True)  # 트레일링 스탑 고점 추적
    # autobot_config_id 제거됨 (autobot 통합, 2026-04-17)
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


class TradingDefaults(models.Model):
    """자동매매 기본값 설정 모델"""

    TRADING_MODES = [
        ("manual", "Manual"),
        ("turtle", "Turtle(ATR)"),
    ]

    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name="trading_defaults"
    )
    # 매매모드 설정
    trading_mode = models.CharField(
        max_length=20, choices=TRADING_MODES, default="turtle"
    )  # 현재 선택된 매매 모드

    # Manual 모드 설정값들
    manual_max_loss = models.FloatField(default=2.0)  # Manual 최대손실(%)
    manual_stop_loss = models.FloatField(default=8.0)  # Manual 손절가(%)
    manual_take_profit = models.FloatField(null=True, blank=True)  # Manual 익절가(%)
    manual_pyramiding_count = models.IntegerField(default=0)  # Manual 피라미딩 횟수
    manual_position_size = models.FloatField(default=100.0)  # Manual 포지션 크기(%)
    manual_positions = models.JSONField(
        default=default_manual_positions, blank=True
    )  # Manual 포지션 비율 배열
    manual_pyramiding_entries = models.JSONField(
        default=list, blank=True
    )  # Manual 진입시점 배열
    manual_use_trailing_stop = models.BooleanField(
        default=True
    )  # Manual 트레일링 스탑 사용
    manual_trailing_stop_trigger = models.FloatField(
        default=8.0
    )  # Manual 트레일링 스탑 시작 조건(%)
    manual_trailing_stop_percent = models.FloatField(
        default=8.0
    )  # Manual 트레일링 스탑 비율(%)

    # Turtle(ATR) 모드 설정값들
    turtle_max_loss = models.FloatField(default=2.0)  # Turtle 최대손실(ATR)
    turtle_stop_loss = models.FloatField(default=2.0)  # Turtle 손절가(ATR)
    turtle_take_profit = models.FloatField(null=True, blank=True)  # Turtle 익절가(ATR)
    turtle_pyramiding_count = models.IntegerField(default=3)  # Turtle 피라미딩 횟수
    turtle_position_size = models.FloatField(default=25.0)  # Turtle 포지션 크기(%)
    turtle_positions = models.JSONField(
        default=default_turtle_positions, blank=True
    )  # Turtle 포지션 비율 배열
    turtle_pyramiding_entries = models.JSONField(
        default=default_turtle_pyramiding_entries, blank=True
    )  # Turtle 진입시점 배열
    turtle_use_trailing_stop = models.BooleanField(
        default=True
    )  # Turtle 트레일링 스탑 사용
    turtle_trailing_stop_trigger = models.FloatField(
        default=2.0
    )  # Turtle 트레일링 스탑 시작 조건(ATR)
    turtle_trailing_stop_percent = models.FloatField(
        default=3.0
    )  # Turtle 트레일링 스탑 비율(ATR)

    # 진입/청산 기본값 (공통)
    default_entry_trigger = models.FloatField(
        default=1.0
    )  # 기본 진입 트리거 (ATR 배수)
    default_exit_trigger = models.FloatField(default=2.0)  # 기본 청산 트리거 (ATR 배수)

    # 메타데이터
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}의 자동매매 기본값 설정"

    class Meta:
        verbose_name = "자동매매 기본값 설정"
        verbose_name_plural = "자동매매 기본값 설정들"


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


class TradingSummary(models.Model):
    """매매복기를 위한 종목별 거래 요약 모델"""
    
    TRADING_MODES = [
        ("manual", "Manual"),
        ("turtle", "Turtle"),
    ]
    
    FINAL_STATUS = [
        ("CLOSED", "Closed"),
        ("HOLDING", "Holding"),
    ]
    
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="trading_summaries"
    )
    stock_code = models.CharField(max_length=10)  # 종목 코드
    stock_name = models.CharField(max_length=100)  # 종목명
    first_entry_date = models.DateTimeField(null=True, blank=True)  # 첫 매수일
    last_exit_date = models.DateTimeField(null=True, blank=True)  # 마지막 매도일
    total_buy_amount = models.BigIntegerField(default=0)  # 총 매수 금액
    total_sell_amount = models.BigIntegerField(default=0)  # 총 매도 금액
    total_profit_loss = models.BigIntegerField(default=0)  # 총 손익
    profit_loss_percent = models.FloatField(default=0.0)  # 손익률 (%)
    max_drawdown = models.FloatField(null=True, blank=True)  # 최대 손실률
    holding_days = models.FloatField(default=0.0)  # 보유 일수
    entry_count = models.IntegerField(default=0)  # 매수 횟수
    exit_count = models.IntegerField(default=0)  # 매도 횟수
    trading_mode = models.CharField(max_length=20, choices=TRADING_MODES)  # 거래 모드
    win_rate = models.FloatField(default=0.0)  # 승률
    avg_holding_days = models.FloatField(default=0.0)  # 평균 보유 일수
    max_profit_percent = models.FloatField(null=True, blank=True)  # 최대 수익률
    final_status = models.CharField(max_length=10, choices=FINAL_STATUS)  # 최종 상태
    memo = models.TextField(blank=True)  # 사용자 메모
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "trading_summary"
        ordering = ["-updated_at"]
        unique_together = ["user", "stock_code", "first_entry_date"]
        indexes = [
            models.Index(fields=["user", "final_status"]),
            models.Index(fields=["user", "trading_mode"]),
            models.Index(fields=["stock_code"]),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.stock_name} ({self.final_status})"


class TradeEntry(models.Model):
    """
    KIS API를 통한 실제 매매 체결 기록.

    과거 autobot(FastAPI)이 관리하던 개별 주문 내역을 Django DB로 통합.
    TradingConfig → 전략 설정, TradingSummary → 종목별 집계와 연결된다.
    """

    TRADE_TYPES = [
        ("BUY", "매수"),
        ("SELL", "매도"),
    ]

    ORDER_STATUS = [
        ("SUBMITTED", "주문접수"),
        ("FILLED", "체결완료"),
        ("PARTIAL", "부분체결"),
        ("CANCELLED", "취소"),
        ("FAILED", "실패"),
    ]

    ENTRY_TYPES = [
        ("INITIAL", "최초진입"),
        ("PYRAMIDING", "피라미딩"),
        ("EXIT_PARTIAL", "부분청산"),
        ("EXIT_FULL", "전량청산"),
        ("STOP_LOSS", "손절"),
        ("TRAILING_STOP", "트레일링스탑"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="trade_entries"
    )
    trading_config = models.ForeignKey(
        TradingConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="entries",
    )
    trading_summary = models.ForeignKey(
        TradingSummary,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="entries",
    )

    stock_code = models.CharField(max_length=10)   # 종목 코드
    stock_name = models.CharField(max_length=100)  # 종목명

    trade_type = models.CharField(max_length=10, choices=TRADE_TYPES)  # 매수/매도
    entry_type = models.CharField(
        max_length=20, choices=ENTRY_TYPES, default="INITIAL"
    )  # 진입/청산 유형

    # ── 주문 정보 ──────────────────────────────
    order_no = models.CharField(max_length=20, blank=True)  # KIS 주문번호
    order_quantity = models.IntegerField(default=0)         # 주문 수량
    order_price = models.DecimalField(
        max_digits=12, decimal_places=2, default=0
    )  # 주문가 (0=시장가)

    # ── 체결 정보 ──────────────────────────────
    filled_quantity = models.IntegerField(default=0)   # 체결 수량
    filled_price = models.DecimalField(
        max_digits=12, decimal_places=2, default=0
    )  # 평균 체결가
    filled_amount = models.DecimalField(
        max_digits=15, decimal_places=2, default=0
    )  # 체결 금액 (filled_price × filled_quantity)

    # ── 손익 (매도 시) ─────────────────────────
    profit_loss = models.DecimalField(
        max_digits=15, decimal_places=2, null=True, blank=True
    )  # 손익 금액
    profit_loss_percent = models.FloatField(null=True, blank=True)  # 손익률 (%)

    status = models.CharField(
        max_length=20, choices=ORDER_STATUS, default="SUBMITTED"
    )  # 주문 상태

    # ── ATR 기반 매매 보조 정보 ───────────────
    atr_value = models.FloatField(null=True, blank=True)    # 진입 시 ATR 값
    stop_price = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )  # 손절가

    note = models.TextField(blank=True)  # 비고/메모

    ordered_at = models.DateTimeField(null=True, blank=True)  # 주문 시각
    filled_at = models.DateTimeField(null=True, blank=True)   # 체결 시각
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "trade_entry"
        ordering = ["-ordered_at"]
        indexes = [
            models.Index(fields=["user", "stock_code"]),
            models.Index(fields=["user", "status"]),
            models.Index(fields=["order_no"]),
        ]

    def __str__(self):
        return (
            f"{self.user.username} - {self.stock_name} "
            f"{self.trade_type} {self.filled_quantity}주 ({self.status})"
        )
