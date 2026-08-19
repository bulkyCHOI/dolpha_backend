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
        ("theme_surge", "급등테마주"),
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
    staged_exit_completed_stages = models.JSONField(default=list, blank=True)  # 완료된 분할 익절 단계 [1, 2]
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


class InvestorFlowSnapshot(models.Model):
    """자동매매 대상 종목의 매매동향(투자자별/외국인·기관/프로그램/회원사) 장중 마지막 스냅샷.

    KIS 매매동향 API는 장중(09:00~15:30 KST)에만 조회 가능하므로, 장 마감 직전에
    자동매매 설정 목록의 종목들에 대해 미리 수집해 두고 장 마감 후 조회할 수 있도록 한다.
    """

    stock_code = models.CharField(max_length=10)
    stock_name = models.CharField(max_length=100, blank=True)
    date = models.DateField()
    investor_today = models.JSONField(default=dict, blank=True)  # 당일 투자자별(개인/외국인/기관) 순매수
    foreign_total = models.JSONField(default=dict, blank=True)  # 외국인/기관 가집계 시간대별
    program_trade = models.JSONField(default=dict, blank=True)  # 프로그램매매 추이
    member_firm = models.JSONField(default=dict, blank=True)  # 회원사별 매매동향
    captured_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["stock_code", "date"]
        ordering = ["-date", "stock_code"]

    def __str__(self):
        return f"{self.stock_code} - {self.date}"


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

    # 분할 익절 설정
    STAGED_EXIT_TYPES = [
        ("none", "미사용"),
        ("ma", "이동평균선"),
        ("dead_cross", "데드크로스"),
        ("new_low", "N일 신저가"),
    ]
    staged_exit_type = models.CharField(
        max_length=20, choices=STAGED_EXIT_TYPES, default="none"
    )  # 분할 익절 방식

    # 이동평균선 분할 익절
    ma_stage1_period   = models.IntegerField(default=5)    # 1단계 이동평균 기간
    ma_stage1_sell_pct = models.FloatField(default=30.0)   # 1단계 매도 비율(%)
    ma_stage2_period   = models.IntegerField(default=20)   # 2단계 이동평균 기간
    ma_stage2_sell_pct = models.FloatField(default=50.0)   # 2단계 매도 비율(%)
    ma_stage3_period   = models.IntegerField(default=60)   # 3단계 이동평균 기간
    ma_stage3_sell_pct = models.FloatField(default=100.0)  # 3단계 매도 비율(%)

    # 데드크로스 분할 익절
    dc_stage1_short    = models.IntegerField(default=5)    # 1단계 단기 MA
    dc_stage1_long     = models.IntegerField(default=10)   # 1단계 장기 MA
    dc_stage1_sell_pct = models.FloatField(default=30.0)   # 1단계 매도 비율(%)
    dc_stage2_short    = models.IntegerField(default=10)   # 2단계 단기 MA
    dc_stage2_long     = models.IntegerField(default=30)   # 2단계 장기 MA
    dc_stage2_sell_pct = models.FloatField(default=50.0)   # 2단계 매도 비율(%)
    dc_stage3_short    = models.IntegerField(default=30)   # 3단계 단기 MA
    dc_stage3_long     = models.IntegerField(default=60)   # 3단계 장기 MA
    dc_stage3_sell_pct = models.FloatField(default=100.0)  # 3단계 매도 비율(%)

    # N일 신저가 분할 익절
    nl_stage1_days     = models.IntegerField(default=5)    # 1단계 기간
    nl_stage1_sell_pct = models.FloatField(default=30.0)   # 1단계 매도 비율(%)
    nl_stage2_days     = models.IntegerField(default=10)   # 2단계 기간
    nl_stage2_sell_pct = models.FloatField(default=50.0)   # 2단계 매도 비율(%)
    nl_stage3_days     = models.IntegerField(default=20)   # 3단계 기간
    nl_stage3_sell_pct = models.FloatField(default=100.0)  # 3단계 매도 비율(%)

    # ── 급등테마주 전략 설정 ─────────────────────────────────
    # 진입만 별도 로직(테마 급등 + 1분봉 눌림목 돌파 + 외국인 수급)을 쓰고,
    # 익절/손절/트레일링스탑/분할익절은 위 Manual 기본값을 그대로 따른다.
    theme_surge_enabled = models.BooleanField(default=False)          # 급등테마주 자동매매 사용
    theme_surge_max_candidates = models.IntegerField(default=3)       # 동시 추적 최대 후보 종목 수
    theme_surge_min_fluctuation = models.FloatField(default=3.0)      # 급등 테마 판정 등락률(%)
    theme_surge_min_trading_value = models.BigIntegerField(
        default=50_000_000_000
    )                                                                  # 급등 테마 판정 거래대금 하한(원)
    theme_surge_use_foreign_filter = models.BooleanField(default=True)  # 외국인 매수세 필터 사용

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


class StockMinuteOhlcv(models.Model):
    """자동매매 대상 종목의 분봉 OHLCV (트레이딩 사이클에서 D-1분 단위 수집)."""

    stock_code = models.CharField(max_length=10, db_index=True)
    bar_datetime = models.DateTimeField()  # KST 기준 분봉 시작 시각 (초=0)
    open = models.FloatField(default=0.0)
    high = models.FloatField(default=0.0)
    low = models.FloatField(default=0.0)
    close = models.FloatField(default=0.0)
    volume = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "stock_minute_ohlcv"
        unique_together = [("stock_code", "bar_datetime")]
        ordering = ["stock_code", "bar_datetime"]
        indexes = [
            models.Index(fields=["stock_code", "bar_datetime"], name="idx_smo_code_dt"),
        ]

    def __str__(self):
        return f"{self.stock_code} {self.bar_datetime}"


class ThemeSnapshot(models.Model):
    """토스증권 '지금 뜨는 산업' 5분 단위 스냅샷 (테마 1개 = 1행).

    09:00~15:30 사이 5분마다 수집하며, 하루치를 모으면 그대로 급등 테마 타임라인이 된다.
    """

    date = models.DateField(db_index=True)
    slot_time = models.TimeField()                        # 5분 슬롯 시각 (09:00 ~ 15:30)
    tics_id = models.IntegerField()                       # 토스 산업분류(TICS) ID
    theme_name = models.CharField(max_length=100)         # 테마명
    rank = models.IntegerField()                          # 해당 슬롯 내 등락률 순위
    fluctuation_rate = models.FloatField(default=0.0)     # 테마 등락률(%) — 예: 7.44
    trading_value = models.BigIntegerField(default=0)     # 테마 거래대금(원)
    market_cap = models.BigIntegerField(default=0)        # 테마 시가총액(원)
    stock_count = models.IntegerField(default=0)          # 테마 구성 종목 수
    is_surge = models.BooleanField(default=False)         # 급등 테마 판정 여부
    surge_reason = models.CharField(max_length=200, blank=True)  # 급등 판정 근거
    momentum = models.FloatField(default=0.0)             # 직전 슬롯 대비 등락률 변화(%p)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "theme_snapshot"
        unique_together = [("date", "slot_time", "tics_id")]
        ordering = ["-date", "slot_time", "rank"]
        indexes = [
            models.Index(fields=["date", "slot_time"], name="idx_ts_date_slot"),
            models.Index(fields=["date", "is_surge"], name="idx_ts_date_surge"),
        ]

    def __str__(self):
        return f"{self.date} {self.slot_time} {self.theme_name} ({self.fluctuation_rate:+.2f}%)"


class ThemeLeaderCandidate(models.Model):
    """급등 테마의 주도주 후보 (거래대금 + 상승률 복합 점수 기준 상위 N개)."""

    snapshot = models.ForeignKey(
        ThemeSnapshot, on_delete=models.CASCADE, related_name="leaders"
    )
    date = models.DateField(db_index=True)      # snapshot.date 비정규화 (타임라인 조회용)
    slot_time = models.TimeField()
    tics_id = models.IntegerField()
    theme_name = models.CharField(max_length=100)

    stock_code = models.CharField(max_length=10)   # 6자리 종목코드
    stock_name = models.CharField(max_length=100)
    price = models.FloatField(default=0.0)         # 스냅샷 시점 가격
    change_rate = models.FloatField(default=0.0)   # 상승률(%)
    trading_value = models.BigIntegerField(default=0)  # 거래대금(원)
    market_cap = models.BigIntegerField(default=0)

    score = models.FloatField(default=0.0)         # 거래대금·상승률 복합 점수 (0~1)
    rank_in_theme = models.IntegerField(default=0)  # 테마 내 순위 (1 = 1등 종목)
    is_selected = models.BooleanField(default=False)  # 자동매매 후보로 등록되었는지
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "theme_leader_candidate"
        unique_together = [("snapshot", "stock_code")]
        ordering = ["-date", "slot_time", "rank_in_theme"]
        indexes = [
            models.Index(fields=["date", "stock_code"], name="idx_tlc_date_code"),
            models.Index(fields=["date", "is_selected"], name="idx_tlc_date_sel"),
        ]

    def __str__(self):
        return f"{self.date} {self.theme_name} #{self.rank_in_theme} {self.stock_name}"


class ThemeEntrySignal(models.Model):
    """급등테마주 후보에 대한 1분봉 진입 조건 판정 로그.

    눌림목 → 전고점 돌파 → 외국인 매수세 3단 조건을 매 사이클 평가한 결과를 남겨
    타임라인에 진입 시점 마커로 표시하고, 미진입 사유를 추적한다.
    """

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="theme_entry_signals"
    )
    date = models.DateField(db_index=True)
    checked_at = models.DateTimeField()
    tics_id = models.IntegerField(default=0)
    theme_name = models.CharField(max_length=100, blank=True)
    stock_code = models.CharField(max_length=10)
    stock_name = models.CharField(max_length=100)

    price = models.FloatField(default=0.0)             # 판정 시점 현재가
    prev_high = models.FloatField(null=True, blank=True)      # 직전 스윙 고점(전고점)
    pullback_low = models.FloatField(null=True, blank=True)   # 눌림목 저점
    pullback_pct = models.FloatField(null=True, blank=True)   # 전고점 대비 눌림 깊이(%)
    volume_ratio = models.FloatField(null=True, blank=True)   # 돌파봉 거래량 / 평균 거래량
    foreign_net_buy = models.BigIntegerField(null=True, blank=True)  # 외국인 순매수(주)

    has_pullback = models.BooleanField(default=False)  # 눌림목 성립
    has_breakout = models.BooleanField(default=False)  # 전고점 돌파
    has_foreign_buying = models.BooleanField(default=False)  # 외국인 매수세 포착
    passed = models.BooleanField(default=False)        # 3단 조건 모두 충족
    executed = models.BooleanField(default=False)      # 실제 매수 주문 실행
    reason = models.CharField(max_length=300, blank=True)  # 판정 사유
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "theme_entry_signal"
        ordering = ["-checked_at"]
        indexes = [
            models.Index(fields=["user", "date"], name="idx_tes_user_date"),
            models.Index(fields=["date", "stock_code"], name="idx_tes_date_code"),
            models.Index(fields=["user", "date", "passed"], name="idx_tes_passed"),
        ]

    def __str__(self):
        status = "진입" if self.executed else ("충족" if self.passed else "대기")
        return f"{self.checked_at:%H:%M} {self.stock_name} [{status}]"


class DailyAccountSnapshot(models.Model):
    """일별 계좌 잔고 스냅샷 (차트용)"""

    user = models.ForeignKey(
        "myweb.User", on_delete=models.CASCADE, related_name="account_snapshots"
    )
    date = models.DateField()
    total_money = models.BigIntegerField(default=0)       # 총 평가금액
    stock_money = models.BigIntegerField(default=0)       # 주식 평가금액
    remain_money = models.BigIntegerField(default=0)      # 예수금
    stock_revenue = models.BigIntegerField(default=0)     # 순이익
    confirmed_capital = models.BigIntegerField(default=0) # 납입원금

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "daily_account_snapshot"
        unique_together = [("user", "date")]
        ordering = ["date"]

    def __str__(self):
        return f"{self.user.username} - {self.date} 총잔고:{self.total_money:,}"
