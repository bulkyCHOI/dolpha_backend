"""급등테마주 전략 파라미터.

모든 임계값을 한 곳에 모아 매직넘버를 제거한다.
유저별로 조정 가능한 값(테마 급등 기준, 외국인 필터 사용 여부 등)은
TradingDefaults 에 저장되며 여기 값은 그 기본값 역할을 한다.
"""

from datetime import time

# ── 타임라인 ────────────────────────────────────────────────
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(15, 30)
SLOT_MINUTES = 1                      # 스캔 주기(분) = 타임라인 슬롯 단위
TOTAL_SLOTS = 391                     # 09:00 ~ 15:30, 1분 간격 슬롯 수

# ── 토스증권 API ────────────────────────────────────────────
TOSS_API_BASE = "https://wts-info-api.tossinvest.com"
TOSS_RANKING_PATH = "/api/v2/dashboard/wts/overview/tics/ranking"
TOSS_STOCKS_PATH = "/api/v2/dashboard/wts/overview/tics/{tics_id}/stocks"
TOSS_FALLBACK_RANKING_PATH = "/api/v1/tics/rankings"
TOSS_NATION = "KR"
TOSS_DURATION = "1d"
TOSS_SORT_BY = "FLUCTUATION_RATE"
TOSS_TIMEOUT_SEC = 10
TOSS_MAX_RETRY = 3
TOSS_RETRY_BACKOFF_SEC = 1.0
TOSS_PAGE_SIZE = 10                   # /stocks 엔드포인트 1페이지 반환 개수
TOSS_MAX_PAGES = 10                   # 테마당 최대 100종목까지 수집
TOSS_REQUEST_INTERVAL_SEC = 0.15      # 연속 호출 간 최소 간격 (과도한 부하 방지)

# ── 급등 테마 판정 ──────────────────────────────────────────
SURGE_MIN_FLUCTUATION_PCT = 3.0       # 테마 등락률 하한(%)
SURGE_MIN_TRADING_VALUE = 50_000_000_000   # 테마 거래대금 하한(원) = 500억
SURGE_MIN_STOCK_COUNT = 3             # 구성 종목 3개 미만 테마는 노이즈로 제외
SURGE_MIN_MOMENTUM_PCT = 0.3          # 모멘텀 비교 구간 대비 등락률 증가폭 하한(%p)
MOMENTUM_LOOKBACK_MINUTES = 5         # 모멘텀 비교 기준 시점(분 전) — 슬롯 주기와 무관하게 고정
SURGE_TOP_N = 10                      # 스냅샷으로 저장할 상위 테마 수
SURGE_MAX_THEMES_PER_SLOT = 3         # 한 슬롯에서 후보를 뽑을 최대 테마 수
THEME_PULLBACK_MOMENTUM_TOLERANCE_PCT = 1.5  # 테마 눌림목 모멘텀 보존 허용 오차(%p)

# ── 1등 종목(주도주) 선정 ───────────────────────────────────
LEADER_WEIGHT_TRADING_VALUE = 0.5     # 거래대금 가중치
LEADER_WEIGHT_CHANGE_RATE = 0.5       # 상승률 가중치
LEADER_MIN_CHANGE_RATE_PCT = 1.0      # 후보 최소 상승률(%) — P1.5: 2.0→1.0 (급등테마 51%가 후보 0명이던 문제)
LEADER_MIN_CHANGE_RATE_THEME_RATIO = 0.35  # P1.5: 테마 등락률의 이 비율 미만인 종목은 후행주로 제외
LEADER_MIN_TRADING_VALUE = 5_000_000_000   # 후보 최소 거래대금(원) = 50억
LEADER_MIN_MARKET_CAP = 30_000_000_000     # 후보 최소 시가총액(원) — P1.5: 500억→300억
LEADER_MAX_CHANGE_RATE_PCT = 25.0     # 상한가 근접(+25% 초과) 종목은 추격 매수 제외
LEADER_STORE_COUNT = 3                # 테마당 저장할 후보 수 (1등 + 추적용 2등·3등)

# ── 시간대별 동적 최소 거래대금 ──────────────────────────────
def get_theme_min_trading_value(slot_or_time: time | None = None) -> int:
    """시간대별 급등 테마 최소 거래대금 기준(원)을 반환한다."""
    if slot_or_time is None:
        return SURGE_MIN_TRADING_VALUE
    if slot_or_time <= time(9, 10):
        return 10_000_000_000
    if slot_or_time <= time(9, 20):
        return 20_000_000_000
    if slot_or_time <= time(9, 30):
        return 35_000_000_000
    return SURGE_MIN_TRADING_VALUE


def get_leader_min_trading_value(slot_or_time: time | None = None) -> int:
    """시간대별 주도주 최소 거래대금 기준(원)을 반환한다."""
    if slot_or_time is None:
        return LEADER_MIN_TRADING_VALUE
    if slot_or_time <= time(9, 10):
        return 1_000_000_000
    if slot_or_time <= time(9, 20):
        return 2_000_000_000
    if slot_or_time <= time(9, 30):
        return 3_500_000_000
    return LEADER_MIN_TRADING_VALUE


# ── 모닝 세션 파라미터 (09:00 ~ 09:45 이전 빠른 진입) ────────
MORNING_MIN_BARS = 6                   # 모닝 판정에 필요한 최소 분봉 개수
MORNING_LOOKBACK_BARS = 30             # 모닝 탐색 구간(분봉 개수)
MORNING_PEAK_MIN_RISE_PCT = 2.0        # 시초가 대비 고점 최소 상승폭(%)
MORNING_PULLBACK_MIN_BARS = 2          # 모닝 눌림 최소 봉 수
MORNING_PULLBACK_MAX_BARS = 12         # 모닝 눌림 최대 봉 수
MORNING_PULLBACK_MIN_PCT = 0.5         # 모닝 눌림 최소 깊이(%) — P1: 1.0→0.5 (얕은 눌림 회복)
MORNING_PULLBACK_MAX_PCT = 4.5         # 모닝 눌림 최대 깊이(%)
MORNING_PULLBACK_VOLUME_RATIO_MAX = 1.0   # 모닝 눌림 거래량 비율 상한 — P1: 0.85→1.0
MORNING_BREAKOUT_VOLUME_RATIO_MIN = 2.0   # 모닝 돌파 거래량 비율 하한

# ── 1분봉 진입 판정 (레귤러 세션) ───────────────────────────
ENTRY_MAX_BAR_AGE_MIN = 5             # 마지막 분봉이 이보다 낡으면 판정 보류 (낡은 데이터 진입 차단)
ENTRY_LOOKBACK_BARS = 120             # 전고점 탐색 구간(분봉 개수)
ENTRY_MIN_BARS = 35                   # 판정에 필요한 최소 분봉 개수 — P1.5: 45→35 (09:35부터 레귤러 판정, "전고점 미탐지" 216건 완화). exclude_last(1)+PIVOT_WINDOW(20)+PIVOT_MIN_LEFT_BARS(3) 이상이면 스윙 고점 확정 가능
PIVOT_WINDOW = 20                     # 스윙 고점 판정 좌우 봉 수
PIVOT_MIN_LEFT_BARS = 3               # 좌측 봉이 부족한 장 초반 고점도 후보로 인정하는 최소 좌측 봉 수
PULLBACK_MIN_PCT = 0.5                # 눌림목 최소 깊이(전고점 대비 %) — P1: 1.0→0.5 (미진입 566건/30일 완화)
PULLBACK_MAX_PCT = 6.0                # 눌림목 최대 깊이(%) — 초과 시 추세 이탈로 간주
PULLBACK_MIN_BARS = 2                 # 눌림 구간 최소 봉 수
PULLBACK_VOLUME_RATIO_MAX = 1.05      # 눌림 구간 평균 거래량 / 상승 구간 평균 — P1: 0.9→1.05 (미진입 284건/30일 완화)
BREAKOUT_BUFFER_PCT = 0.1             # 전고점 돌파 인정 여유(%)
BREAKOUT_VOLUME_RATIO_MIN = 1.5       # 돌파봉 거래량 / 최근 평균 거래량 하한
BREAKOUT_VOLUME_AVG_BARS = 20         # 평균 거래량 산출 구간

# ── 외국인 매수세 ───────────────────────────────────────────
FOREIGN_MIN_NET_BUY_QTY = 0           # 외국인 당일 순매수 수량 하한(주)
FOREIGN_REQUIRE_INCREASING = False    # P1: True→False (증가추세 요구가 과도, 순매수>0 만 확인)

# ── 후보 등록/정리 ──────────────────────────────────────────
CANDIDATE_REGISTER_UNTIL = time(14, 30)   # 이 시각 이후에는 신규 후보를 등록하지 않음
CANDIDATE_MAX_DEFAULT = 5                 # 동시 추적 최대 후보 수 기본값 — P1: 3→5 (30일간 진입판정 대상 28종목뿐)

# ── 청산 (데이 트레이딩 전용) ───────────────────────────────
# T(1T) = 진입 신호의 눌림 저점 → 전고점 돌파가 사이의 상승폭.
# 손절가는 눌림 저점이므로 손절폭 ≈ 1T 가 되어 nT 가 사실상 nR 로 읽힌다.
DEFAULT_MAX_LOSS_PCT = 1.0            # 손절 시 감수할 계좌 손실(%) — 배팅 사이즈 산출 기준
DEFAULT_MAX_POSITION_PCT = 20.0       # 1종목 최대 비중(계좌 대비 %) — 얕은 손절폭 과배팅 차단
DEFAULT_EXIT_STAGES = [               # 분할 익절 차수 (유저가 n차 nT 로 자유 설정)
    # P2: 2T 50% 일괄 익절 → 소량 선익절 + 본체를 3T 이후로 이연해 상방을 남긴다
    {"t": 1.5, "sell_pct": 25.0},
    {"t": 3.0, "sell_pct": 35.0},
    {"t": 4.5, "sell_pct": 20.0},
    # 잔여 20% 는 트레일링스탑이 담당
]
DEFAULT_TRAILING_START_T = 1.5        # 트레일링 추적 시작 배수(T) — P2: 2.0→1.5 (이익 보호 조기화)
DEFAULT_TRAILING_BAR_UNIT = "1m"      # 최저점 판정 봉 단위 — P2: 5m→1m (1분봉 전략에 맞춘 반응성)
DEFAULT_TRAILING_BAR_COUNT = 3        # 직전 N봉 최저점 이탈 시 잔량 청산
DEFAULT_FORCE_EXIT_TIME = time(15, 20)  # 당일 강제 청산 시각 (오버나이트 갭 차단)
FORCE_EXIT_PROFIT_GRACE_T = 1.0       # P2: 강제청산 시각에 평단 대비 +이 배수(T) 이상 수익이면
FORCE_EXIT_PROFIT_GRACE_MIN = 5      #     강제청산을 이 분(分)만큼 연장 (손실 포지션은 즉시 청산)

# ── 오버나이트 보유 (강제청산 시각의 수급 조건 충족 시 익일로 이월) ──
# 강제청산 시각에 아래 조건 중 유저가 지정한 개수 이상이 충족되면 잔량을 익일로
# 이월한다. 다음 날에도 강제청산 시각에 같은 조건을 재평가하고, 보유 거래일이
# OVERNIGHT_MAX_DAYS 에 도달하면 조건과 무관하게 전량 청산한다.
OVERNIGHT_CONDITION_KEYS = ("foreign", "institution", "program", "shinhan_top5")
DEFAULT_OVERNIGHT_ENABLED = False
DEFAULT_OVERNIGHT_CONDITIONS = list(OVERNIGHT_CONDITION_KEYS)
DEFAULT_OVERNIGHT_MIN_COUNT = 2       # 이월에 필요한 충족 조건 개수
DEFAULT_OVERNIGHT_MAX_DAYS = 3        # 이 보유 거래일차에 도달하면 조건 무관 강제청산

# ── 분할 진입 설정 ──────────────────────────────────────────
DEFAULT_ENTRY_STAGES = [{"t": 0.0, "weight_pct": 100.0}]  # 분할 진입 기본값 (1차만, 일괄 진입)
MAX_ENTRY_STAGES = 5                  # 최대 진입 차수
ENTRY_STAGE_CUTOFF_BUFFER_MIN = 10    # 강제 청산 전 추가 진입 차단 여유(분)

# 봉 단위 → (분 수, 표시명). 일봉은 분 수 대신 0.
TRAILING_BAR_UNIT_MINUTES = {
    "1m": (1, "1분봉"),
    "5m": (5, "5분봉"),
    "1d": (0, "일봉"),
}
