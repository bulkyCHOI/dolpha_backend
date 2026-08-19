"""급등테마주 전략 파라미터.

모든 임계값을 한 곳에 모아 매직넘버를 제거한다.
유저별로 조정 가능한 값(테마 급등 기준, 외국인 필터 사용 여부 등)은
TradingDefaults 에 저장되며 여기 값은 그 기본값 역할을 한다.
"""

from datetime import time

# ── 타임라인 ────────────────────────────────────────────────
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(15, 30)
SLOT_MINUTES = 5                      # 스캔 주기(분) = 타임라인 슬롯 단위
TOTAL_SLOTS = 79                      # 09:00 ~ 15:30, 5분 간격 슬롯 수

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
SURGE_MIN_MOMENTUM_PCT = 0.3          # 직전 슬롯 대비 등락률 증가폭 하한(%p)
SURGE_TOP_N = 10                      # 스냅샷으로 저장할 상위 테마 수
SURGE_MAX_THEMES_PER_SLOT = 3         # 한 슬롯에서 후보를 뽑을 최대 테마 수

# ── 1등 종목(주도주) 선정 ───────────────────────────────────
LEADER_WEIGHT_TRADING_VALUE = 0.5     # 거래대금 가중치
LEADER_WEIGHT_CHANGE_RATE = 0.5       # 상승률 가중치
LEADER_MIN_CHANGE_RATE_PCT = 2.0      # 후보 최소 상승률(%)
LEADER_MIN_TRADING_VALUE = 5_000_000_000   # 후보 최소 거래대금(원) = 50억
LEADER_MIN_MARKET_CAP = 50_000_000_000     # 후보 최소 시가총액(원) = 500억
LEADER_MAX_CHANGE_RATE_PCT = 25.0     # 상한가 근접(+25% 초과) 종목은 추격 매수 제외
LEADER_STORE_COUNT = 3                # 테마당 저장할 후보 수 (1등 + 추적용 2등·3등)

# ── 1분봉 진입 판정 ─────────────────────────────────────────
ENTRY_LOOKBACK_BARS = 60              # 전고점 탐색 구간(분봉 개수)
ENTRY_MIN_BARS = 20                   # 판정에 필요한 최소 분봉 개수
PIVOT_WINDOW = 2                      # 스윙 고점 판정 좌우 봉 수
PULLBACK_MIN_PCT = 0.8                # 눌림목 최소 깊이(전고점 대비 %)
PULLBACK_MAX_PCT = 6.0                # 눌림목 최대 깊이(%) — 초과 시 추세 이탈로 간주
PULLBACK_MIN_BARS = 2                 # 눌림 구간 최소 봉 수
PULLBACK_VOLUME_RATIO_MAX = 0.9       # 눌림 구간 평균 거래량 / 상승 구간 평균 (거래량 감소 확인)
BREAKOUT_BUFFER_PCT = 0.1             # 전고점 돌파 인정 여유(%)
BREAKOUT_VOLUME_RATIO_MIN = 1.5       # 돌파봉 거래량 / 최근 평균 거래량 하한
BREAKOUT_VOLUME_AVG_BARS = 20         # 평균 거래량 산출 구간

# ── 외국인 매수세 ───────────────────────────────────────────
FOREIGN_MIN_NET_BUY_QTY = 0           # 외국인 당일 순매수 수량 하한(주)
FOREIGN_REQUIRE_INCREASING = True     # 최근 구간에서 순매수가 증가 추세여야 하는지
FOREIGN_RECENT_SLOTS = 3              # 증가 추세 확인에 사용할 최근 구간 수

# ── 후보 등록/정리 ──────────────────────────────────────────
CANDIDATE_REGISTER_UNTIL = time(14, 30)   # 이 시각 이후에는 신규 후보를 등록하지 않음
CANDIDATE_MAX_DEFAULT = 3                 # 동시 추적 최대 후보 수 기본값
