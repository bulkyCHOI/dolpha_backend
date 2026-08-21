"""myweb 앱 시그널.

분봉(StockMinuteOhlcv)은 TradingConfig 삭제·청산과 함께 지우지 않는다.
진입 판정 차트가 지난 날짜의 분봉을 근거 데이터로 사용하므로,
삭제는 사용자가 급등테마주 탭에서 날짜 단위로 명시적으로 실행할 때만 이루어진다.
(dolpha.theme_surge.scanner.purge_candidate_date)
"""
