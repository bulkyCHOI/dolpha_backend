"""주문 로그 — KIS 주문 요청/결과를 전용 파일에 남긴다.

자동매매 엔진은 백그라운드 스케줄러에서 돌아 stdout 이 어디에도 수집되지 않는다.
그래서 주문이 거부돼도 KIS 가 돌려준 사유(msg_cd / msg1)가 사라지고
TradeEntry 에 "주문 실패" 라는 결과만 남아 원인 추적이 불가능했다.
주문은 건수가 적고 사후 추적 가치가 크므로 전용 파일에 따로 적재한다.
"""

from __future__ import annotations

from datetime import datetime

ORDER_LOG = "/tmp/order.log"


def order_log(msg: str) -> None:
    """주문 이벤트를 타임스탬프와 함께 기록한다. 실패해도 매매를 막지 않는다."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(ORDER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
