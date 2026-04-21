"""
한화시스템(272210) 1,427주 시장가 매도 스크립트
포지션 초과분 정리: 1,902주 보유 → 475주로 감축
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# .env 로드
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# Django 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dolpha.settings")
import django
django.setup()

STOCK_CODE = "272210"  # 한화시스템
SELL_QTY = 1427
LOG_FILE = BASE_DIR / "scripts" / "sell_hanwha_result.json"


def main():
    ts = datetime.now().isoformat()
    print(f"[{ts}] 한화시스템({STOCK_CODE}) {SELL_QTY}주 시장가 매도 시작")

    try:
        from dolpha.kis.trade import MakeSellMarketOrder, GetMyStockList

        # 현재 보유 확인
        holdings = GetMyStockList()
        holding = next((s for s in holdings if s["StockCode"] == STOCK_CODE), None)
        if not holding:
            result = {"success": False, "error": "보유 종목 없음", "ts": ts}
            print(f"[ERROR] {result['error']}")
            LOG_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return

        current_qty = int(holding["StockAmt"])
        print(f"  현재 보유: {current_qty}주")

        if current_qty < SELL_QTY:
            result = {
                "success": False,
                "error": f"보유수량 부족: {current_qty}주 < {SELL_QTY}주",
                "ts": ts,
            }
            print(f"[ERROR] {result['error']}")
            LOG_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return

        # 시장가 매도
        order = MakeSellMarketOrder(STOCK_CODE, SELL_QTY)
        if order:
            result = {
                "success": True,
                "stock_code": STOCK_CODE,
                "sell_qty": SELL_QTY,
                "remaining_qty": current_qty - SELL_QTY,
                "order": order,
                "ts": ts,
            }
            print(f"[OK] 매도 주문 완료: {order}")
        else:
            result = {"success": False, "error": "주문 실패 (KIS 응답 None)", "ts": ts}
            print(f"[ERROR] {result['error']}")

    except Exception as e:
        result = {"success": False, "error": str(e), "ts": ts}
        print(f"[EXCEPTION] {e}")

    LOG_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"결과 저장: {LOG_FILE}")


if __name__ == "__main__":
    main()
