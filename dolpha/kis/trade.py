"""
KIS 국내주식 거래 API 모듈

원본 autobot/tradingBot/KIS_API_Helper_KR.py 에서 필요 함수만 Django 환경으로 포팅.
- Common.* 대신 dolpha.kis.auth 함수 사용
- 환경변수 기반 인증 (YAML 파일 불필요)

공개 API:
    GetHashKey(data)               주문 바디 해시키 발급
    GetBalance()                   계좌 잔고 조회
    GetMyStockList()               보유 주식 목록 조회 (페이지 처리 포함)
    GetCurrentPrice(stock_code)    현재가 조회
    MakeBuyMarketOrder(stock_code, qty)   시장가 매수
    MakeSellMarketOrder(stock_code, qty)  시장가 매도
"""

import time
import json
import requests

from .auth import GetHeaders, get_url_base, get_account_no, get_account_cd, get_mode


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────

def _is_virtual() -> bool:
    return get_mode() == "VIRTUAL"


def _sleep():
    """API Rate Limit 대응 (실계좌 초당 5건 / 모의 초당 2건 제한)"""
    time.sleep(0.21)
    if _is_virtual():
        time.sleep(0.31)


def _account_params() -> dict:
    """계좌 공통 파라미터"""
    return {
        "CANO": get_account_no(),
        "ACNT_PRDT_CD": get_account_cd(),
    }


# ─────────────────────────────────────────────────────────────
# GetHashKey
# ─────────────────────────────────────────────────────────────

def GetHashKey(data: dict) -> str:
    """
    주문 요청 body 데이터에 대한 해시키를 발급합니다.
    KIS 주문 API의 hashkey 헤더에 사용됩니다.

    Args:
        data: 주문 body dict
    Returns:
        hashkey 문자열 (실패 시 "")
    """
    from .auth import get_app_key, get_app_secret
    url = f"{get_url_base()}/uapi/hashkey"
    headers = {
        "content-type": "application/json",
        "appkey": get_app_key(),
        "appsecret": get_app_secret(),
    }
    try:
        res = requests.post(url, headers=headers, json=data, timeout=10, verify=False)
        if res.status_code == 200:
            return res.json().get("HASH", "")
    except Exception as e:
        print(f"[KIS] GetHashKey 오류: {e}")
    return ""


# ─────────────────────────────────────────────────────────────
# 잔고 조회
# ─────────────────────────────────────────────────────────────

def GetBalance() -> dict:
    """
    계좌 잔고를 조회합니다.

    Returns:
        {
            "TotalMoney":   float,  # 총 평가금액
            "RemainMoney":  float,  # 주문가능현금 (예수금)
            "StockMoney":   float,  # 주식 총 평가금액
            "StockRevenue": float,  # 평가 손익금액
        }
    Raises:
        RuntimeError: API 호출 실패 시
    """
    _sleep()

    tr_id = "VTTC8434R" if _is_virtual() else "TTTC8434R"
    path = "uapi/domestic-stock/v1/trading/inquire-balance"
    url = f"{get_url_base()}/{path}"

    headers = GetHeaders(tr_id=tr_id, custtype="P")
    params = {
        **_account_params(),
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",   # 02: 종합집계 (output2 에 총합 있음)
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    res = requests.get(url, headers=headers, params=params, timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        result = res.json()["output2"][0]

        stock_money  = float(result["scts_evlu_amt"])
        stock_rev    = float(result["evlu_pfls_smtl_amt"])
        total_money  = float(result["tot_evlu_amt"])

        # 예수금이 0이거나 전일 기준 총평가금액이 더 정확한 경우 교체
        if float(result["dnca_tot_amt"]) == 0 or total_money == stock_money:
            total_money = float(result["bfdy_tot_asst_evlu_amt"])

        remain_money = total_money - stock_money
        if remain_money == 0:
            remain_money = float(result["dnca_tot_amt"])

        return {
            "TotalMoney":   total_money,
            "RemainMoney":  remain_money,
            "StockMoney":   stock_money,
            "StockRevenue": stock_rev,
        }
    else:
        err = res.json().get("msg_cd", res.text)
        raise RuntimeError(f"GetBalance 실패: {res.status_code} — {err}")


# ─────────────────────────────────────────────────────────────
# 보유 주식 목록
# ─────────────────────────────────────────────────────────────

def GetMyStockList() -> list:
    """
    계좌에서 보유 중인 주식 목록을 조회합니다. (연속조회 지원)

    Returns:
        list of {
            "StockCode":       str,
            "StockName":       str,
            "StockAmt":        str,   # 보유 수량
            "StockAvgPrice":   str,   # 평균 매수가
            "StockOriMoney":   str,   # 매수 금액
            "StockNowMoney":   str,   # 현재 평가금액
            "StockNowPrice":   str,   # 현재가
            "StockRevenueRate":  str, # 수익률
            "StockRevenueMoney": str, # 수익금액
        }
    """
    tr_id = "VTTC8434R" if _is_virtual() else "TTTC8434R"
    path  = "uapi/domestic-stock/v1/trading/inquire-balance"
    url   = f"{get_url_base()}/{path}"

    stock_list: list = []
    fk_key = ""
    nk_key = ""
    prev_nk_key = ""
    tr_cont = ""
    fail_count = 0

    while True:
        _sleep()
        headers = GetHeaders(tr_id=tr_id, custtype="P")
        headers["tr_cont"] = tr_cont

        params = {
            **_account_params(),
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",   # 01: 종목별 상세
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": fk_key,
            "CTX_AREA_NK100": nk_key,
        }

        res = requests.get(url, headers=headers, params=params, timeout=10, verify=False)
        resp_tr_cont = res.headers.get("tr_cont", "")
        tr_cont = "N" if resp_tr_cont in ("M", "F") else ""

        if res.status_code == 200 and res.json().get("rt_cd") == "0":
            nk_key = res.json().get("ctx_area_nk100", "").strip()
            fk_key = res.json().get("ctx_area_fk100", "").strip()

            for s in res.json().get("output1", []):
                if int(s.get("hldg_qty", 0)) <= 0:
                    continue
                code = s["pdno"]
                if any(x["StockCode"] == code for x in stock_list):
                    continue  # 중복 제거
                stock_list.append({
                    "StockCode":       code,
                    "StockName":       s["prdt_name"],
                    "StockAmt":        s["hldg_qty"],
                    "StockAvgPrice":   s["pchs_avg_pric"],
                    "StockOriMoney":   s["pchs_amt"],
                    "StockNowMoney":   s["evlu_amt"],
                    "StockNowPrice":   s["prpr"],
                    "StockRevenueRate":  s["evlu_pfls_rt"],
                    "StockRevenueMoney": s["evlu_pfls_amt"],
                })

            # 연속조회 종료 조건
            if not nk_key or nk_key == prev_nk_key:
                break
            prev_nk_key = nk_key

        else:
            fail_count += 1
            err_code = res.json().get("msg_cd", "")
            print(f"[KIS] GetMyStockList 오류: {err_code}")
            if fail_count >= 3 or err_code == "EGW00123":
                break

    return stock_list


# ─────────────────────────────────────────────────────────────
# 현재가 조회
# ─────────────────────────────────────────────────────────────

def GetCurrentPrice(stock_code: str) -> int:
    """
    국내 주식의 현재가를 조회합니다.

    Args:
        stock_code: 종목코드 (예: "005930")
    Returns:
        현재가 (int, 원)
    Raises:
        RuntimeError: API 호출 실패 시
    """
    _sleep()

    path = "uapi/domestic-stock/v1/quotations/inquire-price"
    url  = f"{get_url_base()}/{path}"

    headers = GetHeaders(tr_id="FHKST01010100")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }

    res = requests.get(url, headers=headers, params=params, timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        return int(res.json()["output"]["stck_prpr"])
    else:
        err = res.json().get("msg_cd", res.text)
        raise RuntimeError(f"GetCurrentPrice({stock_code}) 실패: {err}")


# ─────────────────────────────────────────────────────────────
# 시장가 매수
# ─────────────────────────────────────────────────────────────

def MakeBuyMarketOrder(stock_code: str, qty: int) -> dict | None:
    """
    시장가 매수 주문을 접수합니다.

    Args:
        stock_code: 종목코드
        qty: 매수 수량
    Returns:
        성공: {"OrderNum": str, "OrderNum2": str, "OrderTime": str}
        실패: None
    """
    _sleep()

    tr_id = "VTTC0802U" if _is_virtual() else "TTTC0802U"
    path  = "uapi/domestic-stock/v1/trading/order-cash"
    url   = f"{get_url_base()}/{path}"

    data = {
        **_account_params(),
        "PDNO":     stock_code,
        "ORD_DVSN": "01",          # 시장가
        "ORD_QTY":  str(int(qty)),
        "ORD_UNPR": "0",
    }

    headers = GetHeaders(tr_id=tr_id, custtype="P")
    headers["hashkey"] = GetHashKey(data)

    res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        order = res.json()["output"]
        return {
            "OrderNum":  order["KRX_FWDG_ORD_ORGNO"],
            "OrderNum2": order["ODNO"],
            "OrderTime": order["ORD_TMD"],
        }
    else:
        err = res.json().get("msg_cd", res.text)
        print(f"[KIS] MakeBuyMarketOrder({stock_code}, {qty}) 실패: {err}")
        return None


# ─────────────────────────────────────────────────────────────
# 시장가 매도
# ─────────────────────────────────────────────────────────────

def MakeSellMarketOrder(stock_code: str, qty: int) -> dict | None:
    """
    시장가 매도 주문을 접수합니다.

    Args:
        stock_code: 종목코드
        qty: 매도 수량
    Returns:
        성공: {"OrderNum": str, "OrderNum2": str, "OrderTime": str}
        실패: None
    """
    _sleep()

    tr_id = "VTTC0801U" if _is_virtual() else "TTTC0801U"
    path  = "uapi/domestic-stock/v1/trading/order-cash"
    url   = f"{get_url_base()}/{path}"

    data = {
        **_account_params(),
        "PDNO":     stock_code,
        "ORD_DVSN": "01",          # 시장가
        "ORD_QTY":  str(int(qty)),
        "ORD_UNPR": "0",
    }

    headers = GetHeaders(tr_id=tr_id, custtype="P")
    headers["hashkey"] = GetHashKey(data)

    res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        order = res.json()["output"]
        return {
            "OrderNum":  order["KRX_FWDG_ORD_ORGNO"],
            "OrderNum2": order["ODNO"],
            "OrderTime": order["ORD_TMD"],
        }
    else:
        err = res.json().get("msg_cd", res.text)
        print(f"[KIS] MakeSellMarketOrder({stock_code}, {qty}) 실패: {err}")
        return None
