"""
KIS 국내주식 거래 API 모듈

원본 autobot/tradingBot/KIS_API_Helper_KR.py 에서 필요 함수만 Django 환경으로 포팅.
- Common.* 대신 dolpha.kis.auth 함수 사용
- 계좌 자격증명 기반 인증 (YAML 파일 불필요)

모든 함수의 마지막 인자 account는 호출할 계좌를 지정합니다.
    KisCredential    → 그 계좌 (사용자별 계좌: dolpha.kis.credentials 참고)
    "REAL"/"VIRTUAL" → 서버 환경변수 계좌
    None             → KIS_MODE 환경변수 계좌

공개 API:
    GetHashKey(data, account)              주문 바디 해시키 발급
    GetBalance(account)                    계좌 잔고 조회
    GetMyStockList(account)                보유 주식 목록 조회 (페이지 처리 포함)
    GetCurrentPrice(stock_code, account)   현재가 조회
    MakeBuyMarketOrder(stock_code, qty, account)   시장가 매수
    MakeSellMarketOrder(stock_code, qty, account)  시장가 매도
    GetOrderFill(order_no, account, ...)           주문번호별 체결 현황 조회
"""

import time
import json
import requests

from .auth import GetHeaders, resolve_credential


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────

def _is_virtual(account=None) -> bool:
    return resolve_credential(account).is_virtual


def _sleep(account=None) -> None:
    """API Rate Limit 대응 (실계좌 초당 5건 / 모의 초당 2건 제한)"""
    time.sleep(0.21)
    if _is_virtual(account):
        time.sleep(0.31)


def _account_params(account=None) -> dict:
    """계좌 공통 파라미터"""
    cred = resolve_credential(account)
    return {
        "CANO": cred.account_no,
        "ACNT_PRDT_CD": cred.account_cd,
    }


# ─────────────────────────────────────────────────────────────
# GetHashKey
# ─────────────────────────────────────────────────────────────

def GetHashKey(data: dict, account=None) -> str:
    """
    주문 요청 body 데이터에 대한 해시키를 발급합니다.
    KIS 주문 API의 hashkey 헤더에 사용됩니다.

    Args:
        data: 주문 body dict
    Returns:
        hashkey 문자열 (실패 시 "")
    """
    cred = resolve_credential(account)
    url = f"{cred.url_base}/uapi/hashkey"
    headers = {
        "content-type": "application/json",
        "appkey": cred.app_key,
        "appsecret": cred.app_secret,
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

def GetBalance(account=None) -> dict:
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
    cred = resolve_credential(account)
    _sleep(cred)

    tr_id = "VTTC8434R" if cred.is_virtual else "TTTC8434R"
    path = "uapi/domestic-stock/v1/trading/inquire-balance"
    url = f"{cred.url_base}/{path}"

    headers = GetHeaders(tr_id=tr_id, custtype="P", account=cred)
    params = {
        **_account_params(cred),
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
        cash         = float(result["dnca_tot_amt"])
        stock_cost   = float(result.get("pchs_amt_smtl_amt", 0))

        # 예수금이 0이거나 전일 기준 총평가금액이 더 정확한 경우 교체
        if cash == 0 or total_money == stock_money:
            total_money = float(result["bfdy_tot_asst_evlu_amt"])

        remain_money = total_money - stock_money
        if remain_money == 0:
            remain_money = cash

        # 확정원금 = 예수금 + 보유주식 매수원가 합계 (미실현 손익 제외)
        confirmed_capital = cash + stock_cost

        return {
            "TotalMoney":       total_money,
            "RemainMoney":      remain_money,
            "StockMoney":       stock_money,
            "StockRevenue":     stock_rev,
            "ConfirmedCapital": confirmed_capital,
        }
    else:
        err = res.json().get("msg_cd", res.text)
        raise RuntimeError(f"GetBalance 실패: {res.status_code} — {err}")


# ─────────────────────────────────────────────────────────────
# 보유 주식 목록
# ─────────────────────────────────────────────────────────────

def GetMyStockList(account=None) -> list:
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
    cred  = resolve_credential(account)
    tr_id = "VTTC8434R" if cred.is_virtual else "TTTC8434R"
    path  = "uapi/domestic-stock/v1/trading/inquire-balance"
    url   = f"{cred.url_base}/{path}"

    stock_list: list = []
    fk_key = ""
    nk_key = ""
    prev_nk_key = ""
    tr_cont = ""
    fail_count = 0

    while True:
        _sleep(cred)
        headers = GetHeaders(tr_id=tr_id, custtype="P", account=cred)
        headers["tr_cont"] = tr_cont

        params = {
            **_account_params(cred),
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
            try:
                err_code = res.json().get("msg_cd", "")
            except ValueError:  # 에러 응답이 JSON 이 아닌 경우(502/504 HTML 등)
                err_code = f"HTTP {res.status_code}"
            print(f"[KIS] GetMyStockList 오류: {err_code}")
            # 연속조회 도중 실패했는데 여기까지 모은 목록을 그대로 돌려주면,
            # 호출부는 그것을 '완전한 계좌 스냅샷'으로 믿는다. 그 결과 아직
            # 보유 중인 종목이 '계좌에 없음'으로 오판되어 포지션이 청산된 것처럼
            # 정리된다(_reconcile_positions). 불완전한 스냅샷은 절대 반환하지 않고
            # 예외로 알린다 — 호출부는 조회 실패한 계좌를 건드리지 않는다.
            if fail_count >= 3 or err_code == "EGW00123":
                raise RuntimeError(
                    f"보유종목 조회 실패({err_code}) — 불완전한 목록"
                    f"({len(stock_list)}종목)이라 반환하지 않음"
                )

    return stock_list


# ─────────────────────────────────────────────────────────────
# 현재가 조회
# ─────────────────────────────────────────────────────────────

def GetCurrentPrice(stock_code: str, account=None) -> int:
    """
    국내 주식의 현재가를 조회합니다.

    Args:
        stock_code: 종목코드 (예: "005930")
    Returns:
        현재가 (int, 원)
    Raises:
        RuntimeError: API 호출 실패 시
    """
    cred = resolve_credential(account)
    _sleep(cred)

    path = "uapi/domestic-stock/v1/quotations/inquire-price"
    url  = f"{cred.url_base}/{path}"

    headers = GetHeaders(tr_id="FHKST01010100", account=cred)
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

LAST_ORDER_ERROR: dict[str, str] = {}


def format_user_friendly_order_error(raw_error: str) -> str:
    """
    KIS 증권사 API 응답 에러 코드 및 메시지를 일반 사용자가 이해하기 쉬운 안내 문구로 변환합니다.
    """
    if not raw_error:
        return "증권사 주문 처리에 실패했습니다. 잠시 후 다시 시도해주세요."

    raw = raw_error.strip()

    # 1. 장 운영 시간 관련
    if any(k in raw for k in ["APBK0933", "APBK0939", "주문 가능 시간", "장마감", "장종료", "장운영시간", "주문불가시간", "장개시전", "주문불가"]):
        return "현재 정규장 운영 시간(09:00~15:30)이 아니어서 시장가 주문을 접수할 수 없습니다."

    # 2. 주문 가능 수량 부족 / 잔고 부족 / 미체결 주문 대기
    if any(k in raw for k in ["APBK0937", "주문수량", "매도가능수량", "잔고", "수량초과", "잔고부족"]):
        return "매도 가능한 보유 수량이 부족하거나 이미 다른 주문(미체결)이 대기 중입니다. 증권사 계좌의 미체결 내역을 확인해주세요."

    # 3. 초당 요청 제한 / TPS 초과
    if any(k in raw for k in ["40010000", "EGW00201", "초당", "건수초과", "TPS", "호출제한", "거래건수"]):
        return "증권사 API 요청 한도를 일시적으로 초과했습니다. 잠시 후(몇 초 뒤) 다시 시도해주세요."

    # 4. 토큰 / 인증 / 계좌 자격증명
    if any(k in raw for k in ["EGW00123", "IGW00121", "토큰", "인증", "SecretKey", "AppKey", "유효하지 않은"]):
        return "증권사 계좌 인증 정보(AppKey/SecretKey)가 유효하지 않거나 만료되었습니다. 마이페이지에서 계좌 설정을 확인해주세요."

    # 5. 호가 / 가격 범위 오류
    if any(k in raw for k in ["APBK1091", "호가", "가격범위", "상한가", "하한가"]):
        return "현재 호가 범위를 벗어나 주문을 접수할 수 없습니다. 잠시 후 다시 시도해주세요."

    # 6. 모의투자 관련 제한
    if "모의투자" in raw:
        return f"모의투자 주문 제한: {raw}"

    return f"증권사 주문 실패: {raw}"


def MakeBuyMarketOrder(stock_code: str, qty: int, account=None) -> dict | None:
    """
    시장가 매수 주문을 접수합니다.

    Args:
        stock_code: 종목코드
        qty: 매수 수량
    Returns:
        성공: {"OrderNum": str, "OrderNum2": str, "OrderTime": str}
        실패: None
    """
    cred = resolve_credential(account)
    _sleep(cred)

    tr_id = "VTTC0012U" if cred.is_virtual else "TTTC0012U"
    path  = "uapi/domestic-stock/v1/trading/order-cash"
    url   = f"{cred.url_base}/{path}"

    data = {
        **_account_params(cred),
        "PDNO":     stock_code,
        "ORD_DVSN": "01",          # 시장가
        "ORD_QTY":  str(int(qty)),
        "ORD_UNPR": "0",
    }

    headers = GetHeaders(tr_id=tr_id, custtype="P", account=cred)
    headers["hashkey"] = GetHashKey(data, cred)

    res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        order = res.json()["output"]
        return {
            "OrderNum":  order["KRX_FWDG_ORD_ORGNO"],
            "OrderNum2": order["ODNO"],
            "OrderTime": order["ORD_TMD"],
        }
    else:
        d = res.json()
        err = f"{d.get('msg_cd', '')} {d.get('msg1', res.text[:200])}".strip()
        print(f"[KIS] MakeBuyMarketOrder({stock_code}, {qty}) 실패: {err}")
        LAST_ORDER_ERROR["buy"] = err
        return None


# ─────────────────────────────────────────────────────────────
# 시장가 매도
# ─────────────────────────────────────────────────────────────

def MakeSellMarketOrder(stock_code: str, qty: int, account=None) -> dict | None:
    """
    시장가 매도 주문을 접수합니다.

    Args:
        stock_code: 종목코드
        qty: 매도 수량
    Returns:
        성공: {"OrderNum": str, "OrderNum2": str, "OrderTime": str}
        실패: None
    """
    cred = resolve_credential(account)
    _sleep(cred)

    tr_id = "VTTC0011U" if cred.is_virtual else "TTTC0011U"
    path  = "uapi/domestic-stock/v1/trading/order-cash"
    url   = f"{cred.url_base}/{path}"

    data = {
        **_account_params(cred),
        "PDNO":     stock_code,
        "ORD_DVSN": "01",          # 시장가
        "ORD_QTY":  str(int(qty)),
        "ORD_UNPR": "0",
    }

    headers = GetHeaders(tr_id=tr_id, custtype="P", account=cred)
    headers["hashkey"] = GetHashKey(data, cred)

    res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        order = res.json()["output"]
        return {
            "OrderNum":  order["KRX_FWDG_ORD_ORGNO"],
            "OrderNum2": order["ODNO"],
            "OrderTime": order["ORD_TMD"],
        }
    else:
        d = res.json()
        err = f"{d.get('msg_cd', '')} {d.get('msg1', res.text[:200])}".strip()
        print(f"[KIS] MakeSellMarketOrder({stock_code}, {qty}) 실패: {err}")
        LAST_ORDER_ERROR["sell"] = err
        return None


# ─────────────────────────────────────────────────────────────
# 해외주식 시장가 매수
# ─────────────────────────────────────────────────────────────

def MakeBuyMarketOrderUS(stock_code: str, qty: int, exchange: str = "NASD", account=None) -> dict | None:
    """
    해외주식 시장가 매수 주문.

    Args:
        stock_code: 티커 (예: "NVDA")
        qty: 수량
        exchange: 거래소 코드 NASD(나스닥) / NYSE / AMEX / SEHK(홍콩) 등
    Returns:
        성공: {"OrderNum": str, "OrderTime": str}
        실패: None
    """
    cred = resolve_credential(account)
    _sleep(cred)

    tr_id = "VTTT1002U" if cred.is_virtual else "TTTT1002U"
    path  = "uapi/overseas-stock/v1/trading/order"
    url   = f"{cred.url_base}/{path}"

    # 거래소별 시장가 주문 코드: NASD/NYSE/AMEX="32", 그 외="00"(지정가 0원)
    mkt_dvsn = "32" if exchange in ("NASD", "NYSE", "AMEX") else "00"

    data = {
        **_account_params(cred),
        "OVRS_EXCG_CD":    exchange,
        "PDNO":            stock_code,
        "ORD_DVSN":        mkt_dvsn,
        "ORD_QTY":         str(int(qty)),
        "OVRS_ORD_UNPR":   "0",
        "ORD_SVR_DVSN_CD": "0",
    }

    headers = GetHeaders(tr_id=tr_id, custtype="P", account=cred)
    headers["hashkey"] = GetHashKey(data, cred)

    res = requests.post(url, headers=headers, data=json.dumps(data), timeout=10, verify=False)
    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        order = res.json().get("output", {})
        return {
            "OrderNum":  order.get("ODNO", ""),
            "OrderTime": order.get("ORD_TMD", ""),
        }
    else:
        d = res.json()
        print(f"[KIS] MakeBuyMarketOrderUS({stock_code}) 실패: {d.get('msg_cd')} — {d.get('msg1', res.text[:200])}")
        return None


# ─────────────────────────────────────────────────────────────
# 체결 현황 조회
# ─────────────────────────────────────────────────────────────

def GetOrderFill(
    order_no: str,
    account=None,
    stock_code: str = "",
    order_date: str = "",
) -> dict | None:
    """주문번호로 당일 체결 현황을 조회합니다 (주식일별주문체결조회).

    주문 접수(rt_cd=0)는 '접수'일 뿐 체결이 아니다. 시장가 주문도 호가를
    소진하며 나눠 체결되므로, 실제 체결 수량·평균가는 이 조회로만 알 수 있다.

    Args:
        order_no:   주문번호(ODNO). MakeBuy/SellMarketOrder 의 "OrderNum2"
        stock_code: 종목코드로 한 번 더 좁히고 싶을 때 (선택)
        order_date: 주문 일자 YYYYMMDD. 기본은 오늘(KST)

    Returns:
        조회 성공: {
            "order_no", "stock_code", "ordered_qty", "filled_qty",
            "remain_qty", "avg_price", "filled_amount",
        }
        조회 실패·응답 이상: None
        — None 은 '체결 0' 이 아니라 '알 수 없음' 이다. 호출부는 절대 0 으로
          단정하면 안 된다(그랬다가 미체결을 전량체결로 기록해 왔다).
    """
    if not order_no:
        return None

    cred = resolve_credential(account)
    _sleep(cred)

    if not order_date:
        from datetime import datetime

        from pytz import timezone as pytz_tz

        order_date = datetime.now(pytz_tz("Asia/Seoul")).strftime("%Y%m%d")

    tr_id = "VTTC0081R" if cred.is_virtual else "TTTC0081R"  # 3개월 이내
    url = f"{cred.url_base}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"

    params = {
        **_account_params(cred),
        "INQR_STRT_DT": order_date,
        "INQR_END_DT": order_date,
        "SLL_BUY_DVSN_CD": "00",   # 전체
        "PDNO": stock_code or "",
        "CCLD_DVSN": "00",         # 전체(체결 + 미체결)
        "INQR_DVSN": "00",         # 역순
        "INQR_DVSN_3": "00",       # 전체
        "ORD_GNO_BRNO": "",
        "ODNO": order_no,
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        res = requests.get(
            url,
            headers=GetHeaders(tr_id=tr_id, custtype="P", account=cred),
            params=params,
            timeout=10,
            verify=False,
        )
        body = res.json()
    except Exception as e:  # noqa: BLE001 — 네트워크·JSON 오류 모두 '알 수 없음'
        print(f"[KIS] GetOrderFill({order_no}) 조회 오류: {e}")
        return None

    if res.status_code != 200 or body.get("rt_cd") != "0":
        print(
            f"[KIS] GetOrderFill({order_no}) 실패:"
            f" {body.get('msg_cd', '')} {body.get('msg1', '')}".rstrip()
        )
        return None

    rows = [
        row for row in (body.get("output1") or [])
        if str(row.get("odno", "")).lstrip("0") == str(order_no).lstrip("0")
    ]
    if not rows:
        # 접수 직후라 아직 조회에 잡히지 않는 경우도 여기로 온다 — 0 이 아니라 미상.
        return None

    ordered = filled = amount = 0
    for row in rows:
        ordered += _int(row.get("ord_qty"))
        filled += _int(row.get("tot_ccld_qty"))
        amount += _int(row.get("tot_ccld_amt"))

    # 잔여수량은 응답 값을 우선하되, 없으면 주문-체결로 유도한다.
    remain = sum(_int(row.get("rmn_qty")) for row in rows) or max(ordered - filled, 0)

    return {
        "order_no": order_no,
        "stock_code": rows[0].get("pdno", stock_code),
        "ordered_qty": ordered,
        "filled_qty": filled,
        "remain_qty": remain,
        "avg_price": round(amount / filled) if filled else 0,
        "filled_amount": amount,
    }


def _int(value) -> int:
    """KIS 응답의 숫자 문자열을 int 로. 빈 값·이상값은 0."""
    try:
        return int(float(str(value).replace(",", "").strip() or 0))
    except (TypeError, ValueError):
        return 0
