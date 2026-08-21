"""
KIS API — 투자자별·프로그램·회원사 매매동향 조회

사용 엔드포인트:
  - inquire-investor         (FHKST01010900) : 일자별 투자자(개인/외국인/기관) 순매수
  - comp-program-trade-today (FHPPG04650100) : 종목별 프로그램매매 추이(체결)
  - inquire-member           (FHKST01010600) : 주식현재가 회원사(매도/매수 상위 5개사)

주의: KIS 응답 본문의 데이터 키는 모두 단수형 `output` 이다.
      (output1/output2 형태가 아니므로 그대로 읽으면 항상 빈 값이 된다.)
"""

import warnings
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

_TIMEOUT = 15
_TOP_N = 5  # 회원사 매도/매수 상위 노출 개수


def _base() -> str:
    return get_url_base("REAL")


def _call(path: str, tr_id: str, params: dict, label: str):
    """KIS quotations 엔드포인트를 호출하고 `output` 값을 그대로 반환한다."""
    url = f"{_base()}/uapi/domestic-stock/v1/quotations/{path}"
    res = requests.get(
        url,
        headers=GetHeaders(tr_id=tr_id, mode="REAL"),
        params=params,
        timeout=_TIMEOUT,
        verify=False,
    )
    try:
        data = res.json()
    except ValueError:
        raise RuntimeError(f"[KIS {label}] 응답 파싱 실패 ({res.status_code}): {res.text[:200]}")

    # rt_cd=0: 정상, rt_cd=1: 다음 페이지 있음(데이터는 유효)
    if res.status_code != 200 or data.get("rt_cd") not in ("0", "1"):
        raise RuntimeError(
            f"[KIS {label}] 호출 실패 ({res.status_code}): {data.get('msg1', res.text[:200])}"
        )

    return data.get("output")


def _to_int(val) -> int:
    """KIS 문자열 숫자를 int 로 변환한다. 공란·비정상 값은 0."""
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (TypeError, ValueError):
        return 0


def _to_float(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────
# 1. 일자별 투자자(개인/외국인/기관) 순매수
# ─────────────────────────────────────────────────────────────

def GetInvestorToday(stock_code: str) -> dict:
    """주식현재가 투자자 — 최근 30영업일 개인/외국인/기관 순매수 수량·금액.

    Returns:
        {"rows": [{"date", "close", "change", "prsn_qty", "frgn_qty",
                   "orgn_qty", "prsn_amt", "frgn_amt", "orgn_amt"}, ...]}

    참고: 당일 행은 장중 잠정 집계라 순매수 값이 공란(0)으로 내려올 수 있다.
    """
    output = _call(
        "inquire-investor",
        "FHKST01010900",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
        "investor",
    )
    rows = [
        {
            "date": row.get("stck_bsop_date", ""),
            "close": _to_int(row.get("stck_clpr")),
            "change": _to_int(row.get("prdy_vrss")),
            "prsn_qty": _to_int(row.get("prsn_ntby_qty")),
            "frgn_qty": _to_int(row.get("frgn_ntby_qty")),
            "orgn_qty": _to_int(row.get("orgn_ntby_qty")),
            "prsn_amt": _to_int(row.get("prsn_ntby_tr_pbmn")),
            "frgn_amt": _to_int(row.get("frgn_ntby_tr_pbmn")),
            "orgn_amt": _to_int(row.get("orgn_ntby_tr_pbmn")),
        }
        for row in (output or [])
    ]
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────
# 2. 종목별 프로그램매매 추이 (당일 체결 기준)
# ─────────────────────────────────────────────────────────────

def GetProgramTradeToday(stock_code: str) -> dict:
    """종목별 프로그램매매 추이(체결) — 당일 시간대별 프로그램 순매수.

    Returns:
        {"rows": [{"time", "price", "change_rate", "acml_vol",
                   "seln_vol", "shnu_vol", "ntby_qty",
                   "seln_amt", "shnu_amt", "ntby_amt"}, ...]}
    """
    output = _call(
        "comp-program-trade-today",
        "FHPPG04650100",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
        "program-trade",
    )
    rows = [
        {
            "time": row.get("bsop_hour", ""),
            "price": _to_int(row.get("stck_prpr")),
            "change_rate": _to_float(row.get("prdy_ctrt")),
            "acml_vol": _to_int(row.get("acml_vol")),
            "seln_vol": _to_int(row.get("whol_smtn_seln_vol")),
            "shnu_vol": _to_int(row.get("whol_smtn_shnu_vol")),
            "ntby_qty": _to_int(row.get("whol_smtn_ntby_qty")),
            "seln_amt": _to_int(row.get("whol_smtn_seln_tr_pbmn")),
            "shnu_amt": _to_int(row.get("whol_smtn_shnu_tr_pbmn")),
            "ntby_amt": _to_int(row.get("whol_smtn_ntby_tr_pbmn")),
        }
        for row in (output or [])
    ]
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────
# 3. 회원사(증권사)별 매매동향 — 매도/매수 상위 5개사
# ─────────────────────────────────────────────────────────────

def _member_rows(output: dict, side: str) -> list[dict]:
    """`seln`/`shnu` 접두사로 흩어진 1~5번 필드를 행 리스트로 모은다."""
    rows = []
    for i in range(1, _TOP_N + 1):
        name = output.get(f"{side}_mbcr_name{i}", "")
        if not name:
            continue
        rows.append({
            "name": name,
            "qty": _to_int(output.get(f"total_{side}_qty{i}")),
            "ratio": _to_float(output.get(f"{side}_mbcr_rlim{i}")),
            "change": _to_int(output.get(f"{side}_qty_icdc{i}")),
            "is_foreign": output.get(f"{side}_mbcr_glob_yn_{i}", "N") == "Y",
        })
    return rows


def GetMemberFirmTrading(stock_code: str) -> dict:
    """주식현재가 회원사 — 매도/매수 상위 5개 증권사와 외국계 합계.

    Returns:
        {"sell": [...], "buy": [...], "foreign": {...}, "acml_vol": int}
    """
    output = _call(
        "inquire-member",
        "FHKST01010600",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code},
        "member-firm",
    )
    # 이 TR 은 단건이지만 리스트로 감싸 오는 경우가 있어 둘 다 처리한다.
    if isinstance(output, list):
        output = output[0] if output else {}
    output = output or {}

    return {
        "sell": _member_rows(output, "seln"),
        "buy": _member_rows(output, "shnu"),
        "foreign": {
            "seln_qty": _to_int(output.get("glob_total_seln_qty")),
            "shnu_qty": _to_int(output.get("glob_total_shnu_qty")),
            "ntby_qty": _to_int(output.get("glob_ntby_qty")),
            "seln_ratio": _to_float(output.get("glob_seln_rlim")),
            "shnu_ratio": _to_float(output.get("glob_shnu_rlim")),
        },
        "acml_vol": _to_int(output.get("acml_vol")),
    }
