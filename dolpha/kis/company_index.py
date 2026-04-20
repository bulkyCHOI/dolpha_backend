"""
KIS API를 이용한 Company-StockIndex M2M 관계 재구성 모듈

각 종목의 inquire-price API에서 bstp_kor_isnm(업종명)을 수집하고,
업종명 → KIS 인덱스 코드 매핑 테이블로 myweb_company_indices를 채웁니다.
"""

import time
import warnings
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

_PATH = "uapi/domestic-stock/v1/quotations/inquire-price"
_TR_ID = "FHKST01010100"

# (market, bstp_kor_isnm) → StockIndex.code
# KIS API가 반환하는 업종명과 우리 StockIndex.name의 불일치를 해소하는 매핑 테이블
_SECTOR_MAP = {
    # ── KOSPI 업종 ──────────────────────────────────────────────
    ("KOSPI", "음식료·담배"):       "0005",
    ("KOSPI", "섬유·의류"):         "0006",
    ("KOSPI", "종이·목재"):         "0007",
    ("KOSPI", "화학"):              "0008",
    ("KOSPI", "제약"):              "0009",   # KIS: 제약 = StockIndex: 의약품
    ("KOSPI", "의약품"):            "0009",
    ("KOSPI", "비금속"):            "0010",
    ("KOSPI", "비금속광물"):        "0010",
    ("KOSPI", "금속"):              "0011",
    ("KOSPI", "철강금속"):          "0011",
    ("KOSPI", "기계·장비"):         "0012",
    ("KOSPI", "기계"):              "0012",
    ("KOSPI", "전기·전자"):         "0013",
    ("KOSPI", "전기전자"):          "0013",
    ("KOSPI", "의료·정밀기기"):     "0014",
    ("KOSPI", "의료정밀"):          "0014",
    ("KOSPI", "운송장비·부품"):     "0015",
    ("KOSPI", "운수장비"):          "0015",
    ("KOSPI", "유통"):              "0016",
    ("KOSPI", "유통업"):            "0016",
    ("KOSPI", "전기·가스·수도"):    "0017",
    ("KOSPI", "전기가스업"):        "0017",
    ("KOSPI", "건설"):              "0018",
    ("KOSPI", "건설업"):            "0018",
    ("KOSPI", "운송·창고"):         "0019",
    ("KOSPI", "운수창고업"):        "0019",
    ("KOSPI", "통신방송서비스"):    "0020",
    ("KOSPI", "통신업"):            "0020",
    ("KOSPI", "금융"):              "0021",
    ("KOSPI", "금융업"):            "0021",
    ("KOSPI", "은행"):              "0022",
    ("KOSPI", "증권"):              "0023",
    ("KOSPI", "보험"):              "0024",
    ("KOSPI", "일반서비스"):        "0025",
    ("KOSPI", "서비스업"):          "0025",
    # ── KOSDAQ 업종 ─────────────────────────────────────────────
    ("KOSDAQ", "음식료·담배"):      "1005",
    ("KOSDAQ", "섬유·의류"):        "1006",
    ("KOSDAQ", "종이·목재"):        "1007",
    ("KOSDAQ", "출판·매체복제"):    "1008",
    ("KOSDAQ", "화학"):             "1009",
    ("KOSDAQ", "제약"):             "1010",
    ("KOSDAQ", "비금속"):           "1011",
    ("KOSDAQ", "금속"):             "1012",
    ("KOSDAQ", "기계·장비"):        "1013",
    ("KOSDAQ", "전기·전자부품"):    "1014",
    ("KOSDAQ", "전기·전자"):        "1014",
    ("KOSDAQ", "의료·정밀기기"):    "1015",
    ("KOSDAQ", "운송장비·부품"):    "1016",
    ("KOSDAQ", "기타 제조"):        "1017",
    ("KOSDAQ", "건설"):             "1018",
    ("KOSDAQ", "유통"):             "1019",
    ("KOSDAQ", "운송"):             "1020",
    ("KOSDAQ", "운송·창고"):        "1020",
    ("KOSDAQ", "금융"):             "1021",
    ("KOSDAQ", "보험"):             "1021",   # KOSDAQ 보험은 금융으로 분류
    ("KOSDAQ", "오락·문화"):        "1022",
    ("KOSDAQ", "통신방송서비스"):   "1023",
    ("KOSDAQ", "IT S/W & SVC"):     "1024",
    ("KOSDAQ", "IT H/W"):           "1025",
}

# 시장 전체 → 종합지수 매핑
_MARKET_COMPOSITE = {
    "KOSPI":  "0001",
    "KOSDAQ": "1001",
}


def get_sector_name(stock_code: str) -> str:
    """KIS inquire-price API로 단일 종목의 업종명(bstp_kor_isnm) 조회."""
    url = f"{get_url_base('REAL')}/{_PATH}"
    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10, verify=False)
        if res.status_code == 200 and res.json().get("rt_cd") == "0":
            return res.json().get("output", {}).get("bstp_kor_isnm", "").strip()
    except Exception:
        pass
    return ""


def rebuild_company_indices() -> dict:
    """
    모든 KOSPI/KOSDAQ 종목의 업종명을 KIS API로 조회하고
    myweb_company_indices(Company↔StockIndex M2M) 테이블을 재구성합니다.

    Returns:
        {"status": "OK", "total": N, "mapped": N, "skipped": N, "errors": [...]}
    """
    from django.db import transaction
    from myweb.models import Company, StockIndex

    # 인덱스 코드 → 객체 캐시
    index_map = {idx.code: idx for idx in StockIndex.objects.all()}
    if not index_map:
        return {"status": "ERROR", "message": "StockIndex 데이터가 없습니다. 먼저 인덱스 목록을 수집하세요."}

    companies = list(
        Company.objects.filter(market__in=["KOSPI", "KOSDAQ"])
        .values_list("code", "market")
    )

    total = len(companies)
    mapped = 0
    skipped = 0
    errors = []

    # company_code → [index_obj] 관계 수집
    relations: dict[str, list] = {}

    print(f"[rebuild_company_indices] 총 {total}개 종목 업종 조회 시작...")

    for i, (code, market) in enumerate(companies):
        if i % 100 == 0:
            print(f"  진행: {i}/{total} ({i*100//total}%)")

        sector_name = get_sector_name(code)
        time.sleep(0.22)

        indices = []

        # 종합지수 (전체 시장) 할당
        composite_code = _MARKET_COMPOSITE.get(market)
        if composite_code and composite_code in index_map:
            indices.append(index_map[composite_code])

        # 업종별 인덱스 할당
        index_code = _SECTOR_MAP.get((market, sector_name))
        if index_code and index_code in index_map:
            indices.append(index_map[index_code])
        elif sector_name and sector_name != " ":
            errors.append(f"[{code}/{market}] 미매핑 업종명: {sector_name!r}")

        if indices:
            relations[code] = indices
            mapped += 1
        else:
            skipped += 1

    # DB에 M2M 관계 일괄 저장
    print(f"  DB 저장 시작: {len(relations)}개 종목 관계 설정...")
    with transaction.atomic():
        # 기존 관계 전체 초기화
        for idx in index_map.values():
            idx.companies.clear()

        # 새 관계 설정 (배치)
        for code, idx_list in relations.items():
            try:
                company = Company.objects.get(code=code)
                company.indices.set(idx_list)
            except Company.DoesNotExist:
                pass

    print(f"[rebuild_company_indices] 완료: 매핑 {mapped}개, 스킵 {skipped}개")
    return {
        "status": "OK",
        "total": total,
        "mapped": mapped,
        "skipped": skipped,
        "unmapped_sectors": list(set(errors))[:20],
    }
