"""
DART 재무제표 병렬 수집 핵심 로직.
collect_dart.py, data_trigger.py, api_data.py 에서 공통으로 사용.
"""

from __future__ import annotations

import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from django.db import transaction
from myweb.models import Company, StockFinancialStatement

DART_API_KEY = "b6533c6c78ba430e7c63ef16db7bb893ae440d43"
REPRT_LIST = ["11013", "11012", "11014", "11011"]  # 1Q, 2Q, 3Q, 4Q
QUARTER_MAP = {"11011": "4Q", "11012": "2Q", "11013": "1Q", "11014": "3Q"}

_thread_local = threading.local()


def _get_dart():
    """스레드별 OpenDartReader 인스턴스 반환 (thread-safe)."""
    if not hasattr(_thread_local, "dart"):
        import OpenDartReader
        _thread_local.dart = OpenDartReader(DART_API_KEY)
    return _thread_local.dart


def _parse_amount(val) -> int:
    val = str(val).replace(",", "").strip()
    if val in ("-", "", "None", "nan"):
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def fetch_dart_data(code: str) -> pd.DataFrame | None:
    """
    단일 종목 DART 재무제표 fetch + 4Q 값 조정.
    성공 시 DataFrame, 데이터 없으면 None.
    """
    dart = _get_dart()
    year_now = datetime.now().year
    all_dfs = []

    for year in [year_now, year_now - 1]:
        year_data = []
        for reprt in REPRT_LIST:
            try:
                df = dart.finstate(code, year, reprt_code=reprt)
                if df is None or df.empty or "thstrm_amount" not in df.columns:
                    continue

                if (df["fs_nm"] == "연결재무제표").any():
                    df = df[df["fs_nm"] == "연결재무제표"].copy()
                else:
                    df = df[df["fs_nm"] == "재무제표"].copy()

                df["thstrm_amount"] = df["thstrm_amount"].apply(_parse_amount)
                df["year"] = str(year)
                df["quarter"] = df["reprt_code"].map(QUARTER_MAP)
                year_data.append(
                    df[["year", "quarter", "sj_nm", "account_nm", "thstrm_amount"]]
                )
            except Exception:
                continue

        if not year_data:
            continue

        df_year = pd.concat(year_data, ignore_index=True)

        # 4분기 값 조정: 손익계산서는 연간 누적이므로 1Q+2Q+3Q 합계를 차감
        if "4Q" in df_year["quarter"].values:
            q123_sum = (
                df_year[df_year["quarter"].isin(["1Q", "2Q", "3Q"])]
                .groupby(["sj_nm", "account_nm"])["thstrm_amount"]
                .sum()
                .reset_index()
                .rename(columns={"thstrm_amount": "q123_total"})
            )

            q4 = df_year[df_year["quarter"] == "4Q"].copy()
            q4 = q4.merge(q123_sum, on=["sj_nm", "account_nm"], how="left")
            q4["q123_total"] = q4["q123_total"].fillna(0).astype(int)

            q4_bs = q4[q4["sj_nm"] == "재무상태표"][
                ["sj_nm", "account_nm", "thstrm_amount"]
            ].copy()

            q4_is = q4[q4["sj_nm"] == "손익계산서"].copy()
            q4_is["thstrm_amount"] = q4_is["thstrm_amount"] - q4_is["q123_total"]
            q4_is = q4_is[["sj_nm", "account_nm", "thstrm_amount"]]

            q4_merged = pd.concat([q4_bs, q4_is], ignore_index=True)
            q4_merged["year"] = str(year)
            q4_merged["quarter"] = "4Q"

            df_year = pd.concat(
                [df_year[df_year["quarter"] != "4Q"], q4_merged],
                ignore_index=True,
            )

        all_dfs.append(df_year)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def save_to_db(company_code: str, df: pd.DataFrame) -> int:
    """DataFrame을 DB에 저장. 저장된 레코드 수 반환."""
    objects = [
        StockFinancialStatement(
            code_id=company_code,
            year=row["year"],
            quarter=row["quarter"],
            statement_type=row["sj_nm"],
            account_name=row["account_nm"],
            amount=int(row["thstrm_amount"]),
        )
        for _, row in df.iterrows()
    ]
    with transaction.atomic():
        StockFinancialStatement.objects.filter(code_id=company_code).delete()
        StockFinancialStatement.objects.bulk_create(
            objects, batch_size=500, ignore_conflicts=True
        )
    return len(objects)


def _process_company(code: str) -> tuple[str, int, str | None]:
    """단일 종목 fetch + DB 저장. (code, 저장 레코드 수, 오류메시지) 반환."""
    try:
        df = fetch_dart_data(code)
        if df is None or df.empty:
            return code, 0, "no_data"
        count = save_to_db(code, df)
        return code, count, None
    except Exception as e:
        return code, 0, str(e)


def run_parallel(workers: int = 10, resume: bool = False) -> dict:
    """
    전체 종목 재무제표를 병렬로 수집.

    Args:
        workers: 동시 처리 스레드 수 (기본 10)
        resume: True면 이미 수집된 종목 건너뜀

    Returns:
        {"status": "OK", "message": ..., "count_saved": N,
         "success": N, "skip": N, "fail": N}
    """
    all_codes = list(
        Company.objects.filter(market__in=["KOSPI", "KOSDAQ", "KONEX"])
        .values_list("code", flat=True)
    )

    if resume:
        done_codes = set(
            StockFinancialStatement.objects.values_list("code_id", flat=True).distinct()
        )
        remaining = [c for c in all_codes if c not in done_codes]
    else:
        remaining = all_codes

    success, fail, skip = 0, 0, 0
    total_saved = 0
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_company, code): code for code in remaining}
        with tqdm(total=len(remaining), desc="DART 수집") as pbar:
            for future in as_completed(futures):
                code, count, error = future.result()
                if error == "no_data":
                    skip += 1
                elif error:
                    fail += 1
                    if len(errors) < 10:
                        errors.append(f"[{code}] {error}")
                else:
                    success += 1
                    total_saved += count
                pbar.update(1)
                pbar.set_postfix(success=success, skip=skip, fail=fail)

    return {
        "status": "OK",
        "message": f"성공 {success}개 | 건너뜀 {skip}개 | 실패 {fail}개",
        "count_saved": total_saved,
        "success": success,
        "skip": skip,
        "fail": fail,
        "errors": errors,
    }
