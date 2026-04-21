"""
데이터 수집 현황 및 로그 조회 API (Django Ninja)
- 원천데이터(OHLCV), 기술적 분석, 재무제표 수집 현황
- 날짜별 빈 구간 조회
- 실시간 로그 및 프로세스 상태
"""

import os
import sys
import subprocess
from datetime import date, timedelta
from typing import List, Optional

from ninja import Router, Schema
from django.db.models import Count, Max, Min

from myweb.models import Company, StockOHLCV, StockAnalysis, StockFinancialStatement, IndexOHLCV, StockIndex

data_status_router = Router()

# 트리거 스크립트 경로
TRIGGER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_trigger.py",
)
PYTHON_BIN = sys.executable

# 태스크별 로그 파일 매핑
TASK_LOG_MAP = {
    "daily":     "/tmp/daily_pipeline.log",
    "ohlcv":     "/tmp/ohlcv_collect.log",
    "index":     "/tmp/index_collect.log",
    "analysis":  "/tmp/analysis_collect.log",
    "financial": "/tmp/dart_collect.log",
}

# 태스크별 프로세스 식별 키워드
TASK_PROC_MAP = {
    "daily":     "--task daily",
    "ohlcv":     "--task ohlcv",
    "index":     "--task index",
    "analysis":  "--task analysis",
    "financial": ["collect_dart.py", "--task financial"],
}


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class DataSummarySchema(Schema):
    total_companies: int
    ohlcv_done: int
    ohlcv_total_records: int
    ohlcv_latest_date: Optional[str]
    ohlcv_oldest_date: Optional[str]
    analysis_done: int
    analysis_total_records: int
    analysis_latest_date: Optional[str]
    financial_done: int
    financial_total_records: int
    financial_latest_year: Optional[str]
    financial_latest_quarter: Optional[str]


class DateGapSchema(Schema):
    date: str
    count: int
    total: int
    missing: int
    ratio: float


class AnalysisGapSchema(Schema):
    ohlcv_only: int       # OHLCV는 있으나 분석이 없는 종목 수
    both: int             # 둘 다 있는 종목 수
    ohlcv_latest: Optional[str]
    analysis_latest: Optional[str]


class FinancialGapSchema(Schema):
    done: int
    missing: int
    total: int
    missing_codes: List[str]


class LogSchema(Schema):
    source: str
    lines: List[str]


class ProcessSchema(Schema):
    name: str
    pid: Optional[int]
    running: bool
    elapsed: Optional[str]
    cpu_percent: Optional[float]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@data_status_router.get("/summary", response=DataSummarySchema)
def get_summary(request):
    """전체 데이터 수집 현황 요약"""
    total = Company.objects.filter(market__in=["KOSPI", "KOSDAQ", "KONEX"]).count()

    # OHLCV
    ohlcv_agg = StockOHLCV.objects.aggregate(
        done=Count("code_id", distinct=True),
        total=Count("id"),
        latest=Max("date"),
        oldest=Min("date"),
    )

    # Analysis
    analysis_agg = StockAnalysis.objects.aggregate(
        done=Count("code_id", distinct=True),
        total=Count("id"),
        latest=Max("date"),
    )

    # Financial
    financial_agg = StockFinancialStatement.objects.aggregate(
        done=Count("code_id", distinct=True),
        total=Count("id"),
    )
    fin_latest = (
        StockFinancialStatement.objects.values("year", "quarter")
        .order_by("-year", "-quarter")
        .first()
    )

    return {
        "total_companies": total,
        "ohlcv_done": ohlcv_agg["done"] or 0,
        "ohlcv_total_records": ohlcv_agg["total"] or 0,
        "ohlcv_latest_date": str(ohlcv_agg["latest"]) if ohlcv_agg["latest"] else None,
        "ohlcv_oldest_date": str(ohlcv_agg["oldest"]) if ohlcv_agg["oldest"] else None,
        "analysis_done": analysis_agg["done"] or 0,
        "analysis_total_records": analysis_agg["total"] or 0,
        "analysis_latest_date": str(analysis_agg["latest"]) if analysis_agg["latest"] else None,
        "financial_done": financial_agg["done"] or 0,
        "financial_total_records": financial_agg["total"] or 0,
        "financial_latest_year": fin_latest["year"] if fin_latest else None,
        "financial_latest_quarter": fin_latest["quarter"] if fin_latest else None,
    }


@data_status_router.get("/gaps", response=List[DateGapSchema])
def get_date_gaps(request, days: int = 30):
    """날짜별 OHLCV 수집 종목 수 (최근 N 거래일)"""
    total = Company.objects.filter(market__in=["KOSPI", "KOSDAQ", "KONEX"]).count()

    # 최근 days일 내 날짜별 집계
    cutoff = date.today() - timedelta(days=days * 2)  # 넉넉하게 2배로 조회 후 상위 N개 추출
    rows = (
        StockOHLCV.objects.filter(date__gte=cutoff)
        .values("date")
        .annotate(count=Count("code_id", distinct=True))
        .order_by("-date")[:days]
    )

    result = []
    for row in rows:
        missing = total - row["count"]
        result.append({
            "date": str(row["date"]),
            "count": row["count"],
            "total": total,
            "missing": missing,
            "ratio": round(row["count"] / total * 100, 1) if total else 0,
        })

    return result


class PipelineStepData(Schema):
    key: str
    label: str
    total: int
    by_date: dict  # date str → {count, ratio, missing}


class PipelineGridSchema(Schema):
    dates: List[str]
    steps: List[PipelineStepData]


@data_status_router.get("/pipeline-grid", response=PipelineGridSchema)
def get_pipeline_grid(request, days: int = 30):
    """날짜별 파이프라인 스텝 완료율 격자 데이터"""
    cutoff = date.today() - timedelta(days=days * 2)
    total_stocks = Company.objects.filter(market__in=["KOSPI", "KOSDAQ", "KONEX"]).count()
    total_indices = StockIndex.objects.count()

    # 주식 OHLCV per date
    ohlcv_rows = (
        StockOHLCV.objects.filter(date__gte=cutoff)
        .values("date")
        .annotate(count=Count("code_id", distinct=True))
        .order_by("-date")[:days]
    )
    ohlcv_by_date = {
        str(r["date"]): {
            "count": r["count"],
            "ratio": round(r["count"] / total_stocks * 100, 1) if total_stocks else 0,
            "missing": total_stocks - r["count"],
        }
        for r in ohlcv_rows
    }

    # 기술적 분석 per date
    analysis_rows = (
        StockAnalysis.objects.filter(date__gte=cutoff)
        .values("date")
        .annotate(count=Count("code_id", distinct=True))
        .order_by("-date")[:days]
    )
    analysis_by_date = {
        str(r["date"]): {
            "count": r["count"],
            "ratio": round(r["count"] / total_stocks * 100, 1) if total_stocks else 0,
            "missing": total_stocks - r["count"],
        }
        for r in analysis_rows
    }

    # 인덱스 OHLCV per date
    index_rows = (
        IndexOHLCV.objects.filter(date__gte=cutoff)
        .values("date")
        .annotate(count=Count("code_id", distinct=True))
        .order_by("-date")[:days]
    )
    index_by_date = {
        str(r["date"]): {
            "count": r["count"],
            "ratio": round(r["count"] / total_indices * 100, 1) if total_indices else 0,
            "missing": total_indices - r["count"],
        }
        for r in index_rows
    }

    # 날짜 목록 (세 스텝의 합집합, 내림차순)
    all_dates = sorted(
        set(ohlcv_by_date) | set(analysis_by_date) | set(index_by_date),
        reverse=True,
    )[:days]

    steps = [
        {"key": "ohlcv",        "label": "주식 OHLCV",  "total": total_stocks,  "by_date": ohlcv_by_date},
        {"key": "analysis",     "label": "기술적 분석", "total": total_stocks,  "by_date": analysis_by_date},
        {"key": "index_ohlcv",  "label": "인덱스 OHLCV","total": total_indices, "by_date": index_by_date},
    ]

    return {"dates": all_dates, "steps": steps}


@data_status_router.get("/analysis-gap", response=AnalysisGapSchema)
def get_analysis_gap(request):
    """OHLCV 대비 기술적 분석 누락 현황"""
    ohlcv_codes = set(
        StockOHLCV.objects.values_list("code_id", flat=True).distinct()
    )
    analysis_codes = set(
        StockAnalysis.objects.values_list("code_id", flat=True).distinct()
    )

    ohlcv_agg = StockOHLCV.objects.aggregate(latest=Max("date"))
    analysis_agg = StockAnalysis.objects.aggregate(latest=Max("date"))

    return {
        "ohlcv_only": len(ohlcv_codes - analysis_codes),
        "both": len(ohlcv_codes & analysis_codes),
        "ohlcv_latest": str(ohlcv_agg["latest"]) if ohlcv_agg["latest"] else None,
        "analysis_latest": str(analysis_agg["latest"]) if analysis_agg["latest"] else None,
    }


@data_status_router.get("/financial-gap", response=FinancialGapSchema)
def get_financial_gap(request):
    """재무제표 미수집 종목 현황"""
    all_codes = set(
        Company.objects.filter(market__in=["KOSPI", "KOSDAQ", "KONEX"])
        .values_list("code", flat=True)
    )
    done_codes = set(
        StockFinancialStatement.objects.values_list("code_id", flat=True).distinct()
    )
    missing_codes = sorted(all_codes - done_codes)[:100]  # 최대 100개만 반환

    return {
        "done": len(done_codes),
        "missing": len(all_codes - done_codes),
        "total": len(all_codes),
        "missing_codes": missing_codes,
    }


@data_status_router.get("/logs", response=LogSchema)
def get_logs(request, source: str = "dart", lines: int = 200):
    """
    로그 파일 조회
    source: dart | server
    """
    log_files = {
        "daily":     "/tmp/daily_pipeline.log",
        "dart":      "/tmp/dart_collect.log",
        "ohlcv":     "/tmp/ohlcv_collect.log",
        "index":     "/tmp/index_collect.log",
        "analysis":  "/tmp/analysis_collect.log",
        "scheduler": "/tmp/scheduler.log",
        "server": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "server.log"
        ),
    }

    log_path = log_files.get(source, log_files["dart"])

    try:
        if not os.path.exists(log_path):
            return {"source": source, "lines": [f"[로그 파일 없음: {log_path}]"]}

        result = subprocess.run(
            ["tail", "-n", str(lines), log_path],
            capture_output=True,
            text=True,
        )
        log_lines = result.stdout.splitlines()

        # Warning/RuntimeWarning 등 노이즈 제거
        filtered = [
            line for line in log_lines
            if not any(
                kw in line for kw in [
                    "RuntimeWarning", "UserWarning", "pkg_resources",
                    "Scheduled job", "Scheduler", "KRX 로그인"
                ]
            )
        ]
        return {"source": source, "lines": filtered}
    except Exception as e:
        return {"source": source, "lines": [f"[오류: {str(e)}]"]}


@data_status_router.get("/processes", response=List[ProcessSchema])
def get_processes(request):
    """현재 실행 중인 데이터 수집 프로세스 상태"""
    targets = {
        "collect_dart.py":   "재무제표 수집 (자동)",
        "collect_history.py": "전체 데이터 수집",
        "--task ohlcv":      "OHLCV 수집 (수동)",
        "--task index":      "인덱스 수집 (수동)",
        "--task analysis":   "기술적 분석 (수동)",
        "--task financial":  "재무제표 수집 (수동)",
    }

    results = []
    try:
        ps_output = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
        ).stdout

        for script, label in targets.items():
            found = False
            for line in ps_output.splitlines():
                if script in line and "grep" not in line:
                    parts = line.split()
                    try:
                        pid = int(parts[1])
                        cpu = float(parts[2])
                        elapsed = parts[9]  # ELAPSED (macOS ps aux)
                        results.append({
                            "name": label,
                            "pid": pid,
                            "running": True,
                            "elapsed": elapsed,
                            "cpu_percent": cpu,
                        })
                    except (IndexError, ValueError):
                        results.append({
                            "name": label,
                            "pid": None,
                            "running": True,
                            "elapsed": None,
                            "cpu_percent": None,
                        })
                    found = True
                    break

            if not found:
                results.append({
                    "name": label,
                    "pid": None,
                    "running": False,
                    "elapsed": None,
                    "cpu_percent": None,
                })

    except Exception as e:
        pass

    return results


# ─────────────────────────────────────────────
# 수동 트리거 엔드포인트
# ─────────────────────────────────────────────

class TriggerRequestSchema(Schema):
    task: str                          # daily | ohlcv | index | analysis | financial
    start_date: Optional[str] = None  # YYYY-MM-DD (daily/financial 제외)
    end_date: Optional[str] = None    # YYYY-MM-DD (daily/financial 제외)
    skip_dart: bool = False           # daily 전용: DART 수집 건너뜀


class TriggerResponseSchema(Schema):
    success: bool
    message: str
    log_source: Optional[str] = None


def _is_task_running(task: str) -> bool:
    """해당 태스크가 이미 실행 중인지 확인"""
    keyword = f"--task {task}"
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if keyword in line and "grep" not in line:
                return True
    except Exception:
        pass
    return False


@data_status_router.post("/trigger", response=TriggerResponseSchema)
def trigger_collection(request, payload: TriggerRequestSchema):
    """
    데이터 수집 수동 트리거
    task: daily | ohlcv | index | analysis | financial
    start_date / end_date: YYYY-MM-DD (daily/financial 제외)
    skip_dart: daily 전용, DART 수집 건너뜀
    """
    task = payload.task
    if task not in ("daily", "ohlcv", "index", "analysis", "financial"):
        return {"success": False, "message": f"알 수 없는 태스크: {task}"}

    if _is_task_running(task):
        return {"success": False, "message": f"이미 실행 중입니다: {task}"}

    log_path = TASK_LOG_MAP[task]
    cmd = [PYTHON_BIN, "-u", TRIGGER_SCRIPT, "--task", task]

    if task == "daily":
        if payload.start_date:
            cmd += ["--start", payload.start_date]
        if payload.end_date:
            cmd += ["--end", payload.end_date]
        if payload.skip_dart:
            cmd += ["--skip-dart"]
    else:
        if payload.start_date:
            cmd += ["--start", payload.start_date]
        if payload.end_date:
            cmd += ["--end", payload.end_date]

    try:
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                cwd=os.path.dirname(TRIGGER_SCRIPT),
            )

        task_labels = {
            "daily":     "일간 전체 파이프라인",
            "ohlcv":     "OHLCV 수집",
            "index":     "인덱스 수집",
            "analysis":  "기술적 분석",
            "financial": "재무제표 수집",
        }
        date_info = ""
        if task not in ("daily", "financial") and payload.start_date:
            date_info = f" ({payload.start_date} ~ {payload.end_date or '오늘'})"
        if task == "daily" and payload.skip_dart:
            date_info = " (DART 제외)"

        return {
            "success": True,
            "message": f"{task_labels[task]}{date_info} 시작됐습니다.",
            "log_source": "daily" if task == "daily" else (task if task != "financial" else "dart"),
        }
    except Exception as e:
        return {"success": False, "message": f"실행 오류: {str(e)}"}
