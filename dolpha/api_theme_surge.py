"""급등테마주 전략 API.

엔드포인트:
  GET  /theme-surge/timeline    — 09:00~15:30 급등 테마 타임라인 (당일 또는 지정일)
  GET  /theme-surge/themes      — 특정 슬롯의 테마 랭킹 스냅샷
  GET  /theme-surge/candidates  — 주도주 후보 목록 (테마별 1등 종목)
  GET  /theme-surge/live        — 토스증권 실시간 테마 랭킹 (DB 미경유)
  GET  /theme-surge/positions   — 자동매매 현황: 보유 포지션 + 대기 후보 (로그인 필요)
  GET  /theme-surge/entry-chart — 종목 1분봉 + 진입 판정 좌표 (로그인 필요)
  POST /theme-surge/scan        — 5분 스캔 수동 실행 (로그인 필요)
  DELETE /theme-surge/candidates — 특정 등록일의 후보 설정·분봉·판정 삭제 (로그인 필요)

타임라인은 로그인 없이도 조회 가능하며, 로그인한 경우 해당 유저의 진입 시그널
마커가 함께 내려간다.
"""

from datetime import date as date_cls, datetime

from django.http import JsonResponse
from ninja import Router
from pytz import timezone as pytz_tz

from .api_mypage_ninja import get_authenticated_user
from .theme_surge import (
    EntryChartError,
    TossThemeError,
    build_entry_chart,
    build_positions,
    build_timeline,
    fetch_theme_ranking,
    purge_candidate_date,
    run_theme_scan,
)
from .theme_surge.config import SURGE_TOP_N

theme_surge_router = Router()

_KST = pytz_tz("Asia/Seoul")


def _parse_date(raw: str) -> date_cls | None:
    """YYYY-MM-DD 문자열을 파싱한다. 비어 있으면 오늘(KST)."""
    if not raw:
        return datetime.now(_KST).date()
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


@theme_surge_router.get("/timeline")
def get_timeline(request, date: str = "", only_surge: bool = True):
    """급등 테마 타임라인 조회."""
    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        user = get_authenticated_user(request)
        data = build_timeline(target, user=user, only_surge=only_surge)
        return JsonResponse({"status": "OK", "data": data})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.get("/themes")
def get_themes(request, date: str = "", slot: str = ""):
    """특정 슬롯(미지정 시 당일 최신 슬롯)의 테마 랭킹 스냅샷."""
    from myweb.models import ThemeSnapshot

    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        qs = ThemeSnapshot.objects.filter(date=target)

        if slot:
            try:
                slot_time = datetime.strptime(slot.strip(), "%H:%M").time()
            except ValueError:
                return JsonResponse(
                    {"status": "error", "message": "slot은 HH:MM 형식이어야 합니다."}, status=400
                )
            qs = qs.filter(slot_time=slot_time)
        else:
            latest = qs.order_by("-slot_time").values_list("slot_time", flat=True).first()
            if latest is None:
                return JsonResponse({"status": "OK", "data": [], "slot": None})
            qs = qs.filter(slot_time=latest)

        rows = [
            {
                "tics_id": s.tics_id,
                "theme_name": s.theme_name,
                "rank": s.rank,
                "fluctuation_rate": round(s.fluctuation_rate, 2),
                "trading_value": s.trading_value,
                "market_cap": s.market_cap,
                "stock_count": s.stock_count,
                "momentum": round(s.momentum, 2),
                "is_surge": s.is_surge,
                "surge_reason": s.surge_reason,
                "slot": s.slot_time.strftime("%H:%M"),
            }
            for s in qs.order_by("rank")
        ]
        return JsonResponse(
            {"status": "OK", "data": rows, "slot": rows[0]["slot"] if rows else None}
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.get("/candidates")
def get_candidates(request, date: str = "", selected_only: bool = True):
    """주도주 후보 목록 (기본: 테마별 1등 종목만)."""
    from myweb.models import ThemeLeaderCandidate

    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        qs = ThemeLeaderCandidate.objects.filter(date=target)
        if selected_only:
            qs = qs.filter(is_selected=True)

        rows = [
            {
                "slot": c.slot_time.strftime("%H:%M"),
                "tics_id": c.tics_id,
                "theme_name": c.theme_name,
                "stock_code": c.stock_code,
                "stock_name": c.stock_name,
                "price": c.price,
                "change_rate": round(c.change_rate, 2),
                "trading_value": c.trading_value,
                "market_cap": c.market_cap,
                "score": round(c.score, 4),
                "rank_in_theme": c.rank_in_theme,
            }
            for c in qs.order_by("slot_time", "rank_in_theme")
        ]
        return JsonResponse({"status": "OK", "data": rows})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.get("/live")
def get_live_ranking(request, limit: int = SURGE_TOP_N):
    """토스증권 '지금 뜨는 산업' 실시간 랭킹 (DB 미경유 즉시 조회)."""
    try:
        themes = fetch_theme_ranking(limit=max(1, min(limit, 30)))
        rows = [
            {
                "tics_id": t.tics_id,
                "theme_name": t.name,
                "rank": t.rank,
                "fluctuation_rate": round(t.fluctuation_rate, 2),
                "trading_value": t.trading_value,
                "market_cap": t.market_cap,
                "stock_count": t.stock_count,
                "leading_stock_code": t.leading_stock_code,
                "leading_stock_name": t.leading_stock_name,
            }
            for t in themes
        ]
        return JsonResponse({"status": "OK", "data": rows})
    except TossThemeError as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=502)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.get("/positions")
def get_positions(request, date: str = ""):
    """급등테마주 자동매매 현황 — 보유 포지션과 대기 중인 후보."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"status": "error", "message": "인증이 필요합니다."}, status=401)

    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        return JsonResponse({"status": "OK", "data": build_positions(user, target)})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.get("/entry-chart")
def get_entry_chart(request, code: str, date: str = ""):
    """종목 1분봉과 진입 판정 좌표 — 전고점·눌림 구간을 차트로 보기 위한 데이터."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"status": "error", "message": "인증이 필요합니다."}, status=401)

    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        return JsonResponse({"status": "OK", "data": build_entry_chart(user, target, code)})
    except EntryChartError as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.post("/scan")
def trigger_scan(request):
    """5분 스캔을 수동 실행한다 (로그인 필요)."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"status": "error", "message": "인증이 필요합니다."}, status=401)

    try:
        # 사용자가 명시적으로 누른 스캔이므로 개장일 게이트를 우회한다
        result = run_theme_scan(force=True)
        return JsonResponse({"status": "OK", "data": result})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@theme_surge_router.delete("/candidates")
def delete_candidates(request, date: str = ""):
    """해당 등록일의 급등테마주 후보 설정과 1분봉·판정 이력을 모두 삭제한다.

    후보 설정은 장 마감 후 비활성화만 되고 계속 쌓이므로, 다 본 날짜는
    사용자가 직접 지운다. 보유 포지션이 남은 설정은 청산 관리가 끊기지 않도록 보존한다.
    """
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"status": "error", "message": "인증이 필요합니다."}, status=401)

    target = _parse_date(date)
    if target is None:
        return JsonResponse(
            {"status": "error", "message": "date는 YYYY-MM-DD 형식이어야 합니다."}, status=400
        )

    try:
        return JsonResponse({"status": "OK", "data": purge_candidate_date(user, target)})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
