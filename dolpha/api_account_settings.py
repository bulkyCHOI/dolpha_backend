"""
계좌 설정 API — KIS 모드(실계좌/가상계좌) 조회 및 변경
"""
import os
import re
from pathlib import Path

from ninja import Router
from django.http import JsonResponse

from .api_mypage_ninja import get_authenticated_user

account_settings_router = Router()

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _mask(account_no: str) -> str:
    if not account_no or len(account_no) < 4:
        return account_no or ""
    return account_no[:2] + "*" * (len(account_no) - 4) + account_no[-2:]


@account_settings_router.get("/account-settings")
def get_account_settings(request):
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    kis_mode = os.environ.get("KIS_MODE", "REAL").upper()
    real_no = os.environ.get("KIS_REAL_ACCOUNT_NO", "")
    virtual_no = os.environ.get("KIS_VIRTUAL_ACCOUNT_NO", "")
    current_no = real_no if kis_mode == "REAL" else virtual_no

    return JsonResponse({
        "success": True,
        "data": {
            "kis_mode": kis_mode,
            "real_account_no": _mask(real_no),
            "virtual_account_no": _mask(virtual_no),
            "current_account_no": _mask(current_no),
        },
    })


@account_settings_router.post("/account-settings")
def update_account_settings(request):
    import json

    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    try:
        body = json.loads(request.body)
        new_mode = body.get("kis_mode", "").upper()
    except Exception:
        return JsonResponse({"success": False, "error": "잘못된 요청입니다."}, status=400)

    if new_mode not in ("REAL", "VIRTUAL"):
        return JsonResponse({"success": False, "error": "kis_mode는 REAL 또는 VIRTUAL이어야 합니다."}, status=400)

    # os.environ 즉시 반영 (서버 재시작 없이 적용)
    os.environ["KIS_MODE"] = new_mode

    # .env 파일 영구 저장
    try:
        if _ENV_PATH.exists():
            text = _ENV_PATH.read_text(encoding="utf-8")
            if re.search(r"^KIS_MODE\s*=", text, re.MULTILINE):
                text = re.sub(r"^(KIS_MODE\s*=\s*).*$", f"KIS_MODE={new_mode}", text, flags=re.MULTILINE)
            else:
                text = text.rstrip("\n") + f"\nKIS_MODE={new_mode}\n"
            _ENV_PATH.write_text(text, encoding="utf-8")
    except Exception as e:
        # .env 쓰기 실패해도 os.environ은 이미 반영됐으므로 경고만
        return JsonResponse({
            "success": True,
            "warning": f".env 파일 저장 실패 (서버 재시작 시 초기화됨): {e}",
            "kis_mode": new_mode,
        })

    return JsonResponse({"success": True, "kis_mode": new_mode})
