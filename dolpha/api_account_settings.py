"""
계좌 설정 API

  GET    /mypage/account-settings              계좌 목록 + 전략별 계좌 매핑 조회
  POST   /mypage/kis-accounts                  계좌 등록
  PUT    /mypage/kis-accounts/{account_id}     계좌 수정 (앱키는 입력했을 때만 교체)
  DELETE /mypage/kis-accounts/{account_id}     계좌 삭제
  POST   /mypage/strategy-accounts             전략별 계좌 지정
"""

from typing import Optional

from django.db import transaction
from django.http import JsonResponse
from ninja import Router, Schema

from myweb.models import KisAccount, StrategyAccount, TradingConfig
from .api_mypage_ninja import get_authenticated_user
from .crypto import mask, mask_account_no
from .strategy_account import (
    STRATEGY_CHOICES,
    STRATEGY_KEYS,
    assign_account,
    get_account_map,
    get_default_account,
)

account_settings_router = Router()

ACCOUNT_TYPES = ("REAL", "VIRTUAL")


# ── 스키마 ─────────────────────────────────────────────────────────────────


class KisAccountIn(Schema):
    name: str
    account_type: str = "VIRTUAL"
    account_no: str
    account_cd: str = "01"
    app_key: str = ""
    app_secret: str = ""
    is_default: bool = False
    is_active: bool = True


class KisAccountUpdateIn(Schema):
    name: Optional[str] = None
    account_type: Optional[str] = None
    account_no: Optional[str] = None
    account_cd: Optional[str] = None
    app_key: Optional[str] = None      # 빈 값이면 기존 키 유지
    app_secret: Optional[str] = None   # 빈 값이면 기존 시크릿 유지
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None


class StrategyAccountIn(Schema):
    strategy_type: str
    account_id: Optional[int] = None   # None이면 지정 해제 → 기본 계좌 사용


# ── 직렬화 ─────────────────────────────────────────────────────────────────


def _serialize_account(account: KisAccount) -> dict:
    """계좌 정보를 화면 표시용으로 직렬화한다. 시크릿은 절대 평문으로 내보내지 않는다."""
    return {
        "id": account.pk,
        "name": account.name,
        "account_type": account.account_type,
        "account_no": mask_account_no(account.account_no),
        "account_cd": account.account_cd,
        "app_key_masked": mask(account.decrypted_app_key),
        "is_default": account.is_default,
        "is_active": account.is_active,
    }


def _validate_account_type(account_type: str) -> Optional[str]:
    if account_type not in ACCOUNT_TYPES:
        return "account_type은 REAL 또는 VIRTUAL이어야 합니다."
    return None


@transaction.atomic
def _set_as_default(user, account: KisAccount) -> None:
    """해당 계좌를 유일한 기본 계좌로 만든다."""
    KisAccount.objects.filter(user=user).exclude(pk=account.pk).update(is_default=False)
    if not account.is_default:
        account.is_default = True
        account.save(update_fields=["is_default", "updated_at"])


# ── 조회 ───────────────────────────────────────────────────────────────────


@account_settings_router.get("/account-settings")
def get_account_settings(request):
    """계좌 목록과 전략별 계좌 매핑을 함께 반환합니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    accounts = list(KisAccount.objects.filter(user=user))
    default_account = get_default_account(user)
    account_map = get_account_map(user)
    assigned_ids = {
        row.strategy_type: row.account_id
        for row in StrategyAccount.objects.filter(user=user)
    }

    # 전략별로 실제 사용 중인 활성 설정 수 — 계좌 변경 영향 범위를 화면에 보여준다
    config_counts: dict[str, int] = {}
    for strategy_type in (
        TradingConfig.objects.filter(user=user, is_active=True)
        .values_list("strategy_type", flat=True)
    ):
        config_counts[strategy_type] = config_counts.get(strategy_type, 0) + 1

    strategies = [
        {
            "strategy_type": key,
            "label": label,
            # 명시적으로 지정한 계좌 (없으면 None → 기본 계좌를 따름)
            "account_id": assigned_ids.get(key),
            # 실제로 사용될 계좌
            "effective_account_id": (
                account_map[key].pk if account_map.get(key) else None
            ),
            "effective_account_name": (
                account_map[key].name if account_map.get(key) else None
            ),
            "active_config_count": config_counts.get(key, 0),
        }
        for key, label in STRATEGY_CHOICES
    ]

    return JsonResponse({
        "success": True,
        "data": {
            "accounts": [_serialize_account(a) for a in accounts],
            "default_account_id": default_account.pk if default_account else None,
            "strategies": strategies,
        },
    })


# ── 계좌 등록 / 수정 / 삭제 ────────────────────────────────────────────────


@account_settings_router.post("/kis-accounts")
def create_kis_account(request, data: KisAccountIn):
    """KIS 계좌를 등록합니다. 앱키·시크릿은 암호화되어 저장됩니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    error = _validate_account_type(data.account_type)
    if error:
        return JsonResponse({"success": False, "error": error}, status=400)

    missing = [
        label
        for label, value in (
            ("계좌 별칭", data.name.strip()),
            ("계좌번호", data.account_no.strip()),
            ("앱키", data.app_key.strip()),
            ("앱시크릿", data.app_secret.strip()),
        )
        if not value
    ]
    if missing:
        return JsonResponse(
            {"success": False, "error": f"{', '.join(missing)}을(를) 입력하세요."},
            status=400,
        )

    if KisAccount.objects.filter(user=user, name=data.name.strip()).exists():
        return JsonResponse(
            {"success": False, "error": f"이미 같은 이름의 계좌가 있습니다: {data.name}"},
            status=400,
        )

    with transaction.atomic():
        account = KisAccount(
            user=user,
            name=data.name.strip(),
            account_type=data.account_type,
            account_no=data.account_no.strip(),
            account_cd=(data.account_cd or "01").strip(),
            is_active=data.is_active,
        )
        account.set_app_key(data.app_key.strip())
        account.set_app_secret(data.app_secret.strip())
        account.save()

        # 첫 계좌는 자동으로 기본 계좌가 된다
        is_first = KisAccount.objects.filter(user=user).count() == 1
        if data.is_default or is_first:
            _set_as_default(user, account)

    account.refresh_from_db()
    return JsonResponse({"success": True, "account": _serialize_account(account)})


@account_settings_router.put("/kis-accounts/{int:account_id}")
def update_kis_account(request, account_id: int, data: KisAccountUpdateIn):
    """계좌 정보를 수정합니다. app_key/app_secret은 값을 보낸 경우에만 교체합니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    account = KisAccount.objects.filter(user=user, pk=account_id).first()
    if account is None:
        return JsonResponse({"success": False, "error": "계좌를 찾을 수 없습니다."}, status=404)

    if data.account_type is not None:
        error = _validate_account_type(data.account_type)
        if error:
            return JsonResponse({"success": False, "error": error}, status=400)
        account.account_type = data.account_type

    if data.name is not None and data.name.strip():
        new_name = data.name.strip()
        duplicated = (
            KisAccount.objects.filter(user=user, name=new_name)
            .exclude(pk=account.pk)
            .exists()
        )
        if duplicated:
            return JsonResponse(
                {"success": False, "error": f"이미 같은 이름의 계좌가 있습니다: {new_name}"},
                status=400,
            )
        account.name = new_name

    if data.account_no is not None and data.account_no.strip():
        account.account_no = data.account_no.strip()
    if data.account_cd is not None and data.account_cd.strip():
        account.account_cd = data.account_cd.strip()
    if data.app_key:
        account.set_app_key(data.app_key.strip())
    if data.app_secret:
        account.set_app_secret(data.app_secret.strip())
    if data.is_active is not None:
        account.is_active = data.is_active

    with transaction.atomic():
        account.save()
        if data.is_default:
            _set_as_default(user, account)

    account.refresh_from_db()
    return JsonResponse({"success": True, "account": _serialize_account(account)})


@account_settings_router.delete("/kis-accounts/{int:account_id}")
def delete_kis_account(request, account_id: int):
    """계좌를 삭제합니다. 활성 자동매매 설정이 사용 중이면 거부합니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    account = KisAccount.objects.filter(user=user, pk=account_id).first()
    if account is None:
        return JsonResponse({"success": False, "error": "계좌를 찾을 수 없습니다."}, status=404)

    # 이 계좌를 실제로 쓰고 있는 활성 전략이 있으면 삭제를 막는다
    account_map = get_account_map(user)
    in_use = [
        label
        for key, label in STRATEGY_CHOICES
        if account_map.get(key)
        and account_map[key].pk == account.pk
        and TradingConfig.objects.filter(
            user=user, strategy_type=key, is_active=True
        ).exists()
    ]
    if in_use:
        return JsonResponse({
            "success": False,
            "error": f"이 계좌를 사용 중인 전략이 있습니다: {', '.join(in_use)}."
                     " 해당 전략의 계좌를 먼저 변경하세요.",
        }, status=400)

    was_default = account.is_default
    with transaction.atomic():
        account.delete()
        # 기본 계좌를 지웠으면 남은 계좌 중 하나를 기본으로 승격한다
        if was_default:
            remaining = KisAccount.objects.filter(user=user).order_by("pk").first()
            if remaining:
                _set_as_default(user, remaining)

    return JsonResponse({"success": True})


# ── 전략별 계좌 지정 ───────────────────────────────────────────────────────


@account_settings_router.post("/strategy-accounts")
def set_strategy_account(request, data: StrategyAccountIn):
    """전략이 사용할 계좌를 지정합니다. account_id가 없으면 기본 계좌를 따릅니다."""
    user = get_authenticated_user(request)
    if not user:
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    if data.strategy_type not in STRATEGY_KEYS:
        return JsonResponse(
            {"success": False, "error": f"알 수 없는 전략입니다: {data.strategy_type}"},
            status=400,
        )

    try:
        account = assign_account(user, data.strategy_type, data.account_id)
    except ValueError as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({
        "success": True,
        "strategy_type": data.strategy_type,
        "account_id": account.pk if account else None,
        "account_name": account.name if account else None,
    })
