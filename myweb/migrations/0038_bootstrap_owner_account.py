"""소유자 계정 승인 + 서버 환경변수 KIS 계좌를 사용자 계좌로 이전.

기존에는 서버 환경변수 하나로 전 사용자가 같은 계좌를 공유했다. 계좌가
사용자별 DB 레코드로 바뀌면서, 지금까지 그 계좌를 실제로 쓰던 소유자에게
같은 자격증명을 그대로 옮겨 자동매매가 끊기지 않게 한다.
"""

import os

from django.db import migrations
from django.utils import timezone

# 서비스 소유자 — 승인 절차 없이 바로 사용 가능해야 하는 계정
OWNER_EMAIL = "ppaccomy@gmail.com"

_ENV_ACCOUNTS = (
    ("REAL", "실계좌", "KIS_REAL_APP_KEY", "KIS_REAL_APP_SECRET", "KIS_REAL_ACCOUNT_NO"),
    ("VIRTUAL", "가상계좌", "KIS_VIRTUAL_APP_KEY", "KIS_VIRTUAL_APP_SECRET", "KIS_VIRTUAL_ACCOUNT_NO"),
)


def bootstrap(apps, schema_editor):
    from dolpha.crypto import encrypt

    User = apps.get_model("myweb", "User")
    KisAccount = apps.get_model("myweb", "KisAccount")

    # ── 1) 소유자 및 슈퍼유저 승인 ────────────────────────────
    now = timezone.now()
    User.objects.filter(is_superuser=True, is_approved=False).update(
        is_approved=True, approved_at=now
    )
    User.objects.filter(email__iexact=OWNER_EMAIL, is_approved=False).update(
        is_approved=True, approved_at=now
    )

    owner = User.objects.filter(email__iexact=OWNER_EMAIL).first()
    if owner is None:
        # 아직 로그인한 적 없는 환경(신규 배포/테스트 DB) — 이전할 대상이 없다
        return

    # ── 2) 환경변수 계좌를 소유자 계좌로 이전 ─────────────────
    default_mode = os.environ.get("KIS_MODE", "REAL").upper()

    for account_type, label, key_var, secret_var, account_var in _ENV_ACCOUNTS:
        app_key = os.environ.get(key_var, "")
        app_secret = os.environ.get(secret_var, "")
        account_no = os.environ.get(account_var, "")
        if not (app_key and app_secret and account_no):
            continue
        if KisAccount.objects.filter(user=owner, name=label).exists():
            continue

        KisAccount.objects.create(
            user=owner,
            name=label,
            account_type=account_type,
            account_no=account_no,
            account_cd=os.environ.get("KIS_ACCOUNT_CD", "01"),
            encrypted_app_key=encrypt(app_key),
            encrypted_app_secret=encrypt(app_secret),
            is_default=(account_type == default_mode),
            is_active=True,
        )

    # 기본 계좌가 하나도 없으면 첫 계좌를 기본으로 지정한다
    accounts = KisAccount.objects.filter(user=owner)
    if accounts.exists() and not accounts.filter(is_default=True).exists():
        first = accounts.order_by("pk").first()
        first.is_default = True
        first.save(update_fields=["is_default"])


def unbootstrap(apps, schema_editor):
    """되돌리기: 이전으로 만들어진 계좌만 제거한다. 승인 상태는 건드리지 않는다."""
    User = apps.get_model("myweb", "User")
    KisAccount = apps.get_model("myweb", "KisAccount")

    owner = User.objects.filter(email__iexact=OWNER_EMAIL).first()
    if owner is None:
        return
    KisAccount.objects.filter(user=owner, name__in=["실계좌", "가상계좌"]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("myweb", "0037_user_approved_at_user_is_approved_kisaccount_and_more"),
    ]

    operations = [
        migrations.RunPython(bootstrap, unbootstrap),
    ]
