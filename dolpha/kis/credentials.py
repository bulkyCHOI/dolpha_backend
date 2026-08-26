"""사용자 KisAccount → KisCredential 변환.

kis/auth.py는 Django 모델을 알지 못한다(시세 수집 등 모델 없이 쓰이는 경로가 있다).
모델과 자격증명 사이의 연결은 이 모듈이 담당한다.
"""

from .auth import KisCredential, KisCredentialError


def credential_for_account(account) -> KisCredential:
    """KisAccount 인스턴스를 KIS 호출용 자격증명으로 변환한다.

    Args:
        account: myweb.models.KisAccount
    Raises:
        KisCredentialError: 앱키/시크릿/계좌번호가 비어 있는 경우
    """
    if account is None:
        raise KisCredentialError("계좌가 지정되지 않았습니다.")

    app_key = account.decrypted_app_key
    app_secret = account.decrypted_app_secret

    missing = [
        label
        for label, value in (
            ("앱키", app_key),
            ("앱시크릿", app_secret),
            ("계좌번호", account.account_no),
        )
        if not value
    ]
    if missing:
        raise KisCredentialError(
            f"계좌 '{account.name}'의 {', '.join(missing)}이(가) 비어 있습니다."
        )

    return KisCredential(
        app_key=app_key,
        app_secret=app_secret,
        account_no=account.account_no,
        account_cd=account.account_cd or "01",
        is_virtual=account.is_virtual,
        label=account.name,
    )
