"""민감 정보(KIS 앱키·시크릿) 대칭 암호화 유틸.

DB에 평문으로 저장하면 DB 덤프만으로 계좌를 탈취당하므로 Fernet으로 암호화한다.
암호화 키는 KIS_ENCRYPTION_KEY 환경변수를 우선 사용하고, 없으면 Django
SECRET_KEY에서 파생한다(SECRET_KEY가 바뀌면 기존 암호문을 복호화할 수 없다).
"""

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings

_ENCRYPTED_PREFIX = "enc:"


class DecryptionError(Exception):
    """암호문을 복호화할 수 없을 때 발생. 키 교체 또는 데이터 손상."""


def _fernet() -> Fernet:
    """암호화 키를 로드한다.

    KIS_ENCRYPTION_KEY는 `Fernet.generate_key()` 결과(urlsafe base64 32바이트).
    운영 환경에서는 이 값을 반드시 별도로 지정해 SECRET_KEY 교체와 분리한다.
    """
    raw_key = os.environ.get("KIS_ENCRYPTION_KEY", "")
    if raw_key:
        return Fernet(raw_key.encode("utf-8"))

    derived = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt(plaintext: str) -> str:
    """평문을 'enc:' 접두사가 붙은 암호문으로 변환한다. 빈 값은 그대로 반환."""
    if not plaintext:
        return ""
    token = _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")
    return f"{_ENCRYPTED_PREFIX}{token}"


def decrypt(ciphertext: str) -> str:
    """암호문을 평문으로 되돌린다.

    'enc:' 접두사가 없으면 평문으로 간주해 그대로 반환한다
    (암호화 도입 이전에 저장된 값과의 호환).
    """
    if not ciphertext:
        return ""
    if not ciphertext.startswith(_ENCRYPTED_PREFIX):
        return ciphertext

    token = ciphertext[len(_ENCRYPTED_PREFIX):]
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise DecryptionError(
            "KIS 자격증명을 복호화할 수 없습니다. "
            "KIS_ENCRYPTION_KEY 또는 DJANGO_SECRET_KEY가 변경되었는지 확인하세요."
        ) from e


def mask(value: str, visible_tail: int = 4) -> str:
    """민감 문자열을 마스킹해 화면 표시용으로 변환한다."""
    if not value:
        return ""
    if len(value) <= visible_tail:
        return "*" * len(value)
    return "*" * (len(value) - visible_tail) + value[-visible_tail:]


def mask_account_no(account_no: str) -> str:
    """계좌번호 마스킹: 앞 2자리 + 뒤 2자리만 노출."""
    if not account_no or len(account_no) < 4:
        return account_no or ""
    return account_no[:2] + "*" * (len(account_no) - 4) + account_no[-2:]
