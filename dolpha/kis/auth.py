"""
KIS API 인증 모듈 — 사용자별 계좌 / 서버 환경변수 계좌 듀얼 지원

인증 정보의 출처는 두 가지입니다.

1) 사용자 계좌 (myweb.models.KisAccount)
   자동매매·잔고조회 등 "누구의 돈인가"가 중요한 호출에 사용합니다.
   dolpha.kis.credentials.credential_for_account()로 KisCredential을 만들어
   각 API 함수의 account 인자로 넘깁니다.

2) 서버 환경변수 계좌 (하위 호환 / 시세 수집 전용)
   실계좌: KIS_REAL_APP_KEY, KIS_REAL_APP_SECRET, KIS_REAL_ACCOUNT_NO
   모의계좌: KIS_VIRTUAL_APP_KEY, KIS_VIRTUAL_APP_SECRET, KIS_VIRTUAL_ACCOUNT_NO
   모드 선택: KIS_MODE = REAL (기본) | VIRTUAL
   OHLCV·분봉·지수 등 사용자와 무관한 시세 수집은 항상 mode="REAL"을 씁니다.

API 함수의 account 인자는 다음을 모두 받습니다.
   KisCredential  → 그 계좌로 호출
   "REAL"/"VIRTUAL" → 해당 환경변수 계좌로 호출
   None            → KIS_MODE 환경변수 계좌로 호출
"""

import hashlib
import os
import json
import warnings
import requests
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_REAL_URL    = "https://openapi.koreainvestment.com:9443"
_VIRTUAL_URL = "https://openapivts.koreainvestment.com:29443"


# ─────────────────────────────────────────────────────────────
# 자격증명
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KisCredential:
    """KIS API 호출 1건에 필요한 계좌 자격증명 일체."""

    app_key: str
    app_secret: str
    account_no: str
    account_cd: str
    is_virtual: bool
    label: str = ""          # 로그 표기용 (계좌 별칭 또는 "REAL"/"VIRTUAL")

    @property
    def url_base(self) -> str:
        return _VIRTUAL_URL if self.is_virtual else _REAL_URL

    @property
    def token_cache_key(self) -> str:
        """앱키별 토큰 캐시 식별자. 계좌가 달라도 서로 덮어쓰지 않게 한다."""
        digest = hashlib.sha256(self.app_key.encode("utf-8")).hexdigest()[:12]
        return f"{'virtual' if self.is_virtual else 'real'}_{digest}"


class KisCredentialError(Exception):
    """계좌 자격증명이 없거나 불완전할 때 발생."""


# ─────────────────────────────────────────────────────────────
# 환경변수 읽기
# ─────────────────────────────────────────────────────────────

def _cfg(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise KisCredentialError(f"KIS 환경변수 '{key}'가 설정되지 않았습니다.")
    return val


def _cfg_optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_mode() -> str:
    """서버 환경변수 기본 모드."""
    return _cfg_optional("KIS_MODE", "REAL").upper()


def env_credential(mode: str = None) -> KisCredential:
    """서버 환경변수에 설정된 계좌의 자격증명을 만든다."""
    m = (mode or get_mode()).upper()
    prefix = "KIS_REAL" if m == "REAL" else "KIS_VIRTUAL"
    return KisCredential(
        app_key=_cfg(f"{prefix}_APP_KEY"),
        app_secret=_cfg(f"{prefix}_APP_SECRET"),
        account_no=_cfg(f"{prefix}_ACCOUNT_NO"),
        account_cd=_cfg_optional("KIS_ACCOUNT_CD", "01"),
        is_virtual=(m == "VIRTUAL"),
        label=m,
    )


def resolve_credential(account=None) -> KisCredential:
    """account 인자를 KisCredential로 정규화한다.

    Args:
        account: KisCredential | "REAL" | "VIRTUAL" | None
    """
    if isinstance(account, KisCredential):
        return account
    if account is None:
        return env_credential()
    if isinstance(account, str):
        return env_credential(account)
    raise TypeError(f"지원하지 않는 account 타입입니다: {type(account)!r}")


# ─────────────────────────────────────────────────────────────
# 개별 접근자 (하위 호환)
# ─────────────────────────────────────────────────────────────

def get_url_base(account=None) -> str:
    return resolve_credential(account).url_base


def get_app_key(account=None) -> str:
    return resolve_credential(account).app_key


def get_app_secret(account=None) -> str:
    return resolve_credential(account).app_secret


def get_account_no(account=None) -> str:
    return resolve_credential(account).account_no


def get_account_cd(account=None) -> str:
    return resolve_credential(account).account_cd


def _token_path(cred: KisCredential) -> str:
    base = _cfg_optional("KIS_TOKEN_PATH", "")
    if base:
        return base
    return f"./kis_token_{cred.token_cache_key}.json"


# ─────────────────────────────────────────────────────────────
# 토큰 관리
# ─────────────────────────────────────────────────────────────

def MakeToken(account=None) -> str:
    cred = resolve_credential(account)
    url = f"{cred.url_base}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": cred.app_key,
        "appsecret": cred.app_secret,
    }

    res = requests.post(url, headers=headers, json=body, timeout=10, verify=False)
    if res.status_code != 200:
        raise RuntimeError(
            f"KIS 토큰 발급 실패 ({cred.label}): {res.status_code} — {res.text}"
        )

    data = res.json()
    token = data["access_token"]
    expires_at = data.get("access_token_token_expired", "")

    cache = {"access_token": token, "expires_at": expires_at}
    path = _token_path(cred)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"[KIS] {cred.label} 토큰 발급 완료, 만료: {expires_at}")
    return token


def GetToken(account=None) -> str:
    cred = resolve_credential(account)
    path = _token_path(cred)

    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        token = cache.get("access_token", "")
        expires_at_str = cache.get("expires_at", "")

        if token and expires_at_str:
            try:
                expires_dt = datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S")
                if datetime.now() < expires_dt:
                    return token
                print(f"[KIS] {cred.label} 토큰 만료, 재발급합니다.")
                return MakeToken(cred)
            except ValueError:
                pass

        if token:
            return token

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    return MakeToken(cred)


def GetHeaders(tr_id: str = "", custtype: str = "P", account=None, mode: str = None) -> dict:
    """
    KIS API 공통 요청 헤더.

    Args:
        account: KisCredential | "REAL" | "VIRTUAL" | None
        mode: account의 별칭 (기존 호출부 호환 — 시세 수집은 mode="REAL")
    """
    cred = resolve_credential(account if account is not None else mode)
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {GetToken(cred)}",
        "appkey": cred.app_key,
        "appsecret": cred.app_secret,
        "tr_id": tr_id,
        "custtype": custtype,
    }
