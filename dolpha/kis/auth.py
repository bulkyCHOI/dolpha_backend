"""
KIS API 인증 모듈 — 실계좌 / 모의계좌 듀얼 지원

환경변수:
  실계좌: KIS_REAL_APP_KEY, KIS_REAL_APP_SECRET, KIS_REAL_ACCOUNT_NO
  모의계좌: KIS_VIRTUAL_APP_KEY, KIS_VIRTUAL_APP_SECRET, KIS_VIRTUAL_ACCOUNT_NO
  모드 선택: KIS_MODE = REAL (기본) | VIRTUAL
  토큰 캐시 경로: KIS_TOKEN_PATH (기본: ./kis_token_{mode}.json)

데이터 수집(시세 조회)은 항상 실계좌 API를 사용합니다.
자동매매는 KIS_MODE에 따라 실/모의 계좌를 선택합니다.
"""

import os
import json
import warnings
import requests
from datetime import datetime

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_REAL_URL    = "https://openapi.koreainvestment.com:9443"
_VIRTUAL_URL = "https://openapivts.koreainvestment.com:9443"


# ─────────────────────────────────────────────────────────────
# 환경변수 읽기
# ─────────────────────────────────────────────────────────────

def _cfg(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise EnvironmentError(
            f"KIS 환경변수 '{key}'가 설정되지 않았습니다."
        )
    return val


def _cfg_optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_mode() -> str:
    return _cfg_optional("KIS_MODE", "REAL").upper()


def get_url_base(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    return _REAL_URL if m == "REAL" else _VIRTUAL_URL


def get_app_key(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    return _cfg("KIS_REAL_APP_KEY" if m == "REAL" else "KIS_VIRTUAL_APP_KEY")


def get_app_secret(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    return _cfg("KIS_REAL_APP_SECRET" if m == "REAL" else "KIS_VIRTUAL_APP_SECRET")


def get_account_no(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    return _cfg("KIS_REAL_ACCOUNT_NO" if m == "REAL" else "KIS_VIRTUAL_ACCOUNT_NO")


def get_account_cd() -> str:
    return _cfg_optional("KIS_ACCOUNT_CD", "01")


def _token_path(mode: str) -> str:
    base = _cfg_optional("KIS_TOKEN_PATH", "")
    if base:
        return base
    return f"./kis_token_{mode.lower()}.json"


# ─────────────────────────────────────────────────────────────
# 토큰 관리
# ─────────────────────────────────────────────────────────────

def MakeToken(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    url = f"{get_url_base(m)}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": get_app_key(m),
        "appsecret": get_app_secret(m),
    }

    res = requests.post(url, headers=headers, json=body, timeout=10, verify=False)
    if res.status_code != 200:
        raise RuntimeError(
            f"KIS 토큰 발급 실패 ({m}): {res.status_code} — {res.text}"
        )

    data = res.json()
    token = data["access_token"]
    expires_at = data.get("access_token_token_expired", "")

    cache = {"access_token": token, "expires_at": expires_at}
    path = _token_path(m)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"[KIS] {m} 토큰 발급 완료, 만료: {expires_at}")
    return token


def GetToken(mode: str = None) -> str:
    m = (mode or get_mode()).upper()
    path = _token_path(m)

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
                print(f"[KIS] {m} 토큰 만료, 재발급합니다.")
            except ValueError:
                pass

        if token:
            return token

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    return MakeToken(m)


def GetHeaders(tr_id: str = "", custtype: str = "P", mode: str = None) -> dict:
    """
    KIS API 공통 요청 헤더.
    mode: "REAL" | "VIRTUAL" | None(KIS_MODE 환경변수 따름)
    데이터 수집용으로 항상 실계좌를 쓰려면 mode="REAL" 명시.
    """
    m = (mode or get_mode()).upper()
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {GetToken(m)}",
        "appkey": get_app_key(m),
        "appsecret": get_app_secret(m),
        "tr_id": tr_id,
        "custtype": custtype,
    }
