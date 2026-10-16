"""
API 전역 인증.

NinjaAPI의 기본 auth로 등록되어 모든 엔드포인트가 '승인된 사용자의 유효한 JWT'를
요구하게 만든다. 공개가 필요한 엔드포인트만 개별적으로 auth=None을 지정한다.
"""
from functools import wraps
from typing import Callable, Optional

from django.http import HttpRequest, HttpResponse, JsonResponse
from ninja.security import APIKeyQuery, HttpBearer
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed, InvalidToken, TokenError


def resolve_approved_user(request: HttpRequest, token: str) -> Optional[object]:
    """JWT를 검증해 관리자 승인을 받은 활성 사용자를 반환한다. 실패하면 None.

    승인이 취소된 계정은 access 토큰이 남아 있어도 다음 요청부터 바로 차단된다.
    """
    jwt_auth = JWTAuthentication()
    try:
        validated = jwt_auth.get_validated_token(token)
        user = jwt_auth.get_user(validated)
    except (AuthenticationFailed, InvalidToken, TokenError):
        # 위조·만료 토큰, 삭제·비활성 사용자
        return None

    if not (user and user.is_active and user.has_service_access):
        return None

    # 기존 view들이 request.user를 참조하므로 JWT 사용자로 맞춰 준다
    request.user = user
    return user


class ApprovedJWTAuth(HttpBearer):
    """Authorization: Bearer 헤더의 JWT로 인증한다. 실패 시 Ninja가 401을 응답한다."""

    def authenticate(self, request: HttpRequest, token: str) -> Optional[object]:
        return resolve_approved_user(request, token)


class ApprovedJWTQueryAuth(APIKeyQuery):
    """?access_token= 쿼리의 JWT로 인증한다.

    헤더를 붙일 수 없는 EventSource(SSE) 엔드포인트 전용. URL이 로그에 남을 수 있으므로
    다른 엔드포인트에는 쓰지 않는다.
    """

    param_name = "access_token"

    def authenticate(self, request: HttpRequest, key: Optional[str]) -> Optional[object]:
        if not key:
            return None
        return resolve_approved_user(request, key)


def require_approved_jwt(view: Callable) -> Callable:
    """Ninja 밖의 일반 Django view에 ApprovedJWTAuth와 같은 검사를 적용한다."""
    auth = ApprovedJWTAuth()

    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if auth(request) is None:
            return JsonResponse({"detail": "Unauthorized"}, status=401)
        return view(request, *args, **kwargs)

    return wrapper
