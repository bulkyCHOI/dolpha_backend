"""API 전역 인증(ApprovedJWTAuth) 테스트.

- 토큰 없음 / 위조 토큰 / 미승인 / 비활성 계정은 401
- 승인된 계정은 통과, 슈퍼유저는 승인 없이 통과
- 공개 엔드포인트(/hello, /auth/refresh/)는 토큰 없이 접근 가능
- Ninja 밖의 view는 require_approved_jwt 데코레이터로 같은 규칙을 따른다
"""

from django.http import JsonResponse
from django.test import RequestFactory, TestCase
from rest_framework_simplejwt.tokens import RefreshToken

from dolpha.auth import require_approved_jwt
from myweb.models import User

PROTECTED_URL = "/api/auth/me/"


def bearer(user: User) -> dict:
    return {"HTTP_AUTHORIZATION": f"Bearer {RefreshToken.for_user(user).access_token}"}


class ApprovedJWTAuthTest(TestCase):
    def setUp(self):
        self.approved = User.objects.create_user(
            username="ok@test.com", email="ok@test.com", is_approved=True
        )
        self.pending = User.objects.create_user(
            username="wait@test.com", email="wait@test.com", is_approved=False
        )

    def test_no_token_is_401(self):
        self.assertEqual(self.client.get(PROTECTED_URL).status_code, 401)

    def test_invalid_token_is_401(self):
        res = self.client.get(PROTECTED_URL, HTTP_AUTHORIZATION="Bearer not-a-jwt")
        self.assertEqual(res.status_code, 401)

    def test_pending_user_is_401(self):
        self.assertEqual(self.client.get(PROTECTED_URL, **bearer(self.pending)).status_code, 401)

    def test_inactive_user_is_401(self):
        headers = bearer(self.approved)
        User.objects.filter(pk=self.approved.pk).update(is_active=False)
        self.assertEqual(self.client.get(PROTECTED_URL, **headers).status_code, 401)

    def test_revoked_approval_blocks_existing_token(self):
        headers = bearer(self.approved)
        User.objects.filter(pk=self.approved.pk).update(is_approved=False)
        self.assertEqual(self.client.get(PROTECTED_URL, **headers).status_code, 401)

    def test_approved_user_passes_and_sees_self(self):
        res = self.client.get(PROTECTED_URL, **bearer(self.approved))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["email"], "ok@test.com")

    def test_superuser_passes_without_approval(self):
        admin = User.objects.create_superuser(username="admin@test.com", email="admin@test.com")
        self.assertEqual(self.client.get(PROTECTED_URL, **bearer(admin)).status_code, 200)

    def test_data_router_endpoint_is_protected(self):
        self.assertEqual(self.client.get("/api/trading-summary").status_code, 401)


class PublicEndpointTest(TestCase):
    def test_hello_is_public(self):
        self.assertEqual(self.client.get("/api/hello").status_code, 200)

    def test_refresh_is_reachable_without_access_token(self):
        res = self.client.post("/api/auth/refresh/", data={}, content_type="application/json")
        # 인증 단계에서 막히지 않고 핸들러의 입력 검증(400)까지 도달해야 한다
        self.assertEqual(res.status_code, 400)


class RequireApprovedJwtDecoratorTest(TestCase):
    """market-indices처럼 Ninja 밖에 있는 view를 감싸는 데코레이터."""

    def setUp(self):
        self.view = require_approved_jwt(lambda request: JsonResponse({"ok": True}))
        self.factory = RequestFactory()

    def test_no_token_is_401(self):
        self.assertEqual(self.view(self.factory.get("/")).status_code, 401)

    def test_approved_user_reaches_view(self):
        user = User.objects.create_user(username="m@test.com", email="m@test.com", is_approved=True)
        self.assertEqual(self.view(self.factory.get("/", **bearer(user))).status_code, 200)

    def test_market_indices_url_is_protected(self):
        self.assertEqual(self.client.get("/api/market-indices/").status_code, 401)


class MinuteStreamQueryTokenTest(TestCase):
    """EventSource용 SSE 엔드포인트는 ?access_token= 으로도 인증할 수 있다."""

    URL = "/api/find_stock_minute_stream?code=005930"

    def test_no_token_is_401(self):
        self.assertEqual(self.client.get(self.URL).status_code, 401)

    def test_pending_user_query_token_is_401(self):
        user = User.objects.create_user(username="q@test.com", email="q@test.com", is_approved=False)
        token = RefreshToken.for_user(user).access_token
        self.assertEqual(self.client.get(f"{self.URL}&access_token={token}").status_code, 401)

    def test_query_token_not_accepted_elsewhere(self):
        user = User.objects.create_user(username="r@test.com", email="r@test.com", is_approved=True)
        token = RefreshToken.for_user(user).access_token
        self.assertEqual(self.client.get(f"{PROTECTED_URL}?access_token={token}").status_code, 401)
