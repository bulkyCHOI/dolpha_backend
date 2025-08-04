# standard library
import json
import requests as http_requests

# third-party
from ninja import NinjaAPI, Router
from rest_framework_simplejwt.tokens import RefreshToken

# Django
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

# Google OAuth 관련 import를 try-except로 감싸기
try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
    from google_auth_oauthlib.flow import Flow
    GOOGLE_AUTH_AVAILABLE = True
except ImportError as e:
    GOOGLE_AUTH_AVAILABLE = False
except Exception as e:
    GOOGLE_AUTH_AVAILABLE = False

# 사용자 인증 API 라우터
auth_router = Router()
User = get_user_model()

@auth_router.post("/google/")
def google_oauth_callback(request):
    """
    Google OAuth 콜백 처리 API
    프론트엔드에서 받은 authorization code를 처리하여 JWT 토큰 반환
    """
    if not GOOGLE_AUTH_AVAILABLE:
        return JsonResponse({'error': 'Google 인증 라이브러리가 설치되지 않았습니다.'}, status=500)
    
    try:
        # 요청에서 authorization code와 redirect_uri 추출
        body = json.loads(request.body)
        code = body.get('code')
        redirect_uri = body.get('redirect_uri')
        
        if not code:
            return JsonResponse({'error': 'Authorization code가 필요합니다.'}, status=400)
        
        # Google OAuth2 flow 설정
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": settings.GOOGLE_OAUTH2_CLIENT_ID,
                    "client_secret": settings.GOOGLE_OAUTH2_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [redirect_uri]
                }
            },
            scopes=['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile', 'openid']
        )
        flow.redirect_uri = redirect_uri
        
        # Authorization code를 토큰으로 교환
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flow.fetch_token(code=code)
        
        # ID 토큰에서 사용자 정보 추출
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, 
            google_requests.Request(), 
            settings.GOOGLE_OAUTH2_CLIENT_ID
        )
        
        # 사용자 정보 추출
        google_id = id_info.get('sub')
        email = id_info.get('email')
        name = id_info.get('name')
        picture = id_info.get('picture')
        
        if not google_id or not email:
            return JsonResponse({'error': '사용자 정보를 가져올 수 없습니다.'}, status=400)
        
        # 사용자 조회 또는 생성
        user, created = User.objects.get_or_create(
            google_id=google_id,
            defaults={
                'username': email,
                'email': email,
                'first_name': name or '',
                'profile_picture': picture,
            }
        )
        
        # 기존 사용자의 경우 정보 업데이트
        if not created:
            user.profile_picture = picture
            user.save()
        
        # 사용자 프로필 생성 (존재하지 않는 경우)
        from myweb.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=user)
        
        # JWT 토큰 생성
        refresh = RefreshToken.for_user(user)
        access_token = refresh.access_token
        
        return JsonResponse({
            'access_token': str(access_token),
            'refresh_token': str(refresh),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'profile_picture': user.profile_picture,
                'created_at': user.created_at.isoformat() if user.created_at else None,
            }
        })
        
    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'인증 처리 중 오류가 발생했습니다: {str(e)}'}, status=500)


@auth_router.post("/refresh/")
def refresh_token(request):
    """
    JWT 토큰 갱신 API
    """
    try:
        body = json.loads(request.body)
        refresh_token = body.get('refresh')
        
        if not refresh_token:
            return JsonResponse({'error': 'Refresh token이 필요합니다.'}, status=400)
        
        # 토큰 갱신
        refresh = RefreshToken(refresh_token)
        access_token = refresh.access_token
        
        return JsonResponse({
            'access_token': str(access_token),
        })
        
    except Exception as e:
        return JsonResponse({'error': '토큰 갱신에 실패했습니다.'}, status=401)


@auth_router.post("/logout/")
def logout(request):
    """
    로그아웃 API (토큰 블랙리스트 처리)
    """
    try:
        body = json.loads(request.body)
        refresh_token = body.get('refresh')
        
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        return JsonResponse({'message': '로그아웃이 완료되었습니다.'})
        
    except Exception as e:
        return JsonResponse({'error': '로그아웃 처리 중 오류가 발생했습니다.'}, status=400)


@auth_router.get("/me/")
def get_user_profile(request):
    """
    현재 로그인한 사용자 정보 조회 API
    """
    try:
        # Authorization 헤더에서 토큰 추출
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return JsonResponse({'error': '인증 토큰이 필요합니다.'}, status=401)
        
        token = auth_header.split(' ')[1]
        
        # 토큰 검증 및 사용자 조회는 DRF의 JWTAuthentication이 처리
        # 여기서는 단순히 사용자 정보 반환
        if hasattr(request, 'user') and request.user.is_authenticated:
            user = request.user
            return JsonResponse({
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'profile_picture': user.profile_picture,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                }
            })
        else:
            return JsonResponse({'error': '인증되지 않은 사용자입니다.'}, status=401)
            
    except Exception as e:
        return JsonResponse({'error': '사용자 정보 조회에 실패했습니다.'}, status=500)