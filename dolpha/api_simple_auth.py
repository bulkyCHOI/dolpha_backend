from ninja import Router
from django.http import JsonResponse
import json


# 간단한 인증 API 라우터 (임시 테스트용)
simple_auth_router = Router()


@simple_auth_router.post("/google/")
def google_oauth_callback(request):
    """
    Google OAuth 콜백 처리 API (임시 테스트용)
    """
    try:
        # 요청에서 authorization code와 redirect_uri 추출
        body = json.loads(request.body)
        code = body.get('code')
        redirect_uri = body.get('redirect_uri')
        
        if not code:
            return JsonResponse({'error': 'Authorization code가 필요합니다.'}, status=400)
        
        # 임시로 더미 응답 반환
        return JsonResponse({
            'access_token': 'dummy_access_token_' + code[:10],
            'refresh_token': 'dummy_refresh_token_' + code[:10],
            'user': {
                'id': 1,
                'username': 'test@example.com',
                'email': 'test@example.com',
                'first_name': 'Test User',
                'profile_picture': 'https://example.com/avatar.jpg',
                'created_at': '2024-01-01T00:00:00Z',
            }
        })
        
    except Exception as e:
        print(f"Google OAuth 오류: {str(e)}")
        return JsonResponse({'error': f'인증 처리 중 오류가 발생했습니다: {str(e)}'}, status=500)


@simple_auth_router.post("/refresh/")
def refresh_token(request):
    """
    JWT 토큰 갱신 API (임시 테스트용)
    """
    try:
        body = json.loads(request.body)
        refresh_token = body.get('refresh')
        
        if not refresh_token:
            return JsonResponse({'error': 'Refresh token이 필요합니다.'}, status=400)
        
        # 임시로 더미 응답 반환
        return JsonResponse({
            'access_token': 'new_dummy_access_token',
        })
        
    except Exception as e:
        return JsonResponse({'error': '토큰 갱신에 실패했습니다.'}, status=401)


@simple_auth_router.post("/logout/")
def logout(request):
    """
    로그아웃 API (임시 테스트용)
    """
    return JsonResponse({'message': '로그아웃이 완료되었습니다.'})


@simple_auth_router.get("/me/")
def get_user_profile(request):
    """
    현재 로그인한 사용자 정보 조회 API (임시 테스트용)
    """
    try:
        # Authorization 헤더 확인
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return JsonResponse({'error': '인증 토큰이 필요합니다.'}, status=401)
        
        # 임시로 더미 사용자 정보 반환
        return JsonResponse({
            'user': {
                'id': 1,
                'username': 'test@example.com',
                'email': 'test@example.com',
                'first_name': 'Test User',
                'profile_picture': 'https://example.com/avatar.jpg',
                'created_at': '2024-01-01T00:00:00Z',
            }
        })
            
    except Exception as e:
        return JsonResponse({'error': '사용자 정보 조회에 실패했습니다.'}, status=500)