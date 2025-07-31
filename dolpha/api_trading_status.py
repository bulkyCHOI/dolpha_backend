"""
거래 상태 조회 API 엔드포인트 (Django Ninja)
autobot 서버와 연동하여 실제 거래 진행 상황을 제공
"""

import requests  
from ninja import Router
from django.http import JsonResponse
from myweb.models import UserProfile

# api_mypage_ninja.py에서 가져온 인증 함수
from .api_mypage_ninja import get_authenticated_user

# 라우터 생성
trading_status_router = Router()


@trading_status_router.get("/trading-status")
def get_trading_status(request):
    """
    현재 거래 상태 및 피라미딩 정보를 조회합니다.
    autobot 서버의 trade_history.json과 trading_configs를 연동하여 실제 거래 진행 상황을 제공합니다.
    """
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({"error": "인증이 필요합니다."}, status=401)
        
        # 사용자 프로필에서 autobot 서버 정보 가져오기
        try:
            profile = UserProfile.objects.get(user=user)
        except UserProfile.DoesNotExist:
            return JsonResponse({
                "success": False,
                "error": "PROFILE_NOT_FOUND",
                "message": "사용자 프로필을 찾을 수 없습니다."
            }, status=404)
        
        if not profile.autobot_server_ip:
            return JsonResponse({
                "success": False,
                "error": "SERVER_NOT_CONFIGURED",
                "message": "autobot 서버 설정이 필요합니다."
            }, status=400)
        
        try:
            # autobot 서버에서 거래 상태 조회
            response = requests.get(
                f'http://{profile.autobot_server_ip}:{profile.autobot_server_port}/api/trading-status',
                timeout=10
            )
            
            if response.status_code == 200:
                autobot_data = response.json()
                
                # 성공적으로 데이터를 받아온 경우
                return JsonResponse({
                    "success": True,
                    "data": autobot_data.get("data", {}),
                    "metadata": {
                        "timestamp": autobot_data.get("timestamp"),
                        "total_configs": autobot_data.get("total_configs", 0),
                        "active_positions": autobot_data.get("active_positions", 0),
                        "server_ip": profile.autobot_server_ip,
                        "server_port": profile.autobot_server_port
                    }
                })
            else:
                return JsonResponse({
                    "success": False,
                    "error": "AUTOBOT_SERVER_ERROR",
                    "message": f"autobot 서버 응답 오류: {response.status_code}"
                }, status=502)
                
        except requests.exceptions.ConnectionError:
            return JsonResponse({
                "success": False,
                "error": "CONNECTION_ERROR",
                "message": f"autobot 서버({profile.autobot_server_ip}:{profile.autobot_server_port})에 연결할 수 없습니다."
            }, status=502)
        except requests.exceptions.Timeout:
            return JsonResponse({
                "success": False,
                "error": "TIMEOUT_ERROR",
                "message": "autobot 서버 응답 시간이 초과되었습니다."
            }, status=504)
        except requests.exceptions.RequestException as e:
            return JsonResponse({
                "success": False,
                "error": "REQUEST_ERROR",
                "message": f"autobot 서버 요청 오류: {str(e)}"
            }, status=502)
            
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }, status=500)