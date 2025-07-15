"""
[DEPRECATED] 마이페이지 관련 API 엔드포인트 (Django 클래스 기반 뷰)

⚠️ 주의: 이 파일은 더 이상 사용되지 않습니다.
새로운 API는 api_mypage_ninja.py를 사용합니다 (Django Ninja 기반).

- 사용자 프로필 관리
- autobot 서버 설정 관리  
- 자동매매 설정 관리

이 파일은 참고용으로만 보관되며, 실제 API 요청은 처리하지 않습니다.
"""

import requests
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from django.utils import timezone
from django.db import transaction
import json

from myweb.models import User, UserProfile, TradingConfig


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class UserProfileView(View):
    """사용자 프로필 관리 API"""
    
    def get(self, request):
        """사용자 프로필 정보 조회"""
        try:
            # UserProfile이 없는 경우 자동 생성
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            
            data = {
                'user': {
                    'id': request.user.id,
                    'username': request.user.username,
                    'email': request.user.email,
                    'first_name': request.user.first_name,
                    'last_name': request.user.last_name,
                    'profile_picture': request.user.profile_picture,
                    'date_joined': request.user.date_joined.isoformat() if request.user.date_joined else None,
                },
                'profile': {
                    'autobot_server_ip': profile.autobot_server_ip,
                    'autobot_server_port': profile.autobot_server_port,
                    'server_status': profile.server_status,
                    'last_connection': profile.last_connection.isoformat() if profile.last_connection else None,
                    'created_at': profile.created_at.isoformat() if profile.created_at else None,
                    'updated_at': profile.updated_at.isoformat() if profile.updated_at else None,
                }
            }
            
            return JsonResponse({
                'success': True,
                'data': data
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def put(self, request):
        """사용자 프로필 정보 업데이트"""
        try:
            data = json.loads(request.body)
            
            # 사용자 기본 정보 업데이트
            user_data = data.get('user', {})
            if 'first_name' in user_data:
                request.user.first_name = user_data['first_name']
            if 'last_name' in user_data:
                request.user.last_name = user_data['last_name']
            request.user.save()
            
            # 프로필 정보 업데이트
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            profile_data = data.get('profile', {})
            
            if 'autobot_server_ip' in profile_data:
                profile.autobot_server_ip = profile_data['autobot_server_ip']
            if 'autobot_server_port' in profile_data:
                profile.autobot_server_port = profile_data['autobot_server_port']
            
            profile.save()
            
            return JsonResponse({
                'success': True,
                'message': '프로필이 업데이트되었습니다.'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class ServerSettingsView(View):
    """autobot 서버 설정 관리 API"""
    
    def get(self, request):
        """서버 설정 조회"""
        try:
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            
            data = {
                'autobot_server_ip': profile.autobot_server_ip,
                'autobot_server_port': profile.autobot_server_port,
                'server_status': profile.server_status,
                'last_connection': profile.last_connection.isoformat() if profile.last_connection else None,
            }
            
            return JsonResponse({
                'success': True,
                'data': data
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def post(self, request):
        """서버 설정 저장/업데이트"""
        try:
            data = json.loads(request.body)
            
            ip = data.get('autobot_server_ip')
            port = data.get('autobot_server_port', 8080)
            
            if not ip:
                return JsonResponse({
                    'success': False,
                    'error': 'IP 주소는 필수입니다.'
                }, status=400)
            
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            profile.autobot_server_ip = ip
            profile.autobot_server_port = port
            profile.save()
            
            return JsonResponse({
                'success': True,
                'message': '서버 설정이 저장되었습니다.'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class ServerConnectionTestView(View):
    """autobot 서버 연결 테스트 API"""
    
    def post(self, request):
        """서버 연결 테스트"""
        try:
            data = json.loads(request.body)
            ip = data.get('ip')
            port = data.get('port', 8080)
            
            if not ip:
                return JsonResponse({
                    'success': False,
                    'error': 'IP 주소는 필수입니다.'
                }, status=400)
            
            # autobot 서버 헬스 체크
            try:
                response = requests.get(
                    f'http://{ip}:{port}/health',
                    timeout=5
                )
                
                if response.status_code == 200:
                    # 연결 성공 시 프로필 업데이트
                    profile, created = UserProfile.objects.get_or_create(user=request.user)
                    profile.server_status = 'online'
                    profile.last_connection = timezone.now()
                    profile.save()
                    
                    return JsonResponse({
                        'success': True,
                        'message': '서버 연결 성공',
                        'status': 'connected'
                    })
                else:
                    # 연결 실패
                    profile, created = UserProfile.objects.get_or_create(user=request.user)
                    profile.server_status = 'error'
                    profile.save()
                    
                    return JsonResponse({
                        'success': False,
                        'error': f'서버 응답 오류: {response.status_code}',
                        'status': 'failed'
                    })
                    
            except requests.exceptions.RequestException as e:
                # 네트워크 오류
                profile, created = UserProfile.objects.get_or_create(user=request.user)
                profile.server_status = 'offline'
                profile.save()
                
                return JsonResponse({
                    'success': False,
                    'error': f'서버 연결 실패: {str(e)}',
                    'status': 'failed'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class TradingConfigView(View):
    """자동매매 설정 관리 API"""
    
    def get(self, request):
        """사용자의 자동매매 설정 목록 조회"""
        try:
            configs = TradingConfig.objects.filter(user=request.user)
            
            data = []
            for config in configs:
                data.append({
                    'id': config.id,
                    'stock_code': config.stock_code,
                    'stock_name': config.stock_name,
                    'trading_mode': config.trading_mode,
                    'max_loss': config.max_loss,
                    'stop_loss': config.stop_loss,
                    'take_profit': config.take_profit,
                    'pyramiding_count': config.pyramiding_count,
                    'position_size': config.position_size,
                    'is_active': config.is_active,
                    'autobot_config_id': config.autobot_config_id,
                    'created_at': config.created_at.isoformat(),
                    'updated_at': config.updated_at.isoformat(),
                })
            
            return JsonResponse({
                'success': True,
                'data': data
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def post(self, request):
        """새로운 자동매매 설정 생성 (Presentation 페이지에서 호출)"""
        try:
            data = json.loads(request.body)
            
            # 필수 필드 검증
            required_fields = ['stock_code', 'stock_name', 'trading_mode']
            for field in required_fields:
                if field not in data:
                    return JsonResponse({
                        'success': False,
                        'error': f'{field}는 필수 항목입니다.'
                    }, status=400)
            
            # 기존 활성 설정 비활성화
            TradingConfig.objects.filter(
                user=request.user,
                stock_code=data['stock_code'],
                is_active=True
            ).update(is_active=False)
            
            with transaction.atomic():
                # Django DB에 저장
                trading_config = TradingConfig.objects.create(
                    user=request.user,
                    stock_code=data['stock_code'],
                    stock_name=data['stock_name'],
                    trading_mode=data['trading_mode'],
                    max_loss=data.get('max_loss'),
                    stop_loss=data.get('stop_loss'),
                    take_profit=data.get('take_profit'),
                    pyramiding_count=data.get('pyramiding_count', 0),
                    position_size=data.get('position_size'),
                    is_active=data.get('is_active', True),
                )
                
                # autobot 서버로 설정 전달
                autobot_config_id = self._send_to_autobot_server(request.user, data)
                if autobot_config_id:
                    trading_config.autobot_config_id = autobot_config_id
                    trading_config.save()
                
                return JsonResponse({
                    'success': True,
                    'message': '자동매매 설정이 저장되었습니다.',
                    'data': {
                        'id': trading_config.id,
                        'autobot_config_id': autobot_config_id,
                    }
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _send_to_autobot_server(self, user, config_data):
        """autobot 서버로 설정 전달"""
        try:
            profile = UserProfile.objects.get(user=user)
            
            if not profile.autobot_server_ip:
                print(f"사용자 {user.username}의 autobot 서버 IP가 설정되지 않음")
                return None
            
            # autobot 서버 API 호출
            autobot_data = {
                'stock_code': config_data['stock_code'],
                'stock_name': config_data['stock_name'],
                'trading_mode': config_data['trading_mode'],
                'max_loss': config_data.get('max_loss'),
                'stop_loss': config_data.get('stop_loss'),
                'take_profit': config_data.get('take_profit'),
                'pyramiding_count': config_data.get('pyramiding_count', 0),
                'position_size': config_data.get('position_size'),
                'user_id': user.google_id or str(user.id),  # Google ID 우선, 없으면 User ID
                'is_active': config_data.get('is_active', True),
            }
            
            response = requests.post(
                f'http://{profile.autobot_server_ip}:{profile.autobot_server_port}/trading-configs',
                json=autobot_data,
                timeout=10
            )
            
            if response.status_code == 200:
                autobot_response = response.json()
                return autobot_response.get('id')
            else:
                print(f"autobot 서버 오류: {response.status_code} - {response.text}")
                return None
                
        except UserProfile.DoesNotExist:
            print(f"사용자 {user.username}의 프로필이 없음")
            return None
        except requests.exceptions.RequestException as e:
            print(f"autobot 서버 연결 실패: {str(e)}")
            return None
        except Exception as e:
            print(f"autobot 서버 전달 중 오류: {str(e)}")
            return None


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class TradingConfigDetailView(View):
    """개별 자동매매 설정 관리 API"""
    
    def put(self, request, config_id):
        """자동매매 설정 수정"""
        try:
            data = json.loads(request.body)
            
            config = TradingConfig.objects.get(id=config_id, user=request.user)
            
            # 업데이트 가능한 필드들
            updatable_fields = [
                'trading_mode', 'max_loss', 'stop_loss', 'take_profit',
                'pyramiding_count', 'position_size', 'is_active'
            ]
            
            for field in updatable_fields:
                if field in data:
                    setattr(config, field, data[field])
            
            config.save()
            
            return JsonResponse({
                'success': True,
                'message': '설정이 수정되었습니다.'
            })
            
        except TradingConfig.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': '설정을 찾을 수 없습니다.'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def delete(self, request, config_id):
        """자동매매 설정 삭제"""
        try:
            config = TradingConfig.objects.get(id=config_id, user=request.user)
            config.delete()
            
            return JsonResponse({
                'success': True,
                'message': '설정이 삭제되었습니다.'
            })
            
        except TradingConfig.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': '설정을 찾을 수 없습니다.'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)