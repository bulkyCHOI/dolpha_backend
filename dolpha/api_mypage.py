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
    """서버 설정 API (autobot 통합 후 — 하위 호환 stub)"""

    def get(self, request):
        """서버 설정 조회 (autobot 통합 완료, 별도 서버 정보 없음)"""
        return JsonResponse({
            'success': True,
            'data': {},
            'message': 'autobot이 백엔드에 통합되어 별도 서버 설정이 필요하지 않습니다.'
        })

    def post(self, request):
        """서버 설정 저장 (더 이상 사용하지 않음)"""
        return JsonResponse({
            'success': True,
            'message': 'autobot이 백엔드에 통합되어 별도 서버 설정이 필요하지 않습니다.'
        })


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class ServerConnectionTestView(View):
    """서버 연결 테스트 API (autobot 통합 완료 — stub)"""

    def post(self, request):
        return JsonResponse({
            'success': True,
            'message': 'autobot이 백엔드에 통합되어 별도 서버 연결 테스트가 필요하지 않습니다.',
            'status': 'integrated'
        })


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
                    'entry_point': config.entry_point,
                    'is_active': config.is_active,
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
                trading_config = TradingConfig.objects.create(
                    user=request.user,
                    stock_code=data['stock_code'],
                    stock_name=data['stock_name'],
                    trading_mode=data['trading_mode'],
                    max_loss=data.get('max_loss'),
                    stop_loss=data.get('stop_loss'),
                    take_profit=data.get('take_profit'),
                    pyramiding_count=data.get('pyramiding_count', 0),
                    entry_point=data.get('entry_point'),
                    is_active=data.get('is_active', True),
                )

                return JsonResponse({
                    'success': True,
                    'message': '자동매매 설정이 저장되었습니다.',
                    'data': {'id': trading_config.id}
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    


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
                'pyramiding_count', 'entry_point', 'is_active'
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