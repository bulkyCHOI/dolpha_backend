"""
마이페이지 관련 API 엔드포인트 (Django Ninja)
- 사용자 프로필 관리
- autobot 서버 설정 관리
- 자동매매 설정 관리
"""

import requests
from datetime import datetime
from ninja import Router, Schema
from ninja.security import django_auth
from django.http import JsonResponse
from django.utils import timezone
from django.db import transaction
from typing import Optional, List

from myweb.models import User, UserProfile, TradingConfig


def get_or_create_test_user():
    """테스트용 사용자를 가져오거나 생성합니다."""
    try:
        user = User.objects.get(username='test_user')
        return user
    except User.DoesNotExist:
        # 테스트 사용자 생성
        user = User.objects.create_user(
            username='test_user',
            email='test@example.com',
            first_name='테스트',
            last_name='사용자',
            google_id='test_google_001'
        )
        print(f"테스트 사용자 생성됨: {user.username}")
        return user

mypage_router = Router()

# Pydantic 스키마 정의
class UserProfileSchema(Schema):
    id: int
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_picture: Optional[str] = None
    date_joined: Optional[str] = None

class ProfileSchema(Schema):
    autobot_server_ip: Optional[str] = None
    autobot_server_port: int = 8080
    server_status: str = 'offline'
    last_connection: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class UserProfileResponseSchema(Schema):
    user: UserProfileSchema
    profile: ProfileSchema

class ServerSettingsSchema(Schema):
    autobot_server_ip: str
    autobot_server_port: int = 8080

class ServerConnectionTestSchema(Schema):
    ip: str
    port: int = 8080

class TradingConfigSchema(Schema):
    stock_code: str
    stock_name: str
    trading_mode: str  # 'manual' or 'turtle'
    max_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pyramiding_count: int = 0
    position_size: Optional[float] = None
    pyramiding_entries: List[str] = []  # 피라미딩 진입시점 배열
    positions: List[float] = []         # 포지션 배열
    is_active: bool = True

class TradingConfigResponseSchema(Schema):
    id: int
    stock_code: str
    stock_name: str
    trading_mode: str
    max_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pyramiding_count: int = 0
    position_size: Optional[float] = None
    pyramiding_entries: List[str] = []  # 피라미딩 진입시점 배열
    positions: List[float] = []  # 포지션 배열
    is_active: bool = True
    autobot_config_id: Optional[int] = None
    created_at: str
    updated_at: str

class ResponseSchema(Schema):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


@mypage_router.get("/profile", response=UserProfileResponseSchema, auth=django_auth)
def get_user_profile(request):
    """사용자 프로필 정보 조회"""
    try:
        # UserProfile이 없는 경우 자동 생성
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        return {
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
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.get("/server-settings", response=ProfileSchema)
def get_server_settings(request):
    """서버 설정 조회 - 실제 DB 연동"""
    try:
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        # UserProfile 가져오기 또는 생성
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        if created:
            print(f"새로운 프로필 생성됨: {user.username}")
        
        return {
            'autobot_server_ip': profile.autobot_server_ip,
            'autobot_server_port': profile.autobot_server_port,
            'server_status': profile.server_status,
            'last_connection': profile.last_connection.isoformat() if profile.last_connection else None,
            'created_at': profile.created_at.isoformat() if profile.created_at else None,
            'updated_at': profile.updated_at.isoformat() if profile.updated_at else None,
        }
    except Exception as e:
        print(f"서버 설정 조회 오류: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.post("/server-settings", response=ResponseSchema)
def save_server_settings(request, data: ServerSettingsSchema):
    """서버 설정 저장/업데이트 - 실제 DB 연동"""
    try:
        if not data.autobot_server_ip:
            return {
                'success': False,
                'error': 'IP 주소는 필수입니다.'
            }
        
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        # UserProfile 가져오기 또는 생성하고 저장
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.autobot_server_ip = data.autobot_server_ip
        profile.autobot_server_port = data.autobot_server_port
        profile.save()
        
        print(f"서버 설정 DB 저장 완료: 사용자={user.username}, IP={data.autobot_server_ip}, Port={data.autobot_server_port}")
        
        return {
            'success': True,
            'message': '서버 설정이 데이터베이스에 저장되었습니다!'
        }
    except Exception as e:
        print(f"서버 설정 저장 오류: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@mypage_router.post("/server-connection-test", response=ResponseSchema)
def test_server_connection(request, data: ServerConnectionTestSchema):
    """autobot 서버 연결 테스트 - 실제 DB 연동"""
    try:
        if not data.ip:
            return {
                'success': False,
                'error': 'IP 주소는 필수입니다.'
            }
        
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        # autobot 서버 헬스 체크
        try:
            response = requests.get(
                f'http://{data.ip}:{data.port}/health',
                timeout=5
            )
            
            # UserProfile 가져오기 또는 생성
            profile, created = UserProfile.objects.get_or_create(user=user)
            
            if response.status_code == 200:
                # 연결 성공 시 프로필 업데이트
                profile.server_status = 'online'
                profile.last_connection = timezone.now()
                profile.save()
                
                print(f"서버 연결 성공 - DB 업데이트: 사용자={user.username}, 상태=online")
                
                return {
                    'success': True,
                    'message': '서버 연결 성공'
                }
            else:
                # 연결 실패 시 상태 업데이트
                profile.server_status = 'error'
                profile.save()
                
                print(f"서버 연결 실패 - DB 업데이트: 사용자={user.username}, 상태=error")
                
                return {
                    'success': False,
                    'error': f'서버 응답 오류: {response.status_code}'
                }
                
        except requests.exceptions.RequestException as e:
            # 네트워크 오류 시 상태 업데이트
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.server_status = 'offline'
            profile.save()
            
            print(f"서버 연결 실패 - DB 업데이트: 사용자={user.username}, 상태=offline")
            
            return {
                'success': False,
                'error': f'서버 연결 실패: {str(e)}'
            }
            
    except Exception as e:
        print(f"서버 연결 테스트 오류: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


class TradingConfigSummarySchema(Schema):
    """자동매매 설정 개요 스키마 (두 단계 로딩의 1차 데이터)"""
    id: int
    stock_code: str
    stock_name: str
    trading_mode: str
    stop_loss: Optional[float] = None  # 아코디언 헤더 표시용
    take_profit: Optional[float] = None  # 아코디언 헤더 표시용
    pyramiding_count: int = 0  # 아코디언 헤더 표시용
    position_size: Optional[float] = None  # 아코디언 헤더 표시용
    is_active: bool = True
    created_at: str
    updated_at: str


@mypage_router.get("/trading-configs", response=List[TradingConfigResponseSchema], auth=django_auth)
def get_trading_configs(request):
    """사용자의 자동매매 설정 목록 조회"""
    try:
        configs = TradingConfig.objects.filter(user=request.user)
        
        result = []
        for config in configs:
            result.append({
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
        
        return result
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.get("/trading-configs/summary", response=List[TradingConfigSummarySchema])
def get_trading_configs_summary(request):
    """자동매매 설정 개요 목록 조회 (두 단계 로딩의 1차 데이터)"""
    try:
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        # autobot 서버에서 개요 데이터만 가져오기
        try:
            profile = UserProfile.objects.get(user=user)
            if profile.autobot_server_ip:
                server_ip = profile.autobot_server_ip
                server_port = profile.autobot_server_port
            else:
                server_ip = '127.0.0.1'
                server_port = 8080
        except UserProfile.DoesNotExist:
            server_ip = '127.0.0.1'
            server_port = 8080
        
        try:
            user_id = user.google_id or f"user_{user.id}"
            response = requests.get(
                f'http://{server_ip}:{server_port}/trading-configs/{user_id}',
                timeout=10
            )
            
            if response.status_code == 200:
                configs = response.json()
                
                # 개요 데이터 추출 (아코디언 헤더용 기본 정보 포함)
                summary_data = []
                for config in configs:
                    if config.get('is_active', True):  # 활성 설정만
                        summary_data.append({
                            'id': config.get('id'),
                            'stock_code': config.get('stock_code'),
                            'stock_name': config.get('stock_name'),
                            'trading_mode': config.get('trading_mode'),
                            'stop_loss': config.get('stop_loss'),  # 아코디언 헤더 표시용
                            'take_profit': config.get('take_profit'),  # 아코디언 헤더 표시용
                            'pyramiding_count': config.get('pyramiding_count', 0),  # 아코디언 헤더 표시용
                            'position_size': config.get('position_size'),  # 아코디언 헤더 표시용
                            'is_active': config.get('is_active', True),
                            'created_at': config.get('created_at'),
                            'updated_at': config.get('updated_at'),
                        })
                
                return summary_data
            else:
                print(f"autobot 서버 오류: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"autobot 서버 연결 실패: {str(e)}")
            return []
            
    except Exception as e:
        print(f"개요 데이터 조회 오류: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.post("/trading-configs", response=ResponseSchema)
def create_or_update_trading_config(request, data: TradingConfigSchema):
    """자동매매 설정 생성 또는 업데이트 - 완전한 DB 연동 (Presentation 페이지에서 호출)"""
    try:
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        with transaction.atomic():
            # 기존 활성 설정이 있는지 확인
            existing_config = TradingConfig.objects.filter(
                user=user,
                stock_code=data.stock_code,
                is_active=True
            ).first()
            
            if existing_config:
                # 기존 설정 업데이트
                existing_config.stock_name = data.stock_name
                existing_config.trading_mode = data.trading_mode
                existing_config.max_loss = data.max_loss
                existing_config.stop_loss = data.stop_loss
                existing_config.take_profit = data.take_profit
                existing_config.pyramiding_count = data.pyramiding_count
                existing_config.position_size = data.position_size
                existing_config.is_active = data.is_active
                existing_config.save()
                
                trading_config = existing_config
                action = "업데이트"
                print(f"Django DB 업데이트 완료: TradingConfig ID={trading_config.id}, 사용자={user.username}")
            else:
                # 새로운 설정 생성
                trading_config = TradingConfig.objects.create(
                    user=user,
                    stock_code=data.stock_code,
                    stock_name=data.stock_name,
                    trading_mode=data.trading_mode,
                    max_loss=data.max_loss,
                    stop_loss=data.stop_loss,
                    take_profit=data.take_profit,
                    pyramiding_count=data.pyramiding_count,
                    position_size=data.position_size,
                    is_active=data.is_active,
                )
                action = "생성"
                print(f"Django DB 생성 완료: TradingConfig ID={trading_config.id}, 사용자={user.username}")
            
            # autobot 서버로 설정 전달
            config_dict = {
                'stock_code': data.stock_code,
                'stock_name': data.stock_name,
                'trading_mode': data.trading_mode,
                'max_loss': data.max_loss,
                'stop_loss': data.stop_loss,
                'take_profit': data.take_profit,
                'pyramiding_count': data.pyramiding_count,
                'position_size': data.position_size,
                'pyramiding_entries': data.pyramiding_entries,
                'positions': data.positions,
                'is_active': data.is_active,
            }
            autobot_config_id = send_to_user_autobot_server(user, config_dict)
            if autobot_config_id:
                trading_config.autobot_config_id = autobot_config_id
                trading_config.save()
                print(f"autobot 서버 전달 완료: autobot_config_id={autobot_config_id}")
            
            return {
                'success': True,
                'message': f'자동매매 설정이 성공적으로 {action}되었습니다!\n\n✅ Django DB {action} (ID: {trading_config.id})\n✅ autobot 서버 전달 (ID: {autobot_config_id})'
            }
            
    except Exception as e:
        print(f"자동매매 설정 처리 오류: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@mypage_router.get("/trading-configs/stock/{stock_code}", response=TradingConfigResponseSchema)
def get_trading_config_by_stock(request, stock_code: str):
    """특정 종목의 자동매매 설정을 autobot 서버에서 조회합니다."""
    try:
        # 테스트 사용자 가져오기
        user = get_or_create_test_user()
        
        # 사용자의 autobot 서버 정보 가져오기
        try:
            profile = UserProfile.objects.get(user=user)
            if profile.autobot_server_ip:
                server_ip = profile.autobot_server_ip
                server_port = profile.autobot_server_port
            else:
                server_ip = '127.0.0.1'
                server_port = 8080
        except UserProfile.DoesNotExist:
            server_ip = '127.0.0.1'
            server_port = 8080
        
        # autobot 서버에서 설정 조회
        try:
            user_id = user.google_id or f"user_{user.id}"
            response = requests.get(
                f'http://{server_ip}:{server_port}/trading-configs/user/{user_id}/stock/{stock_code}',
                timeout=10
            )
            
            if response.status_code == 200:
                autobot_response = response.json()
                config = autobot_response.get('config')
                
                if config:
                    print(f"autobot 서버에서 설정 조회 성공: {stock_code}")
                    return {
                        'id': config.get('id'),
                        'stock_code': config.get('stock_code'),
                        'stock_name': config.get('stock_name'),
                        'trading_mode': config.get('trading_mode'),
                        'max_loss': config.get('max_loss'),
                        'stop_loss': config.get('stop_loss'),
                        'take_profit': config.get('take_profit'),
                        'pyramiding_count': config.get('pyramiding_count', 0),
                        'position_size': config.get('position_size'),
                        'pyramiding_entries': config.get('pyramiding_entries', []),  # 피라미딩 진입시점 배열 추가
                        'positions': config.get('positions', []),  # 포지션 배열 추가
                        'is_active': config.get('is_active', True),
                        'autobot_config_id': config.get('id'),
                        'created_at': config.get('created_at'),
                        'updated_at': config.get('updated_at'),
                    }
                else:
                    print(f"autobot 서버에 {stock_code} 설정이 없음")
                    return JsonResponse({
                        'success': False,
                        'error': '해당 종목의 설정이 없습니다.'
                    }, status=404)
            else:
                print(f"autobot 서버 오류: {response.status_code}")
                return JsonResponse({
                    'success': False,
                    'error': f'autobot 서버 오류: {response.status_code}'
                }, status=500)
                
        except requests.exceptions.RequestException as e:
            print(f"autobot 서버 연결 실패: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'autobot 서버 연결 실패: {str(e)}'
            }, status=500)
            
    except Exception as e:
        print(f"설정 조회 오류: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.delete("/trading-configs/{config_id}", response=ResponseSchema, auth=django_auth)
def delete_trading_config(request, config_id: int):
    """자동매매 설정 삭제"""
    try:
        config = TradingConfig.objects.get(id=config_id, user=request.user)
        config.delete()
        
        return {
            'success': True,
            'message': '설정이 삭제되었습니다.'
        }
    except TradingConfig.DoesNotExist:
        return {
            'success': False,
            'error': '설정을 찾을 수 없습니다.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def send_to_user_autobot_server(user, config_data):
    """사용자별 autobot 서버로 설정 전달 - 실제 DB 기반"""
    try:
        # 사용자의 프로필에서 autobot 서버 정보 가져오기
        profile = UserProfile.objects.get(user=user)
        
        if not profile.autobot_server_ip:
            print(f"사용자 {user.username}의 autobot 서버 IP가 설정되지 않음 - 기본 서버 사용")
            # 기본 autobot 서버 사용
            server_ip = '127.0.0.1'
            server_port = 8080
        else:
            server_ip = profile.autobot_server_ip
            server_port = profile.autobot_server_port
        
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
            'pyramiding_entries': config_data.get('pyramiding_entries', []),
            'positions': config_data.get('positions', []),
            'user_id': user.google_id or f"user_{user.id}",  # Google ID 우선, 없으면 User ID
            'is_active': config_data.get('is_active', True),
        }
        
        response = requests.post(
            f'http://{server_ip}:{server_port}/trading-configs',
            json=autobot_data,
            timeout=10
        )
        
        if response.status_code == 200:
            autobot_response = response.json()
            print(f"autobot 서버 전달 성공: 서버={server_ip}:{server_port}, 응답 ID={autobot_response.get('id')}")
            return autobot_response.get('id')
        else:
            print(f"autobot 서버 오류: {response.status_code} - {response.text}")
            return None
            
    except UserProfile.DoesNotExist:
        print(f"사용자 {user.username}의 프로필이 없음 - 기본 서버 사용")
        # 프로필이 없으면 기본 서버로 전달
        return send_to_default_autobot_server(user, config_data)
    except requests.exceptions.RequestException as e:
        print(f"autobot 서버 연결 실패: {str(e)}")
        return None
    except Exception as e:
        print(f"autobot 서버 전달 중 오류: {str(e)}")
        return None


def send_to_default_autobot_server(user, config_data):
    """기본 autobot 서버로 설정 전달"""
    try:
        autobot_data = {
            'stock_code': config_data['stock_code'],
            'stock_name': config_data['stock_name'],
            'trading_mode': config_data['trading_mode'],
            'max_loss': config_data.get('max_loss'),
            'stop_loss': config_data.get('stop_loss'),
            'take_profit': config_data.get('take_profit'),
            'pyramiding_count': config_data.get('pyramiding_count', 0),
            'position_size': config_data.get('position_size'),
            'pyramiding_entries': config_data.get('pyramiding_entries', []),
            'positions': config_data.get('positions', []),
            'user_id': user.google_id or f"user_{user.id}",
            'is_active': config_data.get('is_active', True),
        }
        
        response = requests.post(
            'http://127.0.0.1:8080/trading-configs',  # 기본 서버
            json=autobot_data,
            timeout=10
        )
        
        if response.status_code == 200:
            autobot_response = response.json()
            print(f"기본 autobot 서버 전달 성공: 응답 ID={autobot_response.get('id')}")
            return autobot_response.get('id')
        else:
            print(f"기본 autobot 서버 오류: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"기본 autobot 서버 전달 오류: {str(e)}")
        return None