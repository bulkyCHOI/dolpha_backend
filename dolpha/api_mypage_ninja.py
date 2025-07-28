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
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth.models import AnonymousUser

from myweb.models import User, UserProfile, TradingConfig, TradingDefaults


mypage_router = Router()

# JWT 인증을 위한 헬퍼 함수
def get_authenticated_user(request):
    """
    JWT 토큰을 사용하여 사용자 인증
    """
    try:
        # Authorization 헤더 확인
        auth_header = request.headers.get('Authorization')
        
        jwt_auth = JWTAuthentication()
        auth_result = jwt_auth.authenticate(request)
        
        if auth_result is None:
            return None
            
        user, token = auth_result
        
        if user and user.is_authenticated:
            return user
        else:
            return None
            
    except (InvalidToken, TokenError) as e:
        return None
    except Exception as e:
        return None

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
    strategy_type: str = 'mtt'  # 'mtt' or 'weekly_high'
    max_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pyramiding_count: int = 0
    entry_point: Optional[float] = None
    pyramiding_entries: List[str] = []  # 피라미딩 진입시점 배열
    positions: List[float] = []         # 포지션 배열
    is_active: bool = True

class TradingConfigResponseSchema(Schema):
    id: int
    stock_code: str
    stock_name: str
    trading_mode: str
    strategy_type: str = 'mtt'  # 'mtt' or 'weekly_high'
    max_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pyramiding_count: int = 0
    entry_point: Optional[float] = None
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

# 자동매매 기본값 설정 스키마
class TradingDefaultsSchema(Schema):
    trading_mode: str = "turtle"
    # Manual 모드 설정
    manual_max_loss: float = 8.0
    manual_stop_loss: float = 8.0
    manual_take_profit: Optional[float] = None
    manual_pyramiding_count: int = 0
    manual_position_size: float = 100.0
    manual_positions: List[float] = []
    manual_pyramiding_entries: List[str] = []
    manual_use_trailing_stop: bool = True
    manual_trailing_stop_percent: float = 8.0
    # Turtle 모드 설정
    turtle_max_loss: float = 8.0
    turtle_stop_loss: float = 2.0
    turtle_take_profit: Optional[float] = None
    turtle_pyramiding_count: int = 3
    turtle_position_size: float = 25.0
    turtle_positions: List[float] = []
    turtle_pyramiding_entries: List[str] = []
    turtle_use_trailing_stop: bool = True
    turtle_trailing_stop_percent: float = 3.0
    # 공통 설정
    default_entry_trigger: float = 1.0
    default_exit_trigger: float = 2.0

class TradingDefaultsResponseSchema(Schema):
    id: int
    trading_mode: str
    # Manual 모드 설정
    manual_max_loss: float
    manual_stop_loss: float
    manual_take_profit: Optional[float] = None
    manual_pyramiding_count: int
    manual_position_size: float
    manual_positions: List[float] = []
    manual_pyramiding_entries: List[str] = []
    manual_use_trailing_stop: bool
    manual_trailing_stop_percent: float
    # Turtle 모드 설정
    turtle_max_loss: float
    turtle_stop_loss: float
    turtle_take_profit: Optional[float] = None
    turtle_pyramiding_count: int
    turtle_position_size: float
    turtle_positions: List[float] = []
    turtle_pyramiding_entries: List[str] = []
    turtle_use_trailing_stop: bool
    turtle_trailing_stop_percent: float
    # 공통 설정
    default_entry_trigger: float
    default_exit_trigger: float
    created_at: str
    updated_at: str



@mypage_router.get("/profile", response=UserProfileResponseSchema)
def get_user_profile(request):
    """사용자 프로필 정보 조회"""
    try:
        # JWT 토큰으로 사용자 인증
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        return {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'profile_picture': user.profile_picture,
                'date_joined': user.date_joined.isoformat() if user.date_joined else None,
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
        user = get_authenticated_user(request)
        
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        return {
            'autobot_server_ip': profile.autobot_server_ip,
            'autobot_server_port': profile.autobot_server_port,
            'server_status': profile.server_status,
            'last_connection': profile.last_connection.isoformat() if profile.last_connection else None,
            'created_at': profile.created_at.isoformat() if profile.created_at else None,
            'updated_at': profile.updated_at.isoformat() if profile.updated_at else None,
        }
    except Exception as e:
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
        
        user = get_authenticated_user(request)
        
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.autobot_server_ip = data.autobot_server_ip
        profile.autobot_server_port = data.autobot_server_port
        profile.save()
        
        return {
            'success': True,
            'message': '서버 설정이 데이터베이스에 저장되었습니다!'
        }
    except Exception as e:
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
        
        user = get_authenticated_user(request)
        
        # autobot 서버 헬스 체크
        try:
            response = requests.get(
                f'http://{data.ip}:{data.port}/health',
                timeout=5
            )
            
            profile, created = UserProfile.objects.get_or_create(user=user)
            
            if response.status_code == 200:
                profile.server_status = 'online'
                profile.last_connection = timezone.now()
                profile.save()
                
                return {
                    'success': True,
                    'message': '서버 연결 성공'
                }
            else:
                profile.server_status = 'error'
                profile.save()
                
                return {
                    'success': False,
                    'error': f'서버 응답 오류: {response.status_code}'
                }
                
        except requests.exceptions.RequestException as e:
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.server_status = 'offline'
            profile.save()
            
            return {
                'success': False,
                'error': f'서버 연결 실패: {str(e)}'
            }
            
    except ValueError as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
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
    strategy_type: str = 'mtt'  # 'mtt' or 'weekly_high'
    stop_loss: Optional[float] = None  # 아코디언 헤더 표시용
    take_profit: Optional[float] = None  # 아코디언 헤더 표시용
    pyramiding_count: int = 0  # 아코디언 헤더 표시용
    entry_point: Optional[float] = None  # 아코디언 헤더 표시용
    is_active: bool = True
    created_at: str
    updated_at: str


@mypage_router.get("/trading-configs", response=List[TradingConfigResponseSchema])
def get_trading_configs(request, strategy_type: str = None):
    """사용자의 자동매매 설정 목록 조회 (strategy_type 필터 지원)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
            
        configs = TradingConfig.objects.filter(user=user)
        
        # strategy_type 필터링
        if strategy_type:
            configs = configs.filter(strategy_type=strategy_type)
        
        result = []
        for config in configs:
            result.append({
                'id': config.id,
                'stock_code': config.stock_code,
                'stock_name': config.stock_name,
                'trading_mode': config.trading_mode,
                'strategy_type': config.strategy_type,
                'max_loss': config.max_loss,
                'stop_loss': config.stop_loss,
                'take_profit': config.take_profit,
                'pyramiding_count': config.pyramiding_count,
                'entry_point': config.entry_point,
                'pyramiding_entries': config.pyramiding_entries,  # Django DB에서 직접 가져옴
                'positions': config.positions,  # Django DB에서 직접 가져옴
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
def get_trading_configs_summary(request, strategy_type: str = None):
    """자동매매 설정 개요 목록 조회 (두 단계 로딩의 1차 데이터, strategy_type 필터 지원)"""
    try:
        user = get_authenticated_user(request)
        
        # autobot 서버 설정 확인
        try:
            profile = UserProfile.objects.get(user=user)
            if not profile.autobot_server_ip:
                # 서버 설정이 없으면 빈 배열 반환
                return []
            server_ip = profile.autobot_server_ip
            server_port = profile.autobot_server_port
        except UserProfile.DoesNotExist:
            # 프로필이 없으면 빈 배열 반환
            return []
        
        try:
            user_id = user.google_id or f"user_{user.id}"
            # strategy_type 필터를 쿼리 파라미터로 전달
            query_params = f"?strategy_type={strategy_type}" if strategy_type else ""
            response = requests.get(
                f'http://{server_ip}:{server_port}/trading-configs/{user_id}{query_params}',
                timeout=10
            )
            
            if response.status_code == 200:
                configs = response.json()
                
                # 개요 데이터 추출 (모든 설정 포함, 활성/비활성 구분)
                summary_data = []
                for config in configs:
                    summary_data.append({
                        'id': config.get('id'),
                        'stock_code': config.get('stock_code'),
                        'stock_name': config.get('stock_name'),
                        'trading_mode': config.get('trading_mode'),
                        'strategy_type': config.get('strategy_type', 'mtt'),
                        'stop_loss': config.get('stop_loss'),  # 아코디언 헤더 표시용
                        'take_profit': config.get('take_profit'),  # 아코디언 헤더 표시용
                        'pyramiding_count': config.get('pyramiding_count', 0),  # 아코디언 헤더 표시용
                        'entry_point': config.get('entry_point'),  # 아코디언 헤더 표시용
                        'is_active': config.get('is_active', True),
                        'created_at': config.get('created_at'),
                        'updated_at': config.get('updated_at'),
                    })
                
                return summary_data
            else:
                return []
                
        except requests.exceptions.RequestException as e:
            return []
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.post("/trading-configs", response=ResponseSchema)
def create_or_update_trading_config(request, data: TradingConfigSchema):
    """자동매매 설정 생성 또는 업데이트 - 기존 설정은 그대로 유지"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        if not profile.autobot_server_ip:
            return JsonResponse({
                'success': False,
                'error': 'SERVER_SETTINGS_REQUIRED',
                'message': 'autobot 서버 설정을 먼저 완료해주세요. 마이페이지 > 서버 설정에서 autobot 서버 IP와 포트를 설정한 후 자동매매 설정을 저장할 수 있습니다.'
            }, status=400)
        
        with transaction.atomic():
            # 기존 설정이 있는지 확인 (stock_code + strategy_type 조합으로 확인)
            existing_config = TradingConfig.objects.filter(
                user=user,
                stock_code=data.stock_code,
                strategy_type=data.strategy_type
            ).first()
            
            if existing_config:
                # 기존 설정 업데이트 (그대로 받은 데이터 사용)
                existing_config.stock_name = data.stock_name
                existing_config.trading_mode = data.trading_mode
                existing_config.strategy_type = data.strategy_type
                existing_config.max_loss = data.max_loss
                existing_config.stop_loss = data.stop_loss
                existing_config.take_profit = data.take_profit
                existing_config.pyramiding_count = data.pyramiding_count
                existing_config.entry_point = data.entry_point
                existing_config.pyramiding_entries = data.pyramiding_entries
                existing_config.positions = data.positions
                existing_config.is_active = data.is_active
                existing_config.save()
                
                trading_config = existing_config
                action = "업데이트"
            else:
                # 새로운 설정 생성 (그대로 받은 데이터 사용)
                trading_config = TradingConfig.objects.create(
                    user=user,
                    stock_code=data.stock_code,
                    stock_name=data.stock_name,
                    trading_mode=data.trading_mode,
                    strategy_type=data.strategy_type,
                    max_loss=data.max_loss,
                    stop_loss=data.stop_loss,
                    take_profit=data.take_profit,
                    pyramiding_count=data.pyramiding_count,
                    entry_point=data.entry_point,
                    pyramiding_entries=data.pyramiding_entries,
                    positions=data.positions,
                    is_active=data.is_active,
                )
                action = "생성"
            
            # autobot 서버로 설정 전달 (Django DB가 단일 소스)
            autobot_config_dict = {
                'stock_code': trading_config.stock_code,
                'stock_name': trading_config.stock_name,
                'trading_mode': trading_config.trading_mode,
                'strategy_type': trading_config.strategy_type,
                'max_loss': trading_config.max_loss,
                'stop_loss': trading_config.stop_loss,
                'take_profit': trading_config.take_profit,
                'pyramiding_count': trading_config.pyramiding_count,
                'entry_point': trading_config.entry_point,
                'pyramiding_entries': trading_config.pyramiding_entries,
                'positions': trading_config.positions,
                'is_active': trading_config.is_active,
            }
            autobot_config_id = send_to_user_autobot_server(user, autobot_config_dict)
            if autobot_config_id:
                trading_config.autobot_config_id = autobot_config_id
                trading_config.save()
            
            return {
                'success': True,
                'message': f'자동매매 설정이 성공적으로 {action}되었습니다!\n\n✅ Django DB {action} (ID: {trading_config.id})\n✅ autobot 서버 전달 (ID: {autobot_config_id})'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mypage_router.get("/trading-configs/stock/{stock_code}", response=TradingConfigResponseSchema)
def get_trading_config_by_stock(request, stock_code: str, strategy_type: str = 'mtt'):
    """특정 종목의 자동매매 설정을 Django DB와 autobot 서버에서 조회합니다."""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        # Django DB에서 기본 설정 조회 (stock_code + strategy_type 조합)
        config = TradingConfig.objects.filter(
            user=user, 
            stock_code=stock_code, 
            strategy_type=strategy_type
        ).first()
        
        if config:
            
            return {
                'id': config.id,  # Django DB의 ID 사용
                'stock_code': config.stock_code,
                'stock_name': config.stock_name,
                'trading_mode': config.trading_mode,
                'strategy_type': config.strategy_type,
                'max_loss': config.max_loss,
                'stop_loss': config.stop_loss,
                'take_profit': config.take_profit,
                'pyramiding_count': config.pyramiding_count,
                'entry_point': config.entry_point,
                'pyramiding_entries': config.pyramiding_entries,  # Django DB에서 직접 가져옴
                'positions': config.positions,  # Django DB에서 직접 가져옴
                'is_active': config.is_active,
                'autobot_config_id': config.autobot_config_id,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat(),
            }
        else:
            return JsonResponse({
                'success': False,
                'error': '해당 종목의 설정이 없습니다.'
            }, status=404)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)




@mypage_router.delete("/trading-configs/stock/{stock_code}", response=ResponseSchema)
def delete_trading_config_by_stock_code(request, stock_code: str, strategy_type: str = 'mtt'):
    """자동매매 설정 삭제 - stock_code와 strategy_type을 사용하여 Django DB와 autobot 서버에서 모두 삭제"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
            
        config = TradingConfig.objects.get(
            stock_code=stock_code, 
            strategy_type=strategy_type,
            user=user
        )
        
        # autobot 서버에서도 삭제
        autobot_result = delete_from_autobot_server(user, config.stock_code, config.strategy_type)
        
        # Django DB에서 삭제
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


def delete_from_autobot_server(user, stock_code, strategy_type='mtt'):
    """autobot 서버에서 자동매매 설정 삭제"""
    try:
        # 사용자의 프로필에서 autobot 서버 정보 가져오기
        profile = UserProfile.objects.get(user=user)
        
        if not profile.autobot_server_ip:
            # 서버 설정이 없으면 삭제할 필요 없음
            return True
        
        server_ip = profile.autobot_server_ip
        server_port = profile.autobot_server_port
        
        # autobot 서버에서 설정 삭제 (strategy_type 포함)
        user_id = user.google_id or f"user_{user.id}"
        delete_url = f'http://{server_ip}:{server_port}/trading-configs/user/{user_id}/stock/{stock_code}?strategy_type={strategy_type}'
        
        response = requests.delete(delete_url, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            # 삭제 실패해도 Django DB는 삭제하도록 함
            return False
            
    except UserProfile.DoesNotExist:
        # 프로필이 없으면 삭제할 필요 없음
        return True
    except requests.exceptions.RequestException as e:
        # 네트워크 오류가 있어도 Django DB는 삭제하도록 함
        return False
    except Exception as e:
        return False




def send_to_user_autobot_server(user, config_data):
    """사용자별 autobot 서버로 설정 전달 - 실제 DB 기반"""
    try:
        # 사용자의 프로필에서 autobot 서버 정보 가져오기
        profile = UserProfile.objects.get(user=user)
        
        if not profile.autobot_server_ip:
            # 서버 설정이 없으면 실패 반환
            return None
        
        server_ip = profile.autobot_server_ip
        server_port = profile.autobot_server_port
        
        # autobot 서버 API 호출
        autobot_data = {
            'stock_code': config_data['stock_code'],
            'stock_name': config_data['stock_name'],
            'trading_mode': config_data['trading_mode'],
            'strategy_type': config_data.get('strategy_type', 'mtt'),
            'max_loss': config_data.get('max_loss'),
            'stop_loss': config_data.get('stop_loss'),
            'take_profit': config_data.get('take_profit'),
            'pyramiding_count': config_data.get('pyramiding_count', 0),
            'entry_point': config_data.get('entry_point'),
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
            return autobot_response.get('id')
        else:
            return None
            
    except UserProfile.DoesNotExist:
        # 프로필이 없으면 실패 반환
        return None
    except requests.exceptions.RequestException as e:
        return None
    except Exception as e:
        return None


# 자동매매 기본값 설정 API
@mypage_router.get("/trading-defaults", response=TradingDefaultsResponseSchema)
def get_trading_defaults(request):
    """사용자의 자동매매 기본값 설정 조회"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        # 기본값 설정이 없으면 자동 생성
        defaults, created = TradingDefaults.objects.get_or_create(user=user)
        
        return {
            'id': defaults.id,
            'trading_mode': defaults.trading_mode,
            # Manual 모드 설정
            'manual_max_loss': defaults.manual_max_loss,
            'manual_stop_loss': defaults.manual_stop_loss,
            'manual_take_profit': defaults.manual_take_profit,
            'manual_pyramiding_count': defaults.manual_pyramiding_count,
            'manual_position_size': defaults.manual_position_size,
            'manual_positions': defaults.manual_positions,
            'manual_pyramiding_entries': defaults.manual_pyramiding_entries,
            'manual_use_trailing_stop': defaults.manual_use_trailing_stop,
            'manual_trailing_stop_percent': defaults.manual_trailing_stop_percent,
            # Turtle 모드 설정
            'turtle_max_loss': defaults.turtle_max_loss,
            'turtle_stop_loss': defaults.turtle_stop_loss,
            'turtle_take_profit': defaults.turtle_take_profit,
            'turtle_pyramiding_count': defaults.turtle_pyramiding_count,
            'turtle_position_size': defaults.turtle_position_size,
            'turtle_positions': defaults.turtle_positions,
            'turtle_pyramiding_entries': defaults.turtle_pyramiding_entries,
            'turtle_use_trailing_stop': defaults.turtle_use_trailing_stop,
            'turtle_trailing_stop_percent': defaults.turtle_trailing_stop_percent,
            # 공통 설정
            'default_entry_trigger': defaults.default_entry_trigger,
            'default_exit_trigger': defaults.default_exit_trigger,
            'created_at': defaults.created_at.isoformat(),
            'updated_at': defaults.updated_at.isoformat(),
        }
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@mypage_router.post("/trading-defaults", response=ResponseSchema)
def save_trading_defaults(request, data: TradingDefaultsSchema):
    """사용자의 자동매매 기본값 설정 저장/업데이트"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        # 기본값 설정 가져오기 또는 생성
        defaults, created = TradingDefaults.objects.get_or_create(user=user)
        
        # 데이터 업데이트
        defaults.trading_mode = data.trading_mode
        # Manual 모드 설정
        defaults.manual_max_loss = data.manual_max_loss
        defaults.manual_stop_loss = data.manual_stop_loss
        defaults.manual_take_profit = data.manual_take_profit
        defaults.manual_pyramiding_count = data.manual_pyramiding_count
        defaults.manual_position_size = data.manual_position_size
        defaults.manual_positions = data.manual_positions
        defaults.manual_pyramiding_entries = data.manual_pyramiding_entries
        defaults.manual_use_trailing_stop = data.manual_use_trailing_stop
        defaults.manual_trailing_stop_percent = data.manual_trailing_stop_percent
        # Turtle 모드 설정
        defaults.turtle_max_loss = data.turtle_max_loss
        defaults.turtle_stop_loss = data.turtle_stop_loss
        defaults.turtle_take_profit = data.turtle_take_profit
        defaults.turtle_pyramiding_count = data.turtle_pyramiding_count
        defaults.turtle_position_size = data.turtle_position_size
        defaults.turtle_positions = data.turtle_positions
        defaults.turtle_pyramiding_entries = data.turtle_pyramiding_entries
        defaults.turtle_use_trailing_stop = data.turtle_use_trailing_stop
        defaults.turtle_trailing_stop_percent = data.turtle_trailing_stop_percent
        # 공통 설정
        defaults.default_entry_trigger = data.default_entry_trigger
        defaults.default_exit_trigger = data.default_exit_trigger
        
        defaults.save()
        
        action = "생성" if created else "업데이트"
        return {
            'success': True,
            'message': f'자동매매 기본값 설정이 {action}되었습니다.'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# 기본값 조회 API (프론트엔드에서 새 설정 시 사용)
@mypage_router.get("/trading-defaults/for-new-config", response=TradingConfigSchema)
def get_defaults_for_new_config(request):
    """새로운 자동매매 설정을 위한 기본값 반환 (프론트엔드용)"""
    try:
        user = get_authenticated_user(request)
        if not user:
            return JsonResponse({'error': '인증이 필요합니다.'}, status=401)
        
        try:
            defaults = TradingDefaults.objects.get(user=user)
            
            # 현재 선택된 매매모드에 따라 설정값 결정
            trading_mode = defaults.trading_mode
            
            if trading_mode == 'manual':
                # Manual 모드 설정값 사용
                max_loss = defaults.manual_max_loss
                stop_loss = defaults.manual_stop_loss
                take_profit = defaults.manual_take_profit
                pyramiding_count = defaults.manual_pyramiding_count
                pyramiding_entries = defaults.manual_pyramiding_entries if defaults.manual_pyramiding_entries else [""] * defaults.manual_pyramiding_count
                positions = defaults.manual_positions if defaults.manual_positions else [100]
            else:
                # Turtle 모드 설정값 사용
                max_loss = defaults.turtle_max_loss
                stop_loss = defaults.turtle_stop_loss
                take_profit = defaults.turtle_take_profit
                pyramiding_count = defaults.turtle_pyramiding_count
                pyramiding_entries = defaults.turtle_pyramiding_entries if defaults.turtle_pyramiding_entries else [""] * defaults.turtle_pyramiding_count
                positions = defaults.turtle_positions if defaults.turtle_positions else [25, 25, 25, 25]
            
            # 기본값을 TradingConfigSchema 형태로 변환
            config_template = {
                'stock_code': '',
                'stock_name': '',
                'trading_mode': trading_mode,
                'strategy_type': 'mtt',  # 기본값
                'max_loss': max_loss,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pyramiding_count': pyramiding_count,
                'entry_point': None,
                'pyramiding_entries': pyramiding_entries,
                'positions': positions,
                'is_active': True
            }
            
            return config_template
            
        except TradingDefaults.DoesNotExist:
            # 기본값이 없으면 기본 템플릿 반환 (시스템 기본값)
            default_pyramiding_count = 3
            pyramiding_entries = ["", "", ""]  # 3개의 빈 진입시점
            positions = [25, 25, 25, 25]  # 4개의 25% 포지션
            
            return {
                'stock_code': '',
                'stock_name': '',
                'trading_mode': 'turtle',  # 시스템 기본값
                'strategy_type': 'mtt',
                'max_loss': 8.0,
                'stop_loss': 2.0,
                'take_profit': None,
                'pyramiding_count': default_pyramiding_count,
                'entry_point': None,
                'pyramiding_entries': pyramiding_entries,
                'positions': positions,
                'is_active': True
            }
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
