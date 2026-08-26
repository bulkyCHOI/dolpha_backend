"""전략별 거래 계좌 해석.

전략마다 다른 KIS 계좌로 매매할 수 있도록 TradingConfig.strategy_type →
KisAccount 매핑을 관리한다.

- 매핑이 없는 전략은 사용자의 기본 계좌(KisAccount.is_default)를 따른다.
- 기본 계좌조차 없으면 KisCredentialError를 던진다. 서버 환경변수 계좌로
  조용히 폴백하지 않는다 — 다른 사용자의 계좌로 주문이 나가는 사고를 막는다.
- 매핑은 마이페이지 > 자동매매 기본설정 > 계좌 설정에서 변경한다.
"""

from myweb.models import KisAccount, StrategyAccount, TradingConfig

from .kis.auth import KisCredential, KisCredentialError
from .kis.credentials import credential_for_account

# [("mtt", "MTT (Minervini Trend Template)"), ...]
STRATEGY_CHOICES: list[tuple[str, str]] = list(TradingConfig.STRATEGY_TYPES)
STRATEGY_KEYS: tuple[str, ...] = tuple(key for key, _ in STRATEGY_CHOICES)


def get_default_account(user) -> KisAccount | None:
    """사용자의 기본 계좌. is_default가 없으면 활성 계좌 중 첫 번째."""
    accounts = KisAccount.objects.filter(user=user, is_active=True)
    return accounts.filter(is_default=True).first() or accounts.first()


def get_account_map(user) -> dict[str, KisAccount | None]:
    """전략 → 계좌 전체 매핑. 미지정 전략은 기본 계좌로 채운다."""
    default_account = get_default_account(user)
    assigned = {
        row.strategy_type: row.account
        for row in StrategyAccount.objects.filter(user=user).select_related("account")
    }
    return {
        key: assigned.get(key) or default_account
        for key in STRATEGY_KEYS
    }


def resolve_account(user, strategy_type: str) -> KisAccount:
    """해당 전략이 사용할 계좌를 반환한다.

    Raises:
        KisCredentialError: 지정 계좌도 기본 계좌도 없는 경우
    """
    row = (
        StrategyAccount.objects.filter(user=user, strategy_type=strategy_type)
        .select_related("account")
        .first()
    )
    account = row.account if row and row.account.is_active else None
    account = account or get_default_account(user)

    if account is None:
        raise KisCredentialError(
            "등록된 KIS 계좌가 없습니다. 마이페이지 > 자동매매 기본설정에서"
            " 계좌를 먼저 등록하세요."
        )
    return account


def resolve_credential(user, strategy_type: str) -> KisCredential:
    """전략에 지정된 계좌의 KIS 자격증명을 반환한다."""
    return credential_for_account(resolve_account(user, strategy_type))


def assign_account(user, strategy_type: str, account_id: int | None) -> KisAccount | None:
    """전략에 계좌를 지정한다. account_id가 None이면 지정을 해제(기본 계좌 사용)한다.

    Returns:
        지정된 계좌 (해제 시 None)
    Raises:
        ValueError: 알 수 없는 전략이거나 사용자 소유가 아닌 계좌인 경우
    """
    if strategy_type not in STRATEGY_KEYS:
        raise ValueError(f"알 수 없는 전략입니다: {strategy_type}")

    if account_id is None:
        StrategyAccount.objects.filter(user=user, strategy_type=strategy_type).delete()
        return None

    account = KisAccount.objects.filter(user=user, pk=account_id).first()
    if account is None:
        raise ValueError(f"존재하지 않는 계좌입니다: {account_id}")

    StrategyAccount.objects.update_or_create(
        user=user,
        strategy_type=strategy_type,
        defaults={"account": account},
    )
    return account


def get_active_accounts(user) -> list[KisAccount]:
    """활성 자동매매 설정이 실제로 사용 중인 계좌 목록 (중복 제거).

    매매 사이클에서 계좌별로 잔고·보유종목을 한 번씩만 조회하기 위해 사용한다.
    """
    strategy_types = set(
        TradingConfig.objects.filter(user=user, is_active=True)
        .values_list("strategy_type", flat=True)
        .distinct()
    )
    if not strategy_types:
        return []

    account_map = get_account_map(user)
    by_id: dict[int, KisAccount] = {}
    for strategy_type in strategy_types:
        account = account_map.get(strategy_type)
        if account is not None:
            by_id.setdefault(account.pk, account)
    return list(by_id.values())
