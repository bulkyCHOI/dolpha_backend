"""분할 진입 설정 정규화 및 엔진 필드 매핑.

T 좌표계: T = 진입 신호의 눌림 저점 → 전고점 돌파가 사이의 상승폭(원).
- 1차는 항상 t=0.0 (돌파 즉시 진입)
- 2차부터는 최초 체결가 + t × T 도달 시 추가 진입

weight_pct: 각 차수가 감당할 총 포지션 대비 비중(%).
합계는 정규화 후 정확히 100.00 이어야 함.
"""

from dolpha.theme_surge.config import DEFAULT_ENTRY_STAGES, MAX_ENTRY_STAGES


def normalize_entry_stages(raw: object) -> list[dict]:
    """입력 데이터를 검증하고 정규화된 진입 차수 리스트를 반환한다.

    정규화 규칙 (순서 고정):
    1. dict 아님 / 't'·'weight_pct' 숫자 변환 실패 / t < 0 / weight_pct <= 0 항목 제거
    2. 't' 오름차순 정렬
    3. 't' 중복 제거 (먼저 오는 항목 유지)
    4. 앞에서부터 최대 MAX_ENTRY_STAGES 개로 절단
    5. 1차 항목의 't'를 0.0으로 강제
    6. 'weight_pct' 합계가 100이 되도록 비례 정규화, 소수 2자리 반올림 + 잔차 흡수
    6b. 반올림으로 0이 된 항목 제거
    6c. 0이 된 항목 제거 후 남은 항목 있으면 5~6 재적용
    7. 남은 항목이 없으면 DEFAULT_ENTRY_STAGES 반환

    Args:
        raw: 검증 대상 데이터 (list, dict, None, str 등 어떤 타입이든 가능)

    Returns:
        list[dict]: [{"t": float, "weight_pct": float}, ...] 정규화된 리스트
        모든 값은 float 이며 t는 소수 여러 자리, weight_pct는 소수 정확히 2자리
        모든 비중 > 0, 합계 정확히 100.00, t 오름차순·중복 없음
        입력 객체는 변형되지 않음 (불변 규칙)
    """
    def _apply_normalization(stages: list[dict]) -> list[dict]:
        """5단계(1차 t 강제) + 6단계(비례 정규화 + 잔차 흡수)를 수행한다."""
        if not stages:
            return []

        # 5. 1차 항목의 't'를 0.0으로 강제
        stages[0]["t"] = 0.0

        # 6. weight_pct 합계가 100이 되도록 비례 정규화, 소수 2자리 반올림
        total_weight = sum(item["weight_pct"] for item in stages)
        if total_weight <= 0:
            return []

        # 비례 정규화: (item_weight / total_weight) * 100
        normalized: list[dict] = []
        for item in stages:
            normalized_weight = (item["weight_pct"] / total_weight) * 100.0
            rounded_weight = round(normalized_weight, 2)
            normalized.append({"t": float(item["t"]), "weight_pct": rounded_weight})

        # 반올림 오차로 인해 합계가 100 과 벗어날 수 있으므로, 마지막 항목에 residual 흡수
        current_sum = round(sum(item["weight_pct"] for item in normalized), 2)
        residual = round(100.0 - current_sum, 2)
        if normalized and residual:  # residual은 이미 2자리 반올림됨
            normalized[-1]["weight_pct"] = round(normalized[-1]["weight_pct"] + residual, 2)

        return normalized

    # 1. dict 아님 항목 제거 + t·weight_pct 숫자 변환 실패·범위 초과 제거
    cleaned: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                t_val = float(item.get("t"))
                weight_val = float(item.get("weight_pct"))
            except (TypeError, ValueError):
                continue
            if t_val < 0 or weight_val <= 0:
                continue
            # 새 dict 를 만들어 입력을 변형하지 않음
            cleaned.append({"t": t_val, "weight_pct": weight_val})

    # 2. 't' 오름차순 정렬
    cleaned.sort(key=lambda x: x["t"])

    # 3. 't' 중복 제거 (먼저 오는 항목 유지)
    seen_t: set = set()
    deduped: list[dict] = []
    for item in cleaned:
        t_val = item["t"]
        if t_val not in seen_t:
            deduped.append(item)
            seen_t.add(t_val)

    # 4. 최대 MAX_ENTRY_STAGES 개로 절단
    truncated = deduped[:MAX_ENTRY_STAGES]

    # 7. 남은 항목이 없으면 기본값 반환
    if not truncated:
        return [dict(stage) for stage in DEFAULT_ENTRY_STAGES]

    # 5-6 단계 수행
    normalized = _apply_normalization(truncated)
    if not normalized:
        return [dict(stage) for stage in DEFAULT_ENTRY_STAGES]

    # 6b. 반올림으로 0이 된 항목 제거
    non_zero = [stage for stage in normalized if stage["weight_pct"] > 0]

    # 6c. 0이 된 항목이 있었으면, 5-6 재적용
    if len(non_zero) < len(normalized):
        if not non_zero:
            return [dict(stage) for stage in DEFAULT_ENTRY_STAGES]
        # 남은 항목들에 대해 5-6 재적용
        normalized = _apply_normalization(non_zero)
        if not normalized:
            return [dict(stage) for stage in DEFAULT_ENTRY_STAGES]

    return normalized


def load_entry_stages(defaults: object) -> list[dict]:
    """TradingDefaults 형 객체에서 'theme_surge_entry_stages' 필드를 읽어 정규화한다.

    Args:
        defaults: TradingDefaults 인스턴스 또는 getattr를 지원하는 유사 객체 (None 가능)

    Returns:
        list[dict]: 정규화된 진입 차수 리스트
    """
    raw = getattr(defaults, "theme_surge_entry_stages", None) if defaults else None
    return normalize_entry_stages(raw)


def to_config_fields(stages: list[dict]) -> dict:
    """정규화된 진입 차수를 TradingConfig의 기존 필드에 매핑한다.

    기존 피라미딩 인프라(calculate_pyramiding_amounts / get_current_entry_amount)를
    재사용하기 위해 설정한 회차·비중을 pyramiding_count, pyramiding_entries, positions
    으로 변환한다.

    Args:
        stages: normalize_entry_stages() 결과 ([{"t": float, "weight_pct": float}, ...])
                반드시 1개 이상의 항목을 포함해야 함 (normalize_entry_stages()는 빈 리스트를 반환하지 않음)

    Returns:
        dict: {
            "pyramiding_count": int (stages 길이 - 1),
            "pyramiling_entries": [str, ...] (2차 이후의 T배수 문자열, f-string {t:g} 형식),
            "positions": [float, ...] (각 차수의 weight_pct)
        }
        예) 1차만: pyramiding_count=0, pyramiling_entries=[], positions=[100.0]
        예) 3차: pyramiling_count=2, pyramling_entries=['0.5', '1.0'], positions=[50.0, 30.0, 20.0]
    """
    # 빈 입력에 대한 가드 (normalize_entry_stages는 이를 반환하지 않지만 public 함수이므로 안전)
    if not stages:
        return {
            "pyramiding_count": 0,
            "pyramiding_entries": [],
            "positions": [100.0],
        }

    pyramiding_count = len(stages) - 1
    pyramiding_entries = [f"{stage['t']:g}" for stage in stages[1:]]
    positions = [stage["weight_pct"] for stage in stages]

    return {
        "pyramiding_count": pyramiding_count,
        "pyramiding_entries": pyramiding_entries,
        "positions": positions,
    }
