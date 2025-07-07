from ninja import Schema
from typing import List, Dict, Optional, Any

# =====================================================================
# response를 담는 스키마

class FailedRecord(Schema):
    index: int
    code: str
    error: str

class IndexListResponse(Schema):
    status: str
    count_total: int
    count_created: int
    count_updated: int
    count_failed: int

class StockDescriptionResponse(Schema):
    status: str
    count_total: int
    count_created: int
    count_updated: int
    count_failed: int
    failed_records: Optional[List[FailedRecord]] = None
    
class SuccessResponse(Schema):
    status: str
    message: str
    count_saved: int

class ErrorResponse(Schema):
    status: str
    message: str
    
class SuccessResponseStockDart(Schema):
    status: str
    data: List[Dict[str, Any]]  # JSON 형태를 표현


# =====================================================================
# 데이터를 담는 스키마

class StockIndexSchema(Schema):
    code: str
    name: str
    market: str

class CompanySchema(Schema):
    code: str
    name: str
    market: str
    sector: Optional[str] = None
    industry: Optional[str] = None

class StockAnalysisSchema(Schema):
    code: str
    date: str
    ma50: float
    ma150: float
    ma200: float
    rsScore: float
    rsScore1m: float
    rsScore3m: float
    rsScore6m: float
    rsScore12m: float
    rsRank: float
    rsRank1m: float
    rsRank3m: float
    rsRank6m: float
    rsRank12m: float
    max_52w: float
    min_52w: float
    max_52w_date: str = None
    min_52w_date: str = None
    atr: float = 0.0  # 평균 진폭 (Average True Range)
    is_minervini_trend: bool

class FinancialDataSchema(Schema):
    매출증가율: float = 0.0
    영업이익증가율: float = 0.0
    전전기매출: int
    전기매출: int
    당기매출: int
    전전기영업이익: int
    전기영업이익: int
    당기영업이익: int

class CombinedStockAnalysisSchema(FinancialDataSchema, StockAnalysisSchema, CompanySchema):
    pass

class SuccessResponseStockAnalysis(Schema):
    status: str
    data: List[CombinedStockAnalysisSchema]


class StockOhlcvSchema(Schema):
    code: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float

class SuccessResponseStockOhlcvSchema(Schema):
    status: str
    data: List[StockOhlcvSchema]  # JSON 형태를 표현


class StockFinancialStatementSchema(Schema):
    code: str
    year: str
    quarter: str
    statement_type: str
    account_name: str
    amount: int

class SuccessResponseStockFinancialSchema(Schema):
    status: str
    data: List[StockFinancialStatementSchema]  # JSON 형태를 표현

class SuccessResponseStockAnalysisSchema(Schema):
    status: str
    data: List[StockAnalysisSchema]  # JSON 형태를 표현

class SuccessResponseStockIndexSchema(Schema):
    status: str
    data: List[StockIndexSchema]  # JSON 형태를 표현

class SuccessResponseIndexOhlcvSchema(Schema):
    status: str
    data: List[StockOhlcvSchema]  # JSON 형태를 표현