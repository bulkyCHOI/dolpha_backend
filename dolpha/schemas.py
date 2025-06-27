from ninja import Schema
from typing import List, Dict, Optional, Any

# =====================================================================
# response를 담는 스키마

class FailedRecord(Schema):
    index: int
    code: str
    error: str

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

class CompanySchema(Schema):
    code: str
    name: str
    market: str
    sector: str
    industry: str

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
    is_minervini_trend: bool

class FinancialDataSchema(Schema):
    # '금분기_매출', '전분기_매출', '금분기_영업이익', '전분기_영업이익'
    금분기_매출: float
    전분기_매출: float
    금분기_영업이익: float 
    전분기_영업이익: float    

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