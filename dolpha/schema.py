from ninja import Schema
from typing import List, Dict, Optional, Any

class FailedRecord(Schema):
    index: int
    code: str
    error: str

class StockDescriptionResponse(Schema):
    result: str
    count_total: int
    count_created: int
    count_updated: int
    count_failed: int
    failed_records: Optional[List[FailedRecord]] = None
    
class ErrorResponse(Schema):
    error: str

class SuccessResponse(Schema):
    message: str
    count_saved: int

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

class CombinedStockAnalysisSchema(CompanySchema, StockAnalysisSchema):
    pass

class SuccessResponseStockAnalysis(Schema):
    status: str
    data: List[CombinedStockAnalysisSchema]

class ErrorResponse(Schema):
    status: str
    message: str
    
class SuccessResponseStockDart(Schema):
    status: str
    data: List[Dict[str, Any]]  # JSON 형태를 표현
    

