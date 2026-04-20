import FinanceDataReader as fdr
import yfinance

from datetime import datetime, timedelta
from pytz import timezone
import time
import pandas as pd

from pykrx import stock


############################################################################################################################################################
#한국인지 미국인지 구분해 현재 날짜정보를 리턴해 줍니다!
def GetNowDateStr(area = "KRX", type= "NONE" ):
    timezone_info = timezone('Asia/Seoul')
    if area == "US":
        timezone_info = timezone('America/New_York')

    now = datetime.now(timezone_info)
    if type.upper() == "NONE":
        return now.strftime("%Y%m%d")
    else:
        return now.strftime("%Y-%m-%d")

#현재날짜에서 이전/이후 날짜를 구해서 리턴! (미래의 날짜를 구할 일은 없겠지만..)
def GetFromNowDateStr(area = "KRX", type= "NONE" , days=100):
    timezone_info = timezone('Asia/Seoul')
    if area == "US":
        timezone_info = timezone('America/New_York')

    now = datetime.now(timezone_info)

    if days < 0:
        next = now - timedelta(days=abs(days))
    else:
        next = now + timedelta(days=days)

    if type.upper() == "NONE":
        return next.strftime("%Y%m%d")
    else:
        return next.strftime("%Y-%m-%d")

############################################################################################################################################################
# KIS API 사용 여부 확인 (환경변수 KIS_APP_KEY가 설정된 경우에만 활성화)
def _kis_available() -> bool:
    import os
    return bool(os.environ.get("KIS_REAL_APP_KEY", "") or os.environ.get("KIS_APP_KEY", ""))


############################################################################################################################################################
#OHLCV 값을 가져옴!!
# 우선순위: KRX → KIS(0) → FinanceDataReader(1) → pykrx(2) → yfinance(3)
#           US  → FinanceDataReader(1) → yfinance(2)
#
# KIS는 KIS_APP_KEY 환경변수가 설정된 경우에만 1순위로 사용됩니다.
# start_date / end_date: "YYYY-MM-DD" 형식. None 이면 오늘 하루.
def GetOhlcv(area, stock_code, start_date=None, end_date=None, adj_ok="1"):
    today = GetNowDateStr(area, "BAR")
    if start_date is None:
        start_date = today
    if end_date is None:
        end_date = today

    if area == "KRX":
        sources = []
        if _kis_available():
            sources.append(
                lambda: GetOhlcvKIS(stock_code, start_date, end_date, adj_ok)  # 0순위: KIS API
            )
        sources += [
            lambda: GetOhlcv1(area, stock_code, start_date, end_date, adj_ok),   # 1순위: FinanceDataReader
            lambda: GetOhlcvPykrx(stock_code, start_date, end_date, adj_ok),      # 2순위: pykrx (KRX 공식)
            lambda: GetOhlcv2(area, stock_code, start_date, end_date, adj_ok),    # 3순위: yfinance
        ]
    else:
        sources = [
            lambda: GetOhlcv1(area, stock_code, start_date, end_date, adj_ok),   # 1순위: FinanceDataReader
            lambda: GetOhlcv2(area, stock_code, start_date, end_date, adj_ok),    # 2순위: yfinance
        ]

    df = None
    for i, source in enumerate(sources):
        try:
            df = source()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"OHLCV source {i+1} failed for {stock_code}: {e}")

    return df


# KIS API를 통한 국내 주식 OHLCV 조회 (KIS_APP_KEY 환경변수 필요)
def GetOhlcvKIS(stock_code, start_date, end_date, adj_ok="1"):
    from dolpha.kis.ohlcv import GetOhlcvKR
    return GetOhlcvKR(stock_code, start_date, end_date, adj_ok)





#한국 주식은 KRX 정보데이터시스템에서 가져온다. 그런데 미국주식 크롤링의 경우 investing.com 에서 가져오는데 안전하게 2초 정도 쉬어야 한다!
# https://financedata.github.io/posts/finance-data-reader-users-guide.html
def GetOhlcv1(area, stock_code, start_date, end_date, adj_ok="1"):
    df = fdr.DataReader(stock_code, start_date, end_date)
    if adj_ok == "1":
        try:
            df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        except Exception:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = "Date"

    df.insert(5, 'value', ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0) * df['volume'])
    df.insert(6, 'change', (df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    df[['open', 'high', 'low', 'close', 'volume', 'change']] = df[['open', 'high', 'low', 'close', 'volume', 'change']].apply(pd.to_numeric)

    if area == "US":
        time.sleep(2.0)
    else:
        time.sleep(0.2)

    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    return df




# yfinance 폴백 (KRX/US 공통)
# yfinance의 end는 exclusive이므로 하루 더해서 전달
def GetOhlcv2(area, stock_code, start_date, end_date, adj_ok="1"):
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = yfinance.download(stock_code, start=start_date, end=end_exclusive, timeout=30)

    if adj_ok == "1":
        try:
            df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        except Exception:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df.index.name = "Date"

    df.insert(5, 'value', ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0) * df['volume'])
    df.insert(6, 'change', (df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    df[['open', 'high', 'low', 'close', 'volume', 'change']] = df[['open', 'high', 'low', 'close', 'volume', 'change']].apply(pd.to_numeric)

    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    time.sleep(0.2)
    return df


# pykrx 기반 OHLCV (KRX 공식 데이터, GetOhlcv1 실패 시 2순위 폴백)
# pykrx는 YYYYMMDD 형식을 사용
def GetOhlcvPykrx(stock_code, start_date, end_date, adj_ok="1"):
    start_yyyymmdd = start_date.replace('-', '')
    end_yyyymmdd   = end_date.replace('-', '')

    df = stock.get_market_ohlcv(start_yyyymmdd, end_yyyymmdd, stock_code)

    if df is None or len(df) == 0:
        return df

    df = df[['시가', '고가', '저가', '종가', '거래량', '거래대금']]
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'value']
    df.index.name = "Date"
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'value': float})

    df.insert(6, 'change', (df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    df[['open', 'high', 'low', 'close', 'volume', 'change']] = df[['open', 'high', 'low', 'close', 'volume', 'change']].apply(pd.to_numeric)

    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    time.sleep(0.2)
    return df

def GetStockList(area = "KRX"):
    """
    주식 목록을 가져오는 함수
    1차: finance-datareader 시도 (sector, industry 정보 포함)
    2차: pykrx fallback (기본 정보만)
    
    Args:
        area: "KRX", "KRX-DESC", "KOSPI", "KOSDAQ" 등
    
    Returns:
        DataFrame: Company 모델과 호환되는 형태의 주식 목록
    """
    
    # 1차 시도: finance-datareader (sector, industry 정보 포함)
    try:
        print(f"finance-datareader로 {area} 데이터 조회 시도...")
        df = fdr.StockListing(area)
        
        # Code 컬럼 처리
        if 'Code' in df.columns and len(df) > 0:
            df['Code'] = df['Code'].astype(str)
            df['Code'] = df['Code'].str.replace(r'\D', '', regex=True)
            df['Code'] = df['Code'].apply(lambda x: x.zfill(6) if x.isdigit() and x else x)
        
        print(f"finance-datareader 성공: {len(df)}개 데이터 조회")
        return df
        
    except Exception as fdr_error:
        print(f"finance-datareader 실패: {fdr_error}")
        
        # 2차 시도: pykrx fallback
        try:
            print("pykrx로 fallback 시도...")

            # 최근 영업일 탐색 (공휴일/주말에 0개 반환 방지, 최대 10일 이전까지 시도)
            today = None
            for days_back in range(0, 10):
                candidate = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                try:
                    test = stock.get_market_ticker_list(candidate, market="KOSPI")
                    if len(test) > 0:
                        today = candidate
                        print(f"pykrx 조회 기준일: {today} ({days_back}일 전)")
                        break
                except Exception:
                    continue
            if today is None:
                today = datetime.now().strftime('%Y%m%d')
                print(f"영업일 탐색 실패, 오늘 날짜 사용: {today}")
            
            # area에 따른 처리
            if area in ["KRX", "KRX-DESC"]:
                # 전체 시장 데이터
                print("KOSPI 종목 목록 조회 중...")
                kospi_tickers = stock.get_market_ticker_list(today, market="KOSPI")
                print(f"KOSPI 종목 수: {len(kospi_tickers)}")
                
                print("KOSDAQ 종목 목록 조회 중...")
                kosdaq_tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
                print(f"KOSDAQ 종목 수: {len(kosdaq_tickers)}")
                
                stock_data = []
                
                # KOSPI 종목들 (전체)
                print("KOSPI 종목 이름 조회 중...")
                for i, ticker in enumerate(kospi_tickers):
                    try:
                        name = stock.get_market_ticker_name(ticker)
                        stock_data.append({
                            'Code': ticker,
                            'Name': name,
                            'Market': 'KOSPI',
                            'Sector': None,  # None으로 설정하여 기존 데이터 보존
                            'Industry': None  # None으로 설정하여 기존 데이터 보존
                        })
                        if (i + 1) % 100 == 0:  # 100개마다 진행 상황 출력
                            print(f"KOSPI 진행: {i + 1}/{len(kospi_tickers)}")
                    except Exception as e:
                        print(f"KOSPI 종목 {ticker} 처리 실패: {e}")
                        continue
                
                # KOSDAQ 종목들 (전체)
                print("KOSDAQ 종목 이름 조회 중...")
                for i, ticker in enumerate(kosdaq_tickers):
                    try:
                        name = stock.get_market_ticker_name(ticker)
                        stock_data.append({
                            'Code': ticker,
                            'Name': name,
                            'Market': 'KOSDAQ',
                            'Sector': None,  # None으로 설정하여 기존 데이터 보존
                            'Industry': None  # None으로 설정하여 기존 데이터 보존
                        })
                        if (i + 1) % 100 == 0:  # 100개마다 진행 상황 출력
                            print(f"KOSDAQ 진행: {i + 1}/{len(kosdaq_tickers)}")
                    except Exception as e:
                        print(f"KOSDAQ 종목 {ticker} 처리 실패: {e}")
                        continue
                
                print(f"전체 데이터 수집 완료: KOSPI {len(kospi_tickers)}개 + KOSDAQ {len(kosdaq_tickers)}개 = 총 {len(stock_data)}개")
                df = pd.DataFrame(stock_data)
                
            elif area == "KOSPI":
                print("KOSPI 종목 목록 조회 중...")
                tickers = stock.get_market_ticker_list(today, market="KOSPI")
                print(f"KOSPI 종목 수: {len(tickers)}")
                
                stock_data = []
                for i, ticker in enumerate(tickers):  # 전체 종목
                    try:
                        name = stock.get_market_ticker_name(ticker)
                        stock_data.append({
                            'Code': ticker,
                            'Name': name,
                            'Market': 'KOSPI',
                            'Sector': None,  # None으로 설정하여 기존 데이터 보존
                            'Industry': None  # None으로 설정하여 기존 데이터 보존
                        })
                        if (i + 1) % 100 == 0:
                            print(f"KOSPI 진행: {i + 1}/{len(tickers)}")
                    except Exception as e:
                        print(f"KOSPI 종목 {ticker} 처리 실패: {e}")
                        continue
                df = pd.DataFrame(stock_data)
                
            elif area == "KOSDAQ":
                print("KOSDAQ 종목 목록 조회 중...")
                tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
                print(f"KOSDAQ 종목 수: {len(tickers)}")
                
                stock_data = []
                for i, ticker in enumerate(tickers):  # 전체 종목
                    try:
                        name = stock.get_market_ticker_name(ticker)
                        stock_data.append({
                            'Code': ticker,
                            'Name': name,
                            'Market': 'KOSDAQ',
                            'Sector': None,  # None으로 설정하여 기존 데이터 보존
                            'Industry': None  # None으로 설정하여 기존 데이터 보존
                        })
                        if (i + 1) % 100 == 0:
                            print(f"KOSDAQ 진행: {i + 1}/{len(tickers)}")
                    except Exception as e:
                        print(f"KOSDAQ 종목 {ticker} 처리 실패: {e}")
                        continue
                
                df = pd.DataFrame(stock_data)
                
            else:
                print(f"지원하지 않는 area: {area}")
                return pd.DataFrame()
            
            print(f"pykrx 성공: 총 {len(df)}개 데이터 조회 완료 (sector/industry는 None으로 설정하여 기존 데이터 보존)")
            return df
            
        except Exception as pykrx_error:
            print(f"pykrx도 실패: {pykrx_error}")
            print("모든 데이터 소스 실패. 빈 DataFrame 반환")
            return pd.DataFrame()

def GetSnapDataReader():
    df = fdr.SnapDataReader('KRX/INDEX/LIST')
    return df

def GetSnapDataReader_IndexCode(code):
    # 코스피 대형주 종목 리스트를 가져옵니다.
    # KRX/INDEX/STOCK/1002 는 코스피 대형주 종목 리스트의 코드입니다.
    df = fdr.SnapDataReader(f'KRX/INDEX/STOCK/{code}') # 코스피 대형주 종목 리스트
    return df