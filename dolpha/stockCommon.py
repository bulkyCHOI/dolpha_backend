import FinanceDataReader as fdr
import pandas_datareader.data as web
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
#OHLCV 값을 가져옴!!
def GetOhlcv(area, stock_code, limit = 500, adj_ok = "1"):

    Adjlimit = limit * 1.7 #주말을 감안하면 5개를 가져오려면 적어도 7개는 뒤져야 된다. 1.4가 이상적이지만 혹시 모를 연속 공휴일 있을지 모르므로 1.7로 보정해준다

    df = None

    except_riase = False

    try:

        try:
            # print("----First try----")
            df = GetOhlcv1(area,stock_code,Adjlimit,adj_ok)
            
        except Exception as e:
            # print("")
                
            if df is None or len(df) == 0:

                except_riase = False
                try:
                    # print("----Second try----")
                    df = GetOhlcv2(area,stock_code,Adjlimit,adj_ok)

                    if df is None or len(df) == 0:
                        except_riase = True
                    
                except Exception as e:
                    except_riase = True
                    
    except Exception as e:
        print(e)
        except_riase = True
    

    if except_riase == True:
        return df
    else:
        # print("---", limit)
        return df[-limit:]





#한국 주식은 KRX 정보데이터시스템에서 가져온다. 그런데 미국주식 크롤링의 경우 investing.com 에서 가져오는데 안전하게 2초 정도 쉬어야 한다!
# https://financedata.github.io/posts/finance-data-reader-users-guide.html
def GetOhlcv1(area, stock_code, limit = 500, adj_ok = "1"):
    startDate = GetFromNowDateStr(area,"BAR",-limit)
    endDate = GetNowDateStr(area,"BAR")
    # df = fdr.DataReader(stock_code, startDate, endDate, exchange=area)
    df = fdr.DataReader(stock_code, startDate, endDate)
    if adj_ok == "1":
        
        try :
            df = df[[ 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
        except Exception:
            df = df[[ 'Open', 'High', 'Low', 'Close', 'Volume']]

    else:
        df = df[[ 'Open', 'High', 'Low', 'Close', 'Volume']]



    df.columns = [ 'open', 'high', 'low', 'close', 'volume']
    df.index.name = "Date"

    #거래량과 시가,종가,저가,고가의 평균을 곱해 대략의 거래대금을 구해서 value 라는 항목에 넣는다 ㅎ
    df.insert(5,'value',((df['open'] + df['high'] + df['low'] + df['close'])/4.0) * df['volume'])


    df.insert(6,'change',(df['close'] - df['close'].shift(1)) / df['close'].shift(1))

    df[[ 'open', 'high', 'low', 'close', 'volume', 'change']] = df[[ 'open', 'high', 'low', 'close', 'volume', 'change']].apply(pd.to_numeric)

    #미국주식은 2초를 쉬어주자! 안그러면 24시간 정지당할 수 있다!
    if area == "US":
        time.sleep(2.0)
    else:
        time.sleep(0.2)



    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')


    return df




#https://blog.naver.com/zacra/222986007794
def GetOhlcv2(area, stock_code, limit = 500, adj_ok = "1"):

    df = None

    if area == "KRX":

        df = web.DataReader(stock_code, "naver", GetFromNowDateStr(area,"BAR",-limit),GetNowDateStr(area,"BAR"))


    else:
        df = yfinance.download(stock_code, period='max')

    if adj_ok == "1":
            
        try :
            df = df[[ 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
        except Exception:
            df = df[[ 'Open', 'High', 'Low', 'Close', 'Volume']]

    else:
        df = df[[ 'Open', 'High', 'Low', 'Close', 'Volume']]

    
    df.columns = [ 'open', 'high', 'low', 'close', 'volume']
    df = df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float})
    df.index.name = "Date"


    #거래량과 시가,종가,저가,고가의 평균을 곱해 대략의 거래대금을 구해서 value 라는 항목에 넣는다 ㅎ
    df.insert(5,'value',((df['open'] + df['high'] + df['low'] + df['close'])/4.0) * df['volume'])
    df.insert(6,'change',(df['close'] - df['close'].shift(1)) / df['close'].shift(1))

    df[[ 'open', 'high', 'low', 'close', 'volume', 'change']] = df[[ 'open', 'high', 'low', 'close', 'volume', 'change']].apply(pd.to_numeric)


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
            
            # 기본 오늘 날짜
            today = datetime.now().strftime('%Y%m%d')
            
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