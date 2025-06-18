import FinanceDataReader as fdr
import pandas_datareader.data as web
import yfinance

from datetime import datetime, timedelta
from pytz import timezone
import time
import pandas as pd



############################################################################################################################################################
#한국인지 미국인지 구분해 현재 날짜정보를 리턴해 줍니다!
def GetNowDateStr(area = "KR", type= "NONE" ):
    timezone_info = timezone('Asia/Seoul')
    if area == "US":
        timezone_info = timezone('America/New_York')

    now = datetime.now(timezone_info)
    if type.upper() == "NONE":
        return now.strftime("%Y%m%d")
    else:
        return now.strftime("%Y-%m-%d")

#현재날짜에서 이전/이후 날짜를 구해서 리턴! (미래의 날짜를 구할 일은 없겠지만..)
def GetFromNowDateStr(area = "KR", type= "NONE" , days=100):
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
            print("----First try----")
            df = GetOhlcv1(area,stock_code,Adjlimit,adj_ok)
            
        except Exception as e:
            print("")
                
            if df is None or len(df) == 0:

                except_riase = False
                try:
                    print("----Second try----")
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
        print("---", limit)
        return df[-limit:]





#한국 주식은 KRX 정보데이터시스템에서 가져온다. 그런데 미국주식 크롤링의 경우 investing.com 에서 가져오는데 안전하게 2초 정도 쉬어야 한다!
# https://financedata.github.io/posts/finance-data-reader-users-guide.html
def GetOhlcv1(area, stock_code, limit = 500, adj_ok = "1"):
    startDate = GetFromNowDateStr(area,"BAR",-limit)
    endDate = GetNowDateStr(area,"BAR")
    df = fdr.DataReader(stock_code, startDate, endDate, exchange=area)
    print(df)
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

    if area == "KR":

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
    
    if area == "KR":
        df = fdr.StockListing('KRX')
    elif area == "KOSDAQ":
        df = fdr.StockListing('KOSDAQ')
    elif area == "KOSPI":
        df = fdr.StockListing('KOSPI')
    else:
        df = fdr.StockListing('NASDAQ')

    return df