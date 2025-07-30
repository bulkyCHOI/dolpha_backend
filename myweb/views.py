from django.shortcuts import render
from django.http import JsonResponse
import json
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd

def stock_chart_view(request):
    return render(request, 'stock_chart.html')

def stock_data(request):
    # 예제 주식 데이터 (날짜, 시가, 고가, 저가, 종가, 거래량)
    data = [
        {"date": "2025-06-01", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000},
        {"date": "2025-06-02", "open": 105, "high": 115, "low": 100, "close": 110, "volume": 1200},
        {"date": "2025-06-03", "open": 110, "high": 120, "low": 105, "close": 115, "volume": 1500},
        {"date": "2025-06-04", "open": 115, "high": 125, "low": 110, "close": 120, "volume": 1300},
        {"date": "2025-06-05", "open": 120, "high": 130, "low": 115, "close": 125, "volume": 1400},
    ]
    return JsonResponse(data, safe=False)

def market_indices(request):
    """FinanceDataReader를 사용해서 주요 지수 정보를 가져오는 API"""
    try:
        # 지수 코드 매핑
        indices = {
            'KOSPI': 'KS11',      # 코스피 지수
            'KOSDAQ': 'KQ11',     # 코스닥 지수
            'NASDAQ': 'IXIC'      # 나스닥 종합지수
        }
        
        # 지수별 상세 정보
        index_info = {
            'KOSPI': {
                'name': '코스피',
                'description': '국내 대표 주가지수',
                'marketCap': '2,134조원',
                'volume': '8,456억원',
                'currency': 'KRW'
            },
            'KOSDAQ': {
                'name': '코스닥',
                'description': '중소기업 전용 주가지수',
                'marketCap': '287조원',
                'volume': '6,789억원',
                'currency': 'KRW'
            },
            'NASDAQ': {
                'name': '나스닥',
                'description': '미국 기술주 중심 지수',
                'marketCap': '$21.2T',
                'volume': '$145.6B',
                'currency': 'USD'
            }
        }
        
        result = []
        
        # 각 지수별로 데이터 가져오기
        for display_name, symbol in indices.items():
            try:
                print(f"Fetching data for {display_name} ({symbol})...")
                
                # 최근 30일 데이터 가져오기
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                df = fdr.DataReader(symbol, start_date, end_date)
                
                if df.empty:
                    raise Exception(f"No data available for {symbol}")
                
                # 최신 데이터 (오늘)
                latest = df.iloc[-1]
                
                # 전일 데이터 (변동률 계산용)
                if len(df) > 1:
                    previous = df.iloc[-2]
                    previous_close = previous['Close']
                else:
                    previous_close = latest['Close']
                
                current_price = float(latest['Close'])
                high = float(latest['High'])
                low = float(latest['Low'])
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
                
                # 차트 데이터 생성 (최근 30일)
                chart_data = []
                for idx, row in df.iterrows():
                    chart_data.append({
                        'date': idx.strftime('%m월 %d일'),
                        'price': round(float(row['Close']), 2),
                        'fullDate': idx.strftime('%Y-%m-%d')
                    })
                
                info = index_info[display_name]
                
                result.append({
                    'id': len(result) + 1,
                    'name': info['name'],
                    'code': display_name,
                    'currentPrice': round(current_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'change': round(change, 2),
                    'changePercent': round(change_percent, 2),
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'description': info['description'],
                    'marketCap': info['marketCap'],
                    'volume': info['volume'],
                    'currency': info['currency'],
                    'chartData': chart_data
                })
                
                print(f"Successfully fetched data for {display_name}")
                
            except Exception as e:
                print(f"Error fetching data for {display_name}: {e}")
                
                # 실패 시 더미 데이터 (모두 0으로 설정)
                info = index_info[display_name]
                
                result.append({
                    'id': len(result) + 1,
                    'name': info['name'],
                    'code': display_name,
                    'currentPrice': 0,
                    'high': 0,
                    'low': 0,
                    'change': 0,
                    'changePercent': 0,
                    'time': 'API 오류 - 더미 데이터',
                    'description': info['description'],
                    'marketCap': info['marketCap'],
                    'volume': info['volume'],
                    'currency': info['currency'],
                    'chartData': []
                })
        
        return JsonResponse(result, safe=False)
        
    except Exception as e:
        print(f"General error in market_indices: {e}")
        
        # 전체 실패 시 모든 더미 데이터 반환
        dummy_data = [
            {
                'id': 1,
                'name': '코스피',
                'code': 'KOSPI',
                'currentPrice': 0,
                'high': 0,
                'low': 0,
                'change': 0,
                'changePercent': 0,
                'time': 'API 서비스 불가 - 더미 데이터',
                'description': '국내 대표 주가지수',
                'marketCap': '2,134조원',
                'volume': '8,456억원',
                'currency': 'KRW',
                'chartData': []
            },
            {
                'id': 2,
                'name': '코스닥',
                'code': 'KOSDAQ',
                'currentPrice': 0,
                'high': 0,
                'low': 0,
                'change': 0,
                'changePercent': 0,
                'time': 'API 서비스 불가 - 더미 데이터',
                'description': '중소기업 전용 주가지수',
                'marketCap': '287조원',
                'volume': '6,789억원',
                'currency': 'KRW',
                'chartData': []
            },
            {
                'id': 3,
                'name': '나스닥',
                'code': 'NASDAQ',
                'currentPrice': 0,
                'high': 0,
                'low': 0,
                'change': 0,
                'changePercent': 0,
                'time': 'API 서비스 불가 - 더미 데이터',
                'description': '미국 기술주 중심 지수',
                'marketCap': '$21.2T',
                'volume': '$145.6B',
                'currency': 'USD',
                'chartData': []
            }
        ]
        
        return JsonResponse(dummy_data, safe=False)