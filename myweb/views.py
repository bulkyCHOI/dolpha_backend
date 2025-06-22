from django.shortcuts import render
from django.http import JsonResponse
import json

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