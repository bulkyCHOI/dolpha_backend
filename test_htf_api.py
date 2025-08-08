#!/usr/bin/env python
"""
HTF API 엔드포인트 테스트 스크립트
"""

import requests
import json
from datetime import datetime

# API 베이스 URL
BASE_URL = "http://localhost:8000/api"

def test_htf_api():
    """HTF API 테스트"""
    print("=== HTF API 엔드포인트 테스트 ===\n")
    
    # 1. HTF 종목 리스트 조회 테스트
    print("--- 1. HTF 종목 리스트 조회 테스트 ---")
    try:
        url = f"{BASE_URL}/htf-stocks/"
        params = {
            "min_gain": 50.0,  # 테스트를 위해 기준 낮춤
            "max_pullback": 30.0,
            "limit": 10
        }
        
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ HTF 종목 조회 성공")
            print(f"  - 총 종목 수: {data.get('total_count', 0)}")
            print(f"  - 상태: {data.get('status')}")
            
            if data.get('data'):
                print("  - 샘플 종목:")
                for i, stock in enumerate(data['data'][:3]):
                    print(f"    {i+1}. {stock['name']} ({stock['code']})")
                    print(f"       상승률: {stock['htf_8week_gain']}%, 조정폭: {stock['htf_max_pullback']}%")
        else:
            print(f"❌ 오류: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")
    
    # 2. HTF 상세 분석 테스트
    print(f"\n--- 2. HTF 상세 분석 테스트 ---")
    test_codes = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
    
    for code in test_codes:
        try:
            url = f"{BASE_URL}/htf-analysis/{code}/"
            response = requests.get(url)
            
            print(f"\n{code} 종목:")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 분석 성공")
                
                stock_info = data['data']['stock_info']
                htf_analysis = data['data']['htf_analysis']
                
                print(f"  - 종목명: {stock_info['name']}")
                print(f"  - 분석일: {htf_analysis['analysis_date']}")
                print(f"  - HTF 패턴: {'발견' if htf_analysis.get('htf_8week_gain', 0) > 0 else '미발견'}")
                print(f"  - 8주 상승률: {htf_analysis.get('htf_8week_gain', 0):.2f}%")
                print(f"  - 최대 조정폭: {htf_analysis.get('htf_max_pullback', 0):.2f}%")
                
            elif response.status_code == 400:
                error_data = response.json()
                print(f"⚠️  HTF 데이터 없음: {error_data.get('message')}")
            else:
                print(f"❌ 오류: {response.status_code}")
                print(f"응답: {response.text}")
                
        except Exception as e:
            print(f"❌ 예외 발생 ({code}): {str(e)}")
    
    # 3. HTF 패턴 계산 API 테스트
    print(f"\n--- 3. HTF 패턴 계산 API 테스트 ---")
    try:
        url = f"{BASE_URL}/calculate-htf-patterns/"
        data = {
            "stock_codes": ["005930", "000660"],  # 삼성전자, SK하이닉스만 테스트
            "min_gain_percent": 50.0,  # 테스트를 위해 기준 낮춤
            "max_pullback_percent": 30.0,
            "batch_size": 2
        }
        
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 패턴 계산 성공")
            
            calc_data = result['data']
            print(f"  - 총 종목: {calc_data['total']}")
            print(f"  - 성공: {calc_data['success']}")
            print(f"  - 실패: {calc_data['failed']}")
            print(f"  - 성공률: {calc_data['success_rate']}%")
            
            if calc_data.get('failed_stocks'):
                print(f"  - 실패 종목: {', '.join(calc_data['failed_stocks'])}")
        else:
            print(f"❌ 오류: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")
    
    # 4. API 파라미터 검증 테스트
    print(f"\n--- 4. API 파라미터 검증 테스트 ---")
    
    # 잘못된 파라미터 테스트
    test_cases = [
        {
            "name": "음수 상승률 테스트",
            "url": f"{BASE_URL}/htf-stocks/",
            "params": {"min_gain": -10}
        },
        {
            "name": "잘못된 종목 코드 테스트", 
            "url": f"{BASE_URL}/htf-analysis/XX/",
            "params": {}
        },
        {
            "name": "범위 초과 limit 테스트",
            "url": f"{BASE_URL}/htf-stocks/",
            "params": {"limit": 2000}
        }
    ]
    
    for test_case in test_cases:
        try:
            response = requests.get(test_case["url"], params=test_case["params"])
            print(f"\n{test_case['name']}:")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 400:
                error_data = response.json()
                print(f"✅ 올바른 검증: {error_data.get('message')}")
            else:
                print(f"⚠️  예상과 다른 결과: {response.text[:100]}...")
                
        except Exception as e:
            print(f"❌ 예외 발생 ({test_case['name']}): {str(e)}")
    
    print(f"\n=== API 테스트 완료 ===")

if __name__ == '__main__':
    test_htf_api()