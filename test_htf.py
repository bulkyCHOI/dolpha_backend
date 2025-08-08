#!/usr/bin/env python
"""
HTF 패턴 분석기 테스트 스크립트
"""

import os
import sys
import django

# Django 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dolpha.settings')
django.setup()

from dolpha.htf_analyzer import HTFPatternAnalyzer, get_htf_stocks
from myweb.models import Company

def test_htf_analyzer():
    """HTF 분석기 테스트"""
    print("=== HTF 패턴 분석기 테스트 ===\n")
    
    # 1. 분석기 초기화
    analyzer = HTFPatternAnalyzer()
    print("✅ HTF 분석기 초기화 완료")
    
    # 2. 샘플 종목으로 테스트 (삼성전자)
    test_codes = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
    
    for code in test_codes:
        print(f"\n--- {code} 종목 HTF 패턴 분석 ---")
        try:
            result = analyzer.calculate_htf_pattern(code)
            
            if 'error' in result:
                print(f"❌ 오류: {result['error']}")
            else:
                print(f"✅ 분석 완료:")
                print(f"  - 패턴 발견: {result['pattern_detected']}")
                print(f"  - 8주 상승률: {result.get('htf_8week_gain', 0):.2f}%")
                print(f"  - 최대 조정폭: {result.get('htf_max_pullback', 0):.2f}%")
                print(f"  - 현재 상태: {result.get('htf_current_status', 'none')}")
                
        except Exception as e:
            print(f"❌ 예외 발생: {str(e)}")
    
    # 3. 배치 처리 테스트 (소규모)
    print(f"\n--- 배치 처리 테스트 (3개 종목) ---")
    try:
        result = analyzer.batch_calculate_htf_patterns(
            stock_codes=test_codes,
            batch_size=3
        )
        
        if 'error' in result:
            print(f"❌ 배치 처리 오류: {result['error']}")
        else:
            print(f"✅ 배치 처리 완료:")
            print(f"  - 총 종목: {result['total']}")
            print(f"  - 성공: {result['success']}")
            print(f"  - 실패: {result['failed']}")
            print(f"  - 성공률: {result['success_rate']}%")
            
    except Exception as e:
        print(f"❌ 배치 처리 예외: {str(e)}")
    
    # 4. HTF 종목 조회 테스트
    print(f"\n--- HTF 종목 조회 테스트 ---")
    try:
        htf_stocks = get_htf_stocks(min_gain=50.0, max_pullback=30.0, limit=10)
        print(f"✅ HTF 종목 조회 완료: {len(htf_stocks)}개 종목")
        
        for i, stock in enumerate(htf_stocks[:3]):  # 상위 3개만 출력
            print(f"  {i+1}. {stock['name']} ({stock['code']}) - 상승률: {stock['htf_8week_gain']:.2f}%")
            
    except Exception as e:
        print(f"❌ HTF 종목 조회 예외: {str(e)}")
    
    print(f"\n=== 테스트 완료 ===")

if __name__ == '__main__':
    test_htf_analyzer()