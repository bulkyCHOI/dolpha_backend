"""
TradingSummary 재계산 커맨드

TradeEntry FK 연결이 누락된 레코드를 수정하고 win_rate를 재계산합니다.

max_profit_percent / max_drawdown은 매매 중 실시간 추적값이므로
이 커맨드에서는 수정하지 않습니다 (trading_engine._update_peak_stats 담당).
단, CLOSED 종목 중 값이 완전히 None인 경우에만 한해 청산가 기준으로 초기값을 세팅합니다.
"""

from django.core.management.base import BaseCommand
from myweb.models import TradingSummary, TradeEntry


class Command(BaseCommand):
    help = "TradingSummary의 TradeEntry FK 연결 및 win_rate 재계산"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="실제 저장 없이 결과만 출력",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        summaries = TradingSummary.objects.select_related("user").all()

        updated = 0
        for summary in summaries:
            # 1. TradeEntry FK 연결 (누락된 경우)
            linked_entries = TradeEntry.objects.filter(trading_summary=summary)
            if not linked_entries.exists():
                orphans = TradeEntry.objects.filter(
                    user=summary.user,
                    stock_code=summary.stock_code,
                )
                if orphans.exists():
                    if not dry_run:
                        orphans.update(trading_summary=summary)
                    self.stdout.write(
                        f"  [{summary.stock_name}] {orphans.count()}건 FK 연결"
                        + (" (dry-run)" if dry_run else "")
                    )
                    all_entries = orphans
                else:
                    self.stdout.write(
                        self.style.WARNING(f"  [{summary.stock_name}] 연결할 TradeEntry 없음")
                    )
                    continue
            else:
                all_entries = linked_entries

            sell_entries = all_entries.filter(trade_type="SELL", status="FILLED")
            exit_count = sell_entries.count()

            # 2. win_rate 재계산
            profitable_count = sum(
                1 for e in sell_entries
                if e.profit_loss is not None and float(e.profit_loss) > 0
            )
            new_win_rate = profitable_count / exit_count * 100.0 if exit_count else 0.0

            # 3. max_profit_percent / max_drawdown:
            #    실시간 추적값이므로 기본적으로 건드리지 않음.
            #    단, 값이 완전히 None인 CLOSED 종목에 한해 청산가 기준으로 채워줌 (소급 초기화).
            sell_pcts = [
                float(e.profit_loss_percent)
                for e in sell_entries
                if e.profit_loss_percent is not None
            ]
            new_max_profit  = summary.max_profit_percent
            new_max_drawdown = summary.max_drawdown

            if summary.final_status == "CLOSED" and sell_pcts:
                # 청산가 기준 최고/최저 (실시간 추적 대체값)
                peak_at_sell = max(sell_pcts)
                trough_at_sell = min(sell_pcts)

                if summary.max_profit_percent is None:
                    # 최고 청산 수익률을 최고수익률 초기값으로 설정
                    new_max_profit = peak_at_sell if peak_at_sell > 0 else None

                if summary.max_drawdown is None:
                    # 낙폭 = 최저청산수익률 - 최고수익률 (음수 저장)
                    peak = new_max_profit or peak_at_sell
                    drawdown = trough_at_sell - peak
                    new_max_drawdown = drawdown if drawdown < 0 else None

            update_fields = []
            if abs((summary.win_rate or 0) - new_win_rate) > 0.01:
                summary.win_rate = new_win_rate
                update_fields.append("win_rate")
            if summary.max_profit_percent != new_max_profit:
                summary.max_profit_percent = new_max_profit
                update_fields.append("max_profit_percent")
            if summary.max_drawdown != new_max_drawdown:
                summary.max_drawdown = new_max_drawdown
                update_fields.append("max_drawdown")

            self.stdout.write(
                f"[{summary.stock_name}({summary.final_status})] "
                f"win_rate={new_win_rate:.1f}%, "
                f"max_profit={new_max_profit}, "
                f"max_drawdown={new_max_drawdown}"
                + (" → 변경없음" if not update_fields else f" → {', '.join(update_fields)} 업데이트")
                + (" (dry-run)" if dry_run else "")
            )

            if update_fields and not dry_run:
                summary.save(update_fields=update_fields)
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"\n완료: {updated}건 업데이트"
                + (" (dry-run 모드)" if dry_run else "")
            )
        )
