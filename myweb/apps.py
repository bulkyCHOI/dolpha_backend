from django.apps import AppConfig
from django.conf import settings
import platform

class MywebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myweb'

    def ready(self):
        # 스케줄러는 Linux(Ubuntu)에서만 실행하고, 맥/윈도우에서는 실행하지 않음
        current_os = platform.system()
        is_linux = current_os == 'Linux'
        
        if settings.SCHEDULER_DEFAULT and is_linux:
            from .tasks import start
            start()
            print(f"Scheduler started on {current_os}")
        else:
            print(f"Scheduler disabled on {current_os} (only runs on Linux)")