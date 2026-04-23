from django.apps import AppConfig
from django.conf import settings

class MywebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myweb'

    def ready(self):
        import os
        # runserver의 auto-reloader가 띄우는 두 번째 프로세스에서는 실행하지 않음
        if settings.SCHEDULER_DEFAULT and os.environ.get("RUN_MAIN") == "true":
            from .tasks import start
            start()
            print("Scheduler started")