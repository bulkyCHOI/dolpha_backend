from django.apps import AppConfig
from django.conf import settings

class MywebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myweb'

    def ready(self):
        if settings.SCHEDULER_DEFAULT:
            from .tasks import start
            start()