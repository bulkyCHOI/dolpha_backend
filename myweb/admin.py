from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.utils import timezone

from .models import KisAccount, StrategyAccount, User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    """구글 로그인 사용자 승인 관리.

    is_approved가 False인 사용자는 로그인은 되지만 API를 쓸 수 없다
    (dolpha.api_auth.google_oauth_callback / api_mypage_ninja.get_authenticated_user 참고).
    """

    list_display = (
        "username", "email", "is_approved", "is_superuser", "is_staff", "date_joined",
    )
    list_filter = ("is_approved", "is_staff", "is_superuser")
    search_fields = ("username", "email")
    actions = ["approve_users", "revoke_approval"]

    fieldsets = DjangoUserAdmin.fieldsets + (
        ("서비스 승인", {"fields": ("is_approved", "approved_at", "google_id", "profile_picture")}),
    )

    @admin.action(description="선택한 사용자 승인")
    def approve_users(self, request, queryset):
        updated = queryset.filter(is_approved=False).update(
            is_approved=True, approved_at=timezone.now()
        )
        self.message_user(request, f"{updated}명을 승인했습니다.")

    @admin.action(description="선택한 사용자 승인 취소")
    def revoke_approval(self, request, queryset):
        updated = queryset.update(is_approved=False, approved_at=None)
        self.message_user(request, f"{updated}명의 승인을 취소했습니다.")


@admin.register(KisAccount)
class KisAccountAdmin(admin.ModelAdmin):
    """앱키·시크릿은 암호화되어 있어 관리자 화면에도 노출하지 않는다."""

    list_display = ("user", "name", "account_type", "is_default", "is_active", "updated_at")
    list_filter = ("account_type", "is_default", "is_active")
    search_fields = ("user__username", "user__email", "name", "account_no")
    readonly_fields = ("created_at", "updated_at")
    exclude = ("encrypted_app_key", "encrypted_app_secret")


@admin.register(StrategyAccount)
class StrategyAccountAdmin(admin.ModelAdmin):
    list_display = ("user", "strategy_type", "account", "updated_at")
    list_filter = ("strategy_type",)
    search_fields = ("user__username", "user__email")
