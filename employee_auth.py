"""Firebase employee authentication and adult ISP service/area authorization."""

from dataclasses import dataclass
from functools import lru_cache
import os

import firebase_admin
from firebase_admin import auth
from google.cloud.firestore_v1.base_query import FieldFilter

from firestore_repository import get_firebase_app, get_firestore_client


COMMON_POSITIONS = frozenset({
    "manager", "educareGiver", "physicalTherapy", "occupationalTherapy", "speechTherapy",
})
PHYSICAL_POSITIONS = COMMON_POSITIONS | {"nurse", "nutritionist"}
AUTH_PROJECT_ID = os.getenv("FIREBASE_AUTH_PROJECT_ID", "mumu-care").strip()
EMPLOYEE_EMAIL_DOMAIN = os.getenv("EMPLOYEE_EMAIL_DOMAIN", "maria.org.tw").strip().lower()


class EmployeeAccessDenied(Exception):
    pass


@dataclass(frozen=True)
class RoleMembership:
    position: str
    service_ids: frozenset[str]


@dataclass(frozen=True)
class AuthenticatedEmployee:
    uid: str
    code: str
    roles: tuple[RoleMembership, ...]

    def service_scope(self, domain: str | None = None) -> frozenset[str] | None:
        if any(role.position == "admin" for role in self.roles):
            return None
        positions = PHYSICAL_POSITIONS if domain in {None, "身體福祉"} else COMMON_POSITIONS
        service_ids = frozenset(
            service_id
            for role in self.roles if role.position in positions
            for service_id in role.service_ids
        )
        if "_ALL" in service_ids:
            return None
        if not service_ids:
            raise EmployeeAccessDenied("沒有此領域或服務單位的操作權限。")
        return service_ids


@lru_cache(maxsize=1)
def get_auth_app():
    name = "mumucare-employee-auth"
    try:
        return firebase_admin.get_app(name)
    except ValueError:
        return firebase_admin.initialize_app(
            get_firebase_app().credential,
            options={"projectId": AUTH_PROJECT_ID},
            name=name,
        )


def authenticate_employee(token: str, database=None) -> AuthenticatedEmployee:
    claims = auth.verify_id_token(token, app=get_auth_app(), check_revoked=True)
    email = str(claims.get("email") or "").strip().lower()
    uid = str(claims.get("uid") or "")
    local_part, separator, domain = email.partition("@")
    if (
        not uid or not local_part or separator != "@"
        or domain != EMPLOYEE_EMAIL_DOMAIN
        or claims.get("email_verified") is not True
        or claims.get("firebase", {}).get("sign_in_provider") != "google.com"
    ):
        raise EmployeeAccessDenied("請使用已驗證的公司 Google 員工帳號登入。")

    database = database or get_firestore_client()
    user_snapshot = database.collection("users").document(uid).get(timeout=20)
    user = user_snapshot.to_dict() or {}
    # Derive the code from the verified email, rather than trusting an editable profile.
    code = local_part.replace("m", "0", 1).rjust(5, "0")
    if (
        not user_snapshot.exists or user.get("code") != code
        or user.get("isDeleted") is True or user.get("disabled") is True
        or user.get("isDisabled") is True
    ):
        raise EmployeeAccessDenied("員工帳號資料不存在、已停用或與登入身分不符。")

    employees = database.collection("employees").where(
        filter=FieldFilter("employeeId", "==", code)
    ).stream(timeout=20)
    if not any(
        (data := snapshot.to_dict() or {}).get("jobStatus") == 1
        and data.get("isDeleted") is not True
        for snapshot in employees
    ):
        raise EmployeeAccessDenied("找不到有效的在職員工資料。")

    snapshots = database.collection("roles").where(
        filter=FieldFilter("userCodes", "array_contains", code)
    ).stream(timeout=20)
    roles = []
    for snapshot in snapshots:
        data = snapshot.to_dict() or {}
        if data.get("isDeleted") is True or data.get("disabled") is True:
            continue
        service_ids = data.get("serviceIds")
        roles.append(RoleMembership(
            position=str(data.get("position") or ""),
            service_ids=frozenset(
                item for item in service_ids if isinstance(item, str) and item
            ) if isinstance(service_ids, list) else frozenset(),
        ))
    employee = AuthenticatedEmployee(uid=uid, code=code, roles=tuple(roles))
    employee.service_scope()
    return employee
