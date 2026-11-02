import os
import unittest
from unittest.mock import patch

os.environ.setdefault("FIRESTORE_PROJECT_ID", "mumu-care")
os.environ["APP_ENV"] = "production"
os.environ["OPENAI_API_KEY"] = "test-only-not-a-real-key"
os.environ["CORS_ALLOW_ORIGINS"] = "https://mumu-care.web.app,https://mumu-care.firebaseapp.com"

from firebase_admin import auth
from fastapi.testclient import TestClient

import cloud_run_backend as api
from employee_auth import (
    AuthenticatedEmployee, EmployeeAccessDenied, RoleMembership, authenticate_employee,
)
from firestore_repository import EnrollmentNotFoundError, FirestoreRepository


class Snapshot:
    def __init__(self, data=None):
        self.data = data
        self.exists = data is not None

    def to_dict(self):
        return self.data


class Query:
    def __init__(self, items):
        self.items = items

    def document(self, uid):
        return self

    def get(self, **kwargs):
        return Snapshot(self.items[0] if self.items else None)

    def where(self, *, filter):
        def value_at(item):
            value = item
            for key in filter.field_path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            return value

        if filter.op_string == "array_contains":
            items = [item for item in self.items if filter.value in (value_at(item) or [])]
        else:
            items = [item for item in self.items if value_at(item) == filter.value]
        return Query(items)

    def select(self, fields):
        return self

    def stream(self, **kwargs):
        return iter(Snapshot(item) for item in self.items)


class Database:
    def __init__(self, user=None, employees=None, roles=None, cases=None):
        self.collections = {
            "users": [user] if user is not None else [],
            "employees": employees or [], "roles": roles or [], "cases": cases or [],
        }

    def collection(self, name):
        return Query(self.collections[name])


def employee(position="educareGiver", services=("service-a",)):
    return AuthenticatedEmployee("test-uid", "01615", (
        RoleMembership(position, frozenset(services)),
    ))


class EmployeeAuthTests(unittest.TestCase):
    def setUp(self):
        self.claims = {
            "uid": "test-uid", "email": "m1615@maria.org.tw", "email_verified": True,
            "firebase": {"sign_in_provider": "google.com"},
        }
        self.user = {"code": "01615"}
        self.employees = [{"employeeId": "01615", "jobStatus": 1}]
        self.roles = [{"userCodes": ["01615"], "position": "educareGiver", "serviceIds": ["service-a"]}]

    def authenticate(self):
        database = Database(self.user, self.employees, self.roles)
        with patch("employee_auth.get_auth_app", return_value="auth-app"), patch(
            "employee_auth.auth.verify_id_token", return_value=self.claims,
        ) as verify:
            result = authenticate_employee("test-token", database)
            verify.assert_called_once_with("test-token", app="auth-app", check_revoked=True)
            return result

    def test_verified_active_employee_and_scoped_roles(self):
        result = self.authenticate()
        self.assertEqual(result.code, "01615")
        self.assertEqual(result.service_scope("情緒福祉"), frozenset({"service-a"}))

    def test_identity_and_employee_denials(self):
        for change in (
            {"email": "m1615@example.com"}, {"email_verified": False},
            {"firebase": {"sign_in_provider": "password"}}, {"uid": ""},
        ):
            with self.subTest(change=change):
                original = self.claims
                self.claims = {**original, **change}
                with self.assertRaises(EmployeeAccessDenied):
                    self.authenticate()
                self.claims = original
        for user in (None, {"code": "another-admin"}, {"code": "01615", "disabled": True}):
            with self.subTest(user=user):
                self.user = user
                with self.assertRaises(EmployeeAccessDenied):
                    self.authenticate()
        self.user = {"code": "01615"}
        self.employees[0]["jobStatus"] = 0
        with self.assertRaises(EmployeeAccessDenied):
            self.authenticate()

    def test_no_eligible_roles_is_denied(self):
        self.roles[0]["position"] = "socialWorker"
        with self.assertRaises(EmployeeAccessDenied):
            self.authenticate()

    def test_area_permissions_and_scope_cannot_be_combined_across_roles(self):
        nurse = employee("nurse")
        self.assertEqual(nurse.service_scope("身體福祉"), frozenset({"service-a"}))
        with self.assertRaises(EmployeeAccessDenied):
            nurse.service_scope("情緒福祉")
        combined = AuthenticatedEmployee("uid", "code", (
            RoleMembership("socialWorker", frozenset({"service-b"})),
            RoleMembership("educareGiver", frozenset({"service-a"})),
        ))
        self.assertEqual(combined.service_scope("情緒福祉"), frozenset({"service-a"}))
        self.assertIsNone(employee("admin", ()).service_scope("情緒福祉"))
        self.assertIsNone(employee(services=("_ALL",)).service_scope("情緒福祉"))

    def test_service_list_is_filtered_without_leaking_other_services(self):
        database = Database(cases=[
            {"status": "enrollment", "service": {"id": "service-a", "name": "雲端發展室"},
             "student": {"id": "student-a", "name": "Example A"}},
            {"status": "enrollment", "service": {"id": "service-b", "name": "技能養成第一中心"},
             "student": {"id": "student-b", "name": "Example B"}},
            {"status": "closed", "service": {"id": "service-a", "name": "雲端發展室"},
             "student": {"id": "student-c", "name": "Example C"}},
        ])
        repository = FirestoreRepository(database, allowed_service_ids=frozenset({"service-a"}))
        self.assertEqual(repository.list_enrolled_students(), [{"id": "student-a", "name": "Example A"}])
        self.assertFalse(repository.service_is_allowed({"id": "service-b", "name": "技能養成第一中心"}))

    def test_enrollment_authorization_is_bound_to_the_requested_student(self):
        database = Database(cases=[
            {"status": "enrollment", "service": {"id": "service-a", "name": "雲端發展室"},
             "student": {"id": "student-a"}},
            {"status": "enrollment", "service": {"id": "service-b", "name": "技能養成第一中心"},
             "student": {"id": "student-b"}},
            {"status": "closed", "service": {"id": "service-a", "name": "雲端發展室"},
             "student": {"id": "student-c"}},
        ])
        repository = FirestoreRepository(database, allowed_service_ids=frozenset({"service-a"}))
        repository.ensure_enrolled("student-a")
        for student_id in ("student-b", "student-c", "missing-student"):
            with self.subTest(student_id=student_id), self.assertRaises(EnrollmentNotFoundError):
                repository.ensure_enrolled(student_id)


class ApiAuthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.payload = {"studentId": "student-a", "ispId": "isp-a", "domain": "情緒福祉"}
        with api._rate_limit_lock:
            api._rate_limit_events.clear()

    def test_all_business_routes_require_bearer_and_reject_legacy_key(self):
        requests = [("get", "/v1/service-users", None)] + [
            ("post", path, self.payload) for path in (
                "/v1/isps", "/v1/context", "/v1/long-term-goals", "/v1/references/search",
                "/v1/goals/generate", "/v1/goals/refine",
            )
        ]
        for method, path, payload in requests:
            with self.subTest(path=path), patch(
                "cloud_run_backend.authenticate_employee",
            ) as authenticate:
                response = self.client.request(method, path, json=payload, headers={"X-API-Key": "old-key"})
                self.assertEqual(response.status_code, 401)
                authenticate.assert_not_called()

    def test_invalid_revoked_and_disabled_tokens_are_401(self):
        for error in (auth.InvalidIdTokenError("invalid"), auth.RevokedIdTokenError("revoked"), auth.UserDisabledError("disabled")):
            with self.subTest(error=error), patch("cloud_run_backend.authenticate_employee", side_effect=error):
                response = self.client.get("/v1/service-users", headers={"Authorization": "Bearer bad-token"})
                self.assertEqual(response.status_code, 401)

    def test_employee_denial_is_403_but_verification_failure_is_503(self):
        for error, status in ((EmployeeAccessDenied("denied"), 403), (RuntimeError("IAM unavailable"), 503)):
            with self.subTest(status=status), patch("cloud_run_backend.authenticate_employee", side_effect=error):
                response = self.client.get("/v1/service-users", headers={"Authorization": "Bearer token"})
                self.assertEqual(response.status_code, status)
                self.assertIn("request_id", response.json())

    def test_out_of_scope_request_is_denied_before_openai(self):
        with patch("cloud_run_backend.authenticate_employee", return_value=employee("nurse")), patch(
            "cloud_run_backend.core.api_long_term_goals",
        ) as generate:
            response = self.client.post("/v1/long-term-goals", json=self.payload, headers={"Authorization": "Bearer token"})
            self.assertEqual(response.status_code, 403)
            generate.assert_not_called()

    def test_other_student_is_denied_before_openai(self):
        with patch("cloud_run_backend.authenticate_employee", return_value=employee()), patch(
            "cloud_run_backend.FirestoreRepository",
        ) as repository, patch("cloud_run_backend.core.api_long_term_goals") as generate:
            repository.return_value.load_adult_isp_context.side_effect = EnrollmentNotFoundError("not allowed")
            response = self.client.post("/v1/long-term-goals", json=self.payload, headers={"Authorization": "Bearer token"})
            self.assertEqual(response.status_code, 403)
            repository.assert_called_once_with(allowed_service_ids=frozenset({"service-a"}))
            generate.assert_not_called()

    def test_browser_preflight_does_not_require_login(self):
        response = self.client.options("/v1/long-term-goals", headers={
            "Origin": "https://mumu-care.web.app", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://mumu-care.web.app")
        denied = self.client.options("/v1/long-term-goals", headers={
            "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
        })
        self.assertEqual(denied.status_code, 400)


if __name__ == "__main__":
    unittest.main()
