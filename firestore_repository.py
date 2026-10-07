"""Firestore access for adult ISP generation data."""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter


ALLOWED_SERVICE_NAMES = {"雲端發展室", "技能養成第一中心"}

ADULT_PROFILE_FIELD_BY_DOMAIN = {
    "身體福祉": "physicalWellBeing",
    "情緒福祉": "emotionalWellBeing",
    "物質福祉": "materialWellBeing",
    "個人發展": "personalDevelopment",
    "自我決策": "selfDetermination",
    "人際關係": "interpersonalRelations",
    "權利": "rights",
    "社會融合": "socialInclusion",
}

ADULT_SUMMARY_SOURCES = {
    "身體福祉": [
        ("ispSuggestionId", "adult", "ispSuggestions", ("summary",), "nurses", "護理師", ""),
        ("nutritionistId", "therapy", "nutritionists", ("summary",), "nutritionists", "營養師", ""),
        (
            "physicalTherapyId",
            "therapy",
            "physicalTherapys",
            ("summary", "physicalFitness.summary"),
            "therapists",
            "物理治療師",
            "物理治療評估、體適能評估",
        ),
        (
            "speechTherapyId",
            "therapy",
            "speechTherapys",
            ("feeding.summary",),
            "therapists",
            "語言治療師",
            "吞嚥能力評估",
        ),
    ],
    "物質福祉": [
        (
            "occupationalTherapyId",
            "therapy",
            "occupationalTherapys",
            ("manipulation.summary",),
            "therapists",
            "職能治療師",
            "個人操作技能表現評估",
        ),
    ],
    "個人發展": [
        (
            "occupationalTherapyId",
            "therapy",
            "occupationalTherapys",
            ("independentLiving.summary",),
            "therapists",
            "職能治療師",
            "自立生活表現評估",
        ),
    ],
    "人際關係": [
        (
            "occupationalTherapyId",
            "therapy",
            "occupationalTherapys",
            ("social.summary",),
            "therapists",
            "職能治療師",
            "人際互動與活動表現評估",
        ),
        (
            "speechTherapyId",
            "therapy",
            "speechTherapys",
            ("communication.summary",),
            "therapists",
            "語言治療師",
            "語言治療評估",
        ),
    ],
}


class FirestoreDataError(RuntimeError):
    """Base error for Firestore data lookup failures."""


class EnrollmentNotFoundError(FirestoreDataError):
    """The student is not enrolled in an allowed service."""


class StudentNotFoundError(FirestoreDataError):
    """The student document does not exist."""


class IspNotFoundError(FirestoreDataError):
    """The adult ISP does not exist or was deleted."""


@dataclass(frozen=True)
class AdultIspContext:
    student_id: str
    isp_id: str
    student_name: str
    basic_info: str
    summary: str
    dream: str


@lru_cache(maxsize=1)
def get_firestore_client():
    try:
        app = firebase_admin.get_app()
    except ValueError:
        credential_path = os.getenv("FIREBASE_CREDENTIALS", "").strip()
        if credential_path:
            key_path = Path(credential_path).expanduser().resolve()
            if not key_path.is_file():
                raise FileNotFoundError("找不到 FIREBASE_CREDENTIALS 指定的金鑰。")
            app = firebase_admin.initialize_app(credentials.Certificate(key_path))
        else:
            # Cloud Run uses its service account through Application Default Credentials.
            app = firebase_admin.initialize_app()
    return firestore.client(app=app)


def html_to_plain_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(?:p|div|li|h[1-6])\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def format_disability_card_basic_info(disability_card) -> str:
    if not isinstance(disability_card, dict):
        disability_card = {}

    level = disability_card.get("level") or {}
    level_title = (
        str(level.get("title") or "未提供").strip()
        if isinstance(level, dict)
        else "未提供"
    )
    disability_types = disability_card.get("types") or {}
    if isinstance(disability_types, dict):
        type_items = list(disability_types.values())
    elif isinstance(disability_types, list):
        type_items = disability_types
    else:
        type_items = []

    valid_types = [
        item for item in type_items if isinstance(item, dict) and item.get("title")
    ]
    valid_types.sort(
        key=lambda item: (
            int(item["order"]) if str(item.get("order", "")).isdigit() else 999,
            str(item.get("title")),
        )
    )
    type_titles = list(
        dict.fromkeys(str(item["title"]).strip() for item in valid_types)
    )
    types_text = "、".join(type_titles) if type_titles else "未提供"
    return f"障礙等級：{level_title}\n障礙類別：{types_text}"


def get_nested_value(data: dict, field_path: str):
    value = data
    for key in field_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_summary_source_title(
    data: dict,
    people_field: str,
    role: str,
    assessment: str,
) -> str:
    people = data.get(people_field) or []
    names = []
    if isinstance(people, list):
        for person in people:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    title = f"{'、'.join(names)}{role}" if names else role
    return f"{title}（{assessment}）" if assessment else title


class FirestoreRepository:
    def __init__(self, database=None):
        self.database = database or get_firestore_client()

    def list_enrolled_students(self) -> list[dict[str, str]]:
        snapshots = (
            self.database.collection("cases")
            .where(filter=FieldFilter("status", "==", "enrollment"))
            .select(["service", "student"])
            .stream(timeout=20)
        )
        students_by_id = {}
        for snapshot in snapshots:
            case_data = snapshot.to_dict() or {}
            service = case_data.get("service") or {}
            student = case_data.get("student") or {}
            if not isinstance(service, dict) or not isinstance(student, dict):
                continue
            if service.get("name") not in ALLOWED_SERVICE_NAMES:
                continue
            student_id = str(student.get("id") or "").strip()
            student_name = str(student.get("name") or "").strip()
            if student_id and student_name:
                students_by_id[student_id] = student_name
        return [
            {"id": student_id, "name": name}
            for student_id, name in sorted(
                students_by_id.items(),
                key=lambda item: (item[1], item[0]),
            )
        ]

    def ensure_enrolled(self, student_id: str) -> None:
        snapshots = (
            self.database.collection("cases")
            .where(filter=FieldFilter("student.id", "==", student_id))
            .stream(timeout=20)
        )
        for snapshot in snapshots:
            case_data = snapshot.to_dict() or {}
            service = case_data.get("service") or {}
            if (
                case_data.get("status") == "enrollment"
                and isinstance(service, dict)
                and service.get("name") in ALLOWED_SERVICE_NAMES
            ):
                return
        raise EnrollmentNotFoundError("服務使用者不符合目前在案與服務單位條件。")

    def get_student(self, student_id: str) -> dict:
        self.ensure_enrolled(student_id)
        snapshot = (
            self.database.collection("students")
            .document(student_id)
            .get(timeout=20)
        )
        if not snapshot.exists:
            raise StudentNotFoundError("找不到服務使用者資料。")
        return snapshot.to_dict() or {}

    def list_adult_isps(self, student_id: str) -> tuple[str, list[dict[str, str]]]:
        student_data = self.get_student(student_id)
        snapshots = (
            self.database.collection("students")
            .document(student_id)
            .collection("forms")
            .document("adult")
            .collection("isp")
            .select(["friendlyName", "isDeleted"])
            .stream(timeout=20)
        )
        isps = []
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            if data.get("isDeleted") is True:
                continue
            isps.append({
                "id": snapshot.id,
                "name": str(data.get("friendlyName") or "").strip() or snapshot.id,
            })
        isps.sort(key=lambda item: (item["name"], item["id"]), reverse=True)
        basic_info = format_disability_card_basic_info(
            student_data.get("disabilityCard")
        )
        return basic_info, isps

    def load_adult_isp_context(
        self,
        student_id: str,
        isp_id: str,
        domain: str,
    ) -> AdultIspContext:
        student_data = self.get_student(student_id)
        student_name = str(student_data.get("name") or "").strip()
        basic_info = format_disability_card_basic_info(
            student_data.get("disabilityCard")
        )
        isp_snapshot = (
            self.database.collection("students")
            .document(student_id)
            .collection("forms")
            .document("adult")
            .collection("isp")
            .document(isp_id)
            .get(timeout=20)
        )
        if not isp_snapshot.exists:
            raise IspNotFoundError("找不到所選成人 ISP。")

        isp_data = isp_snapshot.to_dict() or {}
        if isp_data.get("isDeleted") is True:
            raise IspNotFoundError("所選成人 ISP 已刪除。")

        dream = html_to_plain_text(isp_data.get("dream"))
        sections = []
        profile_field = ADULT_PROFILE_FIELD_BY_DOMAIN.get(domain)
        personal_profile = isp_data.get("personalProfile") or {}
        profile_text = (
            html_to_plain_text(personal_profile.get(profile_field))
            if profile_field and isinstance(personal_profile, dict)
            else ""
        )
        if profile_text:
            sections.append(f"【教保員摘述】\n{profile_text}")

        reference = isp_data.get("reference") or {}
        if not isinstance(reference, dict):
            reference = {}
        for source in ADULT_SUMMARY_SOURCES.get(domain, []):
            (
                reference_key,
                directory,
                collection_name,
                fields,
                people_field,
                role,
                assessment,
            ) = source
            document_id = reference.get(reference_key)
            if not document_id:
                continue
            source_snapshot = (
                self.database.collection("students")
                .document(student_id)
                .collection("forms")
                .document(directory)
                .collection(collection_name)
                .document(str(document_id))
                .get(timeout=20)
            )
            if not source_snapshot.exists:
                continue
            source_data = source_snapshot.to_dict() or {}
            summary_parts = [
                html_to_plain_text(get_nested_value(source_data, field_path))
                for field_path in fields
            ]
            summary_parts = [part for part in summary_parts if part]
            if not summary_parts:
                continue
            title = build_summary_source_title(
                source_data,
                people_field,
                role,
                assessment,
            )
            sections.append(f"【{title}】\n" + "\n".join(summary_parts))

        return AdultIspContext(
            student_id=student_id,
            isp_id=isp_id,
            student_name=student_name,
            basic_info=basic_info,
            summary="\n\n".join(sections),
            dream=dream,
        )
