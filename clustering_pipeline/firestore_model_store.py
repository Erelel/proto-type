from __future__ import annotations

import firebase_admin
from firebase_admin import credentials, firestore

from constants import PIPELINE_DIR
from model_params_utils import normalize_model_params, serialize_model_params_for_firestore

SYSTEM_PARAMS_COLLECTION = "system_parameters"
SYSTEM_PARAMS_DOC = "kmeans_v1"
SERVICE_ACCOUNT_PATH = PIPELINE_DIR / "serviceAccountKey.json"


def _get_firestore_client():
    if not firebase_admin._apps:
        if not SERVICE_ACCOUNT_PATH.exists():
            raise FileNotFoundError(
                f"Firebase service account key not found: {SERVICE_ACCOUNT_PATH}"
            )
        cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def load_model_params_from_db() -> dict:
    db = _get_firestore_client()
    snapshot = (
        db.collection(SYSTEM_PARAMS_COLLECTION)
        .document(SYSTEM_PARAMS_DOC)
        .get()
    )
    if not snapshot.exists:
        raise ValueError(
            "Model params document not found: "
            f"{SYSTEM_PARAMS_COLLECTION}/{SYSTEM_PARAMS_DOC}"
        )

    params = snapshot.to_dict()
    if not isinstance(params, dict):
        raise ValueError("Model params document is empty or invalid.")

    return normalize_model_params(params)


def save_model_params_to_db(params: dict) -> None:
    db = _get_firestore_client()
    payload = serialize_model_params_for_firestore(params)
    (
        db.collection(SYSTEM_PARAMS_COLLECTION)
        .document(SYSTEM_PARAMS_DOC)
        .set(payload)
    )
