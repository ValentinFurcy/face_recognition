from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import analyse, main

client = TestClient(main.app)


def jpeg():
    tampon = BytesIO()
    Image.new("RGB", (320, 240), "black").save(tampon, "JPEG")
    return tampon.getvalue()


def serie():
    return {p: (f"{p}.jpg", jpeg(), "image/jpeg") for p in analyse.POSES}


@pytest.fixture(autouse=True)
def vider_enrolements():
    main._enrolements.clear()


def test_page_accueil_servie():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "Vérification d'identité" in reponse.text


def test_pose_sans_visage():
    reponse = client.post("/api/pose", data={"pose": "face"}, files={"image": ("f.jpg", jpeg(), "image/jpeg")})
    assert reponse.status_code == 200
    assert reponse.json()["ok"] is False


def test_pose_inconnue_rejetee():
    reponse = client.post("/api/pose", data={"pose": "dessus"}, files={"image": ("f.jpg", jpeg(), "image/jpeg")})
    assert reponse.status_code == 422


def test_image_illisible_rejetee():
    reponse = client.post("/api/pose", data={"pose": "face"}, files={"image": ("f.jpg", b"xxx", "image/jpeg")})
    assert reponse.status_code == 400


def test_image_trop_grosse_rejetee(monkeypatch):
    monkeypatch.setattr(main, "TAILLE_MAX_IMAGE", 10)
    reponse = client.post("/api/pose", data={"pose": "face"}, files={"image": ("f.jpg", jpeg(), "image/jpeg")})
    assert reponse.status_code == 413


def test_enrolement_refuse_sans_visage():
    corps = client.post("/api/enrolement", files=serie()).json()
    assert corps["ok"] is False
    assert len(corps["erreurs"]) == 3
    assert main._enrolements == {}


def test_enrolement_incomplet_rejete():
    fichiers = serie()
    del fichiers["droite"]
    assert client.post("/api/enrolement", files=fichiers).status_code == 422


def test_verification_enrolement_inconnu():
    reponse = client.post("/api/verification", data={"enrolement_id": "inconnu"}, files=serie())
    assert reponse.status_code == 404


def test_parcours_complet(monkeypatch):
    """Enrôlement puis vérification, avec une validation de série simulée."""
    encodages = iter([
        {p: np.zeros(128) for p in analyse.POSES},       # série identité
        {p: np.full(128, 0.01) for p in analyse.POSES},  # série vérification, même personne
    ])
    monkeypatch.setattr(analyse, "valider_serie", lambda photos: (True, [], next(encodages)))

    enrolement = client.post("/api/enrolement", files=serie()).json()
    assert enrolement["ok"]

    reponse = client.post("/api/verification", data={"enrolement_id": enrolement["enrolement_id"]}, files=serie())
    resultat = reponse.json()["resultat"]
    assert resultat["is_match"] is True
    assert set(resultat["distances"]) == set(analyse.POSES)


def test_enrolement_expire(monkeypatch):
    monkeypatch.setattr(analyse, "valider_serie", lambda photos: (True, [], {p: np.zeros(128) for p in analyse.POSES}))
    enrolement_id = client.post("/api/enrolement", files=serie()).json()["enrolement_id"]

    monkeypatch.setattr(main, "DUREE_VIE_ENROLEMENT", -1)
    reponse = client.post("/api/verification", data={"enrolement_id": enrolement_id}, files=serie())
    assert reponse.status_code == 404
