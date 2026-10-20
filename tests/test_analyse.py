from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app import analyse


def reperes(x_nez):
    """Repères synthétiques : mâchoire de x=100 à x=200, bout du nez en x_nez."""
    return {"chin": [(100, 0)] + [(150, 50)] * 15 + [(200, 0)], "nose_tip": [(x_nez, 60)] * 5}


@pytest.mark.parametrize("x_nez, pose", [
    (150, "face"),     # nez centré
    (158, "face"),     # légère rotation tolérée
    (163, None),       # entre les deux : ni face, ni profil
    (170, "gauche"),   # nez vers la droite de l'image = tête tournée vers SA gauche
    (130, "droite"),
])
def test_pose_detectee_selon_position_du_nez(x_nez, pose):
    assert analyse.pose_detectee(analyse.mesurer_rotation(reperes(x_nez))) == pose


def test_rotation_impossible_si_machoire_degeneree():
    assert analyse.mesurer_rotation({"chin": [(100, 0)] * 17, "nose_tip": [(100, 0)]}) is None


def test_decoder_image_jpeg_et_reduction():
    tampon = BytesIO()
    Image.new("RGB", (2000, 1000), "white").save(tampon, "JPEG")
    image = analyse.decoder_image(tampon.getvalue())
    assert image.shape == (640, 1280, 3)  # réduite à LARGEUR_MAX, proportions gardées


def test_decoder_image_illisible():
    with pytest.raises(analyse.ImageInvalide):
        analyse.decoder_image(b"pas une image")


def test_photo_sans_visage_refusee():
    resultat = analyse.analyser_photo(np.zeros((240, 320, 3), dtype=np.uint8), "face")
    assert not resultat["ok"]
    assert "Aucun visage" in resultat["message"]


def test_serie_incomplete_ou_sans_visage_refusee():
    noir = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, erreurs, encodages = analyse.valider_serie({"face": noir, "gauche": noir})
    assert not ok
    assert len(erreurs) == 3
    assert "Profil droit : photo manquante" in erreurs
    assert encodages == {}


def test_serie_refusee_si_profil_d_une_autre_personne(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(analyse, "analyser_photo", lambda img, pose: {"ok": True, "emplacement": (0, 1, 1, 0)})
    encodages = iter([np.zeros(128), np.ones(128), np.zeros(128)])  # le profil gauche est « quelqu'un d'autre »
    monkeypatch.setattr(analyse.face_recognition, "face_encodings", lambda *a, **k: [next(encodages)])

    ok, erreurs, _ = analyse.valider_serie({"face": image, "gauche": image, "droite": image})
    assert not ok
    assert len(erreurs) == 1
    assert erreurs[0].startswith("Profil gauche : pas la même personne")


def test_comparaison_match_seulement_si_les_3_poses_correspondent():
    base = {p: np.zeros(128) for p in analyse.POSES}
    proche = {p: np.full(128, 0.01) for p in analyse.POSES}       # distance ~0.11
    un_profil_different = {**proche, "droite": np.full(128, 0.1)}  # distance ~1.13

    assert analyse.comparer_series(base, proche)["is_match"]
    resultat = analyse.comparer_series(base, un_profil_different)
    assert not resultat["is_match"]
    assert resultat["distances"]["droite"] > analyse.TOLERANCE
