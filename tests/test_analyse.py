from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from skimage import data

from app import analyse


def unitaire(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def base(i):
    """Vecteur unitaire n° i : deux bases différentes sont « sans rapport » (distance 1)."""
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1
    return v


VIVANT = {"status": "ok", "is_live": True, "live_score": 0.95}


def reperes(x_nez):
    """Repères synthétiques : mâchoire de x=100 à x=200, bas du nez en x_nez."""
    return {"chin": [(100, 0)] + [(150, 50)] * 15 + [(200, 0)], "nose_tip": [(x_nez, 60)] * 5}


# ---------- Pose ----------

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


# ---------- Décodage ----------

def test_decoder_image_jpeg_et_reduction():
    tampon = BytesIO()
    Image.new("RGB", (2000, 1000), "white").save(tampon, "JPEG")
    image = analyse.decoder_image(tampon.getvalue())
    assert image.shape == (640, 1280, 3)  # réduite à LARGEUR_MAX, proportions gardées


def test_decoder_image_illisible():
    with pytest.raises(analyse.ImageInvalide):
        analyse.decoder_image(b"pas une image")


# ---------- Modèle réel (InsightFace) ----------
# Image de test : « astronaut » de scikit-image (photo NASA, domaine public), visage de face.

@pytest.fixture(scope="module")
def visage_de_face():
    return data.astronaut()


def test_photo_sans_visage_refusee():
    resultat = analyse.analyser_photo(np.zeros((240, 320, 3), dtype=np.uint8), "face")
    assert not resultat["ok"]
    assert "Aucun visage" in resultat["message"]


def test_vrai_visage_detecte_de_face(visage_de_face):
    resultat = analyse.analyser_photo(visage_de_face, "face")
    assert resultat["ok"], resultat["message"]
    assert abs(resultat["rotation"]) <= analyse.SEUIL_FACE


def test_vrai_visage_de_face_refuse_comme_profil(visage_de_face):
    resultat = analyse.analyser_photo(visage_de_face, "gauche")
    assert not resultat["ok"]
    assert "détectée « Face »" in resultat["message"]


def test_vrai_visage_empreinte_et_vivacite(visage_de_face):
    resultat = analyse.analyser_photo(visage_de_face, "face")
    empreinte, vivacite = analyse._empreinte_et_vivacite(resultat["bgr"], resultat["visage"])
    assert empreinte.shape == (512,)
    assert np.linalg.norm(empreinte) == pytest.approx(1, abs=1e-3)
    assert vivacite["status"] == "ok"


def test_meme_visage_en_miroir_reste_la_meme_personne(visage_de_face):
    """Retourner l'image ne change pas l'identité : distance bien sous la tolérance."""
    empreintes = []
    for image in (visage_de_face, np.ascontiguousarray(visage_de_face[:, ::-1])):
        r = analyse.analyser_photo(image, "face")
        empreintes.append(analyse._empreinte_et_vivacite(r["bgr"], r["visage"])[0])
    assert analyse.distance(*empreintes) < analyse.TOLERANCE


def test_serie_incomplete_ou_sans_visage_refusee():
    noir = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, erreurs, empreintes = analyse.valider_serie({"face": noir, "gauche": noir})
    assert not ok
    assert len(erreurs) == 3
    assert "Profil droit : photo manquante" in erreurs
    assert empreintes == {}


def test_serie_de_trois_photos_de_face_refusee(visage_de_face):
    """Une photo fixe (de face) ne peut pas servir de profil : la série est refusée."""
    ok, erreurs, _ = analyse.valider_serie({p: visage_de_face for p in analyse.POSES})
    assert not ok
    assert [e.split(" :")[0] for e in erreurs] == ["Profil gauche", "Profil droit"]


# ---------- Règles de validation (modèle simulé) ----------

@pytest.fixture
def poses_correctes(monkeypatch):
    monkeypatch.setattr(analyse, "analyser_photo", lambda img, pose: {"ok": True, "visage": None, "bgr": None})


def simuler_empreintes(monkeypatch, empreintes, vivacites=None):
    empreintes = iter(empreintes)
    vivacites = iter(vivacites or [VIVANT] * 3)
    monkeypatch.setattr(analyse, "_empreinte_et_vivacite", lambda bgr, visage: (next(empreintes), next(vivacites)))


def test_serie_valide(monkeypatch, poses_correctes):
    simuler_empreintes(monkeypatch, [base(0), unitaire(base(0) + 0.1 * base(1)), base(0)])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, erreurs, empreintes = analyse.valider_serie({p: image for p in analyse.POSES})
    assert ok, erreurs
    assert set(empreintes) == set(analyse.POSES)


def test_serie_refusee_si_profil_d_une_autre_personne(monkeypatch, poses_correctes):
    simuler_empreintes(monkeypatch, [base(0), base(1), base(0)])  # le profil gauche est « quelqu'un d'autre »
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, erreurs, _ = analyse.valider_serie({p: image for p in analyse.POSES})
    assert not ok
    assert len(erreurs) == 1
    assert erreurs[0].startswith("Profil gauche : pas la même personne")


@pytest.mark.parametrize("vivacite, extrait", [
    ({"status": "ok", "is_live": False, "live_score": 0.12}, "ne semble pas réel"),
    ({"status": "input_rejected", "is_live": None, "live_score": None}, "centrez le visage"),
])
def test_serie_refusee_si_face_pas_vivante(monkeypatch, poses_correctes, vivacite, extrait):
    simuler_empreintes(monkeypatch, [base(0)] * 3, [vivacite, VIVANT, VIVANT])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, erreurs, _ = analyse.valider_serie({p: image for p in analyse.POSES})
    assert not ok
    assert erreurs[0].startswith("Face :") and extrait in erreurs[0]


def test_vivacite_ignoree_si_desactivee(monkeypatch, poses_correctes):
    monkeypatch.setattr(analyse, "VERIFIER_VIVANT", False)
    simuler_empreintes(monkeypatch, [base(0)] * 3, [{"status": "ok", "is_live": False, "live_score": 0.1}] + [VIVANT] * 2)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert analyse.valider_serie({p: image for p in analyse.POSES})[0]


# ---------- Comparaison ----------

def test_distance_cosinus():
    assert analyse.distance(base(0), base(0)) == pytest.approx(0)
    assert analyse.distance(base(0), base(1)) == pytest.approx(1)


def test_comparaison_match_seulement_si_les_3_poses_correspondent():
    identite = {p: base(0) for p in analyse.POSES}
    proche = {p: unitaire(base(0) + 0.1 * base(1)) for p in analyse.POSES}  # distance ~0.005
    un_profil_different = {**proche, "droite": base(2)}                       # distance 1

    assert analyse.comparer_series(identite, proche)["is_match"]
    resultat = analyse.comparer_series(identite, un_profil_different)
    assert not resultat["is_match"]
    assert resultat["distances"]["droite"] > analyse.TOLERANCE
