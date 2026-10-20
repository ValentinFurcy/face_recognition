"""
Logique de vérification d'identité, indépendante de l'interface :
décodage d'image, estimation de la pose de la tête, validation d'une série
(face / profil gauche / profil droit) et comparaison de deux séries.

Moteur : InsightFace, pack de modèles « buffalo_l » (exécuté avec ONNX Runtime) :
  - détection SCRFD,
  - 68 repères 3D du visage (pose de la tête),
  - empreinte ArcFace (vecteur de 512 nombres),
  - addon « liveness » : détecte une photo ou un écran présenté à la caméra.
Les modèles sont téléchargés au premier lancement dans ~/.insightface
(licence : usage non commercial).
"""
import threading
from io import BytesIO

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from PIL import Image, ImageOps

MODELE = "buffalo_l"
TOLERANCE = 0.6       # distance cosinus max (= similarité min 0.4) pour la même personne
SEUIL_FACE = 0.10     # |rotation| max pour être considéré de face
SEUIL_PROFIL = 0.15   # |rotation| min pour être considéré de profil (tête tournée ~30-45°)
SEUIL_VIVANT = 0.8    # score min de l'addon liveness pour considérer le visage réel
VERIFIER_VIVANT = True
LARGEUR_MAX = 1280    # les images plus grandes sont réduites (temps de calcul)

POSES = ("face", "gauche", "droite")
LIBELLES = {"face": "Face", "gauche": "Profil gauche", "droite": "Profil droit"}


class ImageInvalide(ValueError):
    pass


# ---------- Modèle ----------

_modele = None
_verrou_modele = threading.Lock()


def charger_modele() -> FaceAnalysis:
    """Charge les modèles une seule fois (plusieurs secondes ; téléchargement au premier lancement)."""
    global _modele
    with _verrou_modele:
        if _modele is None:
            modele = FaceAnalysis(
                name=MODELE,
                allowed_modules=["detection", "landmark_3d_68", "recognition"],
                addons=["liveness"],
                liveness_mode="observe",  # on garde l'empreinte même si non vivant : c'est nous qui décidons
                liveness_threshold=SEUIL_VIVANT,
                providers=["CPUExecutionProvider"],
            )
            modele.prepare(ctx_id=-1, det_size=(640, 640))
            _modele = modele
    return _modele


def _detecter(image: np.ndarray):
    """
    Détection + 68 repères 3D, sans empreinte ni vivacité : rapide, pour le suivi en direct.
    Retourne (image BGR, visages) — InsightFace travaille en BGR.
    """
    modele = charger_modele()
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    boites, points = modele.det_model.detect(bgr, max_num=0, metric="default")
    visages = []
    for i in range(boites.shape[0]):
        visage = Face(bbox=boites[i, :4], kps=None if points is None else points[i], det_score=boites[i, 4])
        modele.models["landmark_3d_68"].get(bgr, visage)
        visages.append(visage)
    return bgr, visages


def _empreinte_et_vivacite(bgr: np.ndarray, visage: Face):
    """Calcule l'empreinte ArcFace (normalisée) et le score de vivacité d'un visage déjà détecté."""
    modele = charger_modele()
    modele.addons["liveness"].get(bgr, visage)
    modele.models["recognition"].get(bgr, visage)
    return visage.normed_embedding, visage.liveness


def distance(empreinte_a: np.ndarray, empreinte_b: np.ndarray) -> float:
    """Distance cosinus entre deux empreintes normalisées : 0 = identiques, 1 = sans rapport."""
    return float(1.0 - np.dot(empreinte_a, empreinte_b))


# ---------- Images et pose ----------

def decoder_image(donnees: bytes) -> np.ndarray:
    """Octets d'une image (JPEG, PNG...) -> tableau RGB."""
    try:
        image = Image.open(BytesIO(donnees))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        raise ImageInvalide(f"image illisible : {e}") from e
    image.thumbnail((LARGEUR_MAX, LARGEUR_MAX))
    return np.array(image)


def _reperes(visage: Face) -> dict:
    """Repères 3D (convention iBUG 68 points) -> mâchoire et bas du nez, en 2D."""
    points = visage.landmark_3d_68[:, :2]
    return {"chin": points[0:17].tolist(), "nose_tip": points[31:36].tolist()}


def mesurer_rotation(landmarks):
    """
    Estime la rotation horizontale de la tête : position du bas du nez
    entre les deux bords de la mâchoire.
    ~0 de face ; > 0 tête tournée vers SA gauche ; < 0 vers SA droite (image non miroir).
    """
    machoire = landmarks["chin"]
    x_bord_gauche, x_bord_droit = machoire[0][0], machoire[-1][0]
    largeur = x_bord_droit - x_bord_gauche
    if largeur <= 0:
        return None
    x_nez = np.mean([p[0] for p in landmarks["nose_tip"]])
    return float((x_nez - x_bord_gauche) / largeur - 0.5)


def pose_detectee(rotation):
    if rotation is None:
        return None
    if abs(rotation) <= SEUIL_FACE:
        return "face"
    if rotation >= SEUIL_PROFIL:
        return "gauche"
    if rotation <= -SEUIL_PROFIL:
        return "droite"
    return None  # entre les deux : ni de face, ni assez tourné


def analyser_photo(image: np.ndarray, pose_attendue: str) -> dict:
    """
    Vérifie une photo : un seul visage, dans la pose attendue.
    Retourne {ok, message, pose, rotation} ; si ok, aussi le visage détecté
    et l'image BGR (clés internes « visage » et « bgr »).
    """
    bgr, visages = _detecter(image)
    if not visages:
        return {"ok": False, "pose": None, "rotation": None,
                "message": "Aucun visage détecté (trop loin, mal éclairé ou tête trop tournée ?)"}
    if len(visages) > 1:
        return {"ok": False, "pose": None, "rotation": None,
                "message": f"{len(visages)} visages détectés, une seule personne attendue"}

    rotation = mesurer_rotation(_reperes(visages[0]))
    pose = pose_detectee(rotation)
    if pose != pose_attendue:
        vue = LIBELLES.get(pose, "pose intermédiaire")
        return {"ok": False, "pose": pose, "rotation": rotation,
                "message": f"Pose attendue « {LIBELLES[pose_attendue]} », détectée « {vue} »"}

    return {"ok": True, "pose": pose, "rotation": rotation, "message": "Pose correcte",
            "visage": visages[0], "bgr": bgr}


# ---------- Séries ----------

def _controle_vivacite(vivacite: dict) -> str | None:
    """Message d'erreur si le visage de face ne passe pas le contrôle de vivacité, sinon None."""
    if vivacite["status"] != "ok":
        return "contrôle anti-fraude impossible : centrez le visage et reculez un peu"
    if not vivacite["is_live"]:
        return (f"le visage ne semble pas réel (photo ou écran ?) — "
                f"score de vivacité {vivacite['live_score']:.2f}")
    return None


def valider_serie(photos: dict) -> tuple[bool, list[str], dict]:
    """
    Valide une série {pose: image RGB} :
      - les 3 poses doivent être correctes,
      - la photo de face doit passer le contrôle de vivacité (anti photo / écran),
      - les 3 photos doivent montrer la même personne.
    Retourne (ok, erreurs, empreintes).
    """
    erreurs, empreintes = [], {}
    for pose in POSES:
        image = photos.get(pose)
        if image is None:
            erreurs.append(f"{LIBELLES[pose]} : photo manquante")
            continue
        analyse = analyser_photo(image, pose)
        if not analyse["ok"]:
            erreurs.append(f"{LIBELLES[pose]} : {analyse['message']}")
            continue

        empreinte, vivacite = _empreinte_et_vivacite(analyse["bgr"], analyse["visage"])
        # La vivacité n'est exigée que de face : le modèle est le plus fiable dans cette pose.
        if VERIFIER_VIVANT and pose == "face":
            probleme = _controle_vivacite(vivacite)
            if probleme:
                erreurs.append(f"{LIBELLES[pose]} : {probleme}")
                continue
        empreintes[pose] = empreinte

    # Cohérence interne : les profils doivent être la même personne que la face
    if not erreurs:
        for pose in ("gauche", "droite"):
            d = distance(empreintes["face"], empreintes[pose])
            if d > TOLERANCE:
                erreurs.append(f"{LIBELLES[pose]} : pas la même personne que la photo de face (distance {d:.2f})")

    return not erreurs, erreurs, empreintes


def comparer_series(empreintes_identite: dict, empreintes_verif: dict, tolerance: float = TOLERANCE) -> dict:
    """
    Compare pose par pose (face/face, gauche/gauche, droite/droite).
    Match uniquement si les 3 poses correspondent.
    """
    distances = {pose: round(distance(empreintes_identite[pose], empreintes_verif[pose]), 4) for pose in POSES}
    distance_moyenne = float(np.mean(list(distances.values())))
    # Score intuitif sur la distance moyenne (distance = tolérance -> 50 %)
    score = max(0.0, (1 - distance_moyenne / (tolerance * 2)) * 100)
    return {
        "is_match": all(d <= tolerance for d in distances.values()),
        "tolerance": tolerance,
        "distances": distances,
        "distance_moyenne": round(distance_moyenne, 4),
        "score_pourcentage": round(score, 2),
    }
