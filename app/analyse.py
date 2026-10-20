"""
Logique de vérification d'identité, indépendante de l'interface :
décodage d'image, estimation de la pose de la tête, validation d'une série
(face / profil gauche / profil droit) et comparaison de deux séries.
"""
from io import BytesIO

import face_recognition
import numpy as np
from PIL import Image, ImageOps

TOLERANCE = 0.6       # distance max entre deux visages de la même personne
SEUIL_FACE = 0.10     # |rotation| max pour être considéré de face
SEUIL_PROFIL = 0.15   # |rotation| min pour être considéré de profil (tête tournée ~30-45°)
LARGEUR_MAX = 1280    # les images plus grandes sont réduites (temps de calcul)

POSES = ("face", "gauche", "droite")
LIBELLES = {"face": "Face", "gauche": "Profil gauche", "droite": "Profil droit"}


class ImageInvalide(ValueError):
    pass


def decoder_image(donnees: bytes) -> np.ndarray:
    """Octets d'une image (JPEG, PNG...) -> tableau RGB utilisable par face_recognition."""
    try:
        image = Image.open(BytesIO(donnees))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        raise ImageInvalide(f"image illisible : {e}") from e
    image.thumbnail((LARGEUR_MAX, LARGEUR_MAX))
    return np.array(image)


def mesurer_rotation(landmarks):
    """
    Estime la rotation horizontale de la tête : position du bout du nez
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
    Retourne {ok, message, pose, rotation, emplacement}.
    """
    emplacements = face_recognition.face_locations(image)
    if not emplacements:
        return {"ok": False, "pose": None, "rotation": None,
                "message": "Aucun visage détecté (tête trop tournée, trop loin ou mal éclairée ?)"}
    if len(emplacements) > 1:
        return {"ok": False, "pose": None, "rotation": None,
                "message": f"{len(emplacements)} visages détectés, une seule personne attendue"}

    landmarks = face_recognition.face_landmarks(image, emplacements)[0]
    rotation = mesurer_rotation(landmarks)
    pose = pose_detectee(rotation)
    if pose != pose_attendue:
        vue = LIBELLES.get(pose, "pose intermédiaire")
        return {"ok": False, "pose": pose, "rotation": rotation,
                "message": f"Pose attendue « {LIBELLES[pose_attendue]} », détectée « {vue} »"}

    return {"ok": True, "pose": pose, "rotation": rotation, "message": "Pose correcte",
            "emplacement": emplacements[0]}


def valider_serie(photos: dict) -> tuple[bool, list[str], dict]:
    """
    Valide une série {pose: image} : les 3 poses doivent être correctes
    ET les 3 photos doivent montrer la même personne.
    Retourne (ok, erreurs, encodages).
    """
    erreurs, encodages = [], {}
    for pose in POSES:
        image = photos.get(pose)
        if image is None:
            erreurs.append(f"{LIBELLES[pose]} : photo manquante")
            continue
        analyse = analyser_photo(image, pose)
        if not analyse["ok"]:
            erreurs.append(f"{LIBELLES[pose]} : {analyse['message']}")
            continue
        encodages[pose] = face_recognition.face_encodings(image, [analyse["emplacement"]], num_jitters=5)[0]

    # Cohérence interne : les profils doivent être la même personne que la face
    if not erreurs:
        for pose in ("gauche", "droite"):
            distance = face_recognition.face_distance([encodages["face"]], encodages[pose])[0]
            if distance > TOLERANCE:
                erreurs.append(f"{LIBELLES[pose]} : pas la même personne que la photo de face "
                               f"(distance {distance:.2f})")

    return not erreurs, erreurs, encodages


def comparer_series(encodages_identite: dict, encodages_verif: dict, tolerance: float = TOLERANCE) -> dict:
    """
    Compare pose par pose (face/face, gauche/gauche, droite/droite).
    Match uniquement si les 3 poses correspondent.
    """
    distances = {
        pose: round(float(face_recognition.face_distance([encodages_identite[pose]], encodages_verif[pose])[0]), 4)
        for pose in POSES
    }
    distance_moyenne = float(np.mean(list(distances.values())))
    # Score intuitif sur la distance moyenne (0.6 de distance = 50 %)
    score = max(0.0, (1 - distance_moyenne / (tolerance * 2)) * 100)
    return {
        "is_match": all(d <= tolerance for d in distances.values()),
        "tolerance": tolerance,
        "distances": distances,
        "distance_moyenne": round(distance_moyenne, 4),
        "score_pourcentage": round(score, 2),
    }
