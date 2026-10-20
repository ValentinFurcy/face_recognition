"""
API de vérification d'identité par reconnaissance faciale.

Parcours :
  1. POST /api/pose          -> retour en direct sur la pose pendant la prise de vue
  2. POST /api/enrolement    -> série « identité » (face, profil gauche, profil droit) -> enrolement_id
  3. POST /api/verification  -> série « vérification » comparée à l'enrôlement -> verdict

Aucune image n'est conservée : seules les empreintes faciales (encodages) de
l'enrôlement sont gardées en mémoire, pendant DUREE_VIE_ENROLEMENT secondes.
"""
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import analyse

TAILLE_MAX_IMAGE = 5 * 1024 * 1024   # octets
DUREE_VIE_ENROLEMENT = 10 * 60       # secondes
MAX_ENROLEMENTS = 1000

DOSSIER_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Vérification d'identité")
app.mount("/static", StaticFiles(directory=DOSSIER_STATIC), name="static")

# enrolement_id -> (instant de création, encodages par pose)
_enrolements: dict[str, tuple[float, dict]] = {}
_verrou = threading.Lock()


def _purger_enrolements() -> None:
    limite = time.monotonic() - DUREE_VIE_ENROLEMENT
    for cle in [k for k, (cree, _) in _enrolements.items() if cree < limite]:
        del _enrolements[cle]


def _lire_image(fichier: UploadFile):
    donnees = fichier.file.read(TAILLE_MAX_IMAGE + 1)
    if len(donnees) > TAILLE_MAX_IMAGE:
        raise HTTPException(413, f"Image « {fichier.filename} » trop volumineuse (max 5 Mo)")
    try:
        return analyse.decoder_image(donnees)
    except analyse.ImageInvalide as e:
        raise HTTPException(400, str(e)) from e


def _lire_serie(face: UploadFile, gauche: UploadFile, droite: UploadFile) -> dict:
    return {"face": _lire_image(face), "gauche": _lire_image(gauche), "droite": _lire_image(droite)}


# Les routes sont synchrones (def) : FastAPI les exécute dans un pool de threads,
# le calcul face_recognition (CPU) ne bloque donc pas la boucle asynchrone.

@app.get("/", include_in_schema=False)
def accueil():
    return FileResponse(DOSSIER_STATIC / "index.html")


@app.post("/api/pose")
def verifier_pose(image: UploadFile = File(...), pose: Literal["face", "gauche", "droite"] = Form(...)):
    """Analyse une image du flux vidéo : la pose demandée est-elle tenue ?"""
    resultat = analyse.analyser_photo(_lire_image(image), pose)
    return {"ok": resultat["ok"], "pose": resultat["pose"],
            "rotation": resultat["rotation"], "message": resultat["message"]}


@app.post("/api/enrolement")
def enroler(face: UploadFile = File(...), gauche: UploadFile = File(...), droite: UploadFile = File(...)):
    """Valide la série « identité » et la garde en mémoire comme référence."""
    ok, erreurs, encodages = analyse.valider_serie(_lire_serie(face, gauche, droite))
    if not ok:
        return {"ok": False, "erreurs": erreurs}

    enrolement_id = uuid.uuid4().hex
    with _verrou:
        _purger_enrolements()
        if len(_enrolements) >= MAX_ENROLEMENTS:
            raise HTTPException(503, "Trop d'enrôlements en cours, réessayez plus tard")
        _enrolements[enrolement_id] = (time.monotonic(), encodages)
    return {"ok": True, "enrolement_id": enrolement_id}


@app.post("/api/verification")
def verifier(enrolement_id: str = Form(...), face: UploadFile = File(...),
             gauche: UploadFile = File(...), droite: UploadFile = File(...)):
    """Valide la série « vérification » puis la compare à l'enrôlement."""
    with _verrou:
        _purger_enrolements()
        enrolement = _enrolements.get(enrolement_id)
    if enrolement is None:
        raise HTTPException(404, "Enrôlement inconnu ou expiré, recommencez depuis le début")

    ok, erreurs, encodages = analyse.valider_serie(_lire_serie(face, gauche, droite))
    if not ok:
        return {"ok": False, "erreurs": erreurs}

    return {"ok": True, "resultat": analyse.comparer_series(enrolement[1], encodages)}
