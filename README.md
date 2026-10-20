# Vérification d'identité par reconnaissance faciale

Application web : la page pilote la caméra dans le navigateur, une API FastAPI analyse les images.

1. **Photo d'identité** : 3 photos prises automatiquement (face, profil gauche, profil droit).
2. **Vérification** : les 3 mêmes poses, dans un ordre aléatoire.
3. **Résultat** : chaque pose est comparée à la même pose de l'identité. Il y a match seulement si les 3 correspondent.

Une photo est prise dès que la pose demandée est tenue 1 seconde. La série entière est refusée et à reprendre (3 séries au plus) dans ces cas :
- une photo ne contient pas exactement un visage ;
- une photo n'est pas dans la bonne pose ;
- les profils ne sont pas la même personne que la photo de face.

## Lancer

```bash
venv_face\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ouvrir <http://localhost:8000> dans Chrome ou Edge et autoriser la caméra. Une caméra branchée est sélectionnée par défaut, sinon la caméra intégrée. Le choix peut être changé dans la liste.

> Le navigateur n'autorise la caméra que sur `localhost` ou en **HTTPS**. Pour un accès depuis une autre machine, il faut un certificat (reverse proxy, ou `uvicorn --ssl-keyfile ... --ssl-certfile ...`).

## Installer l'environnement

Python 3.12. dlib n'a pas de paquet précompilé pour 3.12 sous Windows. Il est compilé à l'installation, ce qui demande Visual Studio avec « Développement Desktop en C++ ».

```bash
py -3.12 -m venv venv_face
venv_face\Scripts\python.exe -m pip install cmake
venv_face\Scripts\python.exe -m pip install -r requirements-face.txt
```

## Tests

```bash
venv_face\Scripts\python.exe -m pytest tests -q
```

## API

| Route | Entrée (multipart) | Sortie |
|---|---|---|
| `POST /api/pose` | `image`, `pose` (`face`/`gauche`/`droite`) | `{ok, pose, rotation, message}`, retour en direct pendant la prise |
| `POST /api/enrolement` | `face`, `gauche`, `droite` | `{ok: true, enrolement_id}` ou `{ok: false, erreurs}` |
| `POST /api/verification` | `enrolement_id`, `face`, `gauche`, `droite` | `{ok: true, resultat: {is_match, distances, distance_moyenne, score_pourcentage, tolerance}}` ou `{ok: false, erreurs}` |

Erreurs HTTP : `400` image illisible, `413` image > 5 Mo, `404` enrôlement inconnu ou expiré, `422` champ manquant ou pose inconnue.

## Structure

```
app/analyse.py     logique : pose de la tête, validation d'une série, comparaison
app/main.py        API FastAPI + service de la page
app/static/        page web (HTML, CSS, JS caméra)
tests/             pytest (logique + API)
face_recognition.ipynb   prototype d'origine (fenêtre OpenCV)
```

## Réglages

Dans `app/analyse.py` :

| Réglage | Valeur | Rôle |
|---|---|---|
| `TOLERANCE` | 0.6 | Distance maximale pour considérer deux visages comme la même personne. Plus bas = plus strict. |
| `SEUIL_FACE` | 0.10 | Rotation maximale de la tête pour une photo de face. |
| `SEUIL_PROFIL` | 0.15 | Rotation minimale pour une photo de profil. |

Dans `app/static/app.js` :

| Réglage | Valeur | Rôle |
|---|---|---|
| `DUREE_STABLE_MS` | 1000 | Durée pendant laquelle la pose doit être tenue avant la prise. |
| `MAX_TENTATIVES` | 3 | Nombre de séries autorisées avant abandon. |

## Limites connues

- **Profil = tête tournée d'environ 30 à 45°.** Le détecteur de dlib ne trouve pas un visage vu de profil strict (90°).
- **Protection anti-usurpation partielle.** Une photo affichée sur un téléphone ne peut pas tourner, elle est donc bloquée. En revanche, une vidéo de la personne qui tourne la tête pourrait passer. L'ordre aléatoire des poses complique ce rejeu, mais le serveur ne vérifie pas l'ordre réel de prise. Une vraie détection de vivant demanderait par exemple de vérifier les clignements ou la profondeur, ou d'utiliser un service spécialisé.
- **Données biométriques (RGPD, article 9).** Aucune image n'est conservée. Seules les empreintes faciales de l'enrôlement restent en mémoire, 10 minutes. En production, il faudrait aussi recueillir le consentement, passer en HTTPS, ajouter de l'authentification et limiter le nombre de requêtes.
- **Enrôlement en mémoire.** Il est perdu au redémarrage du serveur et n'est pas partagé entre plusieurs processus uvicorn.
