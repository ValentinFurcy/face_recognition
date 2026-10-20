# Vérification d'identité par reconnaissance faciale

Application web : la page pilote la caméra dans le navigateur, une API FastAPI analyse les images avec **InsightFace**.

1. **Photo d'identité** : 3 photos prises automatiquement (face, profil gauche, profil droit).
2. **Vérification** : les 3 mêmes poses, dans un ordre aléatoire.
3. **Résultat** : chaque pose est comparée à la même pose de l'identité. Il y a match seulement si les 3 correspondent.

Une photo est prise dès que la pose demandée est tenue 1 seconde. La série entière est refusée et à reprendre (3 séries au plus) dans ces cas :
- une photo ne contient pas exactement un visage ;
- une photo n'est pas dans la bonne pose ;
- la photo de face échoue au contrôle de vivacité (photo ou écran présenté à la caméra) ;
- les profils ne sont pas la même personne que la photo de face.

## Lancer

```bash
venv_face\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ouvrir <http://localhost:8000> dans Chrome ou Edge et autoriser la caméra. Une caméra branchée est sélectionnée par défaut, sinon la caméra intégrée. Le choix peut être changé dans la liste.

Au tout premier lancement, les modèles (environ 280 Mo) sont téléchargés dans `~/.insightface`, ce qui peut prendre une à deux minutes. Ensuite, ils sont chargés en quelques secondes au démarrage du serveur.

> Le navigateur n'autorise la caméra que sur `localhost` ou en **HTTPS**. Pour un accès depuis une autre machine, il faut un certificat (reverse proxy, ou `uvicorn --ssl-keyfile ... --ssl-certfile ...`).

## Installer l'environnement

Python 3.12. Aucune compilation n'est nécessaire pour l'application.

```bash
py -3.12 -m venv venv_face
venv_face\Scripts\python.exe -m pip install -r requirements.txt
```

Le notebook prototype (`face_recognition.ipynb`) utilise encore l'ancien moteur, dlib. Ses dépendances sont dans `requirements-notebook.txt` ; dlib y est compilé à l'installation, ce qui demande Visual Studio avec « Développement Desktop en C++ ».

## Tests

```bash
venv_face\Scripts\python.exe -m pytest tests -q
```

Une partie des tests utilise le vrai modèle sur l'image « astronaut » fournie avec scikit-image (photo NASA, domaine public).

## Moteur de reconnaissance

Pack de modèles InsightFace `buffalo_l`, exécuté sur processeur avec ONNX Runtime :

| Étape | Modèle | Usage ici |
|---|---|---|
| Détection | SCRFD | Trouver le visage, y compris tête tournée |
| Repères | 68 points 3D | Estimer la rotation de la tête (position du nez entre les bords de la mâchoire) |
| Empreinte | ArcFace (vecteur de 512 nombres) | Comparer deux visages par distance cosinus |
| Vivacité | Addon `liveness` | Rejeter une photo ou un écran présenté à la caméra |

Le suivi de pose en direct (`/api/pose`) ne lance que la détection et les repères, soit environ 110 ms par image sur processeur.

**Licence :** le code d'InsightFace est open source, mais les modèles pré-entraînés sont réservés à un **usage non commercial**. Un usage commercial demande une licence InsightFace ou d'autres modèles.

## API

| Route | Entrée (multipart) | Sortie |
|---|---|---|
| `POST /api/pose` | `image`, `pose` (`face`/`gauche`/`droite`) | `{ok, pose, rotation, message}`, retour en direct pendant la prise |
| `POST /api/enrolement` | `face`, `gauche`, `droite` | `{ok: true, enrolement_id}` ou `{ok: false, erreurs}` |
| `POST /api/verification` | `enrolement_id`, `face`, `gauche`, `droite` | `{ok: true, resultat: {is_match, distances, distance_moyenne, score_pourcentage, tolerance}}` ou `{ok: false, erreurs}` |

Erreurs HTTP : `400` image illisible, `413` image > 5 Mo, `404` enrôlement inconnu ou expiré, `422` champ manquant ou pose inconnue.

## Structure

```
app/analyse.py     logique : modèles, pose de la tête, vivacité, validation d'une série, comparaison
app/main.py        API FastAPI + service de la page
app/static/        page web (HTML, CSS, JS caméra)
tests/             pytest (logique + API)
face_recognition.ipynb   prototype d'origine (dlib, fenêtre OpenCV)
```

## Réglages

Dans `app/analyse.py` :

| Réglage | Valeur | Rôle |
|---|---|---|
| `TOLERANCE` | 0.6 | Distance cosinus maximale pour considérer deux visages comme la même personne (similarité ≥ 0,4). Plus bas = plus strict. |
| `SEUIL_FACE` | 0.10 | Rotation maximale de la tête pour une photo de face. |
| `SEUIL_PROFIL` | 0.15 | Rotation minimale pour une photo de profil. |
| `SEUIL_VIVANT` | 0.8 | Score minimal de vivacité. |
| `VERIFIER_VIVANT` | `True` | Active le contrôle de vivacité sur la photo de face. |

Dans `app/static/app.js` :

| Réglage | Valeur | Rôle |
|---|---|---|
| `DUREE_STABLE_MS` | 1000 | Durée pendant laquelle la pose doit être tenue avant la prise. |
| `MAX_TENTATIVES` | 3 | Nombre de séries autorisées avant abandon. |

Les seuils n'ont pas encore été calibrés sur de vraies captures caméra : à ajuster après des essais.

## Limites connues

- **Profil = tête tournée d'environ 30 à 45°.** Au-delà, les repères deviennent moins fiables.
- **Vivacité contrôlée seulement de face.** Le modèle est le plus fiable dans cette pose. Une vidéo de la personne qui tourne la tête pourrait encore passer les profils. L'ordre aléatoire des poses complique ce rejeu, mais le serveur ne vérifie pas l'ordre réel de prise.
- **Données biométriques (RGPD, article 9).** Aucune image n'est conservée. Seules les empreintes faciales de l'enrôlement restent en mémoire, 10 minutes. En production, il faudrait aussi recueillir le consentement, passer en HTTPS, ajouter de l'authentification et limiter le nombre de requêtes.
- **Enrôlement en mémoire.** Il est perdu au redémarrage du serveur et n'est pas partagé entre plusieurs processus uvicorn.
