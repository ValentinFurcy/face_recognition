// Vérification d'identité : pilote la caméra et dialogue avec l'API FastAPI.
// La décision (pose correcte, même personne...) est toujours prise côté serveur.

const POSES = ["face", "gauche", "droite"];
const CONSIGNES = {
  face: "Regardez la caméra, de face",
  gauche: "Tournez la tête vers votre gauche",
  droite: "Tournez la tête vers votre droite",
};
const LIBELLES = { face: "Face", gauche: "Profil gauche", droite: "Profil droit" };

const DUREE_STABLE_MS = 1000;       // pose tenue avant la prise automatique
const INTERVALLE_ANALYSE_MS = 200;  // fréquence max des analyses de pose
const LARGEUR_ANALYSE = 480;        // image réduite pour le suivi en direct
const LARGEUR_PHOTO = 1280;         // image envoyée pour l'enrôlement / la vérification
const MAX_TENTATIVES = 3;
const PAUSE_APRES_ECHEC_MS = 4000;

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");

let flux = null;
let annule = false;
let photosIdentite = null;

class Annulation extends Error {}

const attendre = (ms) => new Promise((r) => setTimeout(r, ms));

function melanger(tableau) {
  const t = [...tableau];
  for (let i = t.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [t[i], t[j]] = [t[j], t[i]];
  }
  return t;
}

// ---------- Caméra ----------

async function listerCameras() {
  // Les noms des caméras ne sont visibles qu'après autorisation : on ouvre un flux temporaire.
  const temporaire = await navigator.mediaDevices.getUserMedia({ video: true });
  temporaire.getTracks().forEach((t) => t.stop());

  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  if (!cameras.length) throw new Error("Aucune caméra détectée.");

  const select = $("camera");
  select.innerHTML = "";
  cameras.forEach((c, i) => select.add(new Option(c.label || `Caméra ${i + 1}`, c.deviceId)));

  // Une caméra branchée est privilégiée, la caméra intégrée sert de repli.
  const branchee = cameras.find((c) => !/integrated|intégrée|built-in|facetime/i.test(c.label));
  if (branchee) select.value = branchee.deviceId;
}

async function demarrerCamera(deviceId) {
  arreterCamera();
  flux = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
  video.srcObject = flux;
  await video.play();
}

function arreterCamera() {
  if (flux) flux.getTracks().forEach((t) => t.stop());
  flux = null;
  video.srcObject = null;
}

// Image courante du flux (non miroir), réduite à `largeurMax`, en JPEG.
function capturer(largeurMax) {
  const ratio = Math.min(1, largeurMax / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * ratio);
  canvas.height = Math.round(video.videoHeight * ratio);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
}

// ---------- API ----------

async function appelerApi(url, champs) {
  const formulaire = new FormData();
  for (const [nom, valeur] of Object.entries(champs)) {
    if (valeur instanceof Blob) formulaire.append(nom, valeur, `${nom}.jpg`);
    else formulaire.append(nom, valeur);
  }
  const reponse = await fetch(url, { method: "POST", body: formulaire });
  const donnees = await reponse.json().catch(() => ({}));
  if (!reponse.ok) throw new Error(donnees.detail || `Erreur serveur (${reponse.status})`);
  return donnees;
}

// ---------- Prise de vue ----------

function afficherConsigne(texte, poseOk, progression) {
  $("consigne").textContent = texte;
  $("consigne").classList.toggle("ok", poseOk);
  $("barre").style.width = `${Math.min(100, progression * 100)}%`;
}

function preparerVignettes(ordre) {
  $("vignettes").innerHTML = ordre
    .map((p) => `<li data-pose="${p}"><div class="cadre"></div>${LIBELLES[p]}</li>`)
    .join("");
}

function majVignette(pose, etat, photo) {
  const li = $("vignettes").querySelector(`[data-pose="${pose}"]`);
  li.className = etat;
  if (photo) li.querySelector(".cadre").innerHTML = `<img class="photo" src="${URL.createObjectURL(photo)}" alt="">`;
}

// Attend que la pose soit tenue DUREE_STABLE_MS, puis prend la photo automatiquement.
async function capturerPose(pose) {
  majVignette(pose, "courante");
  let debutPoseOk = null;

  while (true) {
    if (annule) throw new Annulation();
    const debutTour = performance.now();

    const analyse = await appelerApi("/api/pose", { pose, image: await capturer(LARGEUR_ANALYSE) });
    if (!analyse.ok) debutPoseOk = null;
    else if (debutPoseOk === null) debutPoseOk = performance.now();

    const tenue = debutPoseOk === null ? 0 : performance.now() - debutPoseOk;
    afficherConsigne(CONSIGNES[pose], analyse.ok, tenue / DUREE_STABLE_MS);
    $("etat").textContent = analyse.ok ? "Ne bougez plus…" : analyse.message;

    if (analyse.ok && tenue >= DUREE_STABLE_MS) {
      const photo = await capturer(LARGEUR_PHOTO);
      $("flash").classList.remove("actif");
      void $("flash").offsetWidth; // relance l'animation
      $("flash").classList.add("actif");
      majVignette(pose, "prise", photo);
      return photo;
    }
    await attendre(Math.max(0, INTERVALLE_ANALYSE_MS - (performance.now() - debutTour)));
  }
}

// Prend les 3 photos et les fait valider par le serveur. Si la série est refusée,
// les 3 photos sont à reprendre (MAX_TENTATIVES séries au plus).
async function capturerSerie(titre, url, champsSupplementaires, ordreAleatoire) {
  for (let tentative = 1; tentative <= MAX_TENTATIVES; tentative++) {
    $("titre-serie").textContent = `${titre} — série ${tentative}/${MAX_TENTATIVES}`;
    $("erreurs").hidden = true;

    const ordre = ordreAleatoire ? melanger(POSES) : POSES;
    preparerVignettes(ordre);
    const photos = {};
    for (const pose of ordre) photos[pose] = await capturerPose(pose);

    afficherConsigne("Analyse des 3 photos…", true, 1);
    $("etat").textContent = "Vérification de la série en cours…";
    const reponse = await appelerApi(url, { ...champsSupplementaires, ...photos });
    if (reponse.ok) return { photos, reponse };

    $("erreurs").replaceChildren(...reponse.erreurs.map((e) => Object.assign(document.createElement("li"), { textContent: e })));
    $("erreurs").hidden = false;
    afficherConsigne("Série refusée : reprenez les 3 photos", false, 0);
    $("etat").textContent = tentative < MAX_TENTATIVES ? "Nouvelle série dans quelques secondes…" : "";
    await attendre(PAUSE_APRES_ECHEC_MS);
  }
  throw new Error(`Échec après ${MAX_TENTATIVES} séries refusées.`);
}

// ---------- Parcours ----------

function etape(nom) {
  const ordre = ["identite", "verification", "resultat"];
  ordre.forEach((e, i) => {
    $(`etape-${e}`).className = i < ordre.indexOf(nom) ? "fait" : e === nom ? "active" : "";
  });
}

function afficherSection(id) {
  ["accueil", "prise", "resultat"].forEach((s) => ($(s).hidden = s !== id));
}

async function parcours() {
  annule = false;
  afficherSection("prise");
  try {
    await demarrerCamera($("camera").value);

    etape("identite");
    const identite = await capturerSerie("Photo d'identité", "/api/enrolement", {}, false);
    photosIdentite = identite.photos;

    etape("verification");
    const verif = await capturerSerie(
      "Vérification",
      "/api/verification",
      { enrolement_id: identite.reponse.enrolement_id },
      true, // ordre aléatoire : complique le rejeu d'une vidéo préenregistrée
    );

    etape("resultat");
    afficherResultat(verif.reponse.resultat, photosIdentite, verif.photos);
  } catch (e) {
    afficherSection("accueil");
    etape("identite");
    if (!(e instanceof Annulation)) afficherErreurAccueil(e.message);
  } finally {
    arreterCamera();
  }
}

function afficherResultat(resultat, photosA, photosB) {
  afficherSection("resultat");
  const verdict = $("verdict");
  verdict.textContent = resultat.is_match ? "✅ Identité vérifiée" : "❌ Identité non vérifiée";
  verdict.className = resultat.is_match ? "ok" : "ko";
  $("score").textContent =
    `Score de similitude : ${resultat.score_pourcentage.toFixed(1)} % ` +
    `(distance moyenne ${resultat.distance_moyenne.toFixed(2)})`;

  const img = (blob) => `<img class="photo" src="${URL.createObjectURL(blob)}" alt="">`;
  $("tableau").innerHTML = POSES.map((p) => {
    const d = resultat.distances[p];
    const ok = d <= resultat.tolerance;
    return `<tr><td>${LIBELLES[p]}</td><td>${img(photosA[p])}</td><td>${img(photosB[p])}</td>
            <td class="${ok ? "ok" : "ko"}">${d.toFixed(2)}</td></tr>`;
  }).join("");
}

function afficherErreurAccueil(message) {
  $("erreur-camera").textContent = message;
  $("erreur-camera").hidden = !message;
}

// ---------- Initialisation ----------

$("btn-commencer").addEventListener("click", () => {
  afficherErreurAccueil("");
  parcours();
});
$("btn-annuler").addEventListener("click", () => (annule = true));
$("btn-recommencer").addEventListener("click", () => {
  afficherSection("accueil");
  etape("identite");
});

(async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    afficherErreurAccueil("Ce navigateur ne donne pas accès à la caméra (page à ouvrir en https ou sur localhost).");
    return;
  }
  try {
    await listerCameras();
    $("btn-commencer").disabled = false;
  } catch (e) {
    afficherErreurAccueil(`Accès à la caméra impossible : ${e.message}`);
  }
})();
