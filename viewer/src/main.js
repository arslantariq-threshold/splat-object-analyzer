import { SceneManager } from "./SceneManager.js";
import { SplatView } from "./SplatView.js";
import { AnnotationParser } from "./AnnotationParser.js";
import { AnnotationLayer } from "./AnnotationLayer.js";
import { SettingsMenu } from "./SettingsMenu.js";
import { SceneUnderstanding } from "./SceneUnderstanding.js";
import { NavigationLayer } from './NavigationLayer.js';

const configuration = await fetch(new URL('../viewer-config.json', import.meta.url))
  .then(response => response.ok ? response.json() : {scene: null})
  .catch(() => ({scene: null}));
const EXAMPLES = configuration.scenes || (configuration.scene ? [{id: 'analysis', label: configuration.scene.name || 'Analysis result', ...configuration.scene}] : []);
const sceneManager = new SceneManager(document.getElementById('viewport'));
const splatView = new SplatView(sceneManager);
const annotationLayer = new AnnotationLayer(sceneManager, {
  onHoverStart: annotation => splatView.highlight(annotation),
  onHoverEnd: annotation => splatView.clearHighlight(annotation),
});
const hint = document.getElementById('empty-hint');
const loading = document.getElementById('loading');
const fill = document.getElementById('loading-bar-fill');
const loadingLabel = document.getElementById('loading-label');
const status = document.getElementById('analysis-status');
const statusTitle = document.getElementById('analysis-title');
const statusDetail = document.getElementById('analysis-detail');
const progress = document.getElementById('analysis-progress');
const download = document.getElementById('download-results');
let generation = 0;
let loadedAnnotations = [];
let showContents = false;
let showUnconfirmedEnvironment = false;
let sourceHash = null;
let lastCamera = null;
let selectedIds = null;
const understanding = new SceneUnderstanding({
  onBusy: busy => settingsMenu.setBusy(busy),
  onSelection: (ids, index) => {
    selectedIds = ids === null ? null : new Set(ids);
    const descriptions = new Map((index?.objects || []).map(object => [object.id, object]));
    for (const annotation of loadedAnnotations) {
      const description = descriptions.get(annotation.id);
      annotation.resolvedLabel = description?.display_name || annotation.originalLabel;
      annotation.status = annotation.kind === 'object' ? annotation.originalStatus : description?.status || annotation.originalStatus;
    }
    displayAnnotations();
  },
  onView: (camera, ids) => {
    selectedIds = ids === null ? null : new Set(ids);
    displayAnnotations();
    sceneManager.navigationTouched = false;
    applyCamera(camera);
    sceneManager.renderer.domElement.focus({preventScroll:true});
  },
});

function displayAnnotations(annotations = loadedAnnotations) {
  loadedAnnotations = annotations;
  for (const annotation of annotations) {
    annotation.selected = selectedIds !== null && selectedIds.has(annotation.id);
    const unconfirmed = annotation.kind !== 'object' && annotation.status !== 'supported';
    annotation.label = annotation.resolvedLabel + (unconfirmed ? ` (${annotation.status})` : '');
  }
  splatView.clearHighlight();
  const available = annotations.filter(a => a.kind === 'object' || a.status === 'supported' || showUnconfirmedEnvironment);
  const availableIds = new Set(available.map(a => a.id));
  annotationLayer.rebuild(available.filter(a => selectedIds !== null ? selectedIds.has(a.id) :
    (showContents || !a.parentId || !availableIds.has(a.parentId))));
  if (selectedIds !== null) for (const object of annotationLayer.objects) object.box.show();
}
function inventorySummary(result) {
  const entities = result.environment?.entities || [];
  const supported = entities.filter(entity => entity.status === 'supported').length;
  return `${result.objects?.length || 0} objects · ${supported} environment subjects` +
    (entities.length > supported ? ` · ${entities.length - supported} unconfirmed regions` : '');
}
function showStatus(title, detail = '', state = 'running', completed = null, total = null) {
  status.classList.remove('hidden');
  status.dataset.state = state;
  statusTitle.textContent = title;
  statusDetail.textContent = detail;
  progress.classList.toggle('hidden', state === 'complete' || state === 'failed');
  if (total > 0 && completed !== null) { progress.max = total; progress.value = completed; }
  else progress.removeAttribute('value');
  download.classList.add('hidden');
}
function showLoading(label) {
  loading.classList.remove('hidden');
  hint.classList.add('hidden');
  loadingLabel.textContent = label;
  fill.style.width = '0%';
  fill.classList.add('indeterminate');
}
function hideLoading() { loading.classList.add('hidden'); }
function loaderOptions(gen) {
  return {
    onProgress: (loaded, total) => {
      if (gen !== generation) return;
      fill.classList.toggle('indeterminate', !total);
      if (total) fill.style.width = `${Math.min(100, loaded / total * 100)}%`;
      loadingLabel.textContent = `Loading scene: ${(loaded / 1048576).toFixed(1)}${total ? ' / ' + (total / 1048576).toFixed(1) : ''} MB`;
    },
    onProcessing: () => {
      if (gen !== generation) return;
      fill.classList.add('indeterminate');
      loadingLabel.textContent = 'Preparing scene for viewing…';
    },
  };
}
function resetScene() {
  generation += 1;
  understanding.reset();
  navigationLayer.clear();
  selectedIds = null;
  splatView.clear();
  loadedAnnotations = [];
  annotationLayer.clear();
  sourceHash = null;
  lastCamera = null;
  settingsMenu.updateStats({fps: sceneManager.fps, splatCount: 0, labelCount: 0});
  return generation;
}
function applyCamera(camera) {
  if (camera) lastCamera = camera;
  if (!camera || sceneManager.navigationTouched) return;
  sceneManager.camera.position.fromArray(camera.position);
  sceneManager.controls.target.fromArray(camera.target);
  sceneManager.camera.fov = camera.fov || 90;
  sceneManager.camera.near = .01;
  sceneManager.camera.far = 1000;
  sceneManager.camera.updateProjectionMatrix();
  sceneManager.controls.update();
}
async function loadPair(pair, gen, keepScene = false) {
  if (!keepScene) {
    showLoading('Loading scene…');
    await splatView.load(pair.splatUrl, loaderOptions(gen));
  }
  if (gen !== generation) return;
  const response = await fetch(pair.annotationsUrl);
  if (!response.ok) throw new Error('Could not load the analysis results.');
  const result = await response.json();
  if (gen !== generation) return;
  if (pair.sourceHash && result.input?.sha256 !== pair.sourceHash) throw new Error('The annotations belong to a different scene.');
  sourceHash = result.input?.sha256 || null;
  displayAnnotations(AnnotationParser.fromText(JSON.stringify(result)));
  applyCamera(pair.camera);
  annotationLayer.playReveal();
  hideLoading();
  if (configuration.understandingAvailable) understanding.setScene({...pair, sourceHash});
  await navigationLayer.load(pair.navigationUrl, sourceHash);
  return result;
}
async function loadExample(id) {
  const pair = EXAMPLES.find(ex => ex.id === id);
  const gen = resetScene();
  settingsMenu.clearFileSelections();
  status.classList.add('hidden');
  if (!pair) { hideLoading(); hint.classList.remove('hidden'); return; }
  try {
    const result = await loadPair(pair, gen);
    if (gen !== generation || !result) return;
    showStatus('Analysis ready', `${pair.label || pair.name} · ${inventorySummary(result)}`, 'complete');
    download.href = pair.annotationsUrl;
    download.classList.remove('hidden');
  }
  catch (error) { hideLoading(); showStatus('Scene could not be loaded', error.message, 'failed'); }
}
function uploadScene(file, quality, cameraFiles = []) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `/api/jobs?filename=${encodeURIComponent(file.name)}&quality=${quality}`);
    let body = file;
    if (cameraFiles.length) {
      body = new FormData();
      body.append('file', file);
      const images = cameraFiles.filter(item => /^images\.(txt|bin)$/i.test(item.name));
      const calibration = cameraFiles.filter(item => /^cameras\.txt$/i.test(item.name));
      const json = cameraFiles.filter(item => /\.json$/i.test(item.name));
      if (images.length > 1 || calibration.length > 1 || json.length > 1 ||
          (images.length && json.length) || (!images.length && !json.length) || (json.length && calibration.length)) {
        reject(new Error('Choose images.txt or images.bin with optional cameras.txt, or one source-space camera JSON.')); return;
      }
      if (images[0]) body.append('colmap_images', images[0]);
      if (calibration[0]) body.append('colmap_cameras', calibration[0]);
      if (json[0]) body.append('cameras', json[0]);
    } else request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('X-Splat-Viewer', '1');
    request.upload.onprogress = event => showStatus('Uploading scene', file.name, 'running', event.loaded, event.total);
    request.onerror = () => reject(new Error('The upload failed. Check that the local analyzer is running.'));
    request.onload = () => {
      let result;
      try { result = JSON.parse(request.responseText); }
      catch { reject(new Error('The analyzer returned an invalid response.')); return; }
      if (request.status !== 202) { reject(new Error(result.error || 'Could not start analysis.')); return; }
      resolve(result.id);
    };
    request.send(body);
  });
}
async function pollJob(id, gen, sceneReady = Promise.resolve()) {
  let placedPreview = false;
  while (gen === generation) {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Could not read analysis status.');
    if (gen !== generation) return;
    if (job.status === 'failed') throw new Error(job.error || 'Analysis failed.');
    if (job.previewCamera && !placedPreview) {
      placedPreview = true;
      sceneReady.then(() => { if (gen === generation) applyCamera(job.previewCamera); }).catch(() => {});
    }
    if (job.status === 'complete') {
      showStatus('Preparing results', job.name);
      await sceneReady;
      const result = await loadPair(job.config.scene, gen, Boolean(splatView.mesh));
      if (gen !== generation) return;
      if (!EXAMPLES.some(example => example.id === job.id)) {
        const example = {id: job.id, label: job.name, ...job.config.scene};
        EXAMPLES.push(example);
        settingsMenu.addExample(example);
      }
      settingsMenu.setExampleId(job.id);
      showStatus('Analysis complete', `${job.name} · ${inventorySummary(result)}`, 'complete');
      download.href = job.config.scene.annotationsUrl;
      download.classList.remove('hidden');
      return;
    }
    const count = job.total ? ` · ${job.completed}/${job.total}` : '';
    showStatus(job.stage, job.name + count, 'running', job.completed, job.total);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
async function analyzeFile(file) {
  if (!/\.(ply|spz)$/i.test(file.name)) { showStatus('Unsupported file', 'Choose a .ply or .spz Gaussian splat.', 'failed'); return; }
  const gen = resetScene();
  settingsMenu.setExampleId('');
  showLoading('Loading scene…');
  showStatus('Uploading scene', file.name);
  settingsMenu.setBusy(true);
  // Preview preparation and backend progress run independently.
  const preview = splatView.load(file, loaderOptions(gen)).then(() => {
    if (gen === generation) hideLoading();
  });
  preview.catch(() => {}); // Observed again when the job finishes; prevent an unhandled rejection.
  try {
    if (!configuration.analysisAvailable) {
      await preview;
      showStatus('Scene preview loaded', 'Object analysis is not connected to this viewer.', 'failed');
      return;
    }
    const id = await uploadScene(file, settingsMenu.getQuality(), settingsMenu.getCameraFiles());
    await pollJob(id, gen, preview);
  } catch (error) {
    if (gen === generation) { hideLoading(); showStatus('Analysis failed', error.message, 'failed'); }
  } finally {
    if (gen === generation) settingsMenu.setBusy(understanding.busy);
  }
}
const settingsMenu = new SettingsMenu({
  examples: EXAMPLES, defaultExampleId: EXAMPLES[0]?.id || '',
  onShowContents: checked => { showContents = checked; displayAnnotations(); },
  onShowUnconfirmedEnvironment: checked => { showUnconfirmedEnvironment = checked; displayAnnotations(); },
  onExampleSelect: id => loadExample(id),
  onSplatFile: file => analyzeFile(file),
  onAnnotationsFile: async file => {
    try {
      const result = JSON.parse(await file.text());
      if (sourceHash && result.input?.sha256 && sourceHash !== result.input.sha256) throw new Error('These annotations belong to a different scene.');
      understanding.reset();
      selectedIds = null;
      displayAnnotations(AnnotationParser.fromText(JSON.stringify(result)));
    } catch (error) { showStatus('Annotations could not be loaded', error.message, 'failed'); }
  },
});
const navigationLayer = new NavigationLayer(sceneManager, settingsMenu.navigationHost);
document.getElementById('choose-scene').addEventListener('click', () => settingsMenu.chooseSplat());
const playBtn = document.createElement('button');
playBtn.id = 'play-btn'; playBtn.className = 'fab'; playBtn.title = 'Play label appear animation';
playBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>';
playBtn.addEventListener('click', () => annotationLayer.playReveal());
document.body.appendChild(playBtn);
const resetView = document.createElement('button');
resetView.id = 'reset-view'; resetView.className = 'fab'; resetView.title = 'Reset view';
resetView.textContent = '⌂';
resetView.addEventListener('click', () => {
  sceneManager.navigationTouched = false;
  applyCamera(lastCamera);
  sceneManager.renderer.domElement.focus({preventScroll: true});
});
document.body.appendChild(resetView);
sceneManager.start();
let lastStatsUpdate = 0;
sceneManager.addFrameListener(() => {
  const now = performance.now();
  if (now - lastStatsUpdate < 250) return;
  lastStatsUpdate = now;
  settingsMenu.updateStats({fps: sceneManager.fps, splatCount: splatView.numSplats, labelCount: annotationLayer.objects.length});
});
if (configuration.activeJob) {
  const gen = resetScene();
  settingsMenu.setBusy(true);
  try {
    const response = await fetch(`/api/jobs/${configuration.activeJob}`);
    const job = await response.json();
    const ready = job.splatUrl ? splatView.load(job.splatUrl, loaderOptions(gen)) : Promise.resolve();
    ready.then(() => { if (gen === generation) hideLoading(); }).catch(() => {});
    if (job.splatUrl) showLoading('Loading scene…');
    await pollJob(configuration.activeJob, gen, ready);
  } catch (error) { showStatus('Analysis failed', error.message, 'failed'); }
  finally { settingsMenu.setBusy(understanding.busy); }
} else if (EXAMPLES.length) loadExample(EXAMPLES[0].id);
