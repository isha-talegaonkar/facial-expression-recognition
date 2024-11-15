const labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const icons = ['⚡', '◌', '◇', '☺', '☁', '✳', '○'];
const $ = (id) => document.getElementById(id);
const bars = $('prediction-bars');
const video = $('camera');
const photo = $('uploaded-image');
const canvas = $('sample-canvas');
const context = canvas.getContext('2d', {willReadFrequently: true});
let model, stream, timer, photoUrl;
let loading;

labels.forEach((name, index) => {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<span>${name}</span><div class="bar-track"><div class="bar-fill" id="bar-${index}"></div></div><span class="bar-percent" id="value-${index}">0%</span>`;
  bars.append(row);
});

async function loadModel() {
  if (model) return model;
  if (!loading) {
    $('demo-status').textContent = 'Loading the trained model (about 15 MB)…';
    loading = (async () => {
      if (!window.tf) throw new Error('TensorFlow.js did not load. Check your internet connection.');
      const loaded = await tf.loadLayersModel('./model.json');
      model = loaded;
      $('demo-status').textContent = 'Model ready. Align one face inside the guide.';
      return loaded;
    })().catch((error) => { loading = null; throw error; });
  }
  return loading;
}

function stopCamera() {
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = true;
  $('camera-button').textContent = 'Start camera';
  $('camera-state').textContent = 'Camera off';
}

function showError(error) {
  console.error(error);
  $('demo-status').textContent = error.message || String(error);
}

function drawCrop(source, width, height) {
  const stage = $('camera-stage');
  const displayRatio = stage.clientWidth / stage.clientHeight;
  const sourceRatio = width / height;
  let visibleWidth = width, visibleHeight = height;
  if (sourceRatio > displayRatio) visibleWidth = height * displayRatio;
  else visibleHeight = width / displayRatio;
  const guideSize = stage.querySelector('.face-guide').clientWidth / stage.clientWidth;
  const side = Math.min(visibleWidth * guideSize, visibleHeight * guideSize);
  context.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, 48, 48);
}

async function predict(source, width, height) {
  if (!model || !width || !height) return;
  drawCrop(source, width, height);
  const pixels = context.getImageData(0, 0, 48, 48).data;
  const grayscale = new Float32Array(48 * 48);
  for (let i = 0; i < grayscale.length; i++) {
    const offset = i * 4;
    grayscale[i] = (0.299 * pixels[offset] + 0.587 * pixels[offset + 1] + 0.114 * pixels[offset + 2]) / 255;
  }
  const input = tf.tensor4d(grayscale, [1, 48, 48, 1]);
  const output = model.predict(input);
  const scores = await output.data();
  input.dispose();
  output.dispose();
  let best = 0;
  scores.forEach((score, index) => {
    if (score > scores[best]) best = index;
    $('bar-' + index).style.width = (score * 100).toFixed(1) + '%';
    $('value-' + index).textContent = Math.round(score * 100) + '%';
  });
  $('prediction-name').textContent = labels[best];
  $('prediction-confidence').textContent = Math.round(scores[best] * 100) + '% model confidence';
  $('prediction-icon').textContent = icons[best];
}

$('camera-button').addEventListener('click', async () => {
  if (stream) { stopCamera(); return; }
  try {
    $('demo-status').textContent = 'Requesting camera access…';
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs HTTPS or localhost.');
    stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}, audio: false});
    if (photoUrl) { URL.revokeObjectURL(photoUrl); photoUrl = null; }
    photo.hidden = true;
    video.hidden = false;
    video.srcObject = stream;
    await video.play();
    $('camera-placeholder').hidden = true;
    $('camera-button').textContent = 'Stop camera';
    $('camera-state').textContent = 'Live';
    await loadModel();
    timer = setInterval(() => predict(video, video.videoWidth, video.videoHeight).catch(showError), 350);
  } catch (error) { stopCamera(); showError(error); }
});

$('image-upload').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  stopCamera();
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  photo.src = photoUrl;
  photo.hidden = false;
  $('camera-placeholder').hidden = true;
  $('camera-state').textContent = 'Photo preview';
  try {
    await new Promise((resolve, reject) => { photo.onload = resolve; photo.onerror = reject; });
    await loadModel();
    await predict(photo, photo.naturalWidth, photo.naturalHeight);
  } catch (error) { showError(error); }
  event.target.value = '';
});

window.addEventListener('pagehide', stopCamera);
