'use strict';

// One player for the teaser and the four per-type clips; the clip label under the player covers all five.
const clips = {
  teaser: { name: 'All four closure types', text: 'Top: M1 plain thread, M2 push-and-turn. Bottom: M3 squeeze-and-turn, M6 stuck seal.' },
  m1: { name: 'M1 plain thread', text: 'Plain thread: the hand unscrews and lifts the cap; pressing down would jam the thread.' },
  m2: { name: 'M2 push-and-turn', text: 'Push-and-turn: the thread advances only while, or just after, the cap is pushed down (teal arrow).' },
  m3: { name: 'M3 squeeze-and-turn', text: 'Squeeze-and-turn: the thread advances only while, or just after, both tabs are squeezed (orange arrows); in between, the shell turns without moving the thread.' },
  m6: { name: 'M6 stuck seal', text: 'Stuck seal: it looks like the plain thread, but a twist (violet arc) must first break a hidden seal; then the cap unscrews.' }
};

const tabs = Array.from(document.querySelectorAll('.clip-tabs [data-key]'));
const video = document.getElementById('clip-video');
const clipText = document.getElementById('clip-text');
const toggle = document.getElementById('clip-toggle');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (reduceMotion.matches) { video.removeAttribute('autoplay'); video.pause(); }

// play/pause toggle in the top corner (the native control bar would cover the state display at the bottom)
function syncToggle() {
  const paused = video.paused;
  toggle.classList.toggle('is-paused', paused);
  toggle.setAttribute('aria-label', paused ? 'Play the clip' : 'Pause the clip');
}
function playPause() {
  if (video.paused) video.play().catch(() => {}); else video.pause();
}
toggle.hidden = false;
toggle.addEventListener('click', playPause);
video.addEventListener('click', playPause);
video.addEventListener('play', syncToggle);
video.addEventListener('pause', syncToggle);
syncToggle();

function show(key) {
  const clip = clips[key];
  tabs.forEach((b) => {
    const on = b.dataset.key === key;
    b.classList.toggle('is-selected', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  video.pause();
  video.poster = `./static/videos/${key}-poster.jpg?v=v3c`;
  video.setAttribute('aria-label', `${clip.name}: one policy opens the closure in simulation; a state display shows the mechanism.`);
  video.querySelector('source').src = `./static/videos/${key}.mp4?v=v3c`;
  video.querySelector('a').href = `./static/videos/${key}.mp4?v=v3c`;
  clipText.textContent = clip.text;
  video.load();
  if (!reduceMotion.matches) video.play().catch(() => {});
  syncToggle();
}

tabs.forEach((b) => b.addEventListener('click', () => { if (!b.classList.contains('is-selected')) show(b.dataset.key); }));

// close the "More Research" menu on a click elsewhere or on Escape
const menu = document.querySelector('.more-research');
if (menu) {
  document.addEventListener('click', (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); } });
}
