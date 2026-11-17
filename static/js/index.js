'use strict';

// One player for the teaser and the four per-type clips; the clip label under the player covers all five.
const clips = {
  teaser: { name: 'All four closure types', text: 'One policy, four look-alike closures. Clockwise from top left: plain thread, push-and-turn, stuck seal and squeeze-and-turn. The policy is not told which one it is holding.' },
  m1: { name: 'Plain thread (M1)', text: 'Plain thread: the hand unscrews and lifts the cap; pressing down would jam the thread.' },
  m2: { name: 'Push-and-turn (M2)', text: 'Push-and-turn: the thread advances only while the cap is pushed down (teal arrow).' },
  m3: { name: 'Squeeze-and-turn (M3)', text: 'Squeeze-and-turn: the thread advances only while both tabs are squeezed (orange arrows); in between, the shell spins freely.' },
  m6: { name: 'Stuck seal (M6)', text: 'Stuck seal: on screen this looks like the plain thread; the twist (violet arc) breaks the hidden seal, then the cap unscrews.' }
};

const select = document.getElementById('clip-select');
const video = document.getElementById('clip-video');
const clipText = document.getElementById('clip-text');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (reduceMotion.matches) { video.removeAttribute('autoplay'); video.pause(); }

function show(key) {
  const clip = clips[key];
  video.pause();
  video.poster = `./static/videos/${key}-poster.jpg?v=v3b`;
  video.setAttribute('aria-label', `${clip.name}: one policy opens the closure in simulation; a state display shows the mechanism.`);
  video.querySelector('source').src = `./static/videos/${key}.mp4?v=v3b`;
  video.querySelector('a').href = `./static/videos/${key}.mp4?v=v3b`;
  clipText.textContent = clip.text;
  video.load();
  if (!reduceMotion.matches) video.play().catch(() => {});
}

select.addEventListener('change', () => show(select.value));
if (select.value !== 'teaser') show(select.value);   // a browser may restore an earlier choice on reload

// close the "More Research" menu on a click elsewhere or on Escape
const menu = document.querySelector('.more-research');
if (menu) {
  document.addEventListener('click', (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); } });
}
