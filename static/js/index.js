'use strict';

const families = {
  m1: { name: 'M1 plain thread', description: 'Turn to unscrew, then lift. Pressing this cap jams its thread.' },
  m2: { name: 'M2 push-and-turn', description: 'Press the cap down past its threshold while turning; without the press it only freewheels.' },
  m3: { name: 'M3 squeeze-and-turn', description: 'Squeeze the two side tabs past their threshold while turning. Pressing jams it.' },
  m6: { name: 'M6 stuck seal', description: 'Turn hard enough to break the seal free, then unscrew. Pressing jams it.' }
};

const selector = document.getElementById('family-select');
const familyVideo = document.getElementById('family-video');
const teaser = document.getElementById('teaser-video');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (reduceMotion.matches && teaser) { teaser.removeAttribute('autoplay'); teaser.pause(); teaser.controls = true; }

selector.addEventListener('change', () => {
  const key = selector.value;
  const fam = families[key];
  familyVideo.pause();
  familyVideo.poster = `./static/videos/${key}-poster.jpg`;
  familyVideo.setAttribute('aria-label', `${fam.name}: the policy opens and lifts the cap.`);
  familyVideo.querySelector('source').src = `./static/videos/${key}.mp4`;
  familyVideo.querySelector('a').href = `./static/videos/${key}.mp4`;
  document.getElementById('family-description').textContent = fam.description;
  familyVideo.load();
  if (!reduceMotion.matches) familyVideo.play().catch(() => {});
});
