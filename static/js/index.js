'use strict';

const families = {
  m1: { name: 'M1 plain thread', description: 'Turn the cap to unscrew it, then lift it: the display shows shell rotation and thread progress rising together, with no press or squeeze needed (a press past the dashed line would jam this thread), until the cap comes free and lifts off.' },
  m2: { name: 'M2 push-and-turn', description: 'Push the cap down past its gate while turning: the display shows the press depth crossing the gate line as the teal band under the cap disappears, and the thread advancing only while the cap is held down.' },
  m3: { name: 'M3 squeeze-and-turn', description: 'Squeeze the two orange side tabs past their gate while turning: the display shows each squeeze crossing the gate line and the thread advancing only during the squeezes, while in between the shell slips round with clicks (a short press at the start jams the thread for two steps).' },
  m6: { name: 'M6 stuck seal', description: 'Twist hard enough to break the seal, then unscrew: the display shows the torque on the shell reaching the breakaway line, the seal breaking after a third of a second, and then ordinary unscrewing.' }
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
  familyVideo.poster = `./static/videos/${key}-poster.jpg?v=v3`;
  familyVideo.setAttribute('aria-label', `${fam.name}: the policy opens and lifts the cap; a state display shows the mechanism.`);
  familyVideo.querySelector('source').src = `./static/videos/${key}.mp4?v=v3`;
  familyVideo.querySelector('a').href = `./static/videos/${key}.mp4?v=v3`;
  document.getElementById('family-description').textContent = fam.description;
  familyVideo.load();
  if (!reduceMotion.matches) familyVideo.play().catch(() => {});
});
