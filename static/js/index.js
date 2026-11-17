'use strict';

const families = {
  m1: { name: 'M1 plain thread', description: 'Turn the cap to unscrew it, then lift it: the display shows shell rotation and thread progress rising together, with no press or squeeze needed (a press past the dashed line would jam this thread), until the cap comes free and lifts off.' },
  m2: { name: 'M2 push-and-turn', description: 'Push the cap down past its gate while turning: the teal arrow on the cap top grows with the press depth and the teal band under the cap disappears; the display shows the press crossing the gate line and the thread advancing only while the cap is held down (when the hand eases off, the gate locks again until the next push).' },
  m3: { name: 'M3 squeeze-and-turn', description: 'Squeeze the two orange side tabs past their gate while turning: the fingertips cover the tabs, so an orange arrow on each tab shows how far it is pushed in (faint when the tab is behind the cap); the display shows each squeeze crossing the gate line and the thread advancing only during the squeezes, while in between the shell freewheels (the thin shell line climbs while the thick thread line stays flat). A short press at the start jams the thread for two steps.' },
  m6: { name: 'M6 stuck seal', description: 'Twist hard enough to break the seal, then unscrew: on screen this looks like M1, because the seal is not visible; the violet arc over the cap shows the twist while the seal holds, and the display shows the torque on the shell reaching the breakaway line, the seal breaking after a third of a second, and then ordinary unscrewing.' }
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
  familyVideo.poster = `./static/videos/${key}-poster.jpg?v=v3b`;
  familyVideo.setAttribute('aria-label', `${fam.name}: the policy opens and lifts the cap; a state display shows the mechanism.`);
  familyVideo.querySelector('source').src = `./static/videos/${key}.mp4?v=v3b`;
  familyVideo.querySelector('a').href = `./static/videos/${key}.mp4?v=v3b`;
  document.getElementById('family-description').textContent = fam.description;
  familyVideo.load();
  if (!reduceMotion.matches) familyVideo.play().catch(() => {});
});
