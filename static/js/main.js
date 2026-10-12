/* ================================================================
   RailSync AI — main.js  (Home page)
   ================================================================ */

// ── Navbar scroll effect ─────────────────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

// ── Hamburger (mobile) ───────────────────────────────────────────
document.getElementById('hamburger')?.addEventListener('click', () => {
  const links = document.querySelector('.navbar-links');
  if (links) {
    const open = links.style.display === 'flex';
    links.style.cssText = open
      ? ''
      : 'display:flex;flex-direction:column;position:absolute;top:70px;left:0;right:0;background:var(--navy-dark);padding:24px;gap:20px;z-index:999;';
  }
});

// ── Scroll reveal (Intersection Observer) ───────────────────────
const revealEls = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      revealObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
revealEls.forEach(el => revealObserver.observe(el));

// ── Animated counter ────────────────────────────────────────────
function animateCount(el, target, duration = 1600, suffix = '', decimal = false) {
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // cubic ease-out
    const val = Math.round(ease * target);
    el.textContent = decimal ? (val / 100).toFixed(2) + suffix : val.toLocaleString() + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

const statObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target;
    const target  = parseInt(el.dataset.target, 10);
    const suffix  = el.dataset.suffix  || '';
    const decimal = el.dataset.decimal === 'true';
    animateCount(el, target, 1800, suffix, decimal);
    statObserver.unobserve(el);
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-number').forEach(el => statObserver.observe(el));

// ── Hero counter strip ───────────────────────────────────────────
function animateHeroCount(el, target, suffix = '') {
  const start = performance.now();
  const dur = 2000;
  const update = (now) => {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val = Math.round(ease * target);
    el.textContent = val.toLocaleString() + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

window.addEventListener('load', () => {
  setTimeout(() => {
    document.querySelectorAll('.trust-number[data-count]').forEach(el => {
      animateHeroCount(el, parseInt(el.dataset.count), '');
    });
  }, 600);
});

// ── Load trains for preview grid ─────────────────────────────────
async function loadTrainsPreview() {
  const grid = document.getElementById('trainsPreviewGrid');
  if (!grid) return;
  try {
    const res  = await fetch('/api/trains');
    const data = await res.json();
    grid.innerHTML = '';
    data.trains.forEach((t, i) => {
      const card = document.createElement('div');
      card.className = 'train-preview-card reveal';
      card.style.setProperty('--card-color', t.color);
      card.style.transitionDelay = `${i * 0.08}s`;
      const severity = t.avg_delay_min <= 5 ? 'on-time' : t.avg_delay_min <= 15 ? 'minor' : t.avg_delay_min <= 30 ? 'moderate' : 'major';
      card.innerHTML = `
        <div class="tpc-number">${t.train_number}</div>
        <div class="tpc-name">${t.name}</div>
        <div class="tpc-route">${t.source_code} → ${t.dest_code}</div>
        <div class="tpc-delay badge badge-${severity}">Avg. ${t.avg_delay_min} min delay</div>
      `;
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => { window.location.href = '/predict'; });
      grid.appendChild(card);
    });
    // Re-observe new reveal elements
    grid.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));
  } catch (err) {
    console.error('Failed to load trains:', err);
    grid.innerHTML = '<p style="color:var(--navy-muted);padding:24px">Could not load train data.</p>';
  }
}

loadTrainsPreview();
