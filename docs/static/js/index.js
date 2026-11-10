// Page behaviour for the academic project page template. Plain JavaScript, no dependencies.
// Each feature does nothing when its markup is absent, so remove markup freely.
//   - More Works menu      (.more-works)
//   - Copy BibTeX          ([data-copy-target])
//   - Contents and top     (.page-nav; the contents list is built from the page's h2 headings)
//   - Carousels            ([data-carousel]; data-autoplay="6000" opts in to rotation)
// Adapted from the Academic Project Page Template (CC BY-SA 4.0); jQuery and bulma-carousel removed.
(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  // ---------- More Works menu ----------
  document.querySelectorAll('.more-works').forEach(menu => {
    const toggle = menu.querySelector('.more-works-btn');
    const panel = menu.querySelector('.more-works-panel');
    const close = menu.querySelector('.close-btn');
    if (!toggle || !panel) return;

    const setOpen = (open, returnFocus) => {
      toggle.setAttribute('aria-expanded', String(open));
      panel.hidden = !open;
      if (open) panel.querySelector('a, button')?.focus();
      else if (returnFocus) toggle.focus();
    };
    setOpen(false);
    toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
    close?.addEventListener('click', () => setOpen(false, true));
    document.addEventListener('click', event => {
      if (!menu.contains(event.target) && toggle.getAttribute('aria-expanded') === 'true') setOpen(false);
    });
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
        event.stopPropagation();
        setOpen(false, true);
      }
    });
    menu.addEventListener('focusout', event => {
      if (event.relatedTarget && !menu.contains(event.relatedTarget)) setOpen(false);
    });
  });

  // ---------- Copy BibTeX ----------
  document.querySelectorAll('[data-copy-target]').forEach(button => {
    const source = document.getElementById(button.dataset.copyTarget);
    const label = button.querySelector('.copy-text');
    const status = document.getElementById(button.dataset.copyStatus);
    if (!source) return;
    const idleLabel = label ? label.textContent : '';
    let timer;

    const fallbackCopy = text => {
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.append(area);
      area.select();
      const ok = document.execCommand('copy');
      area.remove();
      return ok;
    };

    const announce = (ok) => {
      clearTimeout(timer);
      button.classList.toggle('is-copied', ok);
      if (label) label.textContent = ok ? 'Copied' : 'Copy failed';
      if (status) status.textContent = ok ? 'BibTeX copied to the clipboard.' : 'Copy failed. Select the BibTeX text and copy it manually.';
      timer = setTimeout(() => {
        button.classList.remove('is-copied');
        if (label) label.textContent = idleLabel;
        if (status) status.textContent = '';
      }, 2500);
    };

    button.addEventListener('click', async () => {
      const text = source.textContent;
      try {
        await navigator.clipboard.writeText(text);
        announce(true);
      } catch {
        let ok = false;
        try { ok = fallbackCopy(text); } catch {}
        announce(ok);
      }
    });
  });

  // ---------- Contents menu and back to top ----------
  // The menu is built from the h2 headings in <main>, so it fits any set of sections. Delete the
  // .page-nav block from the page to remove the feature.
  const pageNav = document.querySelector('.page-nav');
  if (pageNav) {
    const menu = pageNav.querySelector('.page-nav-menu');
    const toggle = pageNav.querySelector('.page-nav-toggle');
    const toTop = pageNav.querySelector('.page-nav-top');
    const list = pageNav.querySelector('.page-nav-list');
    const close = pageNav.querySelector('.page-nav-close');
    const header = document.querySelector('.publication-header');
    const used = new Set([...document.querySelectorAll('[id]')].map(el => el.id));

    // One entry per section heading: the link goes to the section's id, else the heading's id (made if absent).
    const sections = [];
    if (menu && toggle && list) {
      document.querySelectorAll('main h2').forEach(heading => {
        const label = heading.textContent.trim().replace(/\s+/g, ' ');
        if (!label || heading.closest('[hidden]')) return;
        const holder = heading.closest('section') || heading;
        if (!holder.id && !heading.id) {
          const base = label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'section';
          let id = base;
          for (let n = 2; used.has(id); n++) id = `${base}-${n}`;
          used.add(id);
          heading.id = id;
        }
        const link = document.createElement('a');
        link.href = `#${holder.id || heading.id}`;
        link.textContent = label;
        const item = document.createElement('li');
        item.append(link);
        list.append(item);
        sections.push({ link, heading });
      });
    }
    // A contents list of fewer than three entries is not worth a button; keep back to top only.
    const hasMenu = sections.length >= 3;
    if (!hasMenu) {
      menu?.remove();
      toggle?.remove();
    }

    const setOpen = (open, returnFocus) => {
      if (!hasMenu || menu.hidden === !open) return;
      toggle.setAttribute('aria-expanded', String(open));
      menu.hidden = !open;
      if (open) (list.querySelector('[aria-current]') || list.querySelector('a')).focus();
      else if (returnFocus) toggle.focus();
    };

    // Mark the section the reader is in: the last heading above a line near the top of the screen.
    const markCurrent = () => {
      const line = window.innerHeight * 0.3;
      let current = null;
      sections.forEach(entry => { if (entry.heading.getBoundingClientRect().top <= line) current = entry; });
      sections.forEach(entry => {
        if (entry === current) entry.link.setAttribute('aria-current', 'location');
        else entry.link.removeAttribute('aria-current');
      });
    };

    // Appears once the header is scrolled past. Where the controls would sit over the text column (phones,
    // tablets, narrow windows) they step aside while the reader scrolls down and return on any scroll up,
    // at the end of the page, or while in use. In a wide gutter they stay.
    const buttons = pageNav.querySelector('.page-nav-buttons') || pageNav;
    const column = document.querySelector('main .container');
    const overColumn = () => !column || buttons.getBoundingClientRect().left < column.getBoundingClientRect().right;
    let lastY = window.scrollY;
    let reading = false;
    const update = () => {
      const y = window.scrollY;
      if (Math.abs(y - lastY) > 4) {
        reading = y > lastY;
        lastY = y;
      }
      const start = header ? header.offsetTop + header.offsetHeight : 400;
      const atEnd = window.innerHeight + y >= document.documentElement.scrollHeight - 80;
      const inUse = (hasMenu && !menu.hidden) || pageNav.contains(document.activeElement);
      const visible = y > start && (!reading || atEnd || inUse || !overColumn());
      pageNav.classList.toggle('is-visible', visible);
      if (!visible) setOpen(false, false);
      if (hasMenu) markCurrent();
    };
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();

    if (hasMenu) {
      toggle.addEventListener('click', () => setOpen(menu.hidden));
      close?.addEventListener('click', () => setOpen(false, true));
      list.addEventListener('click', event => { if (event.target.closest('a')) setOpen(false); });
      document.addEventListener('click', event => {
        if (!menu.hidden && !pageNav.contains(event.target)) setOpen(false);
      });
      pageNav.addEventListener('focusout', event => {
        if (!menu.hidden && event.relatedTarget && !pageNav.contains(event.relatedTarget)) setOpen(false);
      });
      document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !menu.hidden) setOpen(false, pageNav.contains(document.activeElement));
      });
    }

    toTop?.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'auto' : 'smooth' });
      // Move focus to the top of the document so keyboard users continue from there.
      const target = document.getElementById('main-content');
      if (target) target.focus({ preventScroll: true });
    });
  }

  // ---------- Carousels ----------
  // Icons drawn by the carousel: inline SVG from Font Awesome Free 6.7.2 (CC BY 4.0), as in the page.
  const ICONS = {
    'chevron-left': ['0 0 320 512', 'M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l192 192c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256 246.6 86.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-192 192z'],
    'chevron-right': ['0 0 320 512', 'M310.6 233.4c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L242.7 256 73.4 86.6c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0l192 192z'],
    'play': ['0 0 384 512', 'M73 39c-14.8-9.1-33.4-9.4-48.5-.9S0 62.6 0 80L0 432c0 17.4 9.4 33.4 24.5 41.9s33.7 8.1 48.5-.9L361 297c14.3-8.7 23-24.2 23-41s-8.7-32.2-23-41L73 39z'],
    'pause': ['0 0 320 512', 'M48 64C21.5 64 0 85.5 0 112L0 400c0 26.5 21.5 48 48 48l32 0c26.5 0 48-21.5 48-48l0-288c0-26.5-21.5-48-48-48L48 64zm192 0c-26.5 0-48 21.5-48 48l0 288c0 26.5 21.5 48 48 48l32 0c26.5 0 48-21.5 48-48l0-288c0-26.5-21.5-48-48-48l-32 0z'],
  };
  const icon = name => `<svg class="icon-svg" viewBox="${ICONS[name][0]}" aria-hidden="true"><path d="${ICONS[name][1]}"/></svg>`;

  document.querySelectorAll('[data-carousel]:not(.is-ready)').forEach(carousel => {
    const track = carousel.querySelector('.carousel-track');
    const viewport = carousel.querySelector('.carousel-viewport');
    const slides = track ? [...track.children] : [];
    if (slides.length < 2) {
      carousel.classList.add('is-ready');
      return;
    }

    const count = slides.length;
    const interval = Number(carousel.dataset.autoplay) || 0;
    let index = 0;
    let playing = interval > 0 && !reducedMotion.matches;
    let hovered = false;
    let focused = false;
    let inView = false;
    let viewerOpen = false;
    let timer = null;

    slides.forEach((slide, i) => {
      slide.setAttribute('role', 'group');
      slide.setAttribute('aria-roledescription', 'slide');
      slide.setAttribute('aria-label', `${i + 1} of ${count}`);
    });

    // Controls: previous, one dot per slide, next, and a pause/play button only when rotation is enabled.
    const controls = document.createElement('div');
    controls.className = 'carousel-controls';
    controls.innerHTML = `
      <button type="button" class="carousel-button carousel-prev" aria-label="Previous slide">${icon('chevron-left')}</button>
      <div class="carousel-dots" role="group" aria-label="Choose a slide">
        ${slides.map((_, i) => `<button type="button" class="carousel-dot" aria-label="Show slide ${i + 1} of ${count}"></button>`).join('')}
      </div>
      <button type="button" class="carousel-button carousel-next" aria-label="Next slide">${icon('chevron-right')}</button>
      ${interval ? `<button type="button" class="carousel-button carousel-toggle"></button>` : ''}`;
    carousel.append(controls);
    const dots = [...controls.querySelectorAll('.carousel-dot')];
    const toggle = controls.querySelector('.carousel-toggle');

    const videos = slide => [...slide.querySelectorAll('video')];
    // Muted carousel videos play only while their slide is current, the carousel is on screen,
    // and the visitor has not asked for reduced motion. Everything else stays paused.
    const syncVideos = () => {
      slides.forEach((slide, i) => videos(slide).forEach(video => {
        const shouldPlay = i === index && inView && !reducedMotion.matches && video.muted;
        if (shouldPlay && video.paused) video.play().catch(() => {});
        else if (!shouldPlay && !video.paused) video.pause();
      }));
    };

    const render = () => {
      track.style.transform = `translateX(-${index * 100}%)`;
      slides.forEach((slide, i) => {
        const current = i === index;
        // Off-screen slides leave the tab order and the accessibility tree.
        slide.inert = !current;
        if (current) slide.removeAttribute('aria-hidden');
        else slide.setAttribute('aria-hidden', 'true');
      });
      dots.forEach((dot, i) => dot.setAttribute('aria-current', String(i === index)));
      syncVideos();
    };

    const go = (next, fromUser) => {
      index = (next + count) % count;
      render();
      if (fromUser) restart();
    };

    const stop = () => { clearInterval(timer); timer = null; };
    const restart = () => {
      stop();
      if (playing && !hovered && !focused && !viewerOpen && inView && !document.hidden) timer = setInterval(() => go(index + 1), interval);
    };
    const renderToggle = () => {
      if (!toggle) return;
      toggle.setAttribute('aria-label', playing ? `Pause automatic slide show` : `Start automatic slide show`);
      toggle.innerHTML = icon(playing ? 'pause' : 'play');
      // Announce slide changes only when the visitor is driving them.
      track.setAttribute('aria-live', playing ? 'off' : 'polite');
    };

    controls.querySelector('.carousel-prev').addEventListener('click', () => go(index - 1, true));
    controls.querySelector('.carousel-next').addEventListener('click', () => go(index + 1, true));
    dots.forEach((dot, i) => dot.addEventListener('click', () => go(i, true)));
    toggle?.addEventListener('click', () => { playing = !playing; renderToggle(); restart(); });

    carousel.addEventListener('keydown', event => {
      if (event.target.closest('video, input, textarea')) return;
      if (event.key === 'ArrowLeft') { go(index - 1, true); event.preventDefault(); }
      if (event.key === 'ArrowRight') { go(index + 1, true); event.preventDefault(); }
    });

    // Rotation pauses while the pointer or keyboard focus is inside the carousel...
    carousel.addEventListener('mouseenter', () => { hovered = true; restart(); });
    carousel.addEventListener('mouseleave', () => { hovered = false; restart(); });
    carousel.addEventListener('focusin', () => { focused = true; restart(); });
    carousel.addEventListener('focusout', event => { if (!carousel.contains(event.relatedTarget)) { focused = false; restart(); } });
    document.addEventListener('visibilitychange', restart);
    // ...and while the figure viewer shows an enlarged image.
    document.addEventListener('figure-viewer:open', () => { viewerOpen = true; restart(); });
    document.addEventListener('figure-viewer:close', () => { viewerOpen = false; restart(); });

    // Swipe on touch screens.
    let startX = null;
    viewport.addEventListener('pointerdown', event => { if (event.pointerType !== 'mouse') startX = event.clientX; });
    viewport.addEventListener('pointerup', event => {
      if (startX === null) return;
      const dx = event.clientX - startX;
      startX = null;
      if (Math.abs(dx) > 40) go(index + (dx < 0 ? 1 : -1), true);
    });
    viewport.addEventListener('pointercancel', () => { startX = null; });

    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      syncVideos();
      restart();
    }, { threshold: 0.5 }).observe(viewport);

    reducedMotion.addEventListener('change', () => {
      if (reducedMotion.matches) playing = false;
      renderToggle();
      syncVideos();
      restart();
    });

    carousel.classList.add('is-ready');
    track.setAttribute('aria-live', 'polite');
    renderToggle();
    render();
  });
})();
