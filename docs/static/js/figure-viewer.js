// Figure viewer for academic project pages.
// Add data-zoom to an <img> to make it enlargeable. Optional: data-zoom-src (larger file) and
// data-zoom-title (short title, defaults to the alt text). The caption comes from the enclosing
// <figure>'s <figcaption>, else the first <p> in the nearest [data-zoom-context] element.
// The figure grows from its place on the page, fits without upscaling, and zooms around the
// pointer (click, wheel, pinch, double-tap) or the keyboard.
(() => {
  const images = [...document.querySelectorAll('img[data-zoom]')];
  if (!images.length) return;

  const icon = path => `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
  const viewer = document.createElement('dialog');
  viewer.className = 'figure-viewer';
  viewer.setAttribute('aria-labelledby', 'figure-viewer-title');
  viewer.setAttribute('aria-describedby', 'figure-viewer-caption');
  viewer.innerHTML = `
    <button class="figure-viewer-close" type="button" aria-label="Close" title="Close (Esc)" autofocus>${icon('m6 6 12 12M18 6 6 18')}</button>
    <div class="figure-viewer-stage" tabindex="0" role="region" aria-label="Enlarged figure. Use plus and minus to zoom and arrow keys to move."></div>
    <p class="figure-viewer-status" role="status"></p>
    <div class="figure-viewer-footer">
      <p class="figure-viewer-caption"><strong id="figure-viewer-title"></strong> <span id="figure-viewer-caption"></span></p>
      <div class="figure-viewer-zoom" role="group" aria-label="Zoom" hidden>
        <button type="button" data-zoom-step="out" aria-label="Zoom out">${icon('M6 12h12')}</button>
        <output aria-live="polite">100%</output>
        <button type="button" data-zoom-step="in" aria-label="Zoom in">${icon('M6 12h12M12 6v12')}</button>
        <span class="figure-viewer-hint"></span>
      </div>
    </div>`;
  document.body.append(viewer);

  const stage = viewer.querySelector('.figure-viewer-stage');
  const footer = viewer.querySelector('.figure-viewer-footer');
  const title = viewer.querySelector('#figure-viewer-title');
  const caption = viewer.querySelector('#figure-viewer-caption');
  const status = viewer.querySelector('.figure-viewer-status');
  const zoomControls = viewer.querySelector('.figure-viewer-zoom');
  const zoomLevel = zoomControls.querySelector('output');
  const zoomIn = zoomControls.querySelector('[data-zoom-step="in"]');
  const zoomOut = zoomControls.querySelector('[data-zoom-step="out"]');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const coarsePointer = window.matchMedia('(pointer: coarse)');
  viewer.querySelector('.figure-viewer-hint').textContent = coarsePointer.matches ? 'Pinch or double-tap' : 'Scroll to zoom, drag to move';

  const MAX_SCALE = 2;
  const view = { image: null, opener: null, scale: 1, fit: 1, x: 0, y: 0, closing: false };
  const pointers = new Map();
  let gesture = null;
  let lastTap = 0;

  const zoomable = () => view.fit < 1;
  const atFit = () => Math.abs(view.scale - view.fit) < 0.001;

  // The stage fills the screen; a zoomed figure may pass under the caption bar,
  // but fitting and panning limits use only the area above it.
  function viewArea() {
    const pad = stage.clientWidth < 600 ? 12 : 24;
    return { left: pad, top: pad, width: stage.clientWidth - 2 * pad, height: stage.clientHeight - footer.offsetHeight - 2 * pad };
  }

  function clampAxis(position, size, start, length) {
    return size <= length ? start + (length - size) / 2 : Math.min(start, Math.max(start + length - size, position));
  }

  function render() {
    const { image } = view;
    const area = viewArea();
    view.x = clampAxis(view.x, image.naturalWidth * view.scale, area.left, area.width);
    view.y = clampAxis(view.y, image.naturalHeight * view.scale, area.top, area.height);
    // A new zoom or pan replaces the opening animation rather than waiting for it.
    image.getAnimations().forEach(animation => animation.cancel());
    image.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
    image.style.setProperty('--inverse-scale', 1 / view.scale);
    stage.classList.toggle('is-zoomed', !atFit());
    zoomLevel.textContent = `${Math.round(view.scale * 100)}%`;
    zoomOut.disabled = atFit();
    zoomIn.disabled = view.scale >= MAX_SCALE - 0.001;
  }

  function zoomTo(scale, originX, originY) {
    if (!view.image || !zoomable()) return;
    const area = viewArea();
    originX ??= area.left + area.width / 2;
    originY ??= area.top + area.height / 2;
    const next = Math.min(MAX_SCALE, Math.max(view.fit, scale));
    view.x = originX - (originX - view.x) * (next / view.scale);
    view.y = originY - (originY - view.y) * (next / view.scale);
    view.scale = next;
    render();
  }

  function toggleZoom(originX, originY) {
    zoomTo(atFit() ? Math.min(MAX_SCALE, Math.max(1, view.fit * 2)) : view.fit, originX, originY);
  }

  function fitImage() {
    const { image } = view;
    // Measure with the zoom bar shown; hiding it afterwards only adds room.
    zoomControls.hidden = false;
    const area = viewArea();
    view.fit = Math.min(1, area.width / image.naturalWidth, area.height / image.naturalHeight);
    view.scale = view.fit;
    zoomControls.hidden = !zoomable();
    stage.classList.toggle('is-zoomable', zoomable());
    render();
  }

  function animateFromOpener() {
    const thumbnail = view.opener.querySelector('img');
    if (reducedMotion.matches || !thumbnail) return;
    const from = thumbnail.getBoundingClientRect();
    const bounds = stage.getBoundingClientRect();
    if (!from.width) return;
    view.image.animate([
      { transform: `translate(${from.left - bounds.left}px, ${from.top - bounds.top}px) scale(${from.width / view.image.naturalWidth})` },
      { transform: view.image.style.transform },
    ], { duration: 240, easing: 'cubic-bezier(.2, 0, 0, 1)' });
  }

  function stagePoint(event) {
    const bounds = stage.getBoundingClientRect();
    return [event.clientX - bounds.left, event.clientY - bounds.top];
  }

  function onImage(x, y) {
    const { image } = view;
    return x >= view.x && y >= view.y && x <= view.x + image.naturalWidth * view.scale && y <= view.y + image.naturalHeight * view.scale;
  }

  // Caption sources in order: an explicit data-zoom-caption, the figure's figcaption, then a caption-like
  // element in the nearest context (a carousel item's .subtitle, or a paragraph in [data-zoom-context]).
  function describe(image) {
    if (image.dataset.zoomCaption) return image.dataset.zoomCaption.trim();
    const figcaption = image.closest('figure')?.querySelector('figcaption');
    const context = image.closest('[data-zoom-context], .item, .carousel-item');
    return (figcaption || context?.querySelector('figcaption, .subtitle, p'))?.textContent.trim() || '';
  }

  function open(trigger) {
    const source = trigger.querySelector('img');
    const image = new Image();
    image.alt = source.alt;
    image.draggable = false;
    Object.assign(view, { image, opener: trigger, closing: false });
    const heading = (source.dataset.zoomTitle || source.alt || '').replace(/\.$/, '');
    const description = describe(source);
    // A caption that already names the figure ("Figure 2. Cumulative attainment ...") needs no separate title.
    title.textContent = heading && !description.toLowerCase().includes(heading.toLowerCase()) ? `${heading}.` : '';
    caption.textContent = description;
    status.textContent = '';
    zoomControls.hidden = true;
    stage.replaceChildren(image);
    viewer.classList.remove('is-closing');
    viewer.showModal();
    document.body.classList.add('figure-viewer-open');
    // Lets page scripts pause carousels or other motion while a figure is open.
    document.dispatchEvent(new CustomEvent('figure-viewer:open', { detail: { trigger } }));

    const show = () => {
      if (!viewer.open || view.image !== image) return;
      status.textContent = '';
      fitImage();
      animateFromOpener();
    };
    image.addEventListener('error', () => {
      if (viewer.open && view.image === image) status.textContent = 'The image could not be loaded. Close this preview and try again.';
    });
    image.src = source.dataset.zoomSrc || source.currentSrc || source.src;
    if (image.complete && image.naturalWidth) show();
    else {
      status.textContent = 'Loading image…';
      image.addEventListener('load', show, { once: true });
    }
  }

  images.forEach(image => {
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'figure-zoom-trigger';
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.setAttribute('aria-label', `Enlarge ${image.dataset.zoomTitle || image.alt || 'figure'}`);
    image.replaceWith(trigger);
    trigger.append(image);
  });

  // One delegated listener, so copies made later (carousel libraries clone slides) open too.
  document.addEventListener('click', event => {
    const trigger = event.target.closest?.('.figure-zoom-trigger');
    if (trigger && !viewer.contains(trigger)) open(trigger);
  });

  function close() {
    if (!viewer.open || view.closing) return;
    view.closing = true;
    if (reducedMotion.matches) { viewer.close(); return; }
    viewer.classList.add('is-closing');
    setTimeout(() => { if (view.closing) viewer.close(); }, 150);
  }

  viewer.addEventListener('cancel', event => { event.preventDefault(); close(); });
  viewer.querySelector('.figure-viewer-close').addEventListener('click', close);
  viewer.addEventListener('close', () => {
    viewer.classList.remove('is-closing');
    document.body.classList.remove('figure-viewer-open');
    stage.replaceChildren();
    pointers.clear();
    gesture = null;
    view.image = null;
    view.closing = false;
    document.dispatchEvent(new CustomEvent('figure-viewer:close', { detail: { trigger: view.opener } }));
    // A carousel may have moved the opener out of view; return focus to its carousel instead.
    const opener = view.opener;
    if (opener?.checkVisibility() && !opener.closest('[inert], [aria-hidden="true"]')) opener.focus({ preventScroll: true });
    else {
      const fallback = opener?.closest('.carousel, [data-zoom-return]');
      if (fallback) {
        if (!fallback.hasAttribute('tabindex')) fallback.setAttribute('tabindex', '-1');
        fallback.focus({ preventScroll: true });
      }
    }
  });
  viewer.addEventListener('click', event => { if (event.target === viewer) close(); });
  zoomIn.addEventListener('click', () => zoomTo(view.scale * 1.5));
  zoomOut.addEventListener('click', () => zoomTo(view.scale / 1.5));

  stage.addEventListener('pointerdown', event => {
    if (!view.image?.naturalWidth) return;
    stage.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, stagePoint(event));
    if (pointers.size === 2) {
      const [[ax, ay], [bx, by]] = [...pointers.values()];
      gesture = { type: 'pinch', distance: Math.hypot(ax - bx, ay - by), scale: view.scale, midX: (ax + bx) / 2, midY: (ay + by) / 2 };
    } else if (pointers.size === 1) {
      const [x, y] = pointers.get(event.pointerId);
      gesture = { type: 'pan', startX: x, startY: y, lastX: x, lastY: y, moved: false };
    }
  });

  stage.addEventListener('pointermove', event => {
    if (!pointers.has(event.pointerId) || !gesture) return;
    pointers.set(event.pointerId, stagePoint(event));
    if (gesture.type === 'pinch' && pointers.size === 2) {
      const [[ax, ay], [bx, by]] = [...pointers.values()];
      const midX = (ax + bx) / 2;
      const midY = (ay + by) / 2;
      view.x += midX - gesture.midX;
      view.y += midY - gesture.midY;
      Object.assign(gesture, { midX, midY });
      zoomTo(gesture.scale * Math.hypot(ax - bx, ay - by) / gesture.distance, midX, midY);
    } else if (gesture.type === 'pan') {
      const [x, y] = pointers.get(event.pointerId);
      if (Math.hypot(x - gesture.startX, y - gesture.startY) > 6) gesture.moved = true;
      if (gesture.moved && !atFit()) {
        view.x += x - gesture.lastX;
        view.y += y - gesture.lastY;
        render();
      }
      Object.assign(gesture, { lastX: x, lastY: y });
    }
  });

  function endPointer(event) {
    if (!pointers.has(event.pointerId)) return;
    const [x, y] = pointers.get(event.pointerId);
    pointers.delete(event.pointerId);
    if (event.type === 'pointerup' && gesture?.type === 'pan' && !gesture.moved) {
      if (!onImage(x, y) || !zoomable()) close();
      else if (event.pointerType === 'mouse') toggleZoom(x, y);
      else if (event.timeStamp - lastTap < 300) { toggleZoom(x, y); lastTap = 0; }
      else lastTap = event.timeStamp;
    }
    // Lifting one finger of a pinch continues as a pan from the remaining finger.
    if (pointers.size === 1) {
      const [px, py] = [...pointers.values()][0];
      gesture = { type: 'pan', startX: px, startY: py, lastX: px, lastY: py, moved: true };
    } else if (!pointers.size) gesture = null;
  }
  stage.addEventListener('pointerup', endPointer);
  stage.addEventListener('pointercancel', endPointer);

  stage.addEventListener('wheel', event => {
    if (!view.image?.naturalWidth || !zoomable()) return;
    event.preventDefault();
    const [x, y] = stagePoint(event);
    // The wheel always zooms around the cursor; dragging moves. Trackpad pinches arrive as ctrl+wheel.
    const delta = event.deltaY * (event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : 1);
    zoomTo(view.scale * Math.exp(-delta * (event.ctrlKey ? 0.01 : 0.002)), x, y);
  }, { passive: false });

  viewer.addEventListener('keydown', event => {
    if (event.key === 'Tab') {
      const stops = [...viewer.querySelectorAll('button:not(:disabled), [tabindex="0"]')].filter(element => element.offsetParent);
      const first = stops[0];
      const last = stops.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
      return;
    }
    if (!view.image?.naturalWidth || !zoomable()) return;
    const pan = { ArrowLeft: [80, 0], ArrowRight: [-80, 0], ArrowUp: [0, 80], ArrowDown: [0, -80] }[event.key];
    if (event.key === '+' || event.key === '=') zoomTo(view.scale * 1.5);
    else if (event.key === '-') zoomTo(view.scale / 1.5);
    else if (event.key === '0') zoomTo(view.fit);
    else if (pan && !atFit()) {
      view.x += pan[0];
      view.y += pan[1];
      render();
    } else return;
    event.preventDefault();
  });

  window.addEventListener('resize', () => {
    if (!viewer.open || !view.image?.naturalWidth) return;
    const wasFit = atFit();
    const previous = view.scale;
    fitImage();
    if (!wasFit && zoomable()) zoomTo(previous);
  });
})();
