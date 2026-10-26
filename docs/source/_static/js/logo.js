// The waves of the landing page's logo are two Gaussian processes. A click inside the disk adds an
// observation to one of them, picked at random: that wave bends to pass through the point (its
// posterior mean), and its band, the posterior's standard deviation, pinches to nothing there. A
// wave keeps at most three observations; a fourth replaces the oldest. A new fit runs along the
// wave from where it changed and settles like a spring, and the click sends a small ripple along
// the waves. Hovering pushes the waves away from the pointer, but only where they are uncertain: a
// wave holds on to its observations. The waves spring back when the pointer leaves.
//
// The logo is inline in the landing page (index.rst), once for light and once for dark mode, and
// both copies are drawn from the same state. At rest and without observations they show the
// logo's own paths, so the logo looks exactly like the static one; while anything moves, the waves
// are drawn from their formula (docs/design/brand.md). If the logo's paths no longer follow that
// formula (a new logo kit), the logo stays still.
//
// With reduced motion (macOS "Reduce motion", Windows with "Animation effects" off), the waves do
// not follow the pointer and glide to a new fit without rippling or swinging.
(() => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // the waves (docs/design/brand.md) in the logo's units, with y pointing down: the disk has
  // center (100, 100) and radius 92. The tucks move an edge at the right end of a wave (down is
  // positive), so the red-pink wave's top edge comes down 6 and its bottom edge goes up 12.
  const WAVES = [
    { name: "red", amp: 22, phase: 0, tuckTop: 6, tuckBottom: -12 },
    { name: "blue", amp: 16, phase: 1.2, tuckTop: 0, tuckBottom: 0 },
  ];
  const DISK = { x: 100, y: 100, r: 92 };
  const STEP = 1; // spacing of the points a wave is drawn through (units)

  const MAX_POINTS = 3; // observations per wave; a fourth replaces the oldest
  const MIN_GAP = 20; // an observation this close to another on its wave replaces it (units)
  const LENGTHSCALE = 26; // of the squared-exponential kernel: how far an observation bends (units)
  // observation noise, relative to the kernel's variance: only to keep the solve stable, as the
  // observations are at least MIN_GAP apart; the waves pass through them to a hundredth of a unit
  const NOISE = 1e-6;

  const FIT_STIFFNESS = 40; // of the spring that pulls a wave to a new fit (1/s²)
  const FIT_DAMPING = 7.6; // an overshoot of about a tenth, most of the way in about a second (1/s)
  const SPREAD = 220; // speed at which a new fit runs along the wave (units/s)
  const SPREAD_CORE = 12; // the run starts round around the click, without a corner there (units)
  const RAMP = 90; // a point eases to its new target over this time: the edges stay smooth (ms)
  const GLIDE = 0.3; // time constant of the move to a new fit with reduced motion (s)

  const HOVER_STIFFNESS = 260; // of the spring that holds a wave in place against the pointer (1/s²)
  const HOVER_DAMPING = 12; // a bounce or two when the pointer leaves (1/s)
  const DENT = 12; // how far the pointer pushes a wave at most (units)
  const DENT_WIDTH = 16; // how far along the wave the push reaches (units, a standard deviation)
  const REACH = 10; // how far beyond a wave's edge the pointer pushes (units, a standard deviation)

  // steps of the simulation per frame (frames are at most 1/30 s apart): a step of 1/120 s or less
  // keeps the stiffest spring, about 21/s with the pointer's push, far from unstable (2 / step)
  const SUBSTEPS = 4;

  const DOT = 3.2; // radius of an observation (units)
  const RIM = 84; // an observation is at most this far from the disk's center, so it shows whole (units)
  const POP = 300; // how long an observation takes to appear, or to fade (ms)
  const RIPPLE = 600; // how long the ring that marks a click takes to fade (ms)
  const RIPPLE_RADIUS = 18; // how far it spreads (units)

  // the ripple a click sends along the waves, like a drop in water: a short train of waves that
  // runs out from the click both ways and fades, the less the farther a wave is from the click
  const WOBBLE = 1400; // how long it lasts (ms)
  const WOBBLE_HEIGHT = 3; // how high it starts (units)
  const WOBBLE_SPEED = 160; // how fast it runs along the waves (units/s)
  const WOBBLE_LENGTH = 40; // its wavelength (units)
  const WOBBLE_TRAIN = 14; // how long the train is, about a crest and a trough (units, a standard deviation)
  const WOBBLE_FADE = 0.45; // time constant of its fading (s)
  const WOBBLE_REACH = 45; // how far above or below the click a wave ripples (units, a standard deviation)

  // below these, a fraction of a pixel and too slow to see, a wave counts as at rest
  const REST_OFFSET = 0.02; // units
  const REST_SPEED = 0.05; // units/s
  const REST_WIDTH = 0.001; // fraction of the width
  const REST_WIDTH_SPEED = 0.005; // fraction of the width per second

  const smoothstep = (a, b, x) => {
    const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
    return t * t * (3 - 2 * t);
  };
  const easeOutBack = (t) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2; // a slight pop
  const easeOutCubic = (t) => 1 - (1 - t) ** 3;
  const kernel = (a, b) => Math.exp(-((a - b) ** 2) / (2 * LENGTHSCALE ** 2));

  // the top and bottom edge of a wave at x, without observations
  function edges(wave, x) {
    const center = 100 + wave.amp * Math.sin((2 * Math.PI * 1.25 * x) / 200 + wave.phase);
    const half = 5 + 22 * (x / 200) ** 1.6;
    const tuck = smoothstep(150, 195, x);
    return [center - half + wave.tuckTop * tuck, center + half + wave.tuckBottom * tuck];
  }

  const centerAt = (wave, x) => edges(wave, x).reduce((a, b) => a + b) / 2;

  // whether a wave's path in the logo follows the formula: each corner that is not on the disk's
  // edge (within 0.1 units: the path is clipped to the disk) lies on one of the wave's edges, to
  // the 2 decimals the path is written with and some slack
  function follows(path, wave) {
    const numbers = path?.getAttribute("d")?.match(/-?\d*\.?\d+/g)?.map(Number) ?? [];
    let corners = 0;
    for (let i = 0; i + 1 < numbers.length; i += 2) {
      const [x, y] = [numbers[i], numbers[i + 1]];
      if (Math.abs(Math.hypot(x - DISK.x, y - DISK.y) - DISK.r) < 0.1) continue;
      const [top, bottom] = edges(wave, x);
      if (Math.min(Math.abs(y - top), Math.abs(y - bottom)) > 0.2) return false;
      corners += 1;
    }
    return corners > 20; // the logo's waves have more than 200
  }

  // the Cholesky factor of a small symmetric positive definite matrix, and a solve with it
  function cholesky(a) {
    const n = a.length;
    const l = a.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j <= i; j += 1) {
        let sum = a[i][j];
        for (let k = 0; k < j; k += 1) sum -= l[i][k] * l[j][k];
        l[i][j] = i === j ? Math.sqrt(sum) : sum / l[j][j];
      }
    }
    return l;
  }

  function solve(l, b) {
    const n = l.length;
    const y = [];
    for (let i = 0; i < n; i += 1) {
      let sum = b[i];
      for (let k = 0; k < i; k += 1) sum -= l[i][k] * y[k];
      y.push(sum / l[i][i]);
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i -= 1) {
      let sum = y[i];
      for (let k = i + 1; k < n; k += 1) sum -= l[k][i] * x[k];
      x[i] = sum / l[i][i];
    }
    return x;
  }

  function makeWave(spec) {
    const xs = [];
    for (let x = 0; x <= 200; x += STEP) xs.push(x);
    const array = (value) => new Float64Array(xs.length).fill(value);
    const sides = xs.map((x) => edges(spec, x));
    return {
      spec,
      xs,
      mid: sides.map(([top, bottom]) => (top + bottom) / 2),
      half: sides.map(([top, bottom]) => (bottom - top) / 2),
      points: [], // observations: { x, y, born, dying, shrinkFrom, marks }
      certain: array(0), // the fraction of the prior variance the observations explain
      // the offset of the center line from the logo's (fit) and the width as a fraction of the
      // logo's (width), each with its speed and the target it springs to. At each point, the
      // target eases from one fit to the next over RAMP ms from the time in "start", which is
      // later the farther the point is from where the fit changed: so the fit runs along the wave
      fit: array(0),
      fitSpeed: array(0),
      fitTarget: array(0),
      width: array(1),
      widthSpeed: array(0),
      widthTarget: array(1),
      from: { fit: array(0), width: array(1) },
      to: { fit: array(0), width: array(1) },
      start: array(-Infinity),
      // the push of the pointer, an offset of the center line on top of the fit, and the side of
      // the wave the pointer is on (1 below it, -1 above it, 0 while the pointer is away); the
      // pointer pushes the wave the other way
      dent: array(0),
      dentSpeed: array(0),
      side: 0,
    };
  }

  // the posterior of a wave given its observations: at each point, the offset of the mean from
  // the logo's center line, and the fraction of the prior variance the observations explain
  function posterior(wave) {
    const observed = wave.points.filter((p) => p.dying === null);
    const fit = new Float64Array(wave.xs.length);
    const certain = new Float64Array(wave.xs.length);
    if (!observed.length) return { fit, certain };
    const l = cholesky(observed.map((a, i) => observed.map((b, j) => kernel(a.x, b.x) + (i === j ? NOISE : 0))));
    const weights = solve(
      l,
      observed.map((p) => p.y - centerAt(wave.spec, p.x)),
    );
    wave.xs.forEach((x, i) => {
      const k = observed.map((p) => kernel(x, p.x));
      const explained = solve(l, k);
      fit[i] = k.reduce((sum, ki, j) => sum + ki * weights[j], 0);
      certain[i] = k.reduce((sum, ki, j) => sum + ki * explained[j], 0);
    });
    return { fit, certain };
  }

  // the target of the i-th point of a wave at a time: on its way from one fit to the next
  function target(wave, i, now) {
    const eased = smoothstep(0, RAMP, now - wave.start[i]);
    return [
      wave.from.fit[i] + (wave.to.fit[i] - wave.from.fit[i]) * eased,
      wave.from.width[i] + (wave.to.width[i] - wave.from.width[i]) * eased,
    ];
  }

  // a new fit of a wave, which reaches each point of it later the farther it is from x. It starts
  // from the previous fit everywhere, also where that one has not arrived yet: a wave that kept
  // the older fit there would tear where the previous one stopped. Between two observations far
  // apart in height, the mean can swing far out of the disk: the fit lets the wave leave the disk
  // but goes no farther (nor than where the logo has it), which keeps it through the observations
  function refit(wave, x, now) {
    const { fit, certain } = posterior(wave);
    wave.certain = certain;
    const speed = reducedMotion.matches ? Infinity : SPREAD;
    wave.xs.forEach((xi, i) => {
      const chord = Math.sqrt(Math.max(DISK.r ** 2 - (xi - DISK.x) ** 2, 0));
      const lowest = Math.min(DISK.y - chord - wave.half[i], wave.mid[i]);
      const highest = Math.max(DISK.y + chord + wave.half[i], wave.mid[i]);
      wave.from.fit[i] = wave.to.fit[i];
      wave.from.width[i] = wave.to.width[i];
      wave.to.fit[i] = Math.min(Math.max(wave.mid[i] + fit[i], lowest), highest) - wave.mid[i];
      // as for a Gaussian process, the band is the posterior's standard deviation: the logo's band
      // where the wave is uncertain, nothing where it is observed
      wave.to.width[i] = Math.sqrt(Math.max(1 - certain[i], 0));
      const distance = Math.hypot(xi - x, SPREAD_CORE) - SPREAD_CORE;
      wave.start[i] = now + (distance / speed) * 1000;
    });
  }

  function observe(m, click, now) {
    const wave = m.waves[Math.floor(Math.random() * m.waves.length)];
    const out = Math.hypot(click.x - DISK.x, click.y - DISK.y);
    const inward = Math.min(RIM / out, 1); // a click by the rim is pulled in a little
    const point = { x: DISK.x + (click.x - DISK.x) * inward, y: DISK.y + (click.y - DISK.y) * inward };
    const alive = () => wave.points.filter((p) => p.dying === null);
    const dies = (p) => {
      // a dot still popping in shrinks from its size (with reduced motion, it does not pop)
      p.shrinkFrom = reducedMotion.matches ? 1 : Math.min((now - p.born) / POP, 1);
      p.dying = now;
    };
    alive()
      .filter((p) => Math.abs(p.x - point.x) < MIN_GAP)
      .forEach(dies);
    const left = alive();
    if (left.length >= MAX_POINTS) dies(left[0]);
    wave.points.push({ x: point.x, y: point.y, born: now, dying: null, shrinkFrom: 1, marks: [] });
    if (!reducedMotion.matches) {
      // how much each wave ripples: the closer it passes by the click, the more
      const near = m.waves.map((w) => {
        const i = Math.min(Math.max(Math.round(point.x / STEP), 0), w.xs.length - 1);
        return Math.exp(-((point.y - (w.mid[i] + w.fit[i] + w.dent[i])) ** 2) / (2 * WOBBLE_REACH ** 2));
      });
      m.ripples.push({ x: point.x, y: point.y, wave, near, born: now, marks: [] });
    }
    refit(wave, point.x, now);
  }

  // the pointer's push on the i-th point of a wave: away from the side of the wave the pointer is
  // on, the harder the closer the pointer is to the wave's edge, and only as far as the wave is
  // uncertain there
  function push(m, w, i) {
    if (!m.pointer || !w.side || reducedMotion.matches) return 0;
    const off = m.pointer.y - (w.mid[i] + w.fit[i] + w.dent[i]);
    const beyond = Math.max(Math.abs(off) - w.half[i] * w.width[i], 0);
    const along = w.xs[i] - m.pointer.x;
    const reach = Math.exp(-(along ** 2) / (2 * DENT_WIDTH ** 2) - beyond ** 2 / (2 * REACH ** 2));
    return -w.side * HOVER_STIFFNESS * DENT * reach * (1 - w.certain[i]);
  }

  // the side of a wave the pointer is on (1 below, -1 above), taken where the pointer is and kept
  // while the pointer is on the wave, so that the whole dent goes one way
  function side(m, w) {
    if (!m.pointer) return 0;
    const i = Math.min(Math.max(Math.round(m.pointer.x / STEP), 0), w.xs.length - 1);
    const off = m.pointer.y - (w.mid[i] + w.fit[i] + w.dent[i]);
    if (w.side && Math.abs(off) <= w.half[i] * w.width[i]) return w.side;
    return off >= 0 ? 1 : -1;
  }

  // the springs of each point of each wave, advanced by dt seconds
  function step(m, dt, now) {
    const calm = reducedMotion.matches;
    const glide = 1 - Math.exp(-dt / GLIDE);
    for (const w of m.waves) {
      w.side = side(m, w);
      for (let i = 0; i < w.xs.length; i += 1) {
        [w.fitTarget[i], w.widthTarget[i]] = target(w, i, now);
        if (calm) {
          w.fit[i] += (w.fitTarget[i] - w.fit[i]) * glide;
          w.width[i] += (w.widthTarget[i] - w.width[i]) * glide;
          w.dent[i] -= w.dent[i] * glide;
          w.fitSpeed[i] = w.widthSpeed[i] = w.dentSpeed[i] = 0;
          continue;
        }
        w.fitSpeed[i] += (FIT_STIFFNESS * (w.fitTarget[i] - w.fit[i]) - FIT_DAMPING * w.fitSpeed[i]) * dt;
        w.fit[i] += w.fitSpeed[i] * dt;
        w.widthSpeed[i] += (FIT_STIFFNESS * (w.widthTarget[i] - w.width[i]) - FIT_DAMPING * w.widthSpeed[i]) * dt;
        w.width[i] += w.widthSpeed[i] * dt;
        w.dentSpeed[i] += (push(m, w, i) - HOVER_STIFFNESS * w.dent[i] - HOVER_DAMPING * w.dentSpeed[i]) * dt;
        w.dent[i] += w.dentSpeed[i] * dt;
      }
    }
  }

  // whether nothing moves anymore: then the loop stops and the waves snap to their targets. With
  // the pointer on the logo, a wave is at rest where its spring balances the pointer's push
  function settled(m, now) {
    if (m.ripples.length) return false;
    for (const w of m.waves) {
      if (w.points.some((p) => p.dying !== null || now - p.born < POP)) return false;
      for (let i = 0; i < w.xs.length; i += 1) {
        if (now < w.start[i] + RAMP) return false; // the fit has not reached this point yet
        if (Math.abs(w.fitTarget[i] - w.fit[i]) > REST_OFFSET || Math.abs(w.fitSpeed[i]) > REST_SPEED) return false;
        if (Math.abs(w.widthTarget[i] - w.width[i]) > REST_WIDTH || Math.abs(w.widthSpeed[i]) > REST_WIDTH_SPEED) {
          return false;
        }
        const balance = push(m, w, i) / HOVER_STIFFNESS;
        if (Math.abs(balance - w.dent[i]) > REST_OFFSET || Math.abs(w.dentSpeed[i]) > REST_SPEED) return false;
      }
    }
    return true;
  }

  // a band through the points of a wave, between its top and bottom edge at each
  const band = (xs, tops, bottoms) =>
    `M${xs.map((x, i) => `${x},${tops[i].toFixed(2)}`).join(" ")} ` +
    `${xs.map((x, i) => `${x},${bottoms[i].toFixed(2)}`).reverse().join(" ")}Z`;

  // the offset of the i-th point of the j-th wave by the ripples of clicks: a train of waves that
  // runs out from each click both ways, eases in, fades out, and leaves the wave's observations be
  function wobble(m, j, i, now) {
    const w = m.waves[j];
    let offset = 0;
    for (const ripple of m.ripples) {
      const age = now - ripple.born;
      if (age < 0 || age >= WOBBLE) continue;
      const ahead = Math.abs(w.xs[i] - ripple.x) - (WOBBLE_SPEED * age) / 1000; // from the front
      const height =
        WOBBLE_HEIGHT *
        ripple.near[j] *
        Math.exp(-age / 1000 / WOBBLE_FADE) *
        smoothstep(0, 80, age) * // eases in
        (1 - smoothstep(0.7 * WOBBLE, WOBBLE, age)); // and out
      offset += height * Math.exp(-(ahead ** 2) / (2 * WOBBLE_TRAIN ** 2)) * Math.sin((2 * Math.PI * ahead) / WOBBLE_LENGTH);
    }
    return offset * (1 - w.certain[i]);
  }

  // the outlines of the two waves and of their overlap, where both waves cover a point
  function outlines(m, now) {
    const [tops, bottoms] = [[], []];
    for (const [j, w] of m.waves.entries()) {
      const top = new Float64Array(w.xs.length);
      const bottom = new Float64Array(w.xs.length);
      for (let i = 0; i < w.xs.length; i += 1) {
        const center = w.mid[i] + w.fit[i] + w.dent[i] + wobble(m, j, i, now);
        const half = w.half[i] * Math.max(w.width[i], 0); // a spring overshooting past nothing is nothing
        top[i] = center - half;
        bottom[i] = center + half;
      }
      tops.push(top);
      bottoms.push(bottom);
    }
    const xs = m.waves[0].xs;
    const overlapTop = xs.map((_, i) => Math.max(...tops.map((t) => t[i])));
    const overlapBottom = xs.map((_, i) => Math.max(Math.min(...bottoms.map((b) => b[i])), overlapTop[i]));
    return {
      waves: m.waves.map((w, j) => band(xs, tops[j], bottoms[j])),
      overlap: band(xs, overlapTop, overlapBottom),
    };
  }

  const SVG = "http://www.w3.org/2000/svg";
  const element = (name, attributes, parent) => {
    const el = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
    return parent ? parent.appendChild(el) : el;
  };

  // the marks that show an observation or a ripple, one in each copy of the logo, made on first
  // use and removed with it
  function marksOf(m, item, make) {
    while (item.marks.length < m.views.length) {
      const view = m.views[item.marks.length];
      item.marks.push(view.markLayer.appendChild(make(view)));
    }
    return item.marks;
  }

  function render(m, now) {
    const still = !m.running && m.waves.every((w) => !w.points.length && w.dent.every((d) => d === 0));
    const shapes = still ? null : outlines(m, now);
    for (const view of m.views) {
      view.live.style.display = still ? "none" : "";
      for (const path of view.static) path.style.display = still ? "" : "none";
      if (still) continue;
      view.waves.forEach((path, j) => path.setAttribute("d", shapes.waves[j]));
      view.overlap.setAttribute("d", shapes.overlap);
    }
    m.waves.forEach((w, j) => {
      w.points = w.points.filter((p) => {
        // a dot in its wave's color, ringed in the disk's so that it shows on its own wave
        const dots = marksOf(m, p, (view) =>
          element("circle", {
            cx: p.x,
            cy: p.y,
            fill: view.colors[j],
            stroke: view.disk,
            "stroke-width": 1.4,
            "data-wave": w.spec.name, // which wave the observation belongs to, for tests
          }),
        );
        const fading = p.dying !== null ? Math.min((now - p.dying) / POP, 1) : 0;
        if (fading >= 1) {
          dots.forEach((dot) => dot.remove());
          return false;
        }
        const grown = reducedMotion.matches ? 1 : Math.min((now - p.born) / POP, 1); // no pop with reduced motion
        const scale = p.dying !== null ? easeOutBack(p.shrinkFrom) * (1 - fading) : easeOutBack(grown);
        for (const dot of dots) {
          dot.setAttribute("r", (DOT * Math.max(scale, 0)).toFixed(2));
          dot.setAttribute("opacity", (1 - fading).toFixed(2));
        }
        return true;
      });
    });
    m.ripples = m.ripples.filter((ripple) => {
      const t = Math.min((now - ripple.born) / RIPPLE, 1);
      const j = m.waves.indexOf(ripple.wave);
      if (ripple.ringDone) return now - ripple.born < WOBBLE; // the ring is gone, the waves still ripple
      // a ring in the wave's color on a wider one in the disk's, so that it shows on any wave
      const rings = marksOf(m, ripple, (view) => {
        const ring = element("g", { fill: "none" });
        element("circle", { cx: ripple.x, cy: ripple.y, stroke: view.disk, "stroke-width": 2.8 }, ring);
        element("circle", { cx: ripple.x, cy: ripple.y, stroke: view.colors[j], "stroke-width": 1.4 }, ring);
        return ring;
      });
      if (t >= 1) {
        rings.forEach((ring) => ring.remove());
        ripple.ringDone = true;
        return now - ripple.born < WOBBLE;
      }
      const r = (DOT + (RIPPLE_RADIUS - DOT) * easeOutCubic(t)).toFixed(2);
      for (const ring of rings) {
        for (const circle of ring.children) circle.setAttribute("r", r);
        ring.setAttribute("opacity", (1 - t).toFixed(2));
      }
      return true;
    });
  }

  function animate(m) {
    if (m.running) return;
    m.running = true;
    let last = performance.now();
    const frame = (now) => {
      // the first frame's time can be from before this call: no step back in time
      const dt = Math.min(Math.max(now - last, 0) / 1000, 1 / 30);
      last = Math.max(now, last);
      for (let i = 1; i <= SUBSTEPS; i += 1) step(m, dt / SUBSTEPS, last - dt * 1000 * (1 - i / SUBSTEPS));
      if (settled(m, last)) {
        for (const w of m.waves) {
          w.fitTarget.set(w.to.fit);
          w.widthTarget.set(w.to.width);
          w.fit.set(w.to.fit);
          w.width.set(w.to.width);
          w.fitSpeed.fill(0);
          w.widthSpeed.fill(0);
          w.dentSpeed.fill(0);
          if (!m.pointer) w.dent.fill(0);
        }
        m.running = false;
        render(m, last);
        return;
      }
      render(m, last);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // the parts of a copy of the logo, if its waves follow the formula
  function parts(svg) {
    const group = svg.querySelector("g");
    const disk = group?.querySelector(":scope > circle");
    const [red, blue, overlap] = group ? group.querySelectorAll(":scope > path") : [];
    if (!disk || !overlap || !follows(red, WAVES[0]) || !follows(blue, WAVES[1])) return null;
    return { svg, group, disk, static: [red, blue, overlap] };
  }

  // a copy of the logo: its static waves, and the live ones drawn in their place while they move
  function makeView({ svg, group, disk, static: [red, blue, overlap] }) {
    const clip = `probly-logo-disk-${document.querySelectorAll("[id^='probly-logo-disk-']").length + 1}`;
    const live = element("g", { "clip-path": `url(#${clip})`, style: "display: none" });
    overlap.after(live);
    element("circle", { cx: DISK.x, cy: DISK.y, r: DISK.r }, element("clipPath", { id: clip }, live));
    const colors = [red.getAttribute("fill"), blue.getAttribute("fill")];
    svg.querySelector(":scope > title")?.remove(); // a tooltip over a logo that invites the pointer
    svg.classList.add("probly-logo-live");
    return {
      svg,
      group,
      static: [red, blue, overlap],
      live,
      waves: colors.map((fill) => element("path", { fill }, live)),
      overlap: element("path", { fill: overlap.getAttribute("fill") }, live),
      markLayer: element("g", {}, live), // the observations and ripples
      colors,
      disk: disk.getAttribute("fill"),
    };
  }

  function pointerIn(view, event) {
    const matrix = view.group.getScreenCTM();
    return matrix ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()) : null;
  }

  const inDisk = (point) => Math.hypot(point.x - DISK.x, point.y - DISK.y) <= DISK.r;

  function attach(svgs) {
    const found = svgs.map(parts);
    if (!found.length || found.some((p) => !p)) return;
    const views = found.map(makeView);
    const m = { views, waves: WAVES.map(makeWave), ripples: [], pointer: null, running: false };
    for (const view of views) {
      view.svg.addEventListener("pointermove", (event) => {
        const point = pointerIn(view, event);
        if (!point) return;
        if (event.pointerType === "touch" || reducedMotion.matches) {
          if (m.pointer) {
            m.pointer = null; // a dent left by a mouse glides back
            animate(m);
          }
          return;
        }
        m.pointer = { x: point.x, y: point.y };
        animate(m);
      });
      view.svg.addEventListener("pointerleave", () => {
        m.pointer = null;
        animate(m);
      });
      view.svg.addEventListener("click", (event) => {
        const point = pointerIn(view, event);
        if (!point || !inDisk(point)) return;
        observe(m, point, performance.now());
        animate(m);
      });
    }
    reducedMotion.addEventListener("change", () => animate(m)); // a dent glides back at once
  }

  document.addEventListener("DOMContentLoaded", () => {
    attach([...document.querySelectorAll(".probly-hero-logo svg")]);
  });
})();
