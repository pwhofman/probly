// The Milton demo (demos/milton/index.rst): Hurricane Milton as WeatherNext forecast it four days
// ahead, from 7 October 2024, 00 UTC, and what probly makes of the spread of its runs.
//
// WeatherNext 2 is four models trained independently, and every run of a model draws noise of its
// own: the runs of one model scatter as the atmosphere might (aleatoric uncertainty), and the
// models' disagreement shows what they do not know (epistemic uncertainty), as in its paper
// (Alet et al., 2025). The data (data.js, written by the scripts in ../compute) holds the runs'
// tracks, fields on a grid every 6 hours, every run's wind near the storm, and the runs at a few
// places; the page interpolates between the 6-hourly steps and only looks values up.
//
// Three views: one run of the model, taken as the forecast; all runs, with the chance of
// tropical-storm-force wind and the outline of every run's; and the runs' spread, split into its
// aleatoric and epistemic parts by the law of total variance. Pointing at the map shows the values
// there; clicking it, or a place, shows that spot over time.
//
// Without JavaScript, the static figure in the <noscript> shows; when the data does not load, the
// script shows that figure itself.
(() => {
  const root = document.querySelector(".milton-demo");
  if (!root) return;
  // with scripts on, a <noscript>'s content is its HTML as text
  const fallback = root.querySelector("noscript")?.textContent ?? "";
  const showFallback = () => {
    root.classList.remove("milton-demo-ready");
    root.classList.add("milton-demo-failed");
    root.innerHTML = fallback;
  };

  const SVG = "http://www.w3.org/2000/svg";
  const VIEWS = [
    // the view, its label, and the label on small screens
    ["one", "One run", "One"],
    ["ensemble", "All runs", "All"],
    ["split", "Decomposition", "Split"],
  ];
  const WIND = new Set(["single", "mean", "lower", "upper", "truth"]);
  const HOURS_PER_SECOND = 10; // the speed of playback
  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svg = (tag, attributes = {}) => {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  };
  const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
  const lerp = (a, b, w) => a + (b - a) * w;
  const percent = (share) => `${Math.round(share * 100)}%`;
  const ms = (value) => `${value.toFixed(value < 10 ? 1 : 0)} m/s`;
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const decode = (text) => {
    const bytes = atob(text);
    const out = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += 1) out[i] = bytes.charCodeAt(i);
    return out;
  };
  // numpy's default quantile (linear interpolation)
  const quantile = (values, share) => {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * share;
    const low = Math.floor(position);
    const high = Math.ceil(position);
    return lerp(sorted[low], sorted[high], position - low);
  };
  const hex = (color) => {
    const value = color.trim().replace("#", "");
    const full = value.length === 3 ? [...value].map((c) => c + c).join("") : value;
    return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
  };

  function start(data) {
    if (!data || !data.models || !data.times) throw new Error("no data");
    const init = Date.parse(data.init);
    const last = data.times[data.times.length - 1];
    const step = data.times[1] - data.times[0];
    const keys = Object.keys(data.models);
    const state = {
      view: "split",
      model: keys[0],
      hours: 66,
      truth: false,
      place: "Tampa",
      spot: null, // a cell picked on the map: {lat, lon}
      playing: false,
    };
    const model = () => data.models[state.model];
    const splittable = (key) => data.models[key].models > 1;
    // what a model's groups of runs are: WeatherNext's models, or checkpoints that stand in for them
    const NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];
    const groups = (m = model()) => (m.group ?? ["model", "models"])[m.models === 1 ? 0 : 1];
    const counted = (m = model()) => `${NUMBERS[m.models] ?? m.models} ${groups(m)}`;
    // whether the model has what happened to its wind (the observed track is always there)
    const withTruth = () => Boolean(model().fields.truth);

    // -- time -----------------------------------------------------------------------------------
    const when = (hours) => {
      const date = new Date(init + Math.round(hours) * 3600e3);
      const hour = String(date.getUTCHours()).padStart(2, "0");
      return `${DAYS[date.getUTCDay()]} ${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}, ${hour} UTC`;
    };
    const day = (hours) => when(hours).split(",")[0].replace(/ [A-Z][a-z]{2}$/, "");
    const ahead = (hours) => `+${Math.round(hours)} h`;
    // a value of a 6-hourly series at any hour
    const sample = (series, hours) => {
      const x = clamp(hours / step, 0, series.length - 1);
      const i = Math.min(Math.floor(x), series.length - 2);
      return lerp(series[i], series[i + 1], x - i);
    };

    // -- the fields on the grid ---------------------------------------------------------------------
    const decoded = new Map();
    const frames = (key, name) => {
      const id = `${key}/${name}`;
      if (!decoded.has(id)) decoded.set(id, data.models[key].fields[name].map(decode));
      return decoded.get(id);
    };
    const scale = (key, name) => data.models[key].scales[WIND.has(name) ? "wind" : name];
    // a field at any hour, interpolated between its 6-hourly frames, in its units
    const field = (name, hours, key = state.model) => {
      const list = frames(key, name);
      const factor = scale(key, name);
      const x = clamp(hours / step, 0, list.length - 1);
      const i = Math.min(Math.floor(x), list.length - 2);
      const w = x - i;
      const a = list[i];
      const b = list[i + 1];
      const out = new Float32Array(a.length);
      for (let c = 0; c < a.length; c += 1) out[c] = ((1 - w) * a[c] + w * b[c]) / factor;
      return out;
    };
    // the range of a field's colors over all frames: below the median, a value is too common to
    // stand out; above the 99.5th percentile, all is full
    const ranges = new Map();
    const range = (name, key = state.model) => {
      const id = `${key}/${name}`;
      if (!ranges.has(id)) {
        const values = [];
        data.times.forEach((hours) => values.push(...field(name, hours, key)));
        const high = Math.max(quantile(values, 0.995), 1);
        ranges.set(id, { low: Math.min(quantile(values, 0.5), high * 0.5), high });
      }
      return ranges.get(id);
    };
    const strength = (value, { low, high }) => clamp((value - low) / (high - low), 0, 1);
    const cellOf = (lat, lon, key = state.model) => {
      const grid = data.models[key].grid;
      const row = Math.round((grid.north - lat) / grid.step);
      const col = Math.round((lon - grid.west) / grid.step);
      if (row < 0 || col < 0 || row >= grid.rows || col >= grid.cols) return -1;
      return row * grid.cols + col;
    };
    // the wind of one run near the storm (its square of cells), interpolated between 6-hourly steps
    const runFrames = new Map();
    const runField = (run, hours) => {
      const m = model();
      const windows = m.run_fields;
      const size = windows.size;
      const n = size * size;
      if (!runFrames.has(state.model)) runFrames.set(state.model, windows.fields.map(decode));
      const list = runFrames.get(state.model);
      const x = clamp(hours / step, 0, list.length - 1);
      const i = Math.min(Math.floor(x), list.length - 2);
      const w = x - i;
      const [r0, c0] = windows.origins[i];
      const [r1, c1] = windows.origins[i + 1];
      const a = list[i];
      const b = list[i + 1];
      const values = new Float32Array(n);
      for (let r = 0; r < size; r += 1) {
        for (let c = 0; c < size; c += 1) {
          const va = a[run * n + r * size + c];
          const rb = r + r0 - r1;
          const cb = c + c0 - c1;
          const vb = rb >= 0 && rb < size && cb >= 0 && cb < size ? b[run * n + rb * size + cb] : va;
          values[r * size + c] = lerp(va, vb, w) / m.scales.wind;
        }
      }
      return { values, size, origin: [r0, c0] };
    };

    // -- the tracks -----------------------------------------------------------------------------------
    const position = (points, hours) => {
      const x = clamp(hours / step, 0, points.length - 1);
      const i = Math.min(Math.floor(x), points.length - 2);
      const w = x - i;
      return [lerp(points[i][0], points[i + 1][0], w), lerp(points[i][1], points[i + 1][1], w)];
    };
    const modelMeans = new Map();
    // the mean track of every model's runs
    const meanTracks = (key = state.model) => {
      if (!modelMeans.has(key)) {
        const m = data.models[key];
        const means = [];
        for (let g = 0; g < m.models; g += 1) {
          const runs = m.tracks.filter((track) => track.model === g);
          means.push(
            data.times.map((_, t) => [
              runs.reduce((sum, run) => sum + run.points[t][0], 0) / runs.length,
              runs.reduce((sum, run) => sum + run.points[t][1], 0) / runs.length,
            ]),
          );
        }
        modelMeans.set(key, means);
      }
      return modelMeans.get(key);
    };
    // the observed track (IBTrACS) at any hour, from its irregular times
    const observedAt = (hours) => {
      const points = data.observed.points;
      for (let i = 1; i < points.length; i += 1) {
        if (points[i][0] >= hours) {
          const w = (hours - points[i - 1][0]) / (points[i][0] - points[i - 1][0]);
          return [lerp(points[i - 1][1], points[i][1], w), lerp(points[i - 1][2], points[i][2], w)];
        }
      }
      const end = points[points.length - 1];
      return [end[1], end[2]];
    };
    const coastName = (lat, lon) => {
      let best = null;
      let distance = Infinity;
      for (const place of data.coast ?? []) {
        const d = Math.hypot(place.lat - lat, (place.lon - lon) * Math.cos((lat * Math.PI) / 180));
        if (d < distance) {
          best = place.name;
          distance = d;
        }
      }
      return best ?? `${lat.toFixed(1)}° N`;
    };

    // -- a spot over time: a place with its runs, or a cell of the grid ------------------------------
    const series = () => {
      const m = model();
      const runsPerModel = m.runs_per_model;
      const place = state.spot ? null : m.places[state.place];
      if (place) {
        const runs = place.runs;
        const times = data.times.map((_, t) => runs.map((run) => run[t]));
        const means = [];
        for (let g = 0; g < m.models; g += 1) {
          const group = runs.slice(g * runsPerModel, (g + 1) * runsPerModel);
          means.push(data.times.map((_, t) => group.reduce((sum, run) => sum + run[t], 0) / group.length));
        }
        return {
          name: state.place,
          runs,
          modelMeans: means,
          single: runs[0],
          mean: times.map((values) => values.reduce((a, b) => a + b, 0) / values.length),
          lower: times.map((values) => quantile(values, 0.05)),
          upper: times.map((values) => quantile(values, 0.95)),
          gale: times.map((values) => values.filter((v) => v >= data.gale).length / values.length),
          aleatoric: place.aleatoric,
          epistemic: place.epistemic,
          truth: place.truth ?? null,
        };
      }
      const spot = state.spot ?? { lat: 27.3, lon: -82.6 };
      const cell = cellOf(spot.lat, spot.lon);
      const values = (name) => frames(state.model, name).map((frame) => frame[cell] / scale(state.model, name));
      const spread = values("spread");
      const share = values("epistemic");
      return {
        name: `${Math.abs(spot.lat).toFixed(1)}° N, ${Math.abs(spot.lon).toFixed(1)}° W`,
        runs: null,
        modelMeans: null,
        single: values("single"),
        mean: values("mean"),
        lower: values("lower"),
        upper: values("upper"),
        gale: values("gale"),
        aleatoric: spread.map((s, t) => s * s * (1 - share[t])),
        epistemic: spread.map((s, t) => s * s * share[t]),
        truth: withTruth() ? values("truth") : null,
      };
    };

    // -- the page ------------------------------------------------------------------------------------
    root.innerHTML = "";
    const bar = el("div", "milton-demo-bar");
    const switches = el("div", "milton-demo-switches");
    const viewSwitch = el("div", "milton-demo-switch");
    viewSwitch.setAttribute("role", "group");
    viewSwitch.setAttribute("aria-label", "View");
    const viewButtons = new Map();
    for (const [key, long, short] of VIEWS) {
      const button = el("button", "milton-demo-option");
      button.type = "button";
      button.append(el("span", "milton-demo-long", long), el("span", "milton-demo-short", short));
      button.addEventListener("click", () => setView(key));
      viewButtons.set(key, button);
      viewSwitch.append(button);
    }
    const modelSwitch = el("div", "milton-demo-switch");
    modelSwitch.setAttribute("role", "group");
    modelSwitch.setAttribute("aria-label", "Model");
    const modelButtons = new Map();
    for (const key of keys) {
      const m = data.models[key];
      const button = el("button", "milton-demo-option");
      button.type = "button";
      button.append(el("span", "milton-demo-long", `${m.label} · ${m.models} ${groups(m)}`), el("span", "milton-demo-short", m.label));
      button.addEventListener("click", () => setModel(key));
      modelButtons.set(key, button);
      modelSwitch.append(button);
    }
    const info = el("button", "milton-demo-info", "i");
    info.type = "button";
    info.setAttribute("aria-expanded", "false");
    info.setAttribute("aria-label", "About WeatherNext and this demo");
    switches.append(modelSwitch, info);
    bar.append(viewSwitch, switches);

    const about = el("div", "milton-demo-about");
    about.hidden = true;
    about.id = "milton-demo-about";
    info.setAttribute("aria-controls", about.id);
    about.append(
      el(
        "p",
        "",
        "WeatherNext 2 (Google DeepMind) forecasts the weather of the whole globe in steps of 6 hours. " +
          "It is four models, trained independently, and every run of a model draws its own random noise. " +
          "The runs of one model scatter as the atmosphere might: that spread is aleatoric uncertainty. " +
          "Where the four models disagree, they lack knowledge: that is epistemic uncertainty " +
          "(Alet et al., 2025).",
      ),
      el(
        "p",
        "",
        "probly takes the runs as a second-order sample and splits their variance by the law of total " +
          "variance: the mean variance within a model (aleatoric) plus the variance of the models' means " +
          "(epistemic). Mini is a smaller version at 1°; the note under the demo says which runs it shows.",
      ),
    );
    info.addEventListener("click", () => {
      about.hidden = !about.hidden;
      info.setAttribute("aria-expanded", String(!about.hidden));
    });

    const body = el("div", "milton-demo-body");
    const stage = el("div", "milton-demo-stage");
    const map = el("div", "milton-demo-map");
    const canvas = el("canvas");
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    const tip = el("div", "milton-demo-tip");
    tip.hidden = true;
    map.append(canvas, tip);

    const time = el("div", "milton-demo-time");
    const play = el("button", "milton-demo-play");
    play.type = "button";
    const track = el("div", "milton-demo-track");
    const slider = el("input", "milton-demo-slider");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(last);
    slider.step = "1";
    slider.id = "milton-demo-hours";
    slider.setAttribute("aria-label", "Hours after the forecast starts");
    const days = el("div", "milton-demo-days");
    for (let hours = 0; hours <= last; hours += 24) {
      const tick = el("span", "", day(hours));
      tick.style.left = `${(hours / last) * 100}%`;
      days.append(tick);
    }
    track.append(slider, days);
    const now = el("div", "milton-demo-now");
    const clock = el("div", "milton-demo-clock");
    const toggle = el("label", "milton-demo-toggle");
    const truthBox = el("input");
    truthBox.type = "checkbox";
    truthBox.id = "milton-demo-truth";
    toggle.append(truthBox, el("span"), document.createTextNode("Show what happened"));
    now.append(clock, toggle);
    time.append(play, track, now);
    stage.append(map, time);

    const panel = el("div", "milton-demo-panel");
    const title = el("h3", "milton-demo-title");
    const lead = el("p", "milton-demo-lead");
    const shore = el("div", "milton-demo-shore");
    const shoreChart = svg("svg", { class: "milton-demo-shore-chart", role: "img" });
    const shoreNote = el("p", "milton-demo-shore-note");
    shore.append(el("div", "milton-demo-shore-head", "Where the runs come ashore"), shoreChart, shoreNote);
    const chips = el("div", "milton-demo-chips");
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", "Place");
    const chipButtons = new Map();
    for (const place of data.places) {
      const chip = el("button", "milton-demo-chip", place.name);
      chip.type = "button";
      chip.addEventListener("click", () => setPlace(place.name));
      chipButtons.set(place.name, chip);
      chips.append(chip);
    }
    const spot = el("div", "milton-demo-spot");
    const spotHead = el("div", "milton-demo-spot-head");
    const spotName = el("span", "milton-demo-spot-name");
    const spotUnit = el("span", "milton-demo-spot-unit");
    spotHead.append(spotName, spotUnit);
    const chart = svg("svg", { class: "milton-demo-chart", role: "img" });
    const stats = el("dl", "milton-demo-stats");
    spot.append(spotHead, chart, stats);
    const legend = el("div", "milton-demo-legend");
    const note = el("p", "milton-demo-note");
    panel.append(title, lead, shore, chips, spot, legend, note);
    body.append(stage, panel);
    root.append(bar, about, body);

    // -- the map ---------------------------------------------------------------------------------------
    const extent = data.extent;
    const squeeze = Math.cos((((extent.north + extent.south) / 2) * Math.PI) / 180);
    const ratio = (extent.north - extent.south) / ((extent.east - extent.west) * squeeze);
    const context = canvas.getContext("2d");
    let width = 0;
    let height = 0;
    let colors = {};
    const x = (lon) => ((lon - extent.west) / (extent.east - extent.west)) * width;
    const y = (lat) => ((extent.north - lat) / (extent.north - extent.south)) * height;
    const unproject = (px, py) => ({
      lon: extent.west + (px / width) * (extent.east - extent.west),
      lat: extent.north - (py / height) * (extent.north - extent.south),
    });
    const readColors = () => {
      const style = getComputedStyle(root);
      const token = (name) => style.getPropertyValue(name).trim();
      colors = {
        ink: hex(token("--md-ink")),
        red: hex(token("--md-red")),
        blue: hex(token("--md-blue")),
        ocean: token("--md-ocean"),
        land: token("--md-land"),
        coast: token("--md-coast"),
        graticule: token("--md-graticule"),
        surface: token("--md-surface"),
        text: token("--md-text"),
        muted: token("--md-muted"),
      };
    };
    const rgba = ([r, g, b], alpha) => `rgba(${r}, ${g}, ${b}, ${alpha})`;
    // the epistemic share of the variance: blue where it is aleatoric, red where it is epistemic
    const split = (share) => colors.blue.map((c, i) => lerp(c, colors.red[i], share));

    const offscreen = document.createElement("canvas");
    const raster = () => {
      const m = model();
      const grid = m.grid;
      offscreen.width = grid.cols;
      offscreen.height = grid.rows;
      const image = offscreen.getContext("2d").createImageData(grid.cols, grid.rows);
      const pixels = image.data;
      const hours = state.hours;
      const paint = (c, rgb, alpha) => {
        pixels[c * 4] = rgb[0];
        pixels[c * 4 + 1] = rgb[1];
        pixels[c * 4 + 2] = rgb[2];
        pixels[c * 4 + 3] = Math.round(clamp(alpha, 0, 1) * 255);
      };
      if (state.view === "one") {
        // calm wind stays clear: the shading starts at 8 m/s and is full at 33 m/s
        field("single", hours).forEach((v, c) => paint(c, colors.ink, Math.pow(clamp((v - 8) / 25, 0, 1), 0.85) * 0.85));
      } else if (state.view === "ensemble") {
        field("gale", hours).forEach((p, c) => paint(c, colors.ink, p * 0.6));
      } else {
        const share = field("epistemic", hours);
        const spread = range("spread");
        field("spread", hours).forEach((s, c) => paint(c, split(share[c]), Math.pow(strength(s, spread), 0.9) * 0.88));
      }
      offscreen.getContext("2d").putImageData(image, 0, 0);
      const half = grid.step / 2;
      context.save();
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(
        offscreen,
        x(grid.west - half),
        y(grid.north + half),
        x(grid.west + (grid.cols - 0.5) * grid.step) - x(grid.west - half),
        y(grid.north - (grid.rows - 0.5) * grid.step) - y(grid.north + half),
      );
      context.restore();
    };

    // -- the outline of a run's tropical-storm-force wind: marching squares, smoothed ------------------
    // the lines where ``values`` (rows x cols, row-major) crosses ``level``, in fractional (row, col)
    const isolines = (values, rows, cols, level) => {
      const points = new Map(); // a cell edge's id -> where the line crosses it
      const links = new Map(); // a cell edge's id -> the edges it is joined to
      const vertical = rows * (cols - 1); // the ids of vertical edges follow those of horizontal ones
      const value = (r, c) => values[r * cols + c];
      const cross = (id, r0, c0, r1, c1) => {
        if (!points.has(id)) {
          const a = value(r0, c0);
          const w = (level - a) / (value(r1, c1) - a);
          points.set(id, [lerp(r0, r1, w), lerp(c0, c1, w)]);
        }
        return id;
      };
      const join = (a, b) => {
        if (!links.has(a)) links.set(a, []);
        if (!links.has(b)) links.set(b, []);
        links.get(a).push(b);
        links.get(b).push(a);
      };
      for (let r = 0; r < rows - 1; r += 1) {
        for (let c = 0; c < cols - 1; c += 1) {
          const index =
            ((value(r, c) >= level) << 3) |
            ((value(r, c + 1) >= level) << 2) |
            ((value(r + 1, c + 1) >= level) << 1) |
            (value(r + 1, c) >= level);
          if (index === 0 || index === 15) continue;
          const top = () => cross(r * (cols - 1) + c, r, c, r, c + 1);
          const bottom = () => cross((r + 1) * (cols - 1) + c, r + 1, c, r + 1, c + 1);
          const left = () => cross(vertical + r * cols + c, r, c, r + 1, c);
          const right = () => cross(vertical + r * cols + c + 1, r, c + 1, r + 1, c + 1);
          // a saddle is resolved by the cell's mean
          const center = (value(r, c) + value(r, c + 1) + value(r + 1, c + 1) + value(r + 1, c)) / 4 >= level;
          if (index === 1 || index === 14) join(left(), bottom());
          else if (index === 2 || index === 13) join(bottom(), right());
          else if (index === 3 || index === 12) join(left(), right());
          else if (index === 4 || index === 11) join(top(), right());
          else if (index === 6 || index === 9) join(top(), bottom());
          else if (index === 7 || index === 8) join(left(), top());
          else if ((index === 5) === center) {
            join(left(), top());
            join(bottom(), right());
          } else {
            join(left(), bottom());
            join(top(), right());
          }
        }
      }
      const lines = [];
      const seen = new Set();
      const follow = (start) => {
        const ids = [start];
        seen.add(start);
        let previous = -1;
        let current = start;
        for (;;) {
          const next = links.get(current).find((id) => id !== previous && !seen.has(id));
          if (next === undefined) break;
          ids.push(next);
          seen.add(next);
          previous = current;
          current = next;
        }
        return { points: ids.map((id) => points.get(id)), closed: ids.length > 2 && links.get(current).includes(start) };
      };
      // the lines that end at the grid's border first, then the closed ones
      for (const [id, next] of links) if (next.length === 1 && !seen.has(id)) lines.push(follow(id));
      for (const id of links.keys()) if (!seen.has(id)) lines.push(follow(id));
      return lines;
    };
    // two rounds of Chaikin's corner cutting, so that a line on a coarse grid reads as a curve
    const smooth = ({ points, closed }) => {
      let out = points;
      for (let round = 0; round < 2 && out.length > 2; round += 1) {
        const next = closed ? [] : [out[0]];
        const n = out.length;
        for (let i = 0; i < (closed ? n : n - 1); i += 1) {
          const a = out[i];
          const b = out[(i + 1) % n];
          next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]], [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
        }
        if (!closed) next.push(out[n - 1]);
        out = next;
      }
      return { points: out, closed };
    };
    // every run's outline of tropical-storm-force wind
    const outlines = () => {
      const m = model();
      const grid = m.grid;
      context.beginPath();
      for (let run = 0; run < m.tracks.length; run += 1) {
        const { values, size, origin } = runField(run, state.hours);
        for (const line of isolines(values, size, size, data.gale).map(smooth)) {
          line.points.forEach(([r, c], i) => {
            const px = x(grid.west + (c + origin[1]) * grid.step);
            const py = y(grid.north - (r + origin[0]) * grid.step);
            if (i) context.lineTo(px, py);
            else context.moveTo(px, py);
          });
          if (line.closed) context.closePath();
        }
      }
      context.lineJoin = "round";
      context.lineWidth = 1;
      context.strokeStyle = rgba(colors.ink, 0.5);
      context.stroke();
    };

    const path = (points, from, to) => {
      // a track from one hour to another, through its 6-hourly points
      context.beginPath();
      const startAt = position(points, from);
      context.moveTo(x(startAt[1]), y(startAt[0]));
      for (let t = 0; t < points.length; t += 1) {
        const hours = data.times[t];
        if (hours > from && hours < to) context.lineTo(x(points[t][1]), y(points[t][0]));
      }
      const endAt = position(points, to);
      context.lineTo(x(endAt[1]), y(endAt[0]));
    };
    const dot = (lat, lon, radius, fill, ring) => {
      context.beginPath();
      context.arc(x(lon), y(lat), radius, 0, Math.PI * 2);
      if (ring) {
        context.lineWidth = 2;
        context.strokeStyle = ring;
        context.stroke();
      }
      context.fillStyle = fill;
      context.fill();
    };
    const label = (text, px, py, color, align = "left", weight = 500) => {
      context.font = `${weight} 11px system-ui, -apple-system, "Segoe UI", sans-serif`;
      context.textAlign = align;
      context.textBaseline = "middle";
      context.lineJoin = "round";
      context.lineWidth = 3;
      context.strokeStyle = colors.surface;
      context.strokeText(text, px, py);
      context.fillStyle = color;
      context.fillText(text, px, py);
    };
    const tracks = () => {
      const m = model();
      const hours = state.hours;
      context.lineCap = "round";
      context.lineJoin = "round";
      if (state.view === "one") {
        const run = m.tracks[0];
        context.setLineDash([4, 4]);
        context.lineWidth = 1.5;
        context.strokeStyle = rgba(colors.ink, 0.55);
        path(run.points, hours, last);
        context.stroke();
        context.setLineDash([]);
        context.lineWidth = 2.25;
        context.strokeStyle = rgba(colors.ink, 0.95);
        path(run.points, 0, hours);
        context.stroke();
        if (run.landfall) {
          const { lat, lon } = run.landfall;
          context.save();
          context.translate(x(lon), y(lat));
          context.rotate(Math.PI / 4);
          context.fillStyle = colors.surface;
          context.strokeStyle = rgba(colors.ink, 1);
          context.lineWidth = 2;
          context.fillRect(-4, -4, 8, 8);
          context.strokeRect(-4, -4, 8, 8);
          context.restore();
        }
        const [lat, lon] = position(run.points, hours);
        dot(lat, lon, 5.5, rgba(colors.ink, 1), colors.surface);
      } else if (state.view === "ensemble") {
        // the tracks recede a little behind the runs' outlines
        for (const run of m.tracks) {
          context.lineWidth = 1;
          context.strokeStyle = rgba(colors.ink, 0.1);
          path(run.points, hours, last);
          context.stroke();
          context.strokeStyle = rgba(colors.ink, 0.32);
          path(run.points, 0, hours);
          context.stroke();
        }
        for (const run of m.tracks) {
          const [lat, lon] = position(run.points, hours);
          dot(lat, lon, 2.2, rgba(colors.ink, 0.9));
        }
        // where the runs have come ashore so far
        for (const run of m.tracks) {
          if (run.landfall && hours >= run.landfall.hours) {
            dot(run.landfall.lat, run.landfall.lon, 3, rgba(colors.ink, 1), colors.surface);
          }
        }
      } else {
        for (const run of m.tracks) {
          context.lineWidth = 1;
          context.strokeStyle = rgba(colors.blue, 0.13);
          path(run.points, hours, last);
          context.stroke();
          context.strokeStyle = rgba(colors.blue, 0.45);
          path(run.points, 0, hours);
          context.stroke();
        }
        const means = meanTracks();
        for (const mean of means) {
          context.setLineDash([3, 4]);
          context.lineWidth = 1.5;
          context.strokeStyle = rgba(colors.red, 0.6);
          path(mean, hours, last);
          context.stroke();
          context.setLineDash([]);
          context.lineWidth = 2.25;
          context.strokeStyle = rgba(colors.red, 1);
          path(mean, 0, hours);
          context.stroke();
        }
        for (const run of m.tracks) {
          const [lat, lon] = position(run.points, hours);
          dot(lat, lon, 1.9, rgba(colors.blue, 0.95));
        }
        for (const mean of means) {
          const [lat, lon] = position(mean, hours);
          dot(lat, lon, 4, rgba(colors.red, 1), colors.surface);
        }
      }
    };
    const observed = () => {
      const hours = state.hours;
      const points = data.observed.points.filter((point) => point[0] >= 0 && point[0] <= hours);
      const [lat, lon] = observedAt(hours);
      context.beginPath();
      const first = observedAt(0);
      context.moveTo(x(first[1]), y(first[0]));
      for (const point of points) context.lineTo(x(point[2]), y(point[1]));
      context.lineTo(x(lon), y(lat));
      context.lineCap = "round";
      context.lineWidth = 5;
      context.strokeStyle = colors.surface;
      context.stroke();
      context.setLineDash([1, 5]);
      context.lineWidth = 2.5;
      context.strokeStyle = colors.text;
      context.stroke();
      context.setLineDash([]);
      context.beginPath();
      context.arc(x(lon), y(lat), 6.5, 0, Math.PI * 2);
      context.lineWidth = 4.5;
      context.strokeStyle = colors.surface;
      context.stroke();
      context.lineWidth = 2;
      context.strokeStyle = colors.text;
      context.stroke();
      const landfall = data.observed.landfall;
      if (landfall && hours >= landfall.hours) {
        label(`Milton came ashore, ${ahead(landfall.hours)}`, x(landfall.lon) - 12, y(landfall.lat) + 17, colors.text, "right", 700);
      }
    };
    const graticule = () => {
      context.lineWidth = 1;
      context.strokeStyle = colors.graticule;
      context.beginPath();
      for (let lon = Math.ceil(extent.west / 5) * 5; lon < extent.east; lon += 5) {
        context.moveTo(Math.round(x(lon)) + 0.5, 0);
        context.lineTo(Math.round(x(lon)) + 0.5, height);
      }
      for (let lat = Math.ceil(extent.south / 5) * 5; lat < extent.north; lat += 5) {
        context.moveTo(0, Math.round(y(lat)) + 0.5);
        context.lineTo(width, Math.round(y(lat)) + 0.5);
      }
      context.stroke();
    };
    // the sea's name, in the empty water north of Milton's path, under the runs
    const sea = () => {
      context.save();
      context.font = `italic 500 ${clamp(width / 48, 10, 13)}px system-ui, -apple-system, "Segoe UI", sans-serif`;
      context.letterSpacing = "0.12em";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillStyle = colors.coast;
      context.fillText("Gulf of Mexico", x(-90.2), y(26.6));
      context.restore();
    };
    const graticuleLabels = () => {
      context.font = `500 10px system-ui, -apple-system, "Segoe UI", sans-serif`;
      context.fillStyle = colors.muted;
      context.textBaseline = "bottom";
      context.textAlign = "center";
      for (let lon = Math.ceil(extent.west / 5) * 5; lon < extent.east; lon += 5) {
        context.fillText(`${-lon}° W`, x(lon), height - 4);
      }
      context.textAlign = "right";
      context.textBaseline = "middle";
      for (let lat = Math.ceil(extent.south / 5) * 5; lat < extent.north; lat += 5) {
        context.fillText(`${lat}° N`, width - 5, y(lat) - 7);
      }
    };
    const land = (fill) => {
      context.beginPath();
      for (const ring of data.land) {
        ring.forEach(([lon, lat], i) => (i ? context.lineTo(x(lon), y(lat)) : context.moveTo(x(lon), y(lat))));
        context.closePath();
      }
      if (fill) {
        context.fillStyle = colors.land;
        context.fill("evenodd");
      } else {
        context.lineWidth = 0.9;
        context.strokeStyle = colors.coast;
        context.stroke();
      }
    };
    const pins = () => {
      for (const place of data.places) {
        const chosen = !state.spot && state.place === place.name;
        dot(place.lat, place.lon, chosen ? 4.5 : 3.2, colors.surface);
        context.beginPath();
        context.arc(x(place.lon), y(place.lat), chosen ? 4.5 : 3.2, 0, Math.PI * 2);
        context.lineWidth = chosen ? 2.25 : 1.4;
        context.strokeStyle = colors.text;
        context.stroke();
        if (width < 520 && !chosen) continue; // on a small map, only the chosen place has its name
        const west = place.lon < -82.3 && place.lon > -84; // Tampa Bay's places: labels to the west
        label(place.name, x(place.lon) + (west ? -8 : 8), y(place.lat), chosen ? colors.text : colors.muted, west ? "right" : "left", chosen ? 700 : 500);
      }
      if (state.spot) {
        const px = x(state.spot.lon);
        const py = y(state.spot.lat);
        context.lineWidth = 2;
        context.strokeStyle = colors.text;
        context.beginPath();
        context.arc(px, py, 6, 0, Math.PI * 2);
        context.moveTo(px - 11, py);
        context.lineTo(px - 3, py);
        context.moveTo(px + 3, py);
        context.lineTo(px + 11, py);
        context.moveTo(px, py - 11);
        context.lineTo(px, py - 3);
        context.moveTo(px, py + 3);
        context.lineTo(px, py + 11);
        context.stroke();
      }
    };
    const draw = () => {
      if (!width) return;
      const ratioPx = window.devicePixelRatio || 1;
      context.setTransform(ratioPx, 0, 0, ratioPx, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = colors.ocean;
      context.fillRect(0, 0, width, height);
      graticule();
      land(true);
      raster();
      land(false);
      sea();
      tracks();
      if (state.view === "ensemble") outlines();
      if (state.truth) observed();
      pins();
      graticuleLabels();
      if (fade.start) {
        // the map before a switch, fading out over the new one
        const t = clamp((performance.now() - fade.start) / FADE, 0, 1);
        context.save();
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.globalAlpha = 1 - t * (2 - t);
        context.drawImage(fade.image, 0, 0);
        context.restore();
      }
    };
    // a cross-fade from the map as it is to the one after a switch of view or model
    const FADE = 420; // ms
    const fade = { image: document.createElement("canvas"), start: 0 };
    const crossfade = () => {
      if (reducedMotion() || !width) return;
      fade.image.width = canvas.width;
      fade.image.height = canvas.height;
      fade.image.getContext("2d").drawImage(canvas, 0, 0);
      fade.start = performance.now();
      const step = () => {
        if (performance.now() - fade.start >= FADE) fade.start = 0;
        draw();
        if (fade.start) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
      // and the panel's new content fades in
      panel.classList.remove("milton-demo-enter");
      void panel.offsetWidth; // restart the animation
      panel.classList.add("milton-demo-enter");
    };
    const resize = () => {
      const next = map.clientWidth;
      if (!next) return;
      width = next;
      height = Math.round(width * ratio);
      const ratioPx = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratioPx);
      canvas.height = Math.round(height * ratioPx);
      canvas.style.height = `${height}px`;
      draw();
    };

    // -- pointing at the map ---------------------------------------------------------------------------
    const describe = (lat, lon) => {
      const cell = cellOf(lat, lon);
      if (cell < 0) return null;
      const hours = state.hours;
      const value = (name) => field(name, hours)[cell];
      const lines = [];
      if (state.view === "one") lines.push(`This run: ${ms(value("single"))}`);
      if (state.view === "ensemble") {
        lines.push(`Chance of 34-kt wind: ${percent(value("gale"))}`);
        lines.push(`Mean of the runs: ${ms(value("mean"))}`);
      }
      if (state.view === "split") {
        const share = value("epistemic");
        lines.push(`Total spread: ±${ms(value("spread"))}`);
        if (splittable(state.model)) {
          lines.push(`${percent(1 - share)} aleatoric, ${percent(share)} epistemic`);
        }
      }
      if (state.truth && withTruth()) lines.push(`What happened: ${ms(value("truth"))}`);
      return { head: `${Math.abs(lat).toFixed(1)}° N, ${Math.abs(lon).toFixed(1)}° W · ${ahead(hours)}`, lines };
    };
    const nearestPlace = (px, py) => {
      let best = null;
      let distance = 14;
      for (const place of data.places) {
        const d = Math.hypot(x(place.lon) - px, y(place.lat) - py);
        if (d < distance) {
          best = place;
          distance = d;
        }
      }
      return best;
    };
    let pointer = null;
    const showTip = () => {
      if (!pointer) {
        tip.hidden = true;
        return;
      }
      const { lat, lon } = unproject(pointer.x, pointer.y);
      const place = nearestPlace(pointer.x, pointer.y);
      const text = describe(place?.lat ?? lat, place?.lon ?? lon);
      if (!text) {
        tip.hidden = true;
        return;
      }
      tip.replaceChildren(el("strong", "", place ? `${place.name} · ${ahead(state.hours)}` : text.head));
      for (const line of text.lines) tip.append(el("div", "", line));
      tip.hidden = false;
      const left = pointer.x + 14 + tip.offsetWidth > width ? pointer.x - 14 - tip.offsetWidth : pointer.x + 14;
      const top = clamp(pointer.y - tip.offsetHeight / 2, 4, height - tip.offsetHeight - 4);
      tip.style.left = `${Math.max(left, 4)}px`;
      tip.style.top = `${top}px`;
    };
    canvas.addEventListener("pointermove", (event) => {
      if (event.pointerType === "touch") return;
      const box = canvas.getBoundingClientRect();
      pointer = { x: event.clientX - box.left, y: event.clientY - box.top };
      showTip();
    });
    canvas.addEventListener("pointerleave", () => {
      pointer = null;
      showTip();
    });
    canvas.addEventListener("click", (event) => {
      const box = canvas.getBoundingClientRect();
      const px = event.clientX - box.left;
      const py = event.clientY - box.top;
      const place = nearestPlace(px, py);
      if (place) {
        setPlace(place.name);
        return;
      }
      const { lat, lon } = unproject(px, py);
      const grid = model().grid;
      const snapped = {
        lat: grid.north - Math.round((grid.north - lat) / grid.step) * grid.step,
        lon: grid.west + Math.round((lon - grid.west) / grid.step) * grid.step,
      };
      if (cellOf(snapped.lat, snapped.lon) < 0) return;
      state.spot = snapped;
      update();
    });

    // -- the chart of a spot over time ------------------------------------------------------------------
    let chartWidth = 0;
    const chartHeight = 152;
    const pad = { left: 30, right: 10, top: 8, bottom: 20 };
    let cursor = null;
    let chartX = () => 0;
    // the charts' scales, the same for every place and model, so that switching shows which is higher
    const limits = { wind: data.gale * 1.25, variance: 1 };
    for (const m of Object.values(data.models)) {
      for (const place of Object.values(m.places)) {
        for (const run of place.runs) limits.wind = Math.max(limits.wind, ...run);
        if (place.truth) limits.wind = Math.max(limits.wind, ...place.truth);
        place.aleatoric.forEach((a, t) => (limits.variance = Math.max(limits.variance, a + place.epistemic[t])));
      }
    }
    const buildChart = () => {
      chartWidth = chart.clientWidth || 320;
      chart.setAttribute("viewBox", `0 0 ${chartWidth} ${chartHeight}`);
      chart.replaceChildren();
      const s = series();
      const hours = data.times;
      const innerW = chartWidth - pad.left - pad.right;
      const innerH = chartHeight - pad.top - pad.bottom;
      chartX = (h) => pad.left + (h / last) * innerW;
      const view = state.view;
      // a spot on the map may go higher than the places
      let top;
      if (view === "split") {
        top = Math.max(limits.variance, ...s.aleatoric.map((a, t) => a + s.epistemic[t]));
      } else {
        const values = [limits.wind, ...s.single, ...s.upper, ...(state.truth && s.truth ? s.truth : [])];
        if (s.runs && view === "ensemble") values.push(...s.runs.flat());
        top = Math.max(...values);
      }
      const tick = top > 40 ? 20 : top > 16 ? 10 : top > 6 ? 5 : top > 2 ? 1 : 0.5;
      top = Math.ceil(top / tick) * tick;
      const chartY = (v) => pad.top + innerH - (v / top) * innerH;
      const line = (values, className) => {
        const d = values.map((v, t) => `${t ? "L" : "M"}${chartX(hours[t]).toFixed(1)},${chartY(v).toFixed(1)}`).join("");
        chart.append(svg("path", { d, class: className }));
      };
      const area = (lower, upper, className) => {
        const up = upper.map((v, t) => `${t ? "L" : "M"}${chartX(hours[t]).toFixed(1)},${chartY(v).toFixed(1)}`).join("");
        const down = lower
          .map((v, t) => [t, v])
          .reverse()
          .map(([t, v]) => `L${chartX(hours[t]).toFixed(1)},${chartY(v).toFixed(1)}`)
          .join("");
        chart.append(svg("path", { d: `${up}${down}Z`, class: className }));
      };
      // axes and ticks
      for (let v = 0; v <= top + 1e-9; v += tick) {
        chart.append(svg("line", { x1: pad.left, x2: chartWidth - pad.right, y1: chartY(v), y2: chartY(v), class: "axis", opacity: v ? 0.45 : 1 }));
        const text = svg("text", { x: pad.left - 5, y: chartY(v) + 3.5, "text-anchor": "end" });
        text.textContent = Number.isInteger(v) ? v : v.toFixed(1);
        chart.append(text);
      }
      for (let h = 0; h <= last; h += 24) {
        const text = svg("text", { x: chartX(h), y: chartHeight - 5, "text-anchor": h === 0 ? "start" : h === last ? "end" : "middle" });
        text.textContent = day(h);
        chart.append(text);
      }
      if (view !== "split") {
        chart.append(svg("line", { x1: pad.left, x2: chartWidth - pad.right, y1: chartY(data.gale), y2: chartY(data.gale), class: "threshold" }));
        const text = svg("text", { x: chartWidth - pad.right, y: chartY(data.gale) - 4, "text-anchor": "end" });
        text.textContent = "34 kt";
        chart.append(text);
      }
      if (view === "one") {
        line(s.single, "line");
      } else if (view === "ensemble") {
        area(s.lower, s.upper, "band");
        if (s.runs) s.runs.forEach((run) => line(run, "run"));
        line(s.mean, "line");
      } else {
        const total = s.aleatoric.map((a, t) => a + s.epistemic[t]);
        area(s.aleatoric.map(() => 0), s.aleatoric, "aleatoric");
        area(s.aleatoric, total, "epistemic");
        line(total, "total");
      }
      if (state.truth && s.truth && view !== "split") line(s.truth, "truth");
      cursor = svg("line", { y1: pad.top - 2, y2: pad.top + innerH, class: "cursor" });
      chart.append(cursor);
      placeCursor();
      const unit = view === "split" ? "variance of the wind, (m/s)²" : "wind at 10 m, m/s";
      spotUnit.textContent = unit;
      spotName.textContent = s.name;
      chart.setAttribute("aria-label", `${s.name}: ${unit}, over four days`);
      return s;
    };
    const placeCursor = () => {
      if (!cursor) return;
      const px = chartX(state.hours);
      cursor.setAttribute("x1", px);
      cursor.setAttribute("x2", px);
    };
    const scrub = (event) => {
      const box = chart.getBoundingClientRect();
      const share = (event.clientX - box.left - (pad.left / chartWidth) * box.width) / (((chartWidth - pad.left - pad.right) / chartWidth) * box.width);
      setHours(clamp(share, 0, 1) * last);
    };
    chart.addEventListener("pointerdown", (event) => {
      chart.setPointerCapture(event.pointerId);
      stop();
      scrub(event);
    });
    chart.addEventListener("pointermove", (event) => {
      if (chart.hasPointerCapture(event.pointerId)) scrub(event);
    });

    // -- where the runs come ashore: a strip along Florida's Gulf coast, south to north ----------------
    let shoreMarks = []; // the runs' dots and Milton's mark, with the hour they come ashore
    let shoreSummary = "";
    const buildShore = () => {
      shore.hidden = state.view !== "ensemble";
      shoreMarks = [];
      if (shore.hidden) return;
      const m = model();
      const runs = m.tracks
        .map((run, index) => ({ index, landfall: run.landfall }))
        .filter((run) => run.landfall)
        .sort((a, b) => a.landfall.lat - b.landfall.lat);
      const observed = data.observed.landfall;
      const lats = [...runs.map((run) => run.landfall.lat), observed.lat];
      const south = Math.min(26.1, ...lats) - 0.15;
      const north = Math.max(29.1, ...lats) + 0.15;
      const width = shoreChart.clientWidth || 320;
      const radius = 3.6;
      const pitch = 2 * radius + 1.5;
      const x = (lat) => 8 + ((lat - south) / (north - south)) * (width - 16);
      // a dot plot: the runs in columns as wide as a dot, stacked up from the coast's line
      const heights = new Map();
      for (const run of runs) {
        const column = Math.round(x(run.landfall.lat) / pitch);
        run.cx = column * pitch;
        run.level = heights.get(column) ?? 0;
        heights.set(column, run.level + 1);
      }
      const top = state.truth ? 14 : 4; // room for Milton's label
      const coast = top + Math.max(3, ...heights.values()) * pitch + 2;
      const height = coast + 22;
      shoreChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
      shoreChart.style.height = `${height}px`;
      shoreChart.replaceChildren(svg("line", { x1: 2, x2: width - 2, y1: coast, y2: coast, class: "coast" }));
      // the coast's places under the line, as many as fit
      let free = 0;
      for (const place of [...data.coast].sort((a, b) => a.lat - b.lat)) {
        const px = x(place.lat);
        const half = place.name.length * 2.75;
        if (place.lat < south || place.lat > north || px - half < free || px + half > width) continue;
        shoreChart.append(svg("line", { x1: px, x2: px, y1: coast, y2: coast + 4, class: "tick" }));
        const text = svg("text", { x: px, y: coast + 15, "text-anchor": "middle" });
        text.textContent = place.name;
        shoreChart.append(text);
        free = px + half + 10;
      }
      for (const run of runs) {
        const dot = svg("circle", { cx: run.cx, cy: coast - radius - 1.5 - run.level * pitch, r: radius, class: "run" });
        const tip = svg("title");
        tip.textContent = `A run comes ashore near ${coastName(run.landfall.lat, run.landfall.lon)}, ${ahead(run.landfall.hours)}`;
        dot.append(tip);
        shoreChart.append(dot);
        shoreMarks.push({ node: dot, hours: run.landfall.hours, hide: false });
      }
      if (state.truth) {
        // where Milton came ashore, once it has
        const mark = svg("g", { class: "milton" });
        const px = x(observed.lat);
        mark.append(svg("line", { x1: px, x2: px, y1: 11, y2: coast }));
        const text = svg("text", { x: px, y: 8, "text-anchor": "middle" });
        text.textContent = `Milton, ${ahead(observed.hours)}`;
        mark.append(text);
        shoreChart.append(mark);
        shoreMarks.push({ node: mark, hours: observed.hours, hide: true });
      }
      // the places where the most runs come ashore
      const counts = new Map();
      for (const run of runs) {
        const name = coastName(run.landfall.lat, run.landfall.lon);
        counts.set(name, (counts.get(name) ?? 0) + 1);
      }
      const most = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 3);
      const named = most.map(([name, count], i) => (i ? `${name} (${count})` : `${name} (${count} of ${m.tracks.length})`));
      shoreSummary = runs.length
        ? `Most come ashore near ${named.length > 1 ? `${named.slice(0, -1).join(", ")} and ${named.at(-1)}` : named[0]}.`
        : "No run comes ashore.";
      shoreChart.setAttribute("aria-label", `Where the runs come ashore, south to north. ${shoreSummary}`);
    };
    // the runs come ashore as time goes by: until then, a dot is an outline
    const placeShore = () => {
      if (shore.hidden) return;
      let ashore = 0;
      for (const { node, hours, hide } of shoreMarks) {
        node.classList.toggle(hide ? "later" : "ahead", state.hours < hours);
        if (!hide && state.hours >= hours) ashore += 1;
      }
      shoreNote.textContent = `By ${ahead(state.hours)}, ${ashore} of ${model().tracks.length} runs are ashore. ${shoreSummary}`;
    };

    // -- the texts --------------------------------------------------------------------------------------
    let current = null; // the series of the spot shown
    const strong = (text, className = "") => el("strong", className, text);
    const sentence = (...parts) => {
      lead.replaceChildren(...parts.map((part) => (typeof part === "string" ? document.createTextNode(part) : part)));
    };
    const landfalls = () =>
      model()
        .tracks.map((run) => run.landfall)
        .filter(Boolean)
        .sort((a, b) => a.lat - b.lat);
    const texts = () => {
      const m = model();
      const runs = m.tracks.length;
      if (state.view === "one") {
        title.textContent = "One run, one answer";
        const run = m.tracks[0];
        if (run.landfall) {
          sentence(
            `One run of ${m.name} brings Milton ashore near `,
            strong(coastName(run.landfall.lat, run.landfall.lon)),
            ` on ${when(run.landfall.hours)}. On its own, it gives no hint of how sure it is.`,
          );
        } else {
          sentence(`One run of ${m.name} keeps Milton offshore. On its own, it gives no hint of how sure it is.`);
        }
      } else if (state.view === "ensemble") {
        title.textContent = `${runs} runs, many answers`;
        const ashore = landfalls();
        const models = m.models > 1 ? `${counted(m)} with ${m.runs_per_model} runs each` : `one model`;
        if (ashore.length) {
          const times = ashore.map((l) => l.hours);
          sentence(
            `${runs} runs from ${models}. ${ashore.length} of them bring Milton ashore, from `,
            strong(coastName(ashore[0].lat, ashore[0].lon)),
            " to ",
            strong(coastName(ashore[ashore.length - 1].lat, ashore[ashore.length - 1].lon)),
            `, ${Math.min(...times)} to ${Math.max(...times)} hours ahead. Every outline is where one run has tropical-storm-force wind.`,
          );
        } else {
          sentence(`${runs} runs from ${models}. None of them brings Milton ashore.`);
        }
      } else {
        title.textContent = "Where the spread comes from";
        sentence(
          "The runs of one model scatter as the atmosphere might: that part of the spread is ",
          strong("aleatoric", "milton-demo-aleatoric"),
          ", and more training data would ",
          el("em", "", "not"),
          ` remove it. Where the ${counted()} disagree, they lack knowledge: that part is `,
          strong("epistemic", "milton-demo-epistemic"),
          ". Together, they make up the total spread.",
        );
      }
    };
    const stat = (term, value, className = "") => {
      const item = el("div");
      item.append(el("dt", "", term), el("dd", className, value));
      return item;
    };
    const updateStats = () => {
      const s = current;
      if (!s) return;
      const hours = state.hours;
      const at = (values) => sample(values, hours);
      const items = [];
      if (state.view === "one") {
        const peak = Math.max(...s.single);
        const peakAt = data.times[s.single.indexOf(peak)];
        items.push(stat("This run", ms(at(s.single))), stat("Its peak", `${ms(peak)}, ${ahead(peakAt)}`));
        items.push(state.truth && s.truth ? stat("What happened", ms(at(s.truth))) : stat("Runs shown", `1 of ${model().tracks.length}`));
      } else if (state.view === "ensemble") {
        items.push(stat("Chance of 34 kt", percent(at(s.gale))), stat("5–95% of the runs", `${at(s.lower).toFixed(0)}–${at(s.upper).toFixed(0)} m/s`));
        items.push(state.truth && s.truth ? stat("What happened", ms(at(s.truth))) : stat("Mean of the runs", ms(at(s.mean))));
      } else {
        const a = at(s.aleatoric);
        const e = at(s.epistemic);
        const total = a + e;
        items.push(stat("Total spread", `±${ms(Math.sqrt(total))}`));
        if (splittable(state.model)) {
          items.push(
            stat("Aleatoric", total > 0 ? percent(a / total) : "–", "milton-demo-aleatoric"),
            stat("Epistemic", total > 0 ? percent(e / total) : "–", "milton-demo-epistemic"),
          );
        }
      }
      stats.replaceChildren(...items);
    };
    const legends = () => {
      const item = (swatch, text) => {
        const span = el("span");
        span.append(swatch, document.createTextNode(text));
        return span;
      };
      const swatch = (className, background) => {
        const i = el("i", className);
        if (background) i.style.background = background;
        return i;
      };
      const line = (color, thin = false) => {
        const i = el("i", thin ? "line thin" : "line");
        i.style.borderTopColor = color;
        return i;
      };
      const ink = (alpha) => rgba(colors.ink, alpha);
      const items = [];
      if (state.view === "one") {
        items.push(
          item(swatch("ramp", `linear-gradient(90deg, ${ink(0.04)}, ${ink(0.8)})`), "wind of this run, 8 to 33 m/s"),
          item(line(ink(0.95)), "its track, ahead dashed"),
        );
      } else if (state.view === "ensemble") {
        items.push(
          item(swatch("ramp", `linear-gradient(90deg, ${ink(0.03)}, ${ink(0.6)})`), "chance of tropical-storm-force wind (34 kt)"),
          item(line(ink(0.5), true), "where one run has it"),
        );
      } else {
        items.push(
          item(swatch("", rgba(colors.blue, 0.85)), "aleatoric: the runs of one model scatter"),
          item(swatch("", rgba(colors.red, 0.85)), `epistemic: the ${groups()} disagree`),
          item(line(ink(0.9)), "total: sum of aleatoric and epistemic"),
        );
      }
      if (state.truth) items.push(item(swatch("dashed"), withTruth() ? "what happened" : "Milton's observed track"));
      legend.replaceChildren(...items);
    };
    const footnote = () => {
      const m = model();
      const count = m.models > 1 ? `${m.models} ${groups(m)} × ${m.runs_per_model} runs` : `${m.runs_per_model} runs`;
      note.textContent =
        `${m.name} (${m.detail}), ${count}, from ${when(0)}. ${m.about ?? ""} ` +
        `Observed track: ${data.observed.source}. Coastlines: Natural Earth.`;
    };

    // -- state -----------------------------------------------------------------------------------------
    const update = (what = "all") => {
      for (const [key, button] of viewButtons) {
        button.setAttribute("aria-pressed", String(key === state.view));
        const disabled = key === "split" && !splittable(state.model);
        button.disabled = disabled;
        button.title = disabled ? `${model().label} is a single model: there is no second model to disagree with.` : "";
      }
      for (const [key, button] of modelButtons) button.setAttribute("aria-pressed", String(key === state.model));
      for (const [name, chip] of chipButtons) chip.setAttribute("aria-pressed", String(!state.spot && name === state.place));
      truthBox.checked = state.truth;
      slider.value = String(Math.round(state.hours));
      slider.setAttribute("aria-valuetext", `${ahead(state.hours)}, ${when(state.hours)}`);
      clock.replaceChildren(document.createTextNode(`${ahead(state.hours)} `), el("span", "", `· ${when(state.hours)}`));
      const icon = svg("svg", { viewBox: "0 0 10 10", "aria-hidden": "true" });
      icon.append(svg("path", { d: state.playing ? "M1.5 1h2.5v8H1.5zM6 1h2.5v8H6z" : "M2 1l7 4-7 4z" }));
      play.replaceChildren(icon);
      play.setAttribute("aria-label", state.playing ? "Pause" : "Play the four days");
      if (what === "all") {
        current = buildChart();
        buildShore();
        texts();
        legends();
        footnote();
        canvas.setAttribute("aria-label", `${title.textContent}. Map of the Gulf of Mexico and Florida, ${ahead(state.hours)}.`);
      } else {
        placeCursor();
      }
      updateStats();
      placeShore();
      draw();
      if (pointer) showTip();
    };
    const setView = (view) => {
      if (view === "split" && !splittable(state.model)) return;
      if (view === state.view) return;
      crossfade();
      state.view = view;
      update();
    };
    const setModel = (key) => {
      if (key === state.model) return;
      crossfade();
      state.model = key;
      if (state.view === "split" && !splittable(key)) state.view = "ensemble";
      update();
    };
    const setPlace = (name) => {
      state.place = name;
      state.spot = null;
      update();
    };
    const setHours = (hours) => {
      state.hours = clamp(hours, 0, last);
      update("time");
    };
    let frame = 0;
    let previous = 0;
    const tick = (stamp) => {
      if (!state.playing) return;
      const seconds = previous ? (stamp - previous) / 1000 : 0;
      previous = stamp;
      const next = state.hours + seconds * HOURS_PER_SECOND;
      if (next >= last) {
        state.hours = last;
        stop();
        return;
      }
      state.hours = next;
      update("time");
      frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      state.playing = false;
      cancelAnimationFrame(frame);
      previous = 0;
      update("time");
    };
    play.addEventListener("click", () => {
      if (state.playing) {
        stop();
        return;
      }
      if (state.hours >= last) state.hours = 0;
      state.playing = true;
      if (reducedMotion()) state.hours = Math.ceil((state.hours + 1) / step) * step; // step by 6 h
      previous = 0;
      update("time");
      frame = requestAnimationFrame(tick);
    });
    slider.addEventListener("input", () => {
      if (state.playing) stop();
      setHours(Number(slider.value));
    });
    truthBox.addEventListener("change", () => {
      state.truth = truthBox.checked;
      update();
    });

    readColors();
    new ResizeObserver(() => {
      resize();
      if (Math.abs((chart.clientWidth || 0) - chartWidth) > 1) update();
    }).observe(root);
    const retheme = () => {
      readColors();
      update();
    };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
    new MutationObserver(retheme).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    root.classList.add("milton-demo-ready");
    resize();
    update();
  }

  const boot = () => {
    try {
      start(window.PROBLY_DEMO_MILTON);
    } catch (error) {
      console.error("The Milton demo could not start:", error);
      showFallback();
    }
  };
  if (window.PROBLY_DEMO_MILTON) {
    boot();
  } else {
    const loader = document.createElement("script");
    loader.src = `${root.dataset.root ?? ""}data.js`;
    loader.onload = boot;
    loader.onerror = showFallback;
    document.head.appendChild(loader);
  }
})();
