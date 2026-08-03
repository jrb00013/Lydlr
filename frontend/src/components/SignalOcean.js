import React, { useEffect, useRef, useCallback, useState } from 'react';
import './SignalOcean.css';

const MODALITY_COLORS = {
  camera: { r: 34, g: 211, b: 238 },
  lidar: { r: 16, g: 185, b: 129 },
  imu: { r: 245, g: 158, b: 11 },
  audio: { r: 56, g: 189, b: 248 },
};

const MODALITY_ORDER = ['camera', 'lidar', 'imu', 'audio'];

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function inventDemoMetric(t, nodeId = 'node_0') {
  const pulse = 0.5 + 0.5 * Math.sin(t * 0.35);
  const storm = 0.5 + 0.5 * Math.sin(t * 0.11 + 1.7);
  return {
    node_id: nodeId,
    compression_ratio: 2.2 + pulse * 6.5 + storm * 2,
    quality_score: clamp(0.92 - storm * 0.22 - pulse * 0.05, 0.55, 0.98),
    latency_ms: 12 + storm * 55 + pulse * 18,
    bytes_in: Math.floor(180000 + pulse * 420000 + storm * 90000),
    bytes_out: Math.floor(28000 + (1 - pulse) * 40000),
    bandwidth_estimate: 0.35 + storm * 0.45,
    modality_bytes_in: {
      camera: Math.floor(120000 + pulse * 200000),
      lidar: Math.floor(40000 + storm * 80000),
      imu: Math.floor(8000 + pulse * 12000),
      audio: Math.floor(20000 + storm * 35000),
    },
    modality_bytes_out: {
      camera: Math.floor(14000 + pulse * 8000),
      lidar: Math.floor(6000 + storm * 4000),
      imu: Math.floor(1200),
      audio: Math.floor(3000 + pulse * 2000),
    },
  };
}

/* ------------------------------------------------------------------ */
/* simplex noise + fbm                                                  */
/* ------------------------------------------------------------------ */
function makeSimplex(seed = 20260803) {
  const perm = new Uint8Array(512);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) p[i] = i;
  let s = seed;
  const rnd = () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
  for (let i = 255; i > 0; i -= 1) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }
  for (let i = 0; i < 512; i += 1) perm[i] = p[i & 255];

  const F2 = 0.5 * (Math.sqrt(3) - 1);
  const G2 = (3 - Math.sqrt(3)) / 6;
  const grad = (h, x, y) => {
    const g = h & 7;
    const u = g < 4 ? x : y;
    const v = g < 4 ? y : x;
    return ((g & 1) ? -u : u) + ((g & 2) ? -2 * v : 2 * v);
  };

  const noise = (xin, yin) => {
    let n0 = 0;
    let n1 = 0;
    let n2 = 0;
    const s2 = (xin + yin) * F2;
    const i = Math.floor(xin + s2);
    const j = Math.floor(yin + s2);
    const t = (i + j) * G2;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);
    const i1 = x0 > y0 ? 1 : 0;
    const j1 = x0 > y0 ? 0 : 1;
    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;
    const ii = i & 255;
    const jj = j & 255;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 > 0) {
      t0 *= t0;
      n0 = t0 * t0 * grad(perm[ii + perm[jj]], x0, y0);
    }
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 > 0) {
      t1 *= t1;
      n1 = t1 * t1 * grad(perm[ii + i1 + perm[jj + j1]], x1, y1);
    }
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 > 0) {
      t2 *= t2;
      n2 = t2 * t2 * grad(perm[ii + 1 + perm[jj + 1]], x2, y2);
    }
    return 70 * (n0 + n1 + n2);
  };
  return noise;
}

const simplex = makeSimplex(20260803);

function fbm(x, y, oct = 3) {
  let v = 0;
  let a = 0.5;
  let f = 1;
  for (let i = 0; i < oct; i += 1) {
    v += a * simplex(x * f, y * f);
    a *= 0.5;
    f *= 2;
  }
  return v;
}

function mixRgb(a, b, t) {
  return {
    r: Math.round(lerp(a.r, b.r, t)),
    g: Math.round(lerp(a.g, b.g, t)),
    b: Math.round(lerp(a.b, b.b, t)),
  };
}

/* ------------------------------------------------------------------ */
/* precomputed normalized scenery                                       */
/* ------------------------------------------------------------------ */
const STARS = Array.from({ length: 150 }, () => ({
  x: Math.random(),
  y: Math.random() * 0.34,
  r: 0.3 + Math.random() * 0.9,
  ph: Math.random() * Math.PI * 2,
  sp: 0.4 + Math.random() * 1.6,
}));

const GLINTS = Array.from({ length: 120 }, () => ({
  x: Math.random(),
  ph: Math.random() * Math.PI * 2,
  sp: 0.6 + Math.random() * 2.2,
}));

const CAUSTIC_NODES = (() => {
  const cols = 16;
  const rows = 6;
  const arr = [];
  for (let j = 0; j <= rows; j += 1) {
    for (let i = 0; i <= cols; i += 1) {
      arr.push({ u: i / cols, v: 0.68 + (j / rows) * 0.32 });
    }
  }
  return arr;
})();

const CAUSTIC_EDGES = (() => {
  const cols = 16;
  const rows = 6;
  const edges = [];
  for (let j = 0; j <= rows; j += 1) {
    for (let i = 0; i <= cols; i += 1) {
      const idx = j * (cols + 1) + i;
      if (i < cols) edges.push([idx, idx + 1]);
      if (j < rows) edges.push([idx, idx + (cols + 1)]);
    }
  }
  return edges;
})();

/* ------------------------------------------------------------------ */
/* the ocean                                                            */
/* ------------------------------------------------------------------ */
function SignalOcean({
  metric,
  fleetMetrics = [],
  linkHealth,
  selectedNode,
  onSelectNode,
  onHoverModality,
  brand = 'Lydlr',
  compact = false,
  demoFallback = true,
}) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const rafRef = useRef(0);
  const [hoveredMod, setHoveredMod] = useState(null);
  const [usingDemo, setUsingDemo] = useState(false);
  const pointerRef = useRef({ x: 0.5, y: 0.5, glowUntil: 0 });
  const lastMetricAt = useRef(0);

  const stateRef = useRef({
    t: 0,
    fish: [],
    plankton: [],
    sparks: [],
    pings: [],
    rain: [],
    flow: null,
    flash: 0,
    lx: 0,
    ly: 0,
    _demoLatched: false,
    target: inventDemoMetric(0),
    smooth: inventDemoMetric(0),
  });

  const applyMetric = useCallback((m, linkH, nodeId) => {
    if (!m) return;
    lastMetricAt.current = performance.now();
    setUsingDemo(false);
    const s = stateRef.current;
    const mIn = m.modality_bytes_in || {};
    const total = Object.values(mIn).reduce((a, b) => a + (Number(b) || 0), 0) || 1;
    const modalities = {};
    MODALITY_ORDER.forEach((k) => {
      modalities[k] = (Number(mIn[k]) || 0) / total;
    });
    const sum = MODALITY_ORDER.reduce((a, k) => a + modalities[k], 0);
    if (sum < 0.01) {
      Object.assign(modalities, { camera: 0.4, lidar: 0.25, imu: 0.15, audio: 0.2 });
    } else {
      MODALITY_ORDER.forEach((k) => {
        modalities[k] /= sum;
      });
    }

    let budgetFill = 0.4;
    const rows = linkH?.nodes;
    if (Array.isArray(rows) && rows.length) {
      const row = rows.find((n) => n.node_id === nodeId) || rows[0];
      if (row) {
        budgetFill = clamp(
          (Number(row.estimated_throughput_kbps) || 0) /
            Math.max(Number(row.uplink_budget_kbps) || 1, 1),
          0,
          1.25
        );
      }
    } else if (m.bandwidth_estimate != null) {
      budgetFill = clamp(Number(m.bandwidth_estimate), 0, 1.25);
    }

    s.target = {
      compression: clamp(Number(m.compression_ratio) || 2, 0.5, 48),
      quality: clamp(Number(m.quality_score) || 0.85, 0, 1),
      latency: clamp(Number(m.latency_ms) || 20, 1, 500),
      bytesIn: Math.max(1, Number(m.bytes_in) || 1),
      budgetFill,
      modalities,
      nodeId: m.node_id || nodeId,
    };
  }, []);

  useEffect(() => {
    applyMetric(metric, linkHealth, selectedNode);
  }, [metric, linkHealth, selectedNode, applyMetric]);

  const spawnFish = useCallback((w, h, modalities, rate) => {
    const r = Math.random();
    let acc = 0;
    let mod = 'camera';
    for (const k of MODALITY_ORDER) {
      acc += modalities[k] || 0;
      if (r <= acc) {
        mod = k;
        break;
      }
    }
    const layer = MODALITY_ORDER.indexOf(mod);
    return {
      x: -20 - Math.random() * 90,
      y: h * (0.32 + layer * 0.15) + (Math.random() - 0.5) * h * 0.1,
      vx: 0.6 + Math.random() * 0.9,
      vy: (Math.random() - 0.5) * 0.3,
      mod,
      size: 2 + Math.random() * 3.4,
      tail: Math.random() * Math.PI * 2,
      flick: Math.random() * Math.PI * 2,
      surge: 0,
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d', { alpha: false });
    const s = stateRef.current;

    const resize = () => {
      const parent = wrapRef.current || canvas.parentElement;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = parent?.clientWidth || window.innerWidth;
      const h = parent?.clientHeight || (compact ? 280 : 420);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      s.flow = null;
    };
    resize();
    window.addEventListener('resize', resize);

    const buildFlow = (w, h) => {
      const cols = Math.max(4, Math.floor(w / 56));
      const rows = Math.max(4, Math.floor(h / 56));
      const arr = new Float32Array(cols * rows);
      s.flow = { cols, rows, arr };
      for (let j = 0; j < rows; j += 1) {
        for (let i = 0; i < cols; i += 1) {
          const cx = (i + 0.5) / cols;
          const cy = (j + 0.5) / rows;
          arr[j * cols + i] = fbm(cx * 1.4, cy * 1.4) * Math.PI * 1.6;
        }
      }
    };

    const flowAt = (x, y, w, h) => {
      const f = s.flow;
      if (!f) return 0;
      const gx = clamp(x / w, 0, 0.999) * f.cols;
      const gy = clamp(y / h, 0, 0.999) * f.rows;
      const i0 = Math.floor(gx);
      const j0 = Math.floor(gy);
      const i1 = Math.min(i0 + 1, f.cols - 1);
      const j1 = Math.min(j0 + 1, f.rows - 1);
      const fx = gx - i0;
      const fy = gy - j0;
      const a = f.arr[j0 * f.cols + i0];
      const b = f.arr[j0 * f.cols + i1];
      const c = f.arr[j1 * f.cols + i0];
      const d = f.arr[j1 * f.cols + i1];
      const top = a + (b - a) * fx;
      const bot = c + (d - c) * fx;
      return top + (bot - top) * fy;
    };

    const drawFish = (fish, c, dim) => {
      const sp = Math.min(7, Math.hypot(fish.vx, fish.vy) + 0.5);
      const ang = Math.atan2(fish.vy, fish.vx);
      const wig = Math.sin(s.t * 9 + fish.tail);
      ctx.save();
      ctx.translate(fish.x, fish.y);
      ctx.rotate(ang);
      ctx.lineCap = 'round';
      ctx.lineWidth = Math.max(1, fish.size * 0.28);
      ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${0.55 * dim})`;
      ctx.beginPath();
      ctx.moveTo(-fish.size * 0.9, 0);
      ctx.quadraticCurveTo(
        -fish.size * 1.2,
        -fish.size * 0.8 * (1 + wig * 0.4),
        -fish.size * 1.7,
        -fish.size * 0.9 * wig
      );
      ctx.moveTo(-fish.size * 0.9, 0);
      ctx.quadraticCurveTo(
        -fish.size * 1.2,
        fish.size * 0.8 * (1 + wig * 0.4),
        -fish.size * 1.7,
        fish.size * 0.9 * wig
      );
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(fish.size * 1.15, 0);
      ctx.quadraticCurveTo(0, fish.size * 0.62, -fish.size * 0.9, 0);
      ctx.quadraticCurveTo(0, -fish.size * 0.62, fish.size * 1.15, 0);
      ctx.closePath();
      ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${(0.5 + 0.22 * Math.sin(s.t * 6 + fish.flick)) * dim})`;
      ctx.fill();
      ctx.fillStyle = `rgba(248,250,252,${0.5 * dim})`;
      ctx.beginPath();
      ctx.arc(fish.size * 0.55, 0, Math.max(0.8, fish.size * 0.14), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    };

    const tick = () => {
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      s.t += 0.016;

      if (demoFallback && performance.now() - lastMetricAt.current > 2500) {
        if (!s._demoLatched) {
          s._demoLatched = true;
          setUsingDemo(true);
        }
        const demo = inventDemoMetric(s.t, selectedNode || 'node_0');
        const mIn = demo.modality_bytes_in;
        const total = Object.values(mIn).reduce((a, b) => a + b, 0) || 1;
        s.target = {
          compression: demo.compression_ratio,
          quality: demo.quality_score,
          latency: demo.latency_ms,
          bytesIn: demo.bytes_in,
          budgetFill: demo.bandwidth_estimate,
          modalities: {
            camera: mIn.camera / total,
            lidar: mIn.lidar / total,
            imu: mIn.imu / total,
            audio: mIn.audio / total,
          },
          nodeId: demo.node_id,
        };
      } else if (s._demoLatched && performance.now() - lastMetricAt.current < 2000) {
        s._demoLatched = false;
        setUsingDemo(false);
      }

      const sm = s.smooth;
      const tg = s.target;
      sm.compression = lerp(sm.compression, tg.compression, 0.07);
      sm.quality = lerp(sm.quality, tg.quality, 0.07);
      sm.latency = lerp(sm.latency, tg.latency, 0.07);
      sm.bytesIn = lerp(sm.bytesIn, tg.bytesIn, 0.07);
      sm.budgetFill = lerp(sm.budgetFill, tg.budgetFill, 0.07);
      MODALITY_ORDER.forEach((k) => {
        sm.modalities[k] = lerp(sm.modalities[k] || 0, tg.modalities[k] || 0, 0.09);
      });

      const compressionNorm = clamp(sm.compression / 14, 0.04, 1);
      const spawnRate = clamp(Math.log10(sm.bytesIn + 10) / 3.8, 0.35, 2.6);
      const chop = clamp(sm.latency / 70, 0, 2.8);
      const storm = clamp(sm.latency / 240, 0, 1);
      const fog = clamp(sm.latency / 280, 0, 0.9);
      const spray = clamp(1 - sm.quality, 0, 1);
      const q = clamp(sm.quality, 0, 1);
      const focus = hoveredMod;

      const horizonY = h * 0.3;
      const funnelStart = w * 0.42;
      const mx = w * 0.8;
      const my = h * 0.52;
      const maelR = Math.min(w, h) * (0.18 + 0.26 * compressionNorm);

      if (!s.flow) buildFlow(w, h);
      const f = s.flow;

      /* ---- sky ------------------------------------------------------ */
      const glowHi = { r: 34, g: 211, b: 238 };
      const glowLo = { r: 245, g: 158, b: 11 };
      const glowT = clamp((1 - q) * 0.85 + storm * 0.2, 0, 1);
      const glow = mixRgb(glowHi, glowLo, glowT);
      const skyG = ctx.createLinearGradient(0, 0, 0, horizonY);
      skyG.addColorStop(0, '#01030a');
      skyG.addColorStop(0.6, '#060d1e');
      skyG.addColorStop(1, `rgb(${glow.r * 0.35 | 0},${glow.g * 0.5 | 0},${glow.b * 0.6 | 0})`);
      ctx.fillStyle = skyG;
      ctx.fillRect(0, 0, w, horizonY);

      STARS.forEach((st) => {
        const tw = 0.5 + 0.5 * Math.sin(s.t * st.sp + st.ph);
        const sy = st.y * horizonY;
        ctx.fillStyle = `rgba(226,232,240,${(0.08 + tw * 0.5) * (1 - storm * 0.55)})`;
        ctx.fillRect(st.x * w, sy, st.r, st.r);
      });

      const moonX = w * 0.76;
      const moonY = h * 0.1;
      const mr = Math.min(w, h) * 0.05;
      const moonB = 0.35 + q * 0.65;
      ctx.globalCompositeOperation = 'lighter';
      const moonHalo = ctx.createRadialGradient(moonX, moonY, 0, moonX, moonY, mr * 6);
      moonHalo.addColorStop(0, `rgba(190,230,255,${0.28 * moonB})`);
      moonHalo.addColorStop(1, 'rgba(190,230,255,0)');
      ctx.fillStyle = moonHalo;
      ctx.fillRect(moonX - mr * 6, moonY - mr * 6, mr * 12, mr * 12);
      ctx.beginPath();
      ctx.arc(moonX, moonY, mr, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232,244,255,${0.85 * moonB})`;
      ctx.fill();
      const moonPath = ctx.createLinearGradient(0, horizonY, 0, h * 0.62);
      moonPath.addColorStop(0, `rgba(190,225,255,${0.2 * moonB})`);
      moonPath.addColorStop(1, 'rgba(190,225,255,0)');
      ctx.fillStyle = moonPath;
      ctx.fillRect(moonX - mr, horizonY, mr * 2, h * 0.62 - horizonY);
      ctx.globalCompositeOperation = 'source-over';

      /* ---- sea depth ------------------------------------------------- */
      const seaG = ctx.createLinearGradient(0, horizonY, 0, h);
      seaG.addColorStop(0, '#0e2c42');
      seaG.addColorStop(0.4, '#0a1e35');
      seaG.addColorStop(0.75, '#061224');
      seaG.addColorStop(1, '#030a18');
      ctx.fillStyle = seaG;
      ctx.fillRect(0, horizonY, w, h - horizonY);

      /* depth fog from latency */
      const fogG = ctx.createLinearGradient(0, horizonY, 0, h);
      fogG.addColorStop(0, 'rgba(2,8,16,0)');
      fogG.addColorStop(1, `rgba(2,8,16,${0.55 * fog})`);
      ctx.fillStyle = fogG;
      ctx.fillRect(0, horizonY, w, h - horizonY);

      /* ---- spectral tide strata ------------------------------------- */
      MODALITY_ORDER.forEach((mod, i) => {
        const c = MODALITY_COLORS[mod];
        const weight = sm.modalities[mod] || 0;
        if (weight < 0.02 && !focus) return;
        const rise = ((s.t * 2.6 + i * 47) % 90);
        const bandY = horizonY + (h - horizonY) * (0.14 + i * 0.16) - rise;
        const bandH = (h - horizonY) * 0.13;
        const alpha =
          (0.04 + weight * 0.1) * (focus && focus !== mod ? 0.35 : 1);
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let x = 0; x <= w; x += 10) {
          const wob =
            fbm(x * 0.003 + s.t * 0.05, i * 2) * 9 *
            (1 + chop * 0.6);
          ctx.lineTo(x, bandY + wob);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`;
        ctx.fill();
      });

      /* ---- caustics on the seabed ----------------------------------- */
      ctx.globalCompositeOperation = 'lighter';
      CAUSTIC_NODES.forEach((node, ni) => {
        const nx = node.u * w + fbm(node.u * 3 + s.t * 0.32, node.v * 3) * 9;
        const ny = node.v * h + fbm(node.u * 3 + 7, node.v * 3 + s.t * 0.2) * 5;
        CAUSTIC_EDGES.forEach((e) => {
          if (e[0] !== ni) return;
          const o = CAUSTIC_NODES[e[1]];
          const ox = o.u * w + fbm(o.u * 3 + s.t * 0.32, o.v * 3) * 9;
          const oy = o.v * h + fbm(o.u * 3 + 7, o.v * 3 + s.t * 0.2) * 5;
          const bri = Math.max(0, fbm(node.u * 2.4 + s.t * 0.15, node.v * 2.4)) * q;
          if (bri < 0.06) return;
          ctx.strokeStyle = `rgba(56,189,248,${0.06 + bri * 0.14})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(nx, ny);
          ctx.lineTo(ox, oy);
          ctx.stroke();
        });
      });
      ctx.globalCompositeOperation = 'source-over';

      /* ---- surface wave crests + glints ----------------------------- */
      ctx.beginPath();
      for (let x = 0; x <= w; x += 6) {
        const y =
          horizonY + 4 +
          fbm(x * 0.008 + s.t * 0.6, 3.3) * 6 +
          Math.sin(x * 0.02 + s.t * 1.7) * 2;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(148,197,255,${0.22 + q * 0.1})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.globalCompositeOperation = 'lighter';
      GLINTS.forEach((g) => {
        const gx = g.x * w;
        const gy =
          horizonY + 6 + fbm(g.x * 4 + s.t * 0.5, s.t * 0.18) * 11;
        const br = q * Math.max(0, Math.sin(s.t * g.sp + g.ph));
        if (br < 0.4) return;
        ctx.strokeStyle = `rgba(200,235,255,${(br - 0.4) * 0.7})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(gx - 7, gy);
        ctx.lineTo(gx + 7, gy + fbm(g.x * 5, s.t) * 2);
        ctx.stroke();
      });
      ctx.globalCompositeOperation = 'source-over';

      /* ---- flow currents (faint) ------------------------------------ */
      ctx.strokeStyle = 'rgba(148,197,255,0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let j = 0; j < f.rows; j += 1) {
        for (let i = 0; i < f.cols; i += 1) {
          const ang = f.arr[j * f.cols + i];
          const x = ((i + 0.5) / f.cols) * w;
          const y = horizonY + ((j + 0.5) / f.rows) * (h - horizonY);
          ctx.moveTo(x, y);
          ctx.lineTo(x + Math.cos(ang) * 8, y + Math.sin(ang) * 8);
        }
      }
      ctx.stroke();

      /* ---- compression corridor into the maelstrom ------------------- */
      const corridorG = ctx.createLinearGradient(funnelStart, 0, mx, 0);
      corridorG.addColorStop(0, `rgba(34,211,238,${0.02 + compressionNorm * 0.03})`);
      corridorG.addColorStop(1, `rgba(34,211,238,${0.08 + compressionNorm * 0.12})`);
      ctx.fillStyle = corridorG;
      ctx.beginPath();
      ctx.moveTo(funnelStart, my - h * 0.2);
      ctx.lineTo(mx - maelR * 0.4, my - maelR * 0.8);
      ctx.lineTo(mx - maelR * 0.4, my + maelR * 0.8);
      ctx.lineTo(funnelStart, my + h * 0.2);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = `rgba(34,211,238,${0.14 + compressionNorm * 0.3})`;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([14, 18]);
      ctx.lineDashOffset = -s.t * 46;
      ctx.beginPath();
      ctx.moveTo(funnelStart, my - h * 0.2);
      ctx.lineTo(mx - maelR * 0.4, my - maelR * 0.8);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(funnelStart, my + h * 0.2);
      ctx.lineTo(mx - maelR * 0.4, my + maelR * 0.8);
      ctx.stroke();
      ctx.setLineDash([]);

      /* ---- the maelstrom --------------------------------------------- */
      const spin = 0.35 + compressionNorm * 0.95;
      const arms = 6;
      ctx.globalCompositeOperation = 'lighter';
      for (let a = 0; a < arms; a += 1) {
        const a0 = (a / arms) * Math.PI * 2 + s.t * spin;
        ctx.beginPath();
        for (let r = 3; r <= maelR; r += 6) {
          const ang = a0 + r * 0.034 * (2.1 - compressionNorm) + fbm(r * 0.012, s.t * 0.4) * 0.35;
          const x = mx + Math.cos(ang) * r;
          const y = my + Math.sin(ang) * r * 0.9 + fbm(x * 0.004, s.t * 0.3) * 6;
          if (r === 3) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        const fade = 1 - a / arms;
        ctx.strokeStyle = `rgba(34,211,238,${(0.04 + 0.15 * compressionNorm) * fade})`;
        ctx.lineWidth = 1 + compressionNorm;
        ctx.stroke();
      }
      ctx.globalCompositeOperation = 'source-over';

      const coreG = ctx.createRadialGradient(mx, my, 0, mx, my, maelR * 0.55);
      coreG.addColorStop(0, 'rgba(2,6,16,0.96)');
      coreG.addColorStop(0.55, 'rgba(6,18,34,0.62)');
      coreG.addColorStop(1, 'rgba(6,18,34,0)');
      ctx.fillStyle = coreG;
      ctx.beginPath();
      ctx.arc(mx, my, maelR * 0.55, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = `rgba(56,189,248,${0.1 + compressionNorm * 0.22 + Math.sin(s.t * 5) * 0.05})`;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(mx, my, maelR * (0.55 + Math.sin(s.t * 3.2) * 0.04), 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';

      /* ---- uplink beam leaving the sea ------------------------------- */
      ctx.globalCompositeOperation = 'lighter';
      const bw = 3 + compressionNorm * 7 + Math.sin(s.t * 5) * 1.2;
      const bg2 = ctx.createLinearGradient(mx, 0, w, 0);
      bg2.addColorStop(0, 'rgba(16,185,129,0.55)');
      bg2.addColorStop(0.55, 'rgba(56,189,248,0.28)');
      bg2.addColorStop(1, 'rgba(226,232,240,0)');
      ctx.strokeStyle = bg2;
      ctx.lineWidth = bw;
      ctx.beginPath();
      ctx.moveTo(mx + maelR * 0.5, my);
      ctx.lineTo(w + 20, my + Math.sin(s.t * 0.8) * 7);
      ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';

      /* ---- node buoy --------------------------------------------------- */
      const bx = w * 0.055;
      const by = h * 0.42;
      const blink = 0.5 + 0.5 * Math.sin(s.t * 3.4);
      const bCol = q > 0.7 ? { r: 16, g: 185, b: 129 } : q > 0.4 ? { r: 245, g: 158, b: 11 } : { r: 244, g: 63, b: 94 };
      ctx.strokeStyle = 'rgba(148,197,255,0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(bx, by + 6);
      ctx.lineTo(bx, by - 26);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(bx, by, 7, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(148,197,255,0.5)';
      ctx.stroke();
      ctx.fillStyle = 'rgba(148,197,255,0.14)';
      ctx.fill();
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = `rgba(${bCol.r},${bCol.g},${bCol.b},${0.35 + blink * 0.6})`;
      ctx.beginPath();
      ctx.arc(bx, by - 26, 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over';

      s.pingT = (s.pingT || 0) + 1;
      if (s.pingT % 72 === 0) {
        s.pings.push({ x: bx, y: by, r: 8, life: 1 });
      }
      s.pings = s.pings.filter((p) => {
        p.r += 3.4;
        p.life -= 0.014;
        if (p.life <= 0) return false;
        ctx.globalCompositeOperation = 'lighter';
        ctx.strokeStyle = `rgba(${bCol.r},${bCol.g},${bCol.b},${p.life * 0.28})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalCompositeOperation = 'source-over';
        return true;
      });

      /* ---- bioluminescent plankton ------------------------------------ */
      const planktonTarget = Math.floor(
        (compact ? 55 : 100) + spawnRate * (compact ? 55 : 110)
      );
      while (s.plankton.length < planktonTarget) {
        s.plankton.push({
          x: Math.random() * w,
          y: horizonY + Math.random() * (h - horizonY),
          ph: Math.random() * Math.PI * 2,
          sp: 0.3 + Math.random() * 1.1,
          sz: 0.6 + Math.random() * 1.6,
          mod: MODALITY_ORDER[Math.floor(Math.random() * MODALITY_ORDER.length)],
        });
      }
      if (s.plankton.length > planktonTarget) {
        s.plankton.length = planktonTarget;
      }
      ctx.globalCompositeOperation = 'lighter';
      for (const p of s.plankton) {
        const ang = flowAt(p.x, p.y, w, h);
        p.x += Math.cos(ang) * 0.42;
        p.y += Math.sin(ang) * 0.22 + Math.sin(s.t * 0.6 + p.ph) * 0.08;
        if (p.x < -4) p.x = w + 4;
        if (p.x > w + 4) p.x = -4;
        if (p.y < horizonY - 4) p.y = h - 4;
        if (p.y > h + 4) p.y = horizonY + 4;
        const pulse = 0.5 + 0.5 * Math.sin(s.t * p.sp + p.ph);
        const alpha = (0.05 + pulse * 0.4) * (0.4 + q * 0.6) * (1 - fog * 0.6);
        const c = MODALITY_COLORS[p.mod];
        const col = mixRgb({ r: 140, g: 214, b: 255 }, c, 0.32);
        ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.sz * (0.7 + pulse * 0.6), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = 'source-over';

      /* ---- schooling signal fish --------------------------------------- */
      const fishTarget = Math.floor(
        (compact ? 24 : 46) + compressionNorm * (compact ? 14 : 34)
      );
      while (s.fish.length < fishTarget) {
        s.fish.push(spawnFish(w, h, sm.modalities, spawnRate));
      }
      if (s.fish.length > fishTarget) {
        s.fish.length = fishTarget;
      }

      const ptr = pointerRef.current;
      const lure = performance.now() - ptr.glowUntil < 140;

      for (const fish of s.fish) {
        const inCorridor = fish.x > funnelStart;
        const dxm = mx - fish.x;
        const dym = my - fish.y;
        const dm = Math.hypot(dxm, dym) + 0.001;

        const c = MODALITY_COLORS[fish.mod] || MODALITY_COLORS.camera;
        const focused = !focus || focus === fish.mod;
        const dim = focus && !focused ? 0.28 : 1;

        if (focused) {
          fish.surge = lerp(fish.surge, inCorridor ? 1.5 : 0.6, 0.05);
        } else {
          fish.surge = lerp(fish.surge, 0.35, 0.05);
        }

        const fa = flowAt(fish.x, fish.y, w, h);
        fish.vx += Math.cos(fa) * 0.006;
        fish.vy += Math.sin(fa) * 0.004;

        if (inCorridor) {
          const pull = compressionNorm * 0.16 * (maelR * 1.4 / Math.max(dm, 1));
          fish.vx += (dxm / dm) * pull;
          fish.vy += (dym / dm) * pull;
          fish.vx += (-dym / dm) * 0.035 * compressionNorm;
          fish.vy += (dxm / dm) * 0.035 * compressionNorm;
        }

        if (lure) {
          const dxp = ptr.x * w - fish.x;
          const dyp = ptr.y * h - fish.y;
          const dp = Math.hypot(dxp, dyp);
          if (dp < 150 && dp > 1) {
            fish.vx += (dxp / dp) * 0.05;
            fish.vy += (dyp / dp) * 0.05;
          }
        }

        if (fish.surge > 0.6 && focused) {
          fish.vx = clamp(fish.vx, -2.5, 2.5);
        }
        if (fish.x > w + 40) {
          Object.assign(fish, spawnFish(w, h, sm.modalities, spawnRate));
        }
        if (fish.y < horizonY - 30 || fish.y > h + 30) {
          fish.y = clamp(fish.y, horizonY + 4, h - 4);
          fish.vy *= -0.5;
        }

        fish.x += fish.vx;
        fish.y += fish.vy + Math.sin(s.t * 2.2 + fish.x * 0.02) * 0.14 * chop;
        fish.tail += 0.12 + fish.surge * 0.2;
        fish.flick += 0.1;

        if (inCorridor && dm < maelR * 0.22) {
          s.sparks.push({
            x: fish.x,
            y: fish.y,
            vx: (Math.random() - 0.5) * 1.2,
            vy: (Math.random() - 0.5) * 1.2,
            life: 1,
            mod: fish.mod,
          });
          Object.assign(fish, spawnFish(w, h, sm.modalities, spawnRate));
          continue;
        }

        drawFish(fish, c, dim);
        if (fish.surge > 1 && focused) {
          ctx.globalCompositeOperation = 'lighter';
          ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${0.14 + compressionNorm * 0.1})`;
          ctx.beginPath();
          ctx.arc(fish.x, fish.y, fish.size * 2.6, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalCompositeOperation = 'source-over';
        }
      }

      /* ---- compression sparks ------------------------------------------ */
      s.sparks = s.sparks.filter((sp) => {
        sp.life -= 0.035;
        sp.x += sp.vx;
        sp.y += sp.vy;
        sp.vx *= 0.94;
        sp.vy *= 0.94;
        if (sp.life <= 0) return false;
        const c = MODALITY_COLORS[sp.mod] || MODALITY_COLORS.camera;
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${sp.life * 0.8})`;
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 1.5 + sp.life * 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalCompositeOperation = 'source-over';
        return true;
      });

      /* ---- wave spray when quality is low ------------------------------ */
      if (spray > 0.12) {
        ctx.fillStyle = `rgba(226,232,240,${0.05 + spray * 0.2})`;
        for (let i = 0; i < 26 * spray; i += 1) {
          const sx = Math.random() * funnelStart;
          const sy = horizonY + 6 + fbm(sx * 0.008 + s.t, 1.3) * 7;
          ctx.fillRect(sx, sy - Math.random() * 6, 2, 2);
        }
      }

      /* ---- storm: rain + lightning -------------------------------------- */
      if (chop > 0.25) {
        const rainRate = (chop - 0.25) * 3;
        for (let i = 0; i < rainRate && s.rain.length < 240; i += 1) {
          s.rain.push({
            x: Math.random() * w,
            y: -12,
            len: 8 + Math.random() * 14,
            spd: 10 + chop * 15,
          });
        }
      }
      if (s.rain.length) {
        ctx.strokeStyle = `rgba(148,163,184,${0.1 + chop * 0.2})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        s.rain = s.rain.filter((r) => {
          r.y += r.spd;
          if (r.y > h + 20) return false;
          ctx.moveTo(r.x, r.y);
          ctx.lineTo(r.x + 1.5, r.y + r.len);
          return true;
        });
        ctx.stroke();
      }
      if (storm > 0.5 && Math.random() < storm * 0.008) {
        s.flash = 1;
        s.lx = Math.random() * w;
        s.ly = Math.random() * horizonY * 0.55;
      }
      if (s.flash > 0.03) {
        ctx.fillStyle = `rgba(226,240,255,${s.flash * 0.16})`;
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = `rgba(226,240,255,${s.flash * 0.7})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(s.lx, s.ly);
        let lx = s.lx;
        let ly = s.ly;
        while (ly < horizonY) {
          lx += (Math.random() - 0.5) * 26;
          ly += 14 + Math.random() * 12;
          ctx.lineTo(lx, ly);
        }
        ctx.stroke();
        s.flash *= 0.88;
      }

      /* ---- pointer lure glow + ripples ----------------------------------- */
      if (lure) {
        ctx.globalCompositeOperation = 'lighter';
        const lureG = ctx.createRadialGradient(ptr.x * w, ptr.y * h, 0, ptr.x * w, ptr.y * h, 80);
        lureG.addColorStop(0, 'rgba(34,211,238,0.16)');
        lureG.addColorStop(1, 'rgba(34,211,238,0)');
        ctx.fillStyle = lureG;
        ctx.fillRect(ptr.x * w - 80, ptr.y * h - 80, 160, 160);
        ctx.globalCompositeOperation = 'source-over';
      }
      s.ripples = s.ripples || [];
      s.ripples = s.ripples.filter((rp) => {
        rp.r += 2.2;
        rp.life -= 0.03;
        if (rp.life <= 0) return false;
        ctx.globalCompositeOperation = 'lighter';
        ctx.strokeStyle = `rgba(34,211,238,${rp.life * 0.4})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalCompositeOperation = 'source-over';
        return true;
      });

      /* ---- vignette ------------------------------------------------------- */
      const vig = ctx.createRadialGradient(w * 0.5, h * 0.45, Math.min(w, h) * 0.35, w * 0.5, h * 0.5, Math.max(w, h) * 0.75);
      vig.addColorStop(0, 'rgba(0,0,10,0)');
      vig.addColorStop(1, 'rgba(0,0,10,0.45)');
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, w, h);

      /* ---- HUD readout ------------------------------------------------------ */
      ctx.fillStyle = 'rgba(248,250,252,0.82)';
      ctx.font = compact ? '600 11px DM Sans, sans-serif' : '600 13px DM Sans, sans-serif';
      const label = selectedNode || tg.nodeId || 'fleet';
      ctx.fillText(
        `${label}  ·  ${sm.compression.toFixed(1)}×  ·  q ${(q * 100).toFixed(0)}%  ·  ${sm.latency.toFixed(0)} ms`,
        16,
        h - 14
      );
      ctx.fillStyle = 'rgba(148,163,184,0.5)';
      ctx.font = compact ? '600 9px DM Sans, sans-serif' : '600 10px DM Sans, sans-serif';
      ctx.fillText(
        `maelstrom ${Math.round(compressionNorm * 100)}% · weather ${storm > 0.5 ? 'storm' : 'clear'}`,
        16,
        h - (compact ? 26 : 30)
      );

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [
    selectedNode,
    spawnFish,
    compact,
    demoFallback,
    applyMetric,
    linkHealth,
    fleetMetrics,
    hoveredMod,
  ]);

  const onMove = (e) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    pointerRef.current.x = (e.clientX - rect.left) / rect.width;
    pointerRef.current.y = (e.clientY - rect.top) / rect.height;
    pointerRef.current.glowUntil = performance.now() + 140;
  };

  const onClick = (e) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const ny = (e.clientY - rect.top) / rect.height;
    const idx = clamp(Math.floor((ny - 0.12) / 0.14), 0, 3);
    const mod = MODALITY_ORDER[idx];
    setHoveredMod((prev) => (prev === mod ? null : mod));
    onHoverModality?.(mod);
    const state = stateRef.current;
    state.ripples = state.ripples || [];
    state.ripples.push({
      x: ((e.clientX - rect.left) / rect.width) * rect.width,
      y: ny * rect.height,
      r: 4,
      life: 1,
    });
    if (selectedNode) onSelectNode?.(selectedNode);
  };

  return (
    <div
      className={`signal-ocean${compact ? ' signal-ocean--compact' : ''}`}
      ref={wrapRef}
      onMouseMove={onMove}
      onClick={onClick}
      role="img"
      aria-label="Live compression signal ocean"
    >
      <canvas ref={canvasRef} className="signal-ocean__canvas" />
      <div className="signal-ocean__chrome">
        <span className="signal-ocean__brand">{brand}</span>
        <span className="signal-ocean__tag">
          {usingDemo ? 'Simulated field · waiting for edge metrics' : 'Live compression field'}
        </span>
      </div>
      <div className="signal-ocean__legend">
        {MODALITY_ORDER.map((m) => (
          <button
            key={m}
            type="button"
            className={`signal-ocean__chip signal-ocean__chip--${m}${
              hoveredMod === m ? ' is-active' : ''
            }`}
            onClick={(e) => {
              e.stopPropagation();
              setHoveredMod((prev) => (prev === m ? null : m));
              onHoverModality?.(m);
            }}
          >
            {m}
          </button>
        ))}
      </div>
    </div>
  );
}

export default SignalOcean;
export { inventDemoMetric };
