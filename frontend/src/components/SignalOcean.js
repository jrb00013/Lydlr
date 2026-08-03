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

/**
 * Full-bleed canvas ocean of multimodal signals compressed into an uplink.
 */
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
  const pointerRef = useRef({ x: 0.5, y: 0.5, active: false });
  const lastMetricAt = useRef(0);

  const stateRef = useRef({
    t: 0,
    particles: [],
    ripples: [],
    vortices: [],
    foam: [],
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

  const spawnParticle = useCallback((w, h, modalities, spawnRate, wakeY) => {
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
    const baseY = wakeY != null ? wakeY : h * (0.18 + layer * 0.15);
    return {
      x: -12 - Math.random() * 60,
      y: baseY + (Math.random() - 0.5) * h * 0.07,
      vx: 1.1 + Math.random() * 2.2 * spawnRate,
      vy: (Math.random() - 0.5) * 0.55,
      life: 1,
      mod,
      size: 1.2 + Math.random() * 3.2,
      trail: Math.random() > 0.7,
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
    };
    resize();
    window.addEventListener('resize', resize);

    // seed vortices (compression eddies)
    if (!s.vortices.length) {
      for (let i = 0; i < 3; i += 1) {
        s.vortices.push({
          x: 0.45 + i * 0.12,
          y: 0.35 + (i % 2) * 0.2,
          r: 0.04 + Math.random() * 0.03,
          spin: (Math.random() > 0.5 ? 1 : -1) * (0.8 + Math.random()),
        });
      }
    }

    const tick = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      s.t += 0.016;

      // demo fallback if no live metric recently
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
      const funnelStart = w * 0.4;
      const funnelEnd = w * 0.8;
      const horizonX = w * (0.86 - Math.min(sm.budgetFill, 1) * 0.05);
      const spawnRate = clamp(Math.log10(sm.bytesIn + 10) / 3.8, 0.35, 2.6);
      const chop = clamp(sm.latency / 70, 0, 2.8);
      const spray = clamp(1 - sm.quality, 0, 1);
      const ptr = pointerRef.current;

      // night ocean gradient
      const g = ctx.createLinearGradient(0, 0, w * 0.2, h);
      g.addColorStop(0, '#050914');
      g.addColorStop(0.45, '#0a1628');
      g.addColorStop(0.78, '#071820');
      g.addColorStop(1, '#040c14');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // distant fleet wakes (secondary nodes)
      const wakes = (fleetMetrics || []).slice(0, 4);
      wakes.forEach((fm, wi) => {
        if (!fm || fm.node_id === selectedNode) return;
        const wy = h * (0.25 + wi * 0.12);
        ctx.beginPath();
        for (let x = 0; x < funnelStart * 0.7; x += 8) {
          const y = wy + Math.sin(x * 0.02 + s.t + wi) * 6;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(148,163,184,0.18)';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // layered modality seas
      MODALITY_ORDER.forEach((mod, i) => {
        const c = MODALITY_COLORS[mod];
        const weight = sm.modalities[mod] || 0;
        const highlight = hoveredMod === mod ? 1.35 : 1;
        const amp = (0.014 + weight * 0.055) * h * (1 + spray * 0.9) * highlight;
        const baseY = h * (0.2 + i * 0.135);
        const alpha = (0.07 + weight * 0.28) * (hoveredMod && hoveredMod !== mod ? 0.35 : 1);

        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let x = 0; x <= funnelStart + 8; x += 5) {
          const swell =
            Math.sin(x * 0.011 + s.t * (1.15 + i * 0.18) + i * 1.1) * amp +
            Math.sin(x * 0.037 + s.t * 2.4 + chop * i) * amp * 0.4 * chop +
            Math.sin(x * 0.08 + s.t * 0.7) * amp * 0.15;
          const y = baseY + swell;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(funnelStart, h);
        ctx.closePath();
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`;
        ctx.fill();

        ctx.beginPath();
        for (let x = 0; x <= funnelStart; x += 3) {
          const swell =
            Math.sin(x * 0.011 + s.t * (1.15 + i * 0.18) + i) * amp +
            Math.sin(x * 0.037 + s.t * 2.4) * amp * 0.35 * chop;
          const y = baseY + swell;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${0.3 + weight * 0.5})`;
        ctx.lineWidth = hoveredMod === mod ? 2.4 : 1.4;
        ctx.stroke();
      });

      // compression funnel with breathing walls
      const breathe = 1 + Math.sin(s.t * 2.2) * 0.04 * compressionNorm;
      const narrow = lerp(0.3, 0.05, compressionNorm) * breathe;
      const topOpen = h * 0.14;
      const botOpen = h * 0.86;
      const topClose = h * (0.5 - narrow);
      const botClose = h * (0.5 + narrow);

      const funnelGrad = ctx.createLinearGradient(funnelStart, 0, funnelEnd, 0);
      funnelGrad.addColorStop(0, 'rgba(34,211,238,0.03)');
      funnelGrad.addColorStop(0.6, 'rgba(16,185,129,0.1)');
      funnelGrad.addColorStop(1, 'rgba(245,158,11,0.12)');
      ctx.beginPath();
      ctx.moveTo(funnelStart, topOpen);
      ctx.quadraticCurveTo(
        (funnelStart + funnelEnd) / 2,
        topClose - 10 * compressionNorm,
        funnelEnd,
        topClose
      );
      ctx.lineTo(funnelEnd, botClose);
      ctx.quadraticCurveTo(
        (funnelStart + funnelEnd) / 2,
        botClose + 10 * compressionNorm,
        funnelStart,
        botOpen
      );
      ctx.closePath();
      ctx.fillStyle = funnelGrad;
      ctx.fill();
      ctx.strokeStyle = `rgba(34,211,238,${0.22 + compressionNorm * 0.45})`;
      ctx.lineWidth = 1.6;
      ctx.stroke();

      // vortex eddies inside funnel
      s.vortices.forEach((v, vi) => {
        const vx = funnelStart + (funnelEnd - funnelStart) * (0.2 + vi * 0.25);
        const vy = h * v.y + Math.sin(s.t * v.spin + vi) * 12;
        const vr = Math.min(w, h) * v.r * (0.8 + compressionNorm);
        ctx.beginPath();
        ctx.arc(vx, vy, vr, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(56,189,248,${0.12 + compressionNorm * 0.2})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // uplink budget wall
      const wallPulse =
        0.3 + Math.sin(s.t * 4) * 0.15 * (sm.budgetFill > 0.85 ? 1.4 : 0.25);
      const wallW = 3 + sm.budgetFill * 8;
      const wallGrad = ctx.createLinearGradient(0, topClose, 0, botClose);
      wallGrad.addColorStop(0, `rgba(245,158,11,${wallPulse * 0.2})`);
      wallGrad.addColorStop(0.5, `rgba(245,158,11,${wallPulse})`);
      wallGrad.addColorStop(1, `rgba(244,63,94,${wallPulse * sm.budgetFill})`);
      ctx.fillStyle = wallGrad;
      ctx.fillRect(horizonX, topClose - 10, wallW, botClose - topClose + 20);
      ctx.fillStyle = 'rgba(226,232,240,0.55)';
      ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillText('UPLINK', horizonX - 6, topClose - 16);
      ctx.fillText(`${Math.round(sm.budgetFill * 100)}%`, horizonX - 4, botClose + 18);

      // pointer ripples
      if (ptr.active) {
        s.ripples.push({
          x: ptr.x * w,
          y: ptr.y * h,
          r: 4,
          life: 1,
        });
        ptr.active = false;
      }
      s.ripples = s.ripples.filter((rp) => {
        rp.r += 2.2;
        rp.life -= 0.03;
        if (rp.life <= 0) return false;
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(34,211,238,${rp.life * 0.4})`;
        ctx.stroke();
        return true;
      });

      // particles / signal droplets
      const maxParticles = Math.floor((compact ? 60 : 110) + spawnRate * (compact ? 80 : 140));
      while (s.particles.length < maxParticles) {
        s.particles.push(spawnParticle(w, h, sm.modalities, spawnRate));
      }

      const survivors = [];
      for (const p of s.particles) {
        if (hoveredMod && p.mod !== hoveredMod && Math.random() < 0.02) {
          // gently cull non-hovered layers for focus
        }
        const inFunnel = p.x > funnelStart;
        let cullChance = 0;
        if (inFunnel) {
          const progress = clamp((p.x - funnelStart) / (funnelEnd - funnelStart), 0, 1);
          cullChance = progress * compressionNorm * 0.9;
          // vortex swirl
          const mid = h * 0.5;
          const swirl = Math.sin(s.t * 3 + p.x * 0.02) * 0.8 * compressionNorm;
          p.vy += swirl * 0.05;
          p.y = lerp(p.y, mid + (p.y - mid) * (1 - progress * 0.9), 0.09);
          p.vx = lerp(p.vx, 2.8 + compressionNorm * 3.5, 0.06);
        }
        if (Math.random() < cullChance * 0.09) {
          // compression flash
          if (Math.random() < 0.3) {
            s.foam.push({ x: p.x, y: p.y, life: 1, mod: p.mod });
          }
          continue;
        }

        p.x += p.vx;
        p.y += p.vy + Math.sin(s.t * 2.2 + p.x * 0.025) * 0.18 * chop;
        p.life -= 0.0012;

        // pointer attract/repel
        const dx = p.x - ptr.x * w;
        const dy = p.y - ptr.y * h;
        const dist = Math.sqrt(dx * dx + dy * dy) + 1;
        if (dist < 90) {
          p.vx += (dx / dist) * 0.08;
          p.vy += (dy / dist) * 0.08;
        }

        if (p.x > horizonX + 24 || p.life <= 0 || p.y < -10 || p.y > h + 10) continue;

        const c = MODALITY_COLORS[p.mod] || MODALITY_COLORS.camera;
        const dim = hoveredMod && hoveredMod !== p.mod ? 0.25 : 1;
        const alpha = (0.4 + p.life * 0.45) * dim;
        const funnelT = clamp((p.x - funnelStart) / Math.max(funnelEnd - funnelStart, 1), 0, 1);
        const size = p.size * (inFunnel ? 1 - funnelT * 0.55 : 1);

        if (p.trail) {
          ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${alpha * 0.35})`;
          ctx.beginPath();
          ctx.moveTo(p.x - p.vx * 3, p.y - p.vy * 3);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`;
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fill();
        survivors.push(p);
      }
      s.particles = survivors;

      // foam / compression sparks
      s.foam = s.foam.filter((f) => {
        f.life -= 0.04;
        f.y -= 0.6;
        if (f.life <= 0) return false;
        const c = MODALITY_COLORS[f.mod] || MODALITY_COLORS.camera;
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${f.life})`;
        ctx.fillRect(f.x, f.y, 2, 2);
        return true;
      });

      // quality spray
      if (spray > 0.12) {
        ctx.fillStyle = `rgba(226,232,240,${0.06 + spray * 0.22})`;
        for (let i = 0; i < 28 * spray; i += 1) {
          ctx.fillRect(Math.random() * funnelStart, h * 0.18 + Math.random() * h * 0.55, 2, 2);
        }
      }

      // surviving beam past uplink
      ctx.strokeStyle = `rgba(16,185,129,${0.25 + (1 - compressionNorm) * 0.35})`;
      ctx.lineWidth = 2 + (1 - compressionNorm) * 3;
      ctx.beginPath();
      ctx.moveTo(horizonX + wallW, h * 0.5);
      ctx.lineTo(w + 10, h * 0.5 + Math.sin(s.t) * 4);
      ctx.stroke();

      // HUD readout
      ctx.fillStyle = 'rgba(248,250,252,0.82)';
      ctx.font = compact
        ? '600 11px DM Sans, sans-serif'
        : '600 13px DM Sans, sans-serif';
      const label = selectedNode || tg.nodeId || 'fleet';
      ctx.fillText(
        `${label}  ·  ${sm.compression.toFixed(1)}×  ·  q ${(sm.quality * 100).toFixed(0)}%  ·  ${sm.latency.toFixed(0)} ms`,
        16,
        h - 14
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
    spawnParticle,
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
  };

  const onClick = (e) => {
    pointerRef.current.active = true;
    onMove(e);
    // map y to modality for selection feedback
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const ny = (e.clientY - rect.top) / rect.height;
    const idx = clamp(Math.floor((ny - 0.12) / 0.14), 0, 3);
    const mod = MODALITY_ORDER[idx];
    setHoveredMod((prev) => (prev === mod ? null : mod));
    onHoverModality?.(mod);
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
