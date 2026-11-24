import React, { useEffect, useState } from 'react';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import PageHeader from './ui/PageHeader';
import './ProofView.css';

function ProofView() {
  const [proof, setProof] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    fetch('/beat-jpeg-proof.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setProof(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'failed to load proof');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const margin =
    proof?.best_matched_margin_db ??
    proof?.best_margin_db ??
    proof?.best_below_floor_margin_db;
  const matched = (proof?.wins || []).filter((w) => w.kind === 'matched');
  const belowFloor = (proof?.wins || []).filter((w) => w.kind === 'below_floor');
  const kicker = matched.length
    ? 'Matched-rate PSNR margin vs intra JPEG'
    : 'Below-floor PSNR margin vs intra JPEG (fewer bits than JPEG’s lowest rate)';

  return (
    <div className="proof-view">
      <PageHeader
        title="Beat JPEG"
        subtitle="YCbCr 4:2:0 temporal residual + mild freq DCT + countable rANS, measured on the correlated clip"
        icon={FactCheckIcon}
        badge={
          proof ? (
            <span className={`badge ${proof.success ? 'badge--running' : 'badge--registered'}`}>
              {proof.success ? 'claim holds' : 'claim failed'}
            </span>
          ) : null
        }
      />

      {error && <p className="proof-view__error">{error}</p>}
      {!proof && !error && <p className="proof-view__wait">Loading measured proof…</p>}

      {proof && (
        <>
          <section className="proof-hero card">
            <div>
              <p className="proof-hero__kicker">{kicker}</p>
              <p className="proof-hero__value">
                {margin == null ? '—' : `+${Number(margin).toFixed(1)} dB`}
              </p>
            </div>
            <dl className="proof-hero__meta">
              <div>
                <dt>Clip</dt>
                <dd>{proof.clip}</dd>
              </div>
              <div>
                <dt>Frames</dt>
                <dd>{proof.shape ? proof.shape[0] : '—'}</dd>
              </div>
              <div>
                <dt>Rate</dt>
                <dd>payload bytes, header included</dd>
              </div>
            </dl>
          </section>

          <section className="card proof-table-wrap">
            <h2>Operating points</h2>
            <table className="proof-table">
              <thead>
                <tr>
                  <th>qstep</th>
                  <th>bpp</th>
                  <th>Lydlr PSNR</th>
                  <th>JPEG PSNR</th>
                  <th>margin</th>
                  <th>kind</th>
                </tr>
              </thead>
              <tbody>
                {(proof.wins || []).map((row) => (
                  <tr key={`${row.kind}-${row.qstep}`}>
                    <td>{Number(row.qstep).toFixed(0)}</td>
                    <td>{Number(row.bpp).toFixed(3)}</td>
                    <td>{Number(row.psnr).toFixed(2)}</td>
                    <td>{Number(row.jpeg_psnr).toFixed(2)}</td>
                    <td>{Number(row.margin_db) >= 0 ? '+' : ''}{Number(row.margin_db).toFixed(2)} dB</td>
                    <td>{row.kind === 'matched' ? 'matched bpp' : 'below JPEG floor'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {matched.length === 0 && belowFloor.length > 0 && (
              <p className="proof-view__wait">
                All wins are below JPEG’s rate floor — stronger than matched bpp on this clip.
              </p>
            )}
            {matched.length === 0 && belowFloor.length === 0 && (
              <p className="proof-view__wait">No win in this proof file.</p>
            )}
          </section>
        </>
      )}
    </div>
  );
}

export default ProofView;
