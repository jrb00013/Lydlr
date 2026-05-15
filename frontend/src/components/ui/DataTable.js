import React, { useMemo, useState } from 'react';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import UnfoldMoreIcon from '@mui/icons-material/UnfoldMore';
import './DataTable.css';

/**
 * @param {Object} props
 * @param {Array<{key:string,label:string,render?:(row)=>React.ReactNode,sortable?:boolean,width?:string}>} props.columns
 * @param {Array<Object>} props.rows
 * @param {string} [props.keyField]
 * @param {boolean} [props.loading]
 * @param {string} [props.emptyMessage]
 * @param {number} [props.pageSize]
 */
function DataTable({
  columns,
  rows,
  keyField = 'id',
  loading = false,
  emptyMessage = 'No data',
  pageSize = 15,
}) {
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);

  const sorted = useMemo(() => {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    return [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av;
      }
      return sortDir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
  }, [rows, sortKey, sortDir, columns]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageRows = sorted.slice(page * pageSize, (page + 1) * pageSize);

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const SortIcon = ({ colKey }) => {
    if (sortKey !== colKey) return <UnfoldMoreIcon className="data-table__sort-icon data-table__sort-icon--idle" />;
    return sortDir === 'asc' ? (
      <ArrowUpwardIcon className="data-table__sort-icon" />
    ) : (
      <ArrowDownwardIcon className="data-table__sort-icon" />
    );
  };

  if (loading) {
    return <div className="data-table data-table--loading">Loading…</div>;
  }

  return (
    <div className="data-table">
      <div className="data-table__scroll">
        <table className="data-table__table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key} style={col.width ? { width: col.width } : undefined}>
                  {col.sortable !== false ? (
                    <button
                      type="button"
                      className="data-table__th-btn"
                      onClick={() => toggleSort(col.key)}
                    >
                      {col.label}
                      <SortIcon colKey={col.key} />
                    </button>
                  ) : (
                    col.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="data-table__empty">
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              pageRows.map((row, idx) => (
                <tr key={row[keyField] ?? `${page}-${idx}`}>
                  {columns.map((col) => (
                    <td key={col.key}>
                      {col.render ? col.render(row) : formatCell(row[col.key])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {sorted.length > pageSize && (
        <div className="data-table__footer">
          <span>
            {page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)} of {sorted.length}
          </span>
          <div className="data-table__pager">
            <button type="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
              Prev
            </button>
            <span>
              {page + 1} / {pageCount}
            </span>
            <button
              type="button"
              disabled={page >= pageCount - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function formatCell(value) {
  if (value == null || value === '') return '—';
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(3);
  if (typeof value === 'object' && value instanceof Date) return value.toLocaleString();
  const s = String(value);
  if (s.includes('T') && s.length > 18) {
    try {
      return new Date(s).toLocaleString();
    } catch {
      return s;
    }
  }
  return s;
}

export default DataTable;
