import { useMemo } from 'react';

/**
 * Pick the newest metric from a node→metric map, preferring selectedNode.
 */
export function useLatestMetric(metricsByNode, selectedNode) {
  return useMemo(() => {
    if (!metricsByNode || typeof metricsByNode !== 'object') return null;
    if (selectedNode && metricsByNode[selectedNode]) {
      return metricsByNode[selectedNode];
    }
    const values = Object.values(metricsByNode);
    if (!values.length) return null;
    return values.reduce((best, cur) => {
      const bt = Date.parse(best?.timestamp || best?.ts || 0) || 0;
      const ct = Date.parse(cur?.timestamp || cur?.ts || 0) || 0;
      return ct >= bt ? cur : best;
    });
  }, [metricsByNode, selectedNode]);
}

export default useLatestMetric;
