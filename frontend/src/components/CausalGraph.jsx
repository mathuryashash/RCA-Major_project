import React, { useRef, useEffect, useCallback, useState } from 'react';
import * as d3 from 'd3';

/**
 * FR-33: Interactive Causal Graph Viewer
 *
 * Renders a force-directed DAG of causal relationships using D3.js.
 * React owns the SVG element; D3 controls the simulation and positioning.
 */

const COLORS = {
  background: '#1a1a2e',
  nodeGreen: '#2ecc71',
  nodeAmber: '#f39c12',
  nodeRed: '#e74c3c',
  rootGlow: '#ffd700',
  edgeDefault: '#555',
  labelText: '#ffffff',
  edgeLabelText: '#888',
  arrowMarker: '#777',
};

function getNodeColor(anomalyScore) {
  if (anomalyScore < 0.3) return COLORS.nodeGreen;
  if (anomalyScore <= 0.7) return COLORS.nodeAmber;
  return COLORS.nodeRed;
}

function getNodeSize(confidence) {
  const min = 10;
  const max = 40;
  const clamped = Math.max(0, Math.min(1, confidence ?? 0.5));
  return min + clamped * (max - min);
}

function getEdgeWidth(weight) {
  const min = 1;
  const max = 8;
  const clamped = Math.max(0, Math.min(1, (weight ?? 0.5) / 10));
  return min + clamped * (max - min);
}

export default function CausalGraph({ graphData, onNodeClick }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const simulationRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Stable click handler
  const handleNodeClick = useCallback(
    (nodeId) => {
      if (onNodeClick) onNodeClick(nodeId);
    },
    [onNodeClick],
  );

  // Track container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Main D3 rendering
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;

    // Clear previous render
    svg.selectAll('*').remove();

    if (
      !graphData ||
      !graphData.nodes ||
      graphData.nodes.length === 0
    ) {
      return;
    }

    // Deep-copy data so D3 mutation doesn't affect props
    const nodes = graphData.nodes.map((n) => ({ ...n }));
    const links = graphData.links.map((l) => ({
      ...l,
      source: l.source,
      target: l.target,
    }));

    // ── Defs: arrow markers & root-cause glow filter ──

    const defs = svg.append('defs');

    defs
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', COLORS.arrowMarker);

    const glowFilter = defs.append('filter').attr('id', 'glow');
    glowFilter
      .append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur');
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // ── Container group for zoom/pan ──

    const g = svg.append('g');

    svg.call(
      d3
        .zoom()
        .scaleExtent([0.2, 4])
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        }),
    );

    // ── Links (edges) ──

    const linkGroup = g
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', COLORS.edgeDefault)
      .attr('stroke-width', (d) => getEdgeWidth(d.weight))
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', 'url(#arrowhead)');

    // ── Edge labels ──

    const edgeLabelGroup = g
      .append('g')
      .attr('class', 'edge-labels')
      .selectAll('text')
      .data(links)
      .join('text')
      .attr('font-size', '10px')
      .attr('fill', COLORS.edgeLabelText)
      .attr('text-anchor', 'middle')
      .attr('dy', -4)
      .text((d) => d.label || '');

    // ── Nodes ──

    const nodeGroup = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => handleNodeClick(d.id))
      .call(
        d3
          .drag()
          .on('start', (event, d) => {
            if (!event.active) simulationRef.current?.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulationRef.current?.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    // Draw shape per node
    nodeGroup.each(function (d) {
      const el = d3.select(this);
      const size = getNodeSize(d.confidence);
      const color = getNodeColor(d.anomaly_score);

      if (d.shape === 'square') {
        el.append('rect')
          .attr('width', size)
          .attr('height', size)
          .attr('x', -size / 2)
          .attr('y', -size / 2)
          .attr('rx', 3)
          .attr('fill', color)
          .attr('stroke', d.is_root_cause ? COLORS.rootGlow : 'none')
          .attr('stroke-width', d.is_root_cause ? 3 : 0)
          .attr('filter', d.is_root_cause ? 'url(#glow)' : null);
      } else {
        // circle (default)
        el.append('circle')
          .attr('r', size / 2)
          .attr('fill', color)
          .attr('stroke', d.is_root_cause ? COLORS.rootGlow : 'none')
          .attr('stroke-width', d.is_root_cause ? 3 : 0)
          .attr('filter', d.is_root_cause ? 'url(#glow)' : null);
      }
    });

    // Node labels
    nodeGroup
      .append('text')
      .attr('dy', (d) => getNodeSize(d.confidence) / 2 + 14)
      .attr('text-anchor', 'middle')
      .attr('fill', COLORS.labelText)
      .attr('font-size', '11px')
      .attr('pointer-events', 'none')
      .text((d) => d.id);

    // ── Enter transitions ──

    nodeGroup.attr('opacity', 0).transition().duration(600).attr('opacity', 1);
    linkGroup.attr('opacity', 0).transition().duration(600).attr('opacity', 0.7);
    edgeLabelGroup.attr('opacity', 0).transition().duration(600).attr('opacity', 1);

    // ── Force simulation ──

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance(140),
      )
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force(
        'collision',
        d3.forceCollide().radius((d) => getNodeSize(d.confidence) / 2 + 10),
      )
      .on('tick', () => {
        linkGroup
          .attr('x1', (d) => d.source.x)
          .attr('y1', (d) => d.source.y)
          .attr('x2', (d) => d.target.x)
          .attr('y2', (d) => d.target.y);

        edgeLabelGroup
          .attr('x', (d) => (d.source.x + d.target.x) / 2)
          .attr('y', (d) => (d.source.y + d.target.y) / 2);

        nodeGroup.attr('transform', (d) => `translate(${d.x},${d.y})`);
      });

    simulationRef.current = simulation;

    // Cleanup on unmount or re-render
    return () => {
      simulation.stop();
      simulationRef.current = null;
    };
  }, [graphData, dimensions, handleNodeClick]);

  // ── Empty state ──

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          minHeight: 400,
          background: COLORS.background,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 8,
        }}
      >
        <span style={{ color: '#888', fontSize: 16 }}>
          No causal graph available
        </span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        minHeight: 400,
        background: COLORS.background,
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ display: 'block' }}
      />
    </div>
  );
}
