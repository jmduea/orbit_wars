#!/usr/bin/env node
const fs = require('fs');
const graphPath = process.argv[2];
const outputPath = process.argv[3];
const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const issues = [];
const warnings = [];
const nodeIds = new Set();
const seen = new Map();
for (const [i, n] of graph.nodes.entries()) {
  if (!n.id) issues.push(`Node[${i}] missing id`);
  if (!n.type) issues.push(`Node[${i}] '${n.id}' missing type`);
  if (!n.name) issues.push(`Node[${i}] '${n.id}' missing name`);
  if (!n.summary) issues.push(`Node[${i}] '${n.id}' missing summary`);
  if (!n.tags?.length) issues.push(`Node[${i}] '${n.id}' missing tags`);
  if (seen.has(n.id)) issues.push(`Duplicate node ID '${n.id}'`);
  else seen.set(n.id, i);
  nodeIds.add(n.id);
}
for (const [i, e] of graph.edges.entries()) {
  if (!nodeIds.has(e.source)) issues.push(`Edge[${i}] source '${e.source}' not found`);
  if (!nodeIds.has(e.target)) issues.push(`Edge[${i}] target '${e.target}' not found`);
}
const fileLevelTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const fileNodes = graph.nodes.filter((n) => fileLevelTypes.has(n.type)).map((n) => n.id);
const assigned = new Map();
for (const layer of graph.layers ?? []) {
  for (const id of layer.nodeIds ?? []) {
    if (!nodeIds.has(id)) issues.push(`Layer '${layer.id}' refs missing node '${id}'`);
    if (assigned.has(id)) issues.push(`Node '${id}' appears in multiple layers`);
    assigned.set(id, layer.id);
  }
}
for (const id of fileNodes) {
  if (!assigned.has(id)) warnings.push(`File node '${id}' not in any layer`);
}
fs.writeFileSync(outputPath, JSON.stringify({ issues, warnings, stats: { totalNodes: graph.nodes.length, totalEdges: graph.edges.length, totalLayers: graph.layers.length, tourSteps: graph.tour?.length ?? 0 } }, null, 2));
console.log(JSON.stringify({ issueCount: issues.length, warningCount: warnings.length }, null, 2));
