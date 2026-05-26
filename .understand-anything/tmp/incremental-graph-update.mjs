#!/usr/bin/env node
/** Incremental graph refresh using @understand-anything/core structural extraction. */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const PROJECT_ROOT = process.cwd();
const PLUGIN_ROOT = '/home/jmduea/.understand-anything-plugin';
const core = await import(pathToFileURL(path.join(PLUGIN_ROOT, 'packages/core/dist/index.js')).href);
const {
  mergeGraphUpdate,
  saveFingerprints,
  buildFingerprintStore,
  TreeSitterPlugin,
  PluginRegistry,
  builtinLanguageConfigs,
  registerAllParsers,
} = core;

const scan = JSON.parse(readFileSync('.understand-anything/intermediate/scan-result.json', 'utf8'));
const change = JSON.parse(readFileSync('.understand-anything/intermediate/change-analysis.json', 'utf8'));
const existing = JSON.parse(readFileSync('.understand-anything/knowledge-graph.json', 'utf8'));
const currentHash = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();

const structural = [
  ...new Set(
    change.fileChanges
      .filter((f) => f.changeLevel === 'STRUCTURAL')
      .map((f) => f.filePath)
      .filter((p) => !p.startsWith('.understand-anything/') && scan.files.some((f) => f.path === p)),
  ),
];
const fileByPath = Object.fromEntries(scan.files.map((f) => [f.path, f]));

function nodePrefix(file) {
  switch (file.fileCategory) {
    case 'config':
      return 'config';
    case 'docs':
      return 'document';
    case 'infra':
      return file.path.includes('Dockerfile') ? 'service' : 'config';
    default:
      return 'file';
  }
}

function nodeType(file) {
  return nodePrefix(file) === 'service' ? 'service' : nodePrefix(file);
}

function buildNodesAndEdges(structResults) {
  const nodes = [];
  const edges = [];
  for (const item of structResults) {
    const file = fileByPath[item.path];
    if (!file) continue;
    const prefix = nodePrefix(file);
    const type = nodeType(file);
    const fileId = `${prefix}:${item.path}`;
    const fnCount = item.functions?.length ?? 0;
    const clsCount = item.classes?.length ?? 0;
    nodes.push({
      id: fileId,
      type,
      name: path.basename(item.path),
      filePath: item.path,
      summary: `${file.language} ${file.fileCategory} (${item.totalLines} lines, ${fnCount} functions, ${clsCount} classes).`,
      tags: [file.fileCategory, file.language, 'orbit-wars'],
      complexity: item.totalLines > 400 ? 'complex' : item.totalLines > 150 ? 'moderate' : 'simple',
    });
    for (const fn of item.functions ?? []) {
      const fnId = `function:${item.path}:${fn.name}`;
      nodes.push({
        id: fnId,
        type: 'function',
        name: fn.name,
        filePath: item.path,
        summary: `Function ${fn.name} in ${path.basename(item.path)}.`,
        tags: ['function', file.language],
        complexity: fn.endLine - fn.startLine > 80 ? 'complex' : 'moderate',
      });
      edges.push({ source: fileId, target: fnId, type: 'contains', weight: 1.0 });
    }
    for (const cls of item.classes ?? []) {
      const clsId = `class:${item.path}:${cls.name}`;
      nodes.push({
        id: clsId,
        type: 'class',
        name: cls.name,
        filePath: item.path,
        summary: `Class ${cls.name} in ${path.basename(item.path)}.`,
        tags: ['class', file.language],
        complexity: cls.endLine - cls.startLine > 120 ? 'complex' : 'moderate',
      });
      edges.push({ source: fileId, target: clsId, type: 'contains', weight: 1.0 });
    }
    for (const target of scan.importMap[item.path] ?? []) {
      const tgtFile = fileByPath[target];
      if (!tgtFile) continue;
      edges.push({
        source: fileId,
        target: `${nodePrefix(tgtFile)}:${target}`,
        type: 'imports',
        weight: 0.7,
      });
    }
  }
  return { nodes, edges };
}

mkdirSync('.understand-anything/intermediate', { recursive: true });
const tsConfigs = builtinLanguageConfigs.filter((c) => c.treeSitter);
const tsPlugin = new TreeSitterPlugin(tsConfigs);
await tsPlugin.init();
const registry = new PluginRegistry();
registry.register(tsPlugin);
registerAllParsers(registry);

const structResults = [];
for (const rel of structural) {
  const abs = path.join(PROJECT_ROOT, rel);
  if (!existsSync(abs)) continue;
  const content = readFileSync(abs, 'utf8');
  const lines = content.split('\n');
  const totalLines = content.endsWith('\n') ? Math.max(0, lines.length - 1) : lines.length;
  const file = fileByPath[rel];
  const analysis = registry.analyzeFile(rel, content);
  const result = { path: rel, totalLines, functions: [], classes: [] };
  if (analysis?.functions) {
    result.functions = analysis.functions.map((fn) => ({
      name: fn.name,
      startLine: fn.lineRange[0],
      endLine: fn.lineRange[1],
    }));
  }
  if (analysis?.classes) {
    result.classes = analysis.classes.map((cls) => ({
      name: cls.name,
      startLine: cls.lineRange[0],
      endLine: cls.lineRange[1],
    }));
  }
  structResults.push(result);
}

const { nodes: newNodes, edges: newEdges } = buildNodesAndEdges(structResults);
const merged = mergeGraphUpdate(existing, structural, newNodes, newEdges, currentHash);
merged.project.analyzedAt = new Date().toISOString();
merged.project.description = scan.projectDescription;
merged.project.gitCommitHash = currentHash;

const nodeIds = new Set(merged.nodes.map((n) => n.id));
merged.edges = merged.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));
for (const layer of merged.layers ?? []) {
  layer.nodeIds = (layer.nodeIds ?? []).filter((id) => nodeIds.has(id));
}
const assigned = new Set((merged.layers ?? []).flatMap((l) => l.nodeIds ?? []));
const fileLevel = merged.nodes.filter((n) => ['file', 'config', 'document', 'service'].includes(n.type));
let misc = merged.layers.find((l) => l.id === 'layer:misc');
if (!misc) {
  misc = { id: 'layer:misc', name: 'Misc', description: 'Unassigned files', nodeIds: [] };
  merged.layers.push(misc);
}
for (const n of fileLevel) {
  if (!assigned.has(n.id)) {
    misc.nodeIds.push(n.id);
    assigned.add(n.id);
  }
}

writeFileSync('.understand-anything/knowledge-graph.json', JSON.stringify(merged, null, 2));

const sourcePaths = scan.files.map((f) => f.path);
const store = buildFingerprintStore(PROJECT_ROOT, sourcePaths, registry, currentHash);
saveFingerprints(PROJECT_ROOT, store);

writeFileSync(
  '.understand-anything/config.json',
  JSON.stringify({ autoUpdate: true, outputLanguage: 'en' }, null, 2),
);
writeFileSync(
  '.understand-anything/meta.json',
  JSON.stringify(
    {
      lastAnalyzedAt: new Date().toISOString(),
      gitCommitHash: currentHash,
      version: '1.0.0',
      analyzedFiles: sourcePaths.length,
      filteredByIgnore: scan.filteredByIgnore,
      autoUpdate: true,
    },
    null,
    2,
  ),
);

console.log(
  JSON.stringify(
    {
      structuralUpdated: structural.length,
      nodes: merged.nodes.length,
      edges: merged.edges.length,
      layers: merged.layers.length,
      fingerprints: Object.keys(store.files).length,
      commit: currentHash,
    },
    null,
    2,
  ),
);
