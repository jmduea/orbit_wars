import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const PROJECT_ROOT = process.cwd();
const PLUGIN_ROOT = '/home/jmduea/.understand-anything-plugin';
const core = await import(pathToFileURL(path.join(PLUGIN_ROOT, 'packages/core/dist/index.js')).href);
const {
  createIgnoreFilter,
  loadFingerprints,
  analyzeChanges,
  TreeSitterPlugin,
  PluginRegistry,
  builtinLanguageConfigs,
  registerAllParsers,
  extractFileFingerprint,
  contentHash,
} = core;

const lastHash = JSON.parse(readFileSync('.understand-anything/meta.json', 'utf8')).gitCommitHash;
const currentHash = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const changed = execFileSync('git', ['diff', `${lastHash}..HEAD`, '--name-only'], { encoding: 'utf8' })
  .trim()
  .split('\n')
  .filter(Boolean);
const filter = createIgnoreFilter(PROJECT_ROOT);
const kept = changed.filter((p) => !filter.isIgnored(p));
const sourceExts = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.py', '.go', '.rs', '.java', '.rb', '.cpp', '.c', '.h', '.cs', '.swift', '.kt', '.php', '.md', '.yaml', '.yml', '.json', '.toml', '.sql', '.sh',
]);
const sourceFiles = kept.filter(
  (p) => sourceExts.has(path.extname(p)) || p.endsWith('Makefile') || p.endsWith('Dockerfile'),
);

const store = loadFingerprints(PROJECT_ROOT);
const tsConfigs = builtinLanguageConfigs.filter((c) => c.treeSitter);
const tsPlugin = new TreeSitterPlugin(tsConfigs);
await tsPlugin.init();
const registry = new PluginRegistry();
registry.register(tsPlugin);
registerAllParsers(registry);

const analysis = analyzeChanges(PROJECT_ROOT, sourceFiles, store, registry);
writeFileSync(
  '.understand-anything/intermediate/change-analysis.json',
  JSON.stringify({ ...analysis, keptCount: kept.length, sourceCount: sourceFiles.length, lastHash, currentHash }, null, 2),
);
console.log(
  JSON.stringify(
    {
      action: analysis.action,
      structural: analysis.fileChanges.filter((f) => f.changeLevel === 'STRUCTURAL').length,
      cosmetic: analysis.fileChanges.filter((f) => f.changeLevel === 'COSMETIC').length,
      none: analysis.fileChanges.filter((f) => f.changeLevel === 'NONE').length,
      filesToReanalyze: analysis.filesToReanalyze?.length,
      reason: analysis.reason,
      kept: kept.length,
      sourceFiles: sourceFiles.length,
    },
    null,
    2,
  ),
);
