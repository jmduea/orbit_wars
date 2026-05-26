#!/usr/bin/env node
/**
 * Phase 1 project scan for /understand.
 */
import { readFileSync, writeFileSync, existsSync, statSync, mkdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
const PLUGIN_ROOT = "/home/jmduea/.understand-anything-plugin";
const coreModule = await import(
  pathToFileURL(path.join(PLUGIN_ROOT, "packages/core/dist/index.js")).href
);
const ignoreModule = await import(
  pathToFileURL(
    path.join(PLUGIN_ROOT, "packages/core/node_modules/ignore/index.js"),
  ).href
);
const ignoreLib = ignoreModule.default ?? ignoreModule;
const { createIgnoreFilter, DEFAULT_IGNORE_PATTERNS } = coreModule;

const PROJECT_ROOT = path.resolve(process.argv[2] ?? process.cwd());
const OUTPUT_PATH = path.resolve(
  process.argv[3] ??
    path.join(PROJECT_ROOT, ".understand-anything/intermediate/scan-result.json"),
);

const EXT_LANGUAGE = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  py: "python",
  go: "go",
  rs: "rust",
  java: "java",
  rb: "ruby",
  cpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  h: "cpp",
  hpp: "cpp",
  c: "c",
  cs: "csharp",
  swift: "swift",
  kt: "kotlin",
  php: "php",
  vue: "vue",
  svelte: "svelte",
  sh: "shell",
  bash: "shell",
  ps1: "powershell",
  bat: "batch",
  cmd: "batch",
  md: "markdown",
  rst: "markdown",
  yaml: "yaml",
  yml: "yaml",
  json: "json",
  jsonc: "jsonc",
  toml: "toml",
  sql: "sql",
  graphql: "graphql",
  gql: "graphql",
  proto: "protobuf",
  tf: "terraform",
  tfvars: "terraform",
  html: "html",
  htm: "html",
  css: "css",
  scss: "css",
  sass: "css",
  less: "css",
  xml: "xml",
  cfg: "config",
  ini: "config",
  env: "config",
  csv: "data",
  prisma: "data",
};

const BASENAME_LANGUAGE = {
  Dockerfile: "dockerfile",
  Makefile: "makefile",
  Jenkinsfile: "jenkinsfile",
};

const EXTENSION_VARIANTS = [
  "",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  "/index.ts",
  "/index.js",
  "/index.tsx",
  "/index.jsx",
  ".py",
  ".go",
  ".rs",
  ".rb",
];

const PYTHON_FRAMEWORKS = {
  django: "Django",
  djangorestframework: "Django REST Framework",
  fastapi: "FastAPI",
  flask: "Flask",
  sqlalchemy: "SQLAlchemy",
  alembic: "Alembic",
  celery: "Celery",
  pydantic: "Pydantic",
  uvicorn: "Uvicorn",
  gunicorn: "Gunicorn",
  aiohttp: "aiohttp",
  tornado: "Tornado",
  starlette: "Starlette",
  pytest: "pytest",
  hypothesis: "Hypothesis",
  channels: "Django Channels",
  flax: "Flax",
  jax: "JAX",
  optax: "Optax",
  "hydra-core": "Hydra",
  hydra: "Hydra",
  wandb: "Weights & Biases",
};

const JS_FRAMEWORKS = {
  react: "React",
  vue: "Vue",
  svelte: "Svelte",
  "@angular/core": "Angular",
  express: "Express",
  fastify: "Fastify",
  koa: "Koa",
  next: "Next.js",
  nuxt: "Nuxt",
  vite: "Vite",
  vitest: "Vitest",
  jest: "Jest",
  mocha: "Mocha",
  tailwindcss: "Tailwind CSS",
  prisma: "Prisma",
  typeorm: "TypeORM",
  sequelize: "Sequelize",
  mongoose: "Mongoose",
  redux: "Redux",
  zustand: "Zustand",
  mobx: "MobX",
  "@modelcontextprotocol/sdk": "Model Context Protocol",
  typescript: "TypeScript",
};

function normalizeRel(relPath) {
  return relPath.replace(/\\/g, "/");
}

function detectLanguage(filePath) {
  const base = path.basename(filePath);
  if (BASENAME_LANGUAGE[base]) return BASENAME_LANGUAGE[base];
  const ext = path.extname(filePath).slice(1).toLowerCase();
  if (EXT_LANGUAGE[ext]) return EXT_LANGUAGE[ext];
  return ext || "unknown";
}

function detectFileCategory(filePath) {
  const norm = normalizeRel(filePath);
  const base = path.basename(norm);
  const ext = path.extname(norm).slice(1).toLowerCase();

  if (base === "Dockerfile" || base === "Makefile" || base === "Jenkinsfile" || base === "Procfile" || base === "Vagrantfile" || base.startsWith("docker-compose.") || [".tf", ".tfvars"].includes("." + ext) || norm.startsWith(".github/workflows/") || base === ".gitlab-ci.yml" || norm.startsWith(".circleci/") || base.endsWith(".k8s.yaml") || base.endsWith(".k8s.yml") || norm.includes("/k8s/") || norm.includes("/kubernetes/")) {
    return "infra";
  }
  if (["md", "rst"].includes(ext) || (ext === "txt" && base !== "LICENSE")) return "docs";
  if (["yaml", "yml", "json", "jsonc", "toml", "xml", "cfg", "ini", "env"].includes(ext) || ["tsconfig.json", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"].includes(base)) {
    return "config";
  }
  if (["sql", "graphql", "gql", "proto", "prisma"].includes(ext) || base.endsWith(".schema.json") || ext === "csv") {
    return "data";
  }
  if (["sh", "bash", "ps1", "bat"].includes(ext)) return "script";
  if (["html", "htm", "css", "scss", "sass", "less"].includes(ext)) return "markup";
  return "code";
}

function createDefaultsOnlyFilter() {
  const ig = ignoreLib();
  ig.add(DEFAULT_IGNORE_PATTERNS);
  return { isIgnored: (p) => ig.ignores(p) };
}

function hasUserIgnoreFiles() {
  return (
    existsSync(path.join(PROJECT_ROOT, ".understand-anything", ".understandignore")) ||
    existsSync(path.join(PROJECT_ROOT, ".understandignore"))
  );
}

function discoverFiles() {
  try {
    const out = execFileSync("git", ["ls-files"], {
      cwd: PROJECT_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return out
      .trim()
      .split("\n")
      .filter(Boolean)
      .map(normalizeRel);
  } catch (err) {
    throw new Error(`git ls-files failed: ${err.message}`);
  }
}

function countLines(absPath) {
  const content = readFileSync(absPath, "utf8");
  if (content.length === 0) return 0;
  return content.split("\n").length;
}

function estimateComplexity(totalFiles) {
  if (totalFiles <= 30) return "small";
  if (totalFiles <= 150) return "moderate";
  if (totalFiles <= 500) return "large";
  return "very-large";
}

function readText(relPath) {
  const abs = path.join(PROJECT_ROOT, relPath);
  if (!existsSync(abs)) return "";
  return readFileSync(abs, "utf8");
}

function extractProjectName() {
  const pkg = readText("package.json");
  if (pkg) {
    try {
      const parsed = JSON.parse(pkg);
      if (parsed.name) return parsed.name;
    } catch {
      /* ignore */
    }
  }
  const pyproject = readText("pyproject.toml");
  const nameMatch = pyproject.match(/^\s*name\s*=\s*"([^"]+)"/m);
  if (nameMatch) return nameMatch[1];
  return path.basename(PROJECT_ROOT);
}

function extractRawDescription() {
  const pkg = readText("package.json");
  if (pkg) {
    try {
      const parsed = JSON.parse(pkg);
      if (parsed.description) return parsed.description;
    } catch {
      /* ignore */
    }
  }
  const pyproject = readText("pyproject.toml");
  const descMatch = pyproject.match(/^\s*description\s*=\s*"([^"]+)"/m);
  if (descMatch) return descMatch[1];
  return "";
}

function extractReadmeHead() {
  for (const candidate of ["README.md", "README.rst", "readme.md"]) {
    const content = readText(candidate);
    if (content) return content.split("\n").slice(0, 10).join("\n");
  }
  return "";
}

function synthesizeDescription(rawDescription, readmeHead, totalFiles) {
  let description = rawDescription.trim();
  if (!description && readmeHead.trim()) {
    const lines = readmeHead
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"));
    description = lines.slice(0, 2).join(" ");
  }
  if (!description) description = "No description available";
  if (totalFiles > 100) {
    description +=
      " Note: this project has over 100 source files; consider scoping analysis to a subdirectory for faster results.";
  }
  return description;
}

function extractPyprojectDependencies(pyproject) {
  const deps = [];
  const lines = pyproject.split("\n");
  let inDeps = false;
  for (const line of lines) {
    if (/^\[project\]\s*$/.test(line.trim())) {
      inDeps = false;
    }
    if (/^\s*dependencies\s*=\s*\[\s*$/.test(line)) {
      inDeps = true;
      continue;
    }
    if (inDeps) {
      if (/^\s*\]\s*$/.test(line)) break;
      const match = line.match(/^\s*"([^"]+)"/);
      if (match) deps.push(match[1].split(/[<>=!;\[]/)[0].trim());
    }
  }
  return deps;
}

function addFramework(frameworks, name) {
  if (name && !frameworks.includes(name)) frameworks.push(name);
}

function detectFrameworks(allPaths) {
  const frameworks = [];
  const pathSet = new Set(allPaths);

  const pyproject = readText("pyproject.toml");
  if (pyproject) {
    addFramework(frameworks, "Python");
    for (const depName of extractPyprojectDependencies(pyproject)) {
      const label = PYTHON_FRAMEWORKS[depName];
      if (label) addFramework(frameworks, label);
    }
    if (/^\[tool\.pytest\.ini_options\]/m.test(pyproject)) addFramework(frameworks, "pytest");
    if (/^\[tool\.django\]/m.test(pyproject)) addFramework(frameworks, "Django");
  }

  for (const rel of allPaths) {
    if (rel.endsWith("package.json")) {
      try {
        const parsed = JSON.parse(readText(rel));
        const deps = { ...(parsed.dependencies ?? {}), ...(parsed.devDependencies ?? {}) };
        for (const [dep, label] of Object.entries(JS_FRAMEWORKS)) {
          if (deps[dep]) addFramework(frameworks, label);
        }
      } catch {
        /* ignore */
      }
    }
    if (rel.endsWith("tsconfig.json")) addFramework(frameworks, "TypeScript");
    if (path.basename(rel) === "Dockerfile") addFramework(frameworks, "Docker");
    if (path.basename(rel).startsWith("docker-compose.")) addFramework(frameworks, "Docker Compose");
    if (rel.endsWith(".tf") || rel.endsWith(".tfvars")) addFramework(frameworks, "Terraform");
    if (rel.startsWith(".github/workflows/") && (rel.endsWith(".yml") || rel.endsWith(".yaml"))) {
      addFramework(frameworks, "GitHub Actions");
    }
    if (path.basename(rel) === ".gitlab-ci.yml") addFramework(frameworks, "GitLab CI");
    if (path.basename(rel) === "Jenkinsfile") addFramework(frameworks, "Jenkins");
  }

  if (pathSet.has("Makefile")) addFramework(frameworks, "Make");
  return frameworks.sort((a, b) => a.localeCompare(b));
}

function buildFileSet(filePaths) {
  const set = new Set(filePaths);
  return {
    has(candidate) {
      return set.has(candidate);
    },
    addResolved(resolved, bucket) {
      if (set.has(resolved) && !bucket.includes(resolved)) bucket.push(resolved);
    },
  };
}

function resolveCandidate(basePath, importPath, fileSet, bucket) {
  const joined = normalizeRel(path.join(path.dirname(basePath), importPath));
  for (const variant of EXTENSION_VARIANTS) {
    const candidate = joined.endsWith("/") ? joined.slice(0, -1) + variant : joined + variant;
    fileSet.addResolved(candidate, bucket);
  }
}

function resolvePythonModule(modulePath, fileSet, bucket) {
  const parts = modulePath.split(".").filter(Boolean);
  if (parts.length === 0) return;
  const moduleCandidates = [
    parts.join("/") + ".py",
    parts.join("/") + "/__init__.py",
  ];
  for (const candidate of moduleCandidates) {
    fileSet.addResolved(candidate, bucket);
  }
}

function resolvePythonFromImport(modulePath, importedNames, fileSet, bucket) {
  resolvePythonModule(modulePath, fileSet, bucket);
  const parts = modulePath.split(".").filter(Boolean);
  const initPath = parts.join("/") + "/__init__.py";
  if (fileSet.has(initPath)) {
    for (const name of importedNames) {
      if (name === "*") continue;
      resolvePythonModule(`${modulePath}.${name}`, fileSet, bucket);
    }
  }
}

function resolveRelativePythonModule(importerPath, modulePath, fileSet, bucket) {
  const importerDir = path.dirname(importerPath);
  const segments = modulePath.split(".").filter(Boolean);
  let relDir = importerDir;
  while (segments.length > 0 && segments[0] === "") {
    segments.shift();
  }
  while (segments.length > 0 && (segments[0] === "" || segments[0] === ".")) {
    if (segments[0] === "..") {
      relDir = path.dirname(relDir);
    }
    segments.shift();
  }
  const remainder = segments.join("/");
  const base = normalizeRel(path.join(relDir, remainder));
  for (const suffix of ["", ".py", "/__init__.py"]) {
    const candidate = suffix ? base + suffix : base;
    fileSet.addResolved(candidate, bucket);
  }
}

function extractPythonImports(content, importerPath, fileSet) {
  const resolved = [];
  const importRe = /^(?:from\s+(\.+[\w.]*|\.*[\w.]+)\s+import\s+([\w.*,\s]+)|import\s+([\w.]+))/gm;
  for (const match of content.matchAll(importRe)) {
    const bucket = [];
    if (match[1] !== undefined) {
      const modulePath = match[1];
      const imported = match[2]
        .split(",")
        .map((s) => s.trim().split(/\s+/)[0])
        .filter(Boolean);
      if (modulePath.startsWith(".")) {
        resolveRelativePythonModule(importerPath, modulePath, fileSet, bucket);
        for (const name of imported) {
          if (name === "*") continue;
          resolveRelativePythonModule(importerPath, `${modulePath}.${name}`, fileSet, bucket);
        }
      } else {
        resolvePythonFromImport(modulePath, imported, fileSet, bucket);
      }
    } else if (match[3]) {
      resolvePythonModule(match[3], fileSet, bucket);
    }
    for (const item of bucket) {
      if (!resolved.includes(item)) resolved.push(item);
    }
  }
  return resolved;
}

function loadTsconfigPaths() {
  const configs = [];
  for (const rel of ["tsconfig.json", "mcp-server/tsconfig.json"]) {
    const content = readText(rel);
    if (!content) continue;
    try {
      const parsed = JSON.parse(content);
      const compilerOptions = parsed.compilerOptions ?? {};
      const baseUrl = compilerOptions.baseUrl ?? ".";
      const paths = compilerOptions.paths ?? {};
      const configDir = path.dirname(rel);
      configs.push({ configDir, baseUrl, paths });
    } catch {
      /* ignore */
    }
  }
  return configs;
}

function resolveTsAlias(importPath, tsConfigs, fileSet, bucket) {
  for (const { configDir, baseUrl, paths } of tsConfigs) {
    for (const [prefix, targets] of Object.entries(paths)) {
      const starIdx = prefix.indexOf("*");
      if (starIdx === -1) {
        if (importPath === prefix.replace(/\/$/, "")) {
          for (const target of targets) {
            const resolvedBase = normalizeRel(path.join(configDir, baseUrl, target.replace(/\*$/, "")));
            resolveCandidate(resolvedBase + "/dummy.ts", ".", fileSet, bucket);
          }
        }
        continue;
      }
      const head = prefix.slice(0, starIdx);
      const tail = prefix.slice(starIdx + 1);
      if (importPath.startsWith(head) && importPath.endsWith(tail)) {
        const middle = importPath.slice(head.length, importPath.length - tail.length);
        for (const target of targets) {
          const mapped = target.replace("*", middle);
          const resolvedBase = normalizeRel(path.join(configDir, baseUrl, mapped));
          for (const variant of EXTENSION_VARIANTS) {
            const candidate = resolvedBase + (variant || "");
            fileSet.addResolved(candidate, bucket);
          }
        }
      }
    }
  }
}

function extractTsImports(content, importerPath, fileSet, tsConfigs) {
  const resolved = [];
  const patterns = [
    /import\s+(?:type\s+)?(?:[\w*{}\s,$]+?\s+from\s+)?["']([^"']+)["']/g,
    /export\s+(?:type\s+)?(?:[\w*{}\s,$]+?\s+from\s+)?["']([^"']+)["']/g,
    /require\s*\(\s*["']([^"']+)["']\s*\)/g,
  ];
  for (const re of patterns) {
    for (const match of content.matchAll(re)) {
      const spec = match[1];
      const bucket = [];
      if (spec.startsWith(".")) {
        resolveCandidate(importerPath, spec, fileSet, bucket);
      } else if (!spec.startsWith("node:") && !spec.includes("://")) {
        resolveTsAlias(spec, tsConfigs, fileSet, bucket);
      }
      for (const item of bucket) {
        if (!resolved.includes(item)) resolved.push(item);
      }
    }
  }
  return resolved;
}

function buildImportMap(files, fileSet) {
  const importMap = {};
  const tsConfigs = loadTsconfigPaths();
  for (const file of files) {
    importMap[file.path] = [];
  }
  for (const file of files) {
    if (file.fileCategory !== "code") continue;
    const content = readText(file.path);
    if (!content) continue;
    let resolved = [];
    if (file.language === "python") {
      resolved = extractPythonImports(content, file.path, fileSet);
    } else if (file.language === "typescript" || file.language === "javascript") {
      resolved = extractTsImports(content, file.path, fileSet, tsConfigs);
    }
    importMap[file.path] = resolved.filter((target) => target !== file.path);
  }
  return importMap;
}

function main() {
  if (!existsSync(PROJECT_ROOT)) {
    throw new Error(`Project root does not exist: ${PROJECT_ROOT}`);
  }

  const allTracked = discoverFiles();
  const defaultsFilter = createDefaultsOnlyFilter();
  const step2Files = allTracked.filter((rel) => {
    const abs = path.join(PROJECT_ROOT, rel);
    return !defaultsFilter.isIgnored(rel) && existsSync(abs) && statSync(abs).isFile();
  });

  let finalPaths = step2Files;
  let filteredByIgnore = 0;
  if (hasUserIgnoreFiles()) {
    const unifiedFilter = createIgnoreFilter(PROJECT_ROOT);
    finalPaths = allTracked.filter((rel) => {
      const abs = path.join(PROJECT_ROOT, rel);
      return !unifiedFilter.isIgnored(rel) && existsSync(abs) && statSync(abs).isFile();
    });
    const finalSet = new Set(finalPaths);
    filteredByIgnore = step2Files.filter((rel) => !finalSet.has(rel)).length;
  }

  const files = finalPaths
    .map((rel) => {
      const abs = path.join(PROJECT_ROOT, rel);
      return {
        path: rel,
        language: detectLanguage(rel),
        sizeLines: countLines(abs),
        fileCategory: detectFileCategory(rel),
      };
    })
    .sort((a, b) => a.path.localeCompare(b.path));

  const fileSet = buildFileSet(files.map((f) => f.path));
  const importMap = buildImportMap(files, fileSet);
  const languages = [...new Set(files.map((f) => f.language))].sort();
  const frameworks = detectFrameworks(finalPaths);
  const rawDescription = extractRawDescription();
  const readmeHead = extractReadmeHead();
  const name = extractProjectName();
  const totalFiles = files.length;
  const description = synthesizeDescription(rawDescription, readmeHead, totalFiles);

  const result = {
    name,
    description,
    languages,
    frameworks,
    files,
    totalFiles,
    filteredByIgnore,
    estimatedComplexity: estimateComplexity(totalFiles),
    importMap,
  };

  mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
  writeFileSync(OUTPUT_PATH, JSON.stringify(result, null, 2));

  console.log(
    JSON.stringify(
      {
        name,
        description,
        totalFiles,
        filteredByIgnore,
        languages,
        frameworks,
        outputPath: OUTPUT_PATH,
      },
      null,
      2,
    ),
  );
}

try {
  main();
} catch (err) {
  console.error(err.message ?? err);
  process.exit(1);
}
