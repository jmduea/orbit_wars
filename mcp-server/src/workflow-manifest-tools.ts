import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  getWorkspaceRoot,
  ensureDir,
  safeReadFile,
  safeWriteFile,
  safeJsonParse,
  errorResponse,
} from "./utils.js";

const MANIFEST_RELATIVE_PATH = path.join(".omg", "workflow-manifest.json");

const ACTIVE_STATUSES = ["draft", "approved", "planned", "executing"] as const;
const TERMINAL_STATUSES = ["complete", "superseded", "deferred"] as const;
const ALL_STATUSES = [...ACTIVE_STATUSES, ...TERMINAL_STATUSES] as const;

type WorkflowStatus = (typeof ALL_STATUSES)[number];
type WorkflowKind = "spec" | "plan" | "trace";

export interface WorkflowManifestEntry {
  id: string;
  kind: WorkflowKind;
  path: string;
  workflow: string;
  status: WorkflowStatus;
  title: string;
  spec_id?: string | null;
  plan_id?: string | null;
  related_ids?: string[];
  supersedes_id?: string | null;
  completed_at?: string | null;
  evidence?: string | null;
  notes?: string | null;
}

interface WorkflowManifest {
  schema_version: number;
  _agent?: Record<string, unknown>;
  entries: WorkflowManifestEntry[];
}

function getManifestPath(): string {
  return path.join(getWorkspaceRoot(), MANIFEST_RELATIVE_PATH);
}

function defaultManifest(): WorkflowManifest {
  return {
    schema_version: 1,
    _agent: {
      canonical_source:
        "This manifest is the single source of truth for spec/plan lifecycle.",
      before_brownfield_scan:
        "Call omg_workflow_manifest_list with active_only=true before reading .omg/specs/ or .omg/plans/.",
      active_statuses: [...ACTIVE_STATUSES],
    },
    entries: [],
  };
}

function isWorkflowStatus(value: unknown): value is WorkflowStatus {
  return typeof value === "string" && (ALL_STATUSES as readonly string[]).includes(value);
}

function isWorkflowKind(value: unknown): value is WorkflowKind {
  return value === "spec" || value === "plan" || value === "trace";
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function parseEntry(raw: unknown): WorkflowManifestEntry | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (typeof value.id !== "string" || !value.id.trim()) return null;
  if (!isWorkflowKind(value.kind)) return null;
  if (typeof value.path !== "string" || !value.path.trim()) return null;
  if (typeof value.workflow !== "string" || !value.workflow.trim()) return null;
  if (!isWorkflowStatus(value.status)) return null;
  if (typeof value.title !== "string" || !value.title.trim()) return null;

  return {
    id: value.id.trim(),
    kind: value.kind,
    path: value.path.trim(),
    workflow: value.workflow.trim(),
    status: value.status,
    title: value.title.trim(),
    spec_id: optionalString(value.spec_id),
    plan_id: optionalString(value.plan_id),
    related_ids: Array.isArray(value.related_ids)
      ? value.related_ids.filter((item): item is string => typeof item === "string")
      : [],
    supersedes_id: optionalString(value.supersedes_id),
    completed_at: optionalString(value.completed_at),
    evidence: optionalString(value.evidence),
    notes: optionalString(value.notes),
  };
}

function parseManifest(raw: string): WorkflowManifest {
  const parsed = safeJsonParse(raw);
  if (!parsed.ok) {
    throw new Error(`Malformed workflow manifest: ${parsed.error}`);
  }

  const data = parsed.data;
  const schemaVersion = typeof data.schema_version === "number" ? data.schema_version : 1;
  if (!Array.isArray(data.entries)) {
    throw new Error("Malformed workflow manifest: entries must be an array");
  }

  const entries: WorkflowManifestEntry[] = [];
  for (const item of data.entries) {
    const entry = parseEntry(item);
    if (!entry) {
      throw new Error("Malformed workflow manifest: invalid entry object");
    }
    entries.push(entry);
  }

  const ids = new Set<string>();
  for (const entry of entries) {
    if (ids.has(entry.id)) {
      throw new Error(`Malformed workflow manifest: duplicate id ${entry.id}`);
    }
    ids.add(entry.id);
  }

  return {
    schema_version: schemaVersion,
    _agent:
      typeof data._agent === "object" && data._agent !== null && !Array.isArray(data._agent)
        ? (data._agent as Record<string, unknown>)
        : undefined,
    entries,
  };
}

export function readWorkflowManifest(): WorkflowManifest {
  const manifestPath = getManifestPath();
  const raw = safeReadFile(manifestPath);
  if (!raw) {
    return defaultManifest();
  }
  return parseManifest(raw);
}

export function writeWorkflowManifest(manifest: WorkflowManifest): void {
  const manifestPath = getManifestPath();
  ensureDir(path.dirname(manifestPath));
  safeWriteFile(manifestPath, JSON.stringify(manifest, null, 2));
}

export function isActiveStatus(status: WorkflowStatus): boolean {
  return (ACTIVE_STATUSES as readonly string[]).includes(status);
}

export function listWorkflowManifestEntries(options?: {
  activeOnly?: boolean;
  kind?: WorkflowKind;
  status?: WorkflowStatus;
}): WorkflowManifestEntry[] {
  const manifest = readWorkflowManifest();
  let entries = manifest.entries;

  if (options?.kind) {
    entries = entries.filter((entry) => entry.kind === options.kind);
  }
  if (options?.status) {
    entries = entries.filter((entry) => entry.status === options.status);
  }
  if (options?.activeOnly) {
    entries = entries.filter((entry) => isActiveStatus(entry.status));
  }

  return entries;
}

export function getWorkflowManifestEntry(id: string): WorkflowManifestEntry | null {
  return readWorkflowManifest().entries.find((entry) => entry.id === id) ?? null;
}

export function validateWorkflowManifest(): {
  ok: boolean;
  missing_paths: string[];
  broken_links: string[];
  unregistered_markdown: string[];
} {
  const manifest = readWorkflowManifest();
  const workspaceRoot = getWorkspaceRoot();
  const missing_paths: string[] = [];
  const broken_links: string[] = [];
  const registeredPaths = new Set(manifest.entries.map((entry) => entry.path));

  for (const entry of manifest.entries) {
    const absolutePath = path.join(workspaceRoot, entry.path);
    if (!fs.existsSync(absolutePath)) {
      missing_paths.push(entry.path);
    }
    if (entry.spec_id && !manifest.entries.some((candidate) => candidate.id === entry.spec_id)) {
      broken_links.push(`${entry.id} -> spec_id ${entry.spec_id}`);
    }
    if (entry.plan_id && !manifest.entries.some((candidate) => candidate.id === entry.plan_id)) {
      broken_links.push(`${entry.id} -> plan_id ${entry.plan_id}`);
    }
    for (const relatedId of entry.related_ids ?? []) {
      if (!manifest.entries.some((candidate) => candidate.id === relatedId)) {
        broken_links.push(`${entry.id} -> related_ids ${relatedId}`);
      }
    }
  }

  const unregistered_markdown: string[] = [];
  for (const dirName of ["specs", "plans"]) {
    const dirPath = path.join(workspaceRoot, ".omg", dirName);
    if (!fs.existsSync(dirPath)) continue;
    for (const fileName of fs.readdirSync(dirPath)) {
      if (!fileName.endsWith(".md")) continue;
      const relativePath = path.join(".omg", dirName, fileName);
      if (!registeredPaths.has(relativePath)) {
        unregistered_markdown.push(relativePath);
      }
    }
  }

  return {
    ok: missing_paths.length === 0 && broken_links.length === 0,
    missing_paths,
    broken_links,
    unregistered_markdown,
  };
}

export function registerWorkflowManifestEntry(
  entry: Omit<WorkflowManifestEntry, "related_ids" | "completed_at" | "evidence" | "notes"> &
    Partial<Pick<WorkflowManifestEntry, "related_ids" | "completed_at" | "evidence" | "notes">>
): WorkflowManifestEntry {
  const manifest = readWorkflowManifest();
  if (manifest.entries.some((existing) => existing.id === entry.id)) {
    throw new Error(`Workflow manifest entry already exists: ${entry.id}`);
  }
  if (manifest.entries.some((existing) => existing.path === entry.path)) {
    throw new Error(`Workflow manifest path already registered: ${entry.path}`);
  }

  const normalized: WorkflowManifestEntry = {
    id: entry.id.trim(),
    kind: entry.kind,
    path: entry.path.trim(),
    workflow: entry.workflow.trim(),
    status: entry.status,
    title: entry.title.trim(),
    spec_id: entry.spec_id ?? null,
    plan_id: entry.plan_id ?? null,
    related_ids: entry.related_ids ?? [],
    supersedes_id: entry.supersedes_id ?? null,
    completed_at: entry.completed_at ?? null,
    evidence: entry.evidence ?? null,
    notes: entry.notes ?? null,
  };

  manifest.entries.push(normalized);
  writeWorkflowManifest(manifest);
  return normalized;
}

export function updateWorkflowManifestEntry(
  id: string,
  patch: Partial<
    Pick<
      WorkflowManifestEntry,
      "status" | "evidence" | "notes" | "completed_at" | "plan_id" | "spec_id" | "title" | "path"
    >
  >
): WorkflowManifestEntry {
  const manifest = readWorkflowManifest();
  const index = manifest.entries.findIndex((entry) => entry.id === id);
  if (index < 0) {
    throw new Error(`Workflow manifest entry not found: ${id}`);
  }

  const current = manifest.entries[index];
  const nextStatus = patch.status ?? current.status;
  if (!isWorkflowStatus(nextStatus)) {
    throw new Error(`Invalid workflow status: ${String(patch.status)}`);
  }

  const updated: WorkflowManifestEntry = {
    ...current,
    ...patch,
    status: nextStatus,
    completed_at:
      patch.completed_at !== undefined
        ? patch.completed_at
        : (TERMINAL_STATUSES as readonly string[]).includes(nextStatus) && !current.completed_at
          ? new Date().toISOString().slice(0, 10)
          : current.completed_at ?? null,
  };

  manifest.entries[index] = updated;
  writeWorkflowManifest(manifest);
  return updated;
}

export function registerWorkflowManifestTools(server: McpServer): void {
  server.tool(
    "omg_workflow_manifest_list",
    "List workflow spec/plan/trace entries from .omg/workflow-manifest.json. Use active_only=true before brownfield scans.",
    {
      active_only: z
        .boolean()
        .optional()
        .describe("When true, return only draft/approved/planned/executing entries."),
      kind: z.enum(["spec", "plan", "trace"]).optional().describe("Filter by entry kind."),
      status: z.enum(ALL_STATUSES).optional().describe("Filter by exact status."),
    },
    async ({ active_only, kind, status }) => {
      try {
        const entries = listWorkflowManifestEntries({
          activeOnly: active_only,
          kind,
          status,
        });
        const manifest = readWorkflowManifest();
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                success: true,
                schema_version: manifest.schema_version,
                active_statuses: ACTIVE_STATUSES,
                agent_rules: manifest._agent ?? null,
                total: entries.length,
                entries,
              }),
            },
          ],
        };
      } catch (error) {
        return errorResponse(error instanceof Error ? error.message : String(error));
      }
    }
  );

  server.tool(
    "omg_workflow_manifest_get",
    "Get one workflow manifest entry by id.",
    {
      id: z.string().describe("Manifest entry id, e.g. jax-ppo-split"),
    },
    async ({ id }) => {
      try {
        const entry = getWorkflowManifestEntry(id);
        if (!entry) {
          return errorResponse(`Workflow manifest entry not found: ${id}`);
        }
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, entry }) }],
        };
      } catch (error) {
        return errorResponse(error instanceof Error ? error.message : String(error));
      }
    }
  );

  server.tool(
    "omg_workflow_manifest_register",
    "Register a new spec/plan/trace entry in .omg/workflow-manifest.json.",
    {
      id: z.string().describe("Stable slug id, e.g. jax-ppo-split"),
      kind: z.enum(["spec", "plan", "trace"]).describe("Entry kind"),
      path: z.string().describe("Repo-relative markdown path"),
      workflow: z.string().describe("Originating workflow, e.g. deep-interview"),
      status: z.enum(ALL_STATUSES).default("draft").describe("Initial lifecycle status"),
      title: z.string().describe("Short human title"),
      spec_id: z.string().optional().describe("Linked spec id for plans/traces"),
      plan_id: z.string().optional().describe("Linked plan id for specs"),
      related_ids: z.array(z.string()).optional().describe("Related manifest ids"),
      notes: z.string().optional().describe("Optional notes"),
    },
    async ({ id, kind, path: entryPath, workflow, status, title, spec_id, plan_id, related_ids, notes }) => {
      try {
        const entry = registerWorkflowManifestEntry({
          id,
          kind,
          path: entryPath,
          workflow,
          status,
          title,
          spec_id: spec_id ?? null,
          plan_id: plan_id ?? null,
          related_ids: related_ids ?? [],
          notes: notes ?? null,
        });
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                success: true,
                entry,
                active: isActiveStatus(entry.status),
              }),
            },
          ],
        };
      } catch (error) {
        return errorResponse(error instanceof Error ? error.message : String(error));
      }
    }
  );

  server.tool(
    "omg_workflow_manifest_update",
    "Update lifecycle status or evidence for a workflow manifest entry.",
    {
      id: z.string().describe("Manifest entry id"),
      status: z.enum(ALL_STATUSES).optional().describe("New lifecycle status"),
      evidence: z.string().optional().describe("Completion or progress evidence"),
      notes: z.string().optional().describe("Agent notes"),
      completed_at: z.string().optional().describe("ISO date when work completed"),
      plan_id: z.string().optional().describe("Linked plan entry id"),
      spec_id: z.string().optional().describe("Linked spec entry id"),
    },
    async ({ id, status, evidence, notes, completed_at, plan_id, spec_id }) => {
      try {
        if (
          status === undefined &&
          evidence === undefined &&
          notes === undefined &&
          completed_at === undefined &&
          plan_id === undefined &&
          spec_id === undefined
        ) {
          return errorResponse("Provide at least one field to update.");
        }

        const entry = updateWorkflowManifestEntry(id, {
          status,
          evidence,
          notes,
          completed_at,
          plan_id,
          spec_id,
        });

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                success: true,
                entry,
                active: isActiveStatus(entry.status),
              }),
            },
          ],
        };
      } catch (error) {
        return errorResponse(error instanceof Error ? error.message : String(error));
      }
    }
  );

  server.tool(
    "omg_workflow_manifest_validate",
    "Validate workflow manifest paths and links; report unregistered .omg markdown files.",
    {},
    async () => {
      try {
        const result = validateWorkflowManifest();
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                success: true,
                ...result,
              }),
            },
          ],
        };
      } catch (error) {
        return errorResponse(error instanceof Error ? error.message : String(error));
      }
    }
  );
}
