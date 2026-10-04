/* Exercise the actual inline publisher against a mocked GitHub API. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, '.github/workflows/publish-preview.yml'), 'utf8');
const scripts = [...source.matchAll(/^          script: \|\n((?:            .*\n|\n)+)/gm)]
  .map((match) => match[1].replace(/^            /gm, ''));
assert.equal(scripts.length, 2, 'Both publisher scripts must be tested');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const digest = (data) => `sha256:${crypto.createHash('sha256').update(data).digest('hex')}`;
const sha = 'a'.repeat(40);
const branch = 'main';
const tag = `jarvis-preview-${sha.slice(0, 12)}`;
const repo = { owner: 'owner', repo: 'repository' };

function fixture() {
  const manifest = {
    commit_sha: sha, release_tag: tag, artifact_name: 'JARVIS-Setup.exe',
    artifact_sha256: 'b'.repeat(64), update_class: 'major',
  };
  const bytes = Buffer.from(JSON.stringify(manifest));
  const state = {
    manifest, bytes, uploads: 0, updates: 0, reads: [], summary: [],
    assets: [
      { name: 'JARVIS-Setup.exe', digest: `sha256:${manifest.artifact_sha256}`, size: 100, state: 'uploaded' },
      { name: 'JARVIS-Companion.apk', digest: `sha256:${'c'.repeat(64)}`, size: 100, state: 'uploaded' },
      { name: 'jarvis-update.json', digest: digest(bytes), size: bytes.length, state: 'uploaded' },
    ],
    release: { id: 7, draft: false, prerelease: true, html_url: 'https://github.com/owner/repository/releases/tag/' + tag },
  };
  const summary = {};
  for (const method of ['addHeading', 'addRaw', 'addLink', 'addTable']) {
    summary[method] = (...args) => { state.summary.push([method, ...args]); return summary; };
  }
  summary.write = async () => {};
  const github = {
    rest: {
      git: { getRef: async ({ ref }) => ({ data: { object: { sha, type: 'commit' } } }) },
      actions: { getWorkflowRun: async () => ({ data: {
        head_sha: sha, head_branch: branch, status: 'completed', conclusion: 'success',
      } }) },
      repos: {
        get: async () => ({ data: { private: false } }),
        getReleaseByTag: async () => ({ data: state.release }),
        listReleaseAssets: () => {},
        uploadReleaseAsset: async () => { state.uploads++; throw new Error('Unexpected replacement'); },
        updateRelease: async () => { state.updates++; throw new Error('Unexpected release mutation'); },
      },
    },
    paginate: async () => state.assets,
  };
  const files = {
    readFileSync: (name) => {
      state.reads.push(name);
      if (name === 'release-assets/jarvis-update.json') return JSON.stringify(manifest);
      if (name === 'validated-runs.json') return JSON.stringify(Array.from({ length: 9 }, (_, id) => ({ id, workflow: `workflow-${id}` })));
      throw new Error(`Unexpected payload read: ${name}`);
    },
    writeFileSync: (name, value) => { state.savedRuns = JSON.parse(value); },
  };
  const core = { summary, info: () => {}, setOutput: () => {} };
  const fetch = async (url) => {
    assert.equal(url, `https://github.com/owner/repository/releases/download/${tag}/jarvis-update.json`);
    return { ok: true, arrayBuffer: async () => state.bytes };
  };
  const run = (index = 1, clock = Date) => new AsyncFunction(
    'require', 'context', 'github', 'core', 'process', 'fetch', 'Buffer', 'Date', 'setTimeout', scripts[index],
  )((name) => name === 'node:fs' ? files : require(name), { repo }, github, core,
    { env: { PREVIEW_SHA: sha, PREVIEW_BRANCH: branch } }, fetch, Buffer, clock, (fn) => fn());
  return { state, github, run };
}

test('main promotion reuses a complete immutable release despite different new build bytes', async () => {
  const { state, run } = fixture();
  await run();
  assert.equal(state.uploads, 0);
  assert.equal(state.updates, 0);
  assert.deepEqual(state.reads, ['release-assets/jarvis-update.json', 'validated-runs.json']);
  assert.ok(state.summary.some((entry) => entry.includes('Existing validated JARVIS preview retained')));
});

test('missing or duplicate public assets fail without mutation', async () => {
  for (const duplicate of [false, true]) {
    const { state, run } = fixture();
    if (duplicate) state.assets.push({ ...state.assets[0] }); else state.assets.pop();
    await assert.rejects(run(), /invalid asset metadata/);
    assert.equal(state.uploads + state.updates, 0);
  }
});

test('published manifest bytes must match the GitHub digest', async () => {
  const { state, run } = fixture();
  state.bytes = Buffer.from('changed');
  await assert.rejects(run(), /does not match its GitHub asset digest/);
});

test('published manifest must pin the expected commit and installer', async () => {
  for (const field of ['commit_sha', 'artifact_sha256', 'release_tag', 'artifact_name', 'update_class']) {
    const { state, run } = fixture();
    state.bytes = Buffer.from(JSON.stringify({ ...state.manifest, [field]: 'wrong' }));
    Object.assign(state.assets[2], { size: state.bytes.length, digest: digest(state.bytes) });
    await assert.rejects(run(), /does not pin the expected source and installer/, field);
  }
});

test('reusing a release still rejects validation from another branch', async () => {
  const { github, run } = fixture();
  github.rest.actions.getWorkflowRun = async () => ({ data: {
    head_sha: sha, head_branch: 'other-branch', status: 'completed', conclusion: 'success',
  } });
  await assert.rejects(run(), /Validation changed before publication/);
});

test('validation gate chooses the selected branch instead of a newer canceled duplicate', async () => {
  const { state, github, run } = fixture();
  github.rest.actions.listWorkflowRuns = async (query) => {
    assert.equal(query.branch, branch);
    return { data: { workflow_runs: [
      { id: 200, head_sha: sha, head_branch: 'product/jarvis-brain', event: 'push', status: 'completed', conclusion: 'cancelled' },
      { id: 100, head_sha: sha, head_branch: branch, event: 'push', status: 'completed', conclusion: 'success' },
    ] } };
  };
  await run(0);
  assert.equal(state.savedRuns.length, 9);
  assert.ok(state.savedRuns.every((run) => run.id === 100));
});

test('validation gate refuses to borrow another branch results', async () => {
  const { state, github, run } = fixture();
  github.rest.actions.listWorkflowRuns = async () => ({ data: { workflow_runs: [
    { id: 100, head_sha: sha, head_branch: 'another-branch', event: 'push', status: 'completed', conclusion: 'success' },
  ] } });
  let calls = 0;
  await assert.rejects(run(0, { now: () => ++calls > 2 ? 71 * 60 * 1000 : 0 }), /Timed out/);
  assert.equal(state.savedRuns, undefined);
});

test('every release gate runs on main and product pushes without path omissions', () => {
  const workflows = [...scripts[0].matchAll(/'([\w-]+\.yml)'/g)].map((match) => match[1]);
  for (const workflow of new Set(workflows)) {
    const contents = fs.readFileSync(path.join(root, '.github/workflows', workflow), 'utf8');
    const push = contents.match(/^  push:\n((?:    .*\n|\n)+)/m)?.[1];
    assert.ok(push, `${workflow} needs a push trigger`);
    assert.match(push, /\bmain\b/, `${workflow} needs main`);
    assert.match(push, /product\//, `${workflow} needs product`);
    assert.doesNotMatch(push, /paths(?:-ignore)?:/, `${workflow} must not omit release commits`);
  }
});
