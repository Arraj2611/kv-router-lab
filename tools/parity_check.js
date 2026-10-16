#!/usr/bin/env node
// Extracts the simulator engine from index.html and, using the shared config
// in tools/parity_config.json:
//   1. dumps a golden sequence (first 1000 raw RNG draws + first 200 request
//      fields) for tools/parity_check.py to diff bit-for-bit against the
//      Python port, and
//   2. runs all six strategies at every configured rate and writes the
//      resulting metrics as JSON for tools/parity_check.py to compare.
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const TOOLS = __dirname;

const cfgFile = JSON.parse(fs.readFileSync(path.join(TOOLS, "parity_config.json"), "utf8"));

const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const m = html.match(/<script type="text\/plain" id="engine-src">([\s\S]*?)<\/script>/);
if (!m) {
  console.error("could not find #engine-src script block in index.html");
  process.exit(2);
}
const E = new Function(m[1] + "\nreturn { makeWorkload, Fleet, STRATS, BLOCK, rng };")();

function baseCfg(rate, cpuTokens = 0) {
  return {
    seed: cfgFile.seed, products: cfgFile.products, zipf: cfgFile.zipf,
    sysMin: cfgFile.sysMin, sysMax: cfgFile.sysMax, maxTurns: cfgFile.maxTurns, think: cfgFile.think,
    tps: cfgFile.tps, overhead: cfgFile.overhead, cachedTps: cfgFile.cachedTps, thresh: cfgFile.thresh, slack: cfgFile.slack,
    duration: cfgFile.duration, replicas: cfgFile.replicas, cacheTokens: cfgFile.cacheTokens, rate,
    cpuTokens, reloadTps: cfgFile.reloadTps,
  };
}

// ---------- 1. golden sequence (raw RNG draws + first 200 request fields)
const r = E.rng(cfgFile.seed);
const rVals = [];
for (let i = 0; i < 1000; i++) rVals.push(r());

const goldenCfg = baseCfg(cfgFile.rates[0]);
const goldenW = E.makeWorkload(goldenCfg);
const goldenReqs = goldenW.reqs.slice(0, 200).map(q => ({
  t: q.t, sid: q.sid, p: q.p, tokens: q.tokens, sysFull: q.sysFull, n: q.n,
}));
fs.writeFileSync(path.join(TOOLS, "golden_js.json"), JSON.stringify({ rVals, requests: goldenReqs }));

// ---------- 2. full metrics at every configured rate
const rows = [];
for (const cpu of cfgFile.cpuTokensList) {
  for (const rate of cfgFile.rates) {
    const cfg = baseCfg(rate, cpu);
    const W = E.makeWorkload(cfg);
    for (const s of E.STRATS) {
      const f = new E.Fleet(cfg, s.id);
      for (const q of W.reqs) f.serve(q);
      const sum = f.summary(cfg.duration);
      rows.push({ id: s.id, rate, cpu, hit_rate: sum.hit, ttft_p50: sum.p50, ttft_p99: sum.p99, imbalance: sum.imb, cpu_hit_share: sum.cpuShare });
    }
  }
}
const outPath = path.join(TOOLS, "parity_js.json");
fs.writeFileSync(outPath, JSON.stringify(rows, null, 2));

console.log(`wrote golden_js.json (1000 r() draws, 200 requests) and ${rows.length} rows to parity_js.json`);
for (const row of rows) {
  console.log(`${row.id.padEnd(8)} rate=${String(row.rate).padEnd(4)} cpu=${String(row.cpu).padEnd(6)} hit=${(row.hit_rate * 100).toFixed(1)}%  p50=${row.ttft_p50.toFixed(4)}s  p99=${row.ttft_p99.toFixed(4)}s  imb=${row.imbalance.toFixed(4)}x  cpuShare=${(row.cpu_hit_share * 100).toFixed(1)}%`);
}
