#!/usr/bin/env node
// Runs every case in tools/trace_cases.json through the JS trace parser (extracted from index.html)
// and prints one result per case as JSON for tools/trace_check.py to compare with the Python parser.
"use strict";
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const m = html.match(/<script type="text\/plain" id="engine-src">([\s\S]*?)<\/script>/);
if (!m) { console.error("could not find #engine-src script block in index.html"); process.exit(2); }
const E = new Function(m[1] + "\nreturn { parseTrace };")();

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "trace_cases.json"), "utf8")).cases;
const out = cases.map(c => {
  try {
    const w = E.parseTrace(c.text);
    return { name: c.name, ok: true,
      reqs: w.reqs.map(q => [q.t, q.sid, q.p, q.tokens, q.sysFull, q.n]),
      info: [w.sessions, w.prefixes, w.duration, w.fallbackPrefixes] };
  } catch (e) { return { name: c.name, ok: false, error: e.message }; }
});
process.stdout.write(JSON.stringify(out));
