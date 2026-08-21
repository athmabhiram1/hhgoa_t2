# Fast-path latency — 2026-08-21T07:44:50.478028+00:00

Local bge-m3 (cuda) + brute-force cosine over 31259 chunks. No Vertex, no Neo4j.

| metric | ms |
|---|---|
| P50 | 45.53 |
| P70 | 50.61 |
| P100 | 80.52 |
| mean | 47.33 |
| n | 70 |

Per-lang P50 (ms):

| lang | P50 |
|---|---|
| as | 46.79 |
| bn | 48.45 |
| gu | 42.42 |
| hi | 51.36 |
| kn | 43.79 |
| ml | 45.67 |
| mr | 42.38 |
| ne | 56.19 |
| or | 43.35 |
| pa | 45.66 |
| sa | 45.23 |
| ta | 38.66 |
| te | 46.59 |
| ur | 53.31 |
