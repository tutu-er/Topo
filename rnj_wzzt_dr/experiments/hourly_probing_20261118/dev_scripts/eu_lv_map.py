"""Map the Non-Synthetic European LV Test System (Mendeley 685vgp64sm) into transformers, feeders and
service points, and judge which feeders fit the hourly probing method (11-18 metered terminals in the
case feeders). Reads the OpenDSS text tables as data only."""
import collections, json, re, sys
from pathlib import Path

R = Path(sys.argv[1])
RX = re.compile(r"New Line\.(\S+)\s+Bus1=([^.\s]+)\S*\s+Bus2=([^.\s]+)\S*.*?Linecode=(\S+)\s+Length=([\d.]+)", re.I)


def lines(path, closed_only=True):
    out = []
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip()
        if not s or (closed_only and s.startswith("!")):
            continue
        m = RX.search(s.lstrip("!"))
        if m:
            out.append((m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5)), s.startswith("!")))
    return out


codes = {}
txt = (R/"linecode.txt").read_text()
for m in re.finditer(r"New LineCode\.(\S+).*?rmatrix\s*=\s*\[\s*([\d.]+)", txt, re.S | re.I):
    codes[m.group(1)] = float(m.group(2))                 # ohm/km, phase conductor
L = lines(R/"Line_indexed_check.txt")
F_closed = [x for x in lines(R/"fuse_indexed.txt") if not x[5]]
F_open = [x for x in lines(R/"fuse_indexed.txt", closed_only=False) if x[5]]
tx = {}
for raw in (R/"Transformer_ind.txt").read_text().splitlines():
    m = re.search(r"Transformer\.(\S+).*?Buses=\[\S+\s+([^.\]\s]+).*?kVAs=\[(\d+)", raw)
    if m:
        tx[m.group(2)] = (m.group(1), int(m.group(3)))
loads = collections.defaultdict(list)
for raw in (R/"Load_indexed.txt").read_text().splitlines():
    m = re.search(r"Load\.(\S+).*?Bus1=([^.\s]+)(\S*)", raw)
    if m:
        loads[m.group(2)].append((m.group(1), "3ph" if m.group(3).count(".") >= 4 else "1ph"))
print(f"lines {len(L)}, closed fuses {len(F_closed)}, open fuses {len(F_open)}, transformers {len(tx)}, "
      f"load buses {len(loads)}, customers {sum(map(len, loads.values()))}, line codes {len(codes)}")
missing_codes = {x[3] for x in L if x[3] not in codes}
print("line codes without R:", sorted(missing_codes)[:10])

adj = collections.defaultdict(list)
for name, a, b, code, length, _ in L+F_closed:
    r = codes.get(code, float("nan"))*length/1000.
    adj[a].append((b, r, length, name)); adj[b].append((a, r, length, name))

rows = []
seen_global = set()
for lv, (tname, kva) in sorted(tx.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
    for nb, r0, l0, lname in adj[lv]:
        # one feeder = the subtree reached from the LV bus through this first line
        stack, seen, parent = [(nb, r0, l0)], {lv, nb}, {nb: lv}
        n_edges, Rmax, Lmax, cyc, branch = 0, 0., 0., 0, 0
        terms, cust, rterm, short = [], 0, [], 0
        while stack:
            u, ru, lu = stack.pop()
            Rmax, Lmax = max(Rmax, ru), max(Lmax, lu)
            if u in loads:
                terms.append(u); cust += len(loads[u]); rterm.append(ru)
            kids = 0
            for v, r, l, _ in adj[u]:
                if v == parent.get(u):
                    continue
                if v in seen:
                    cyc += 1; continue
                seen.add(v); parent[v] = u; stack.append((v, ru+r, lu+l)); n_edges += 1; kids += 1
                if r < .01:
                    short += 1
            if kids >= 2:
                branch += 1
        overlap = len(seen & seen_global); seen_global |= seen
        rows.append(dict(tx=tname, kva=kva, feeder=lname, buses=len(seen)-1, terminals=len(terms), customers=cust,
                         branch_nodes=branch, loops=cyc//2, max_len_m=round(Lmax), max_R_ohm=round(Rmax, 4),
                         short_edges=short, overlap=overlap))
print(f"feeders {len(rows)}")
fits = [r for r in rows if 8 <= r["terminals"] <= 25 and r["loops"] == 0]
by_t = collections.Counter(min(r["terminals"]//5*5, 60) for r in rows)
print("feeders by number of service points (bin of 5):", dict(sorted(by_t.items())))
print(f"feeders with 8-25 service points and no loop: {len(fits)}; with 11-18: "
      f"{sum(11 <= r['terminals'] <= 18 and r['loops'] == 0 for r in rows)}")
print(f"feeders with loops (closed fuses between feeders): {sum(r['loops'] > 0 for r in rows)}; any bus in two feeders: "
      f"{sum(r['overlap'] > 0 for r in rows)}")
for r in sorted(fits, key=lambda r: r["terminals"])[:40]:
    print(r)
json.dump(rows, open(sys.argv[2], "w"), indent=0)
