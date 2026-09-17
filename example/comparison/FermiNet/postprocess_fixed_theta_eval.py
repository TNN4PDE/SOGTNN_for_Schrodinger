#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, math
from pathlib import Path
import numpy as np

def read_csv_energy(path):
    energies, variances, pmoves, steps = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "energy" not in reader.fieldnames:
            raise ValueError("No energy column. Columns: " + str(reader.fieldnames))
        for row in reader:
            try:
                energies.append(float(row["energy"]))
                steps.append(int(float(row["step"])))
                if "variance" in row and row["variance"] != "":
                    variances.append(float(row["variance"]))
                if "pmove" in row and row["pmove"] != "":
                    pmoves.append(float(row["pmove"]))
            except Exception:
                pass
    return np.asarray(steps), np.asarray(energies), np.asarray(variances), np.asarray(pmoves)

def blocking_table(x):
    x = np.asarray(x, dtype=np.float64)
    table, block_size, level = [], 1, 0
    y = x.copy()
    while y.size >= 8:
        n = y.size
        mean = float(np.mean(y))
        se = float(np.std(y, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        table.append((level, n, block_size, mean, se))
        if y.size % 2:
            y = y[:-1]
        y = 0.5 * (y[0::2] + y[1::2])
        block_size *= 2
        level += 1
    return table

def choose_conservative_se(table, min_blocks=16):
    valid = [row for row in table if row[1] >= min_blocks and np.isfinite(row[4])]
    if not valid:
        valid = [row for row in table if np.isfinite(row[4])]
    if not valid:
        return float("nan"), None
    best = max(valid, key=lambda r: r[4])
    return best[4], best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--discard", type=int, default=0, help="discard logged rows, not MCMC steps")
    ap.add_argument("--min-blocks", type=int, default=16)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    steps, energy, variance, pmove = read_csv_energy(Path(args.csv))
    if energy.size == 0:
        raise RuntimeError("No energy data read.")
    if args.discard >= energy.size:
        raise ValueError(f"discard={args.discard} >= rows={energy.size}")

    e = energy[args.discard:]
    st = steps[args.discard:] if steps.size == energy.size else np.arange(e.size)
    table = blocking_table(e)
    block_se, chosen = choose_conservative_se(table, args.min_blocks)
    mean = float(np.mean(e))
    naive_se = float(np.std(e, ddof=1) / math.sqrt(e.size)) if e.size > 1 else float("nan")

    print("============================================================")
    print("Fixed-theta FermiNet VMC postprocessing")
    print("============================================================")
    print(f"CSV file                 : {args.csv}")
    print(f"Total logged rows         : {energy.size}")
    print(f"Discarded logged rows     : {args.discard}")
    print(f"Used logged rows          : {e.size}")
    print(f"Step range used           : {int(st[0])} ... {int(st[-1])}")
    print(f"Mean energy               : {mean:.16f} Eh")
    print(f"Naive SE                  : {naive_se:.6e} Eh")
    print(f"Blocking conservative SE  : {block_se:.6e} Eh")
    if chosen:
        print(f"Chosen blocking level     : level={chosen[0]}, n_blocks={chosen[1]}, block_size={chosen[2]}")
    if variance.size == energy.size:
        v = variance[args.discard:]
        print(f"Mean local-energy variance: {np.mean(v):.6e} Eh^2")
        print(f"Mean local-energy std     : {math.sqrt(max(np.mean(v), 0.0)):.6e} Eh")
    if pmove.size == energy.size:
        p = pmove[args.discard:]
        print(f"Mean pmove                : {np.mean(p):.6f}")
    if args.table:
        print("\nBlocking table:")
        print("level,n_blocks,block_size,mean,se")
        for row in table:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]:.16f},{row[4]:.6e}")

if __name__ == "__main__":
    main()
