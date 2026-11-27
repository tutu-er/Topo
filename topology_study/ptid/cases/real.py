"""The real EU LV feeders (PROTOCOL.md, section 2): 33 feeders with real topologies, start days 0 and 6, so 66
development inputs; 8 days of hourly readings from real customer profiles through an AC power flow, without the PCC
power. The truth is their identifiable truth (lines of at least 3 milliohm).

The input files were written by the earlier probing study as objects of its own classes. They are read here without
that code: every class from rnj_wzzt_dr is replaced by a plain record that keeps the attributes.

    names()           the 66 input names
    open_real(name)   dict(name, group, n, hist (P, Q, Y, V0sq), truth (meter groupings))
"""
import pickle

from .paths import REAL


class _Record:
    """Stands in for a class of the earlier studies; unpickling fills its attributes."""


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("rnj_wzzt_dr"):
            return _Record
        return super().find_class(module, name)


def names():
    return sorted(f.stem for f in REAL.glob("*.pkl"))


def open_real(name):
    with open(REAL/f"{name}.pkl", "rb") as fh:
        inp = _Reader(fh).load()
    n = int(inp.n)
    hist = {k: inp.hist[k] for k in ("P", "Q", "Y", "V0sq")}
    truth = {tuple(sorted(int(i) for i in C)) for C in inp.truth_key if 1 < len(C) < n}
    return dict(name=name, group=inp.case, n=n, hist=hist, truth=truth)
