"""The study's cases (PROTOCOL.md, section 2).

    paths.py      where the case files and results are (outputs/, outside Git)
    physics.py    AC power flow and meters, as the cases were made (a copy; the generator is not here)
    library.py    the development library: open a case (hash-checked), its meter readings and its truth
    real.py       the real EU LV feeders (development inputs): readings and identifiable truth
    world.py      a simulated probing day on a development case (Algorithm 2's world)
"""
