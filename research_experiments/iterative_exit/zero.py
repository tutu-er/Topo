"""Release provisional zero atoms before another structural search is built."""
from research_experiments.local_rebuild.model import canonical


def release_zero_atoms(bank, point, *, hard=(), enabled=True, tolerance=1e-9, loss_tolerance=1e-10, stage='fit'):
    """Refit and check loss; repeat if that refit makes other atoms inactive.

    Singleton/full response bases and explicit physical priors are retained.
    Failed numerical-zero tests leave the entire current family unchanged.
    """
    if not enabled:return point, []
    protected=set(canonical(hard));family=canonical(point['supports'])
    if not protected <= set(family):raise ValueError('A hard prior is absent from the current family.')
    events=[];current=point
    for _ in range(len(family)+1):
        keep=canonical(C for C,r,x in zip(current['supports'],current['r'],current['x'])
                       if len(C) in (1,bank.data.n) or tuple(C) in protected or max(r,x)>tolerance)
        removed=canonical(C for C in current['supports'] if tuple(C) not in keep)
        if not removed:break
        proposed=bank.fit(keep,preserve=True)
        accepted=proposed['objective'] <= current['objective']+loss_tolerance
        events.append(dict(kind='zero_release',stage=stage,removed=removed,accepted=accepted,
            before=current['objective'],after=proposed['objective'],tolerance=tolerance,loss_tolerance=loss_tolerance))
        if not accepted:break
        current=proposed
    return current,events


def snapshot_zero_support(source, *, with_ordinary_search=False):
    """Include the cleanup dependency when refreshing a historical snapshot."""
    from pathlib import Path
    import shutil
    root=Path(__file__).resolve().parents[2];source=Path(source)
    names=['research_experiments/iterative_exit/__init__.py','research_experiments/iterative_exit/zero.py']
    if with_ordinary_search:names.append('research_experiments/joint_five/search.py')
    for name in names:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
