"""Geometry and raw-domain checks, plus independent tiny native Gurobi fits."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from research_experiments.rnj.rx75_edge_floor import (
    calibrate_rx75_floor, classify_rx75_atoms,
)


def test_calibration_uses_positive_distance_means_and_is_scale_consistent():
    r = np.diag([.2, .4, .6]) + .3
    x = np.diag([.1, .3, .2]) + .2
    c = calibrate_rx75_floor(r, x)
    dr = np.diag(r)[:, None] + np.diag(r)[None, :] - 2*r
    dx = np.diag(x)[:, None] + np.diag(x)[None, :] - 2*x
    a, b = .75/dr[dr > 0].mean(), .25/dx[dx > 0].mean()
    assert c["a"] == pytest.approx(a)
    assert c["b"] == pytest.approx(b)
    assert c["tau"] == pytest.approx(.16 * np.median(np.diag(a*r+b*x)))
    scaled = calibrate_rx75_floor(3*r, 7*x)
    assert scaled["tau"] == pytest.approx(c["tau"])
    assert scaled["a"]*3 == pytest.approx(c["a"])
    assert scaled["b"]*7 == pytest.approx(c["b"])
    empty_distances = calibrate_rx75_floor(np.zeros((2, 2)), np.zeros((2, 2)))
    assert (empty_distances["mu_r"], empty_distances["mu_x"]) == (1., 1.)
    assert empty_distances["tau"] == pytest.approx(.16e-12)


def test_classification_keeps_leaves_and_equality_including_full_root_stem(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("geometry helper must not read files or truth")
    monkeypatch.setattr("builtins.open", forbidden)
    calibration = {"a": .75, "b": .25, "tau": .1}
    before = deepcopy(calibration)
    family = [[0], [1], [0, 1], [0, 1, 2]]
    result = classify_rx75_atoms(family, [0., 0., .1, .1], [0., 0., .1, .099], calibration)
    assert result["keep"] == [True, True, True, False]
    assert result["contracted_indices"] == [3]
    assert result["absolute_tolerance"] == 1e-10
    calibration.update(truth="not read", family_clade_f1=1., validation_mae=-999.)
    assert result == classify_rx75_atoms(family, [0., 0., .1, .1], [0., 0., .1, .099], calibration)
    assert {key: calibration[key] for key in before} == before
    near = classify_rx75_atoms([[0, 1]], [.1-5e-11], [.1-5e-11], calibration)
    strict = classify_rx75_atoms([[0, 1]], [.1-5e-11], [.1-5e-11], calibration, absolute_tolerance=0.)
    assert near["keep"] == [True] and strict["keep"] == [False]


@pytest.mark.parametrize("r,x,factor", [([[1., 2.]], [[1., 2.]], .16),
    (np.eye(2), np.eye(3), .16), (np.array([[np.nan]]), np.eye(1), .16),
    (np.eye(2), np.eye(2), -.1), (np.eye(2), np.eye(2), np.inf)])
def test_invalid_calibration_is_rejected(r, x, factor):
    with pytest.raises(ValueError):
        calibrate_rx75_floor(r, x, factor)


@pytest.mark.parametrize("r,x", [([.1], []), ([np.inf], [.1]), ([-.1], [.1]), ([[.1]], [.1])])
def test_invalid_atom_dimensions_are_rejected(r, x):
    with pytest.raises(ValueError):
        classify_rx75_atoms([[0, 1]], r, x, {"a": 1., "b": 1., "tau": .1})


BUILD_AND_SOLVE = r'''
import json,sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from research_experiments.rnj.prepare_source_ac288_runtime import activate_runtime
activate_runtime(sys.argv[1])
from rnj_wzzt.estimation import laminar_l1_milp as core,laminar_l1_source as source
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs,prepare_source_inputs
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from research_experiments.rnj.rx75_edge_floor import (
 calibrate_rx75_floor,classify_rx75_atoms,enforce_rx75_floor,FloorConfigurationError)

def dense(model):
 a=np.zeros((model.rows.size,model.variables.size))
 np.add.at(a,(model.rows.row_index,model.rows.col_index),model.rows.data)
 return a,np.asarray(model.rows.lower),np.asarray(model.rows.upper)
def feasible(a,lo,hi,v):return bool(np.all(a@v>=lo-1e-10) and np.all(a@v<=hi+1e-10))

r=np.diag([.2,.3,.4,.5])+.1;x=np.diag([.1,.2,.3,.4])+.08
c=calibrate_rx75_floor(r,x)
geometry=sensitivity_geometry(r,x)
assert abs(c['tau']-.16*max(float(np.median(geometry.root_depths)),1e-12))<1e-14
scenarios=[{'P_terminal':pd.DataFrame(np.ones((2,4))*.1),'Q_terminal':pd.DataFrame(np.ones((2,4))*.04),'drop_target':pd.DataFrame(np.zeros((2,4)))}]
prepared=core._prepare_scenarios(scenarios)
supports=((0,),(1,),(2,),(3,),(0,1),(0,1,2,3))
floor={'a':.75,'b':.25,'tau':.1}
saved={(module.__name__,name):getattr(module,name) for module,names in
 ((core,('build_fixed_model','build_extension_model')),
  (source,('build_fixed_model','build_extension_model','build_fixed_source_model','build_extension_source_model')))
 for name in names}
baseline=core.build_fixed_model(prepared,supports,r_upper_bound=2.,x_upper_bound=2.)
with enforce_rx75_floor(core,source,floor):
 model=core.build_fixed_model(prepared,supports,r_upper_bound=2.,x_upper_bound=2.)
 assert model.variables.slices['edge_active'].stop-model.variables.slices['edge_active'].start==2
 assert model.rows.size==baseline.rows.size+6
 a,lo,hi=dense(model); a,lo,hi=a[-6:],lo[-6:],hi[-6:]
 v=np.zeros(model.variables.size); assert feasible(a,lo,hi,v)
 v[model.r.start+4]=.01; assert not feasible(a,lo,hi,v)
 e=model.variables.slices['edge_active'].start;v[e]=1.
 assert not feasible(a,lo,hi,v) # positive but below the floor
 v[model.r.start+4]=.2; assert feasible(a,lo,hi,v)
 # Full root stem gets the same floor, while leaf coordinates never touch e.
 assert all(not np.any(a[:,model.r.start+i]) for i in range(4))
 v[model.r.start+5]=.2; assert not feasible(a,lo,hi,v)
 v[e+1]=1.; assert feasible(a,lo,hi,v)
 fallback,blocks=source.build_fixed_source_model(prepared,supports,None,r_upper_bound=2.,x_upper_bound=2.)
 assert blocks is None and fallback.variables.size==model.variables.size
 assert fallback.rows.size==model.rows.size # cached alias + outer wrapper added once
 leafs=tuple((i,) for i in range(4))
 ext=core.build_extension_model(prepared,leafs,r_upper_bound=2.,x_upper_bound=2.,terminal_groups=[(0,1,2)])
 aa,ll,hh=dense(ext)
 assert aa[-1,ext.z.start:ext.z.stop].tolist()==[3.,1.]
 assert ll[-1]==2. and aa[-2,ext.new_r.start]==.75 and aa[-2,ext.new_x.start]==.25
 # One 3-terminal group passes original-terminal cardinality; one singleton does not.
 vv=np.zeros(ext.variables.size);vv[ext.z.start]=1.
 assert aa[-1]@vv>=ll[-1]
 vv[ext.z.start]=0.;vv[ext.z.start+1]=1.;assert aa[-1]@vv<ll[-1]
 try:core.build_extension_model(prepared,leafs,r_upper_bound=.01,x_upper_bound=.01)
 except FloorConfigurationError:pass
 else:raise AssertionError('capacity configuration error was reported as a search outcome')
 # Equality with the weighted box capacity is a valid configuration.
 boundary=core.build_fixed_model(prepared,supports,r_upper_bound=.1,x_upper_bound=.1)
 assert boundary._rx75_edge_floor['calibration'][2]==.1
 # Source h remains the original full path, independently of edge activation.
 idx=pd.Index([0,1]);power=[SourceInputs(pd.Series([.03,.06],index=idx),pd.Series([.01,.005],index=idx),'unit')]
 src=prepare_source_inputs(scenarios,power)
 sm,sb=source.build_fixed_source_model(prepared,supports,src,r_upper_bound=2.,x_upper_bound=2.)
 assert sb.h.stop-sb.h.start==len(supports)
 sa,slo,shi=dense(sm)
 assert not np.any(sa[-6:,sb.h])
 sv=np.zeros(sm.variables.size);sv[sb.h.start+4]=1.;sv[sb.h.start+5]=1.
 assert feasible(sa,slo,shi,sv) # zero old edges may still form a nonempty h chain
 se,seb=source.build_extension_source_model(prepared,leafs,src,r_upper_bound=2.,x_upper_bound=2.,terminal_groups=[(0,1,2)])
 assert seb.h.stop-seb.h.start==len(leafs)+1
 assert se.z.stop-se.z.start==2 and se._rx75_edge_floor['extension']
 zero=prepare_source_inputs(scenarios,[SourceInputs(pd.Series([0.,0.],index=idx),pd.Series([0.,0.],index=idx),'zero')])
 ze,zb=source.build_extension_source_model(prepared,supports,zero,r_upper_bound=2.,x_upper_bound=2.)
 assert zb is None and ze.variables.slices['edge_active'].stop-ze.variables.slices['edge_active'].start==2
for module,names in ((core,('build_fixed_model','build_extension_model')),
 (source,('build_fixed_model','build_extension_model','build_fixed_source_model','build_extension_source_model'))):
 for name in names: assert getattr(module,name) is saved[(module.__name__,name)]
# tau=0 is literally the original builder, with no added integer variables/rows.
with enforce_rx75_floor(core,source,{'a':.75,'b':.25,'tau':0.}):
 assert core.build_fixed_model is saved[(core.__name__,'build_fixed_model')]
 identical=core.build_fixed_model(prepared,supports,r_upper_bound=2.,x_upper_bound=2.)
 assert identical.variables.slices==baseline.variables.slices
 np.testing.assert_array_equal(dense(identical)[0],dense(baseline)[0])
try:
 with enforce_rx75_floor(core,source,floor): raise RuntimeError('deliberate context interruption')
except RuntimeError:pass
assert core.build_fixed_model is saved[(core.__name__,'build_fixed_model')]

if sys.argv[2]=='solve':
 rng=np.random.default_rng(20261018);T,n=32,3
 p=rng.uniform(.01,.35,(T,n));q=rng.uniform(-.04,.2,(T,n))
 R=np.diag([.08,.10,.12])+.12;X=np.diag([.05,.07,.06])+.08
 idx=pd.RangeIndex(T);labels=[101,102,103]
 answers=[]
 for enabled in (False,True):
  up=rng.uniform(.01,.2,T) if enabled else np.zeros(T)
  uq=rng.uniform(-.08,.09,T) if enabled else np.zeros(T)
  y=p@R.T+q@X.T+up[:,None]*.12+uq[:,None]*.08
  ss=[{'P_terminal':pd.DataFrame(p,index=idx,columns=labels),
       'Q_terminal':pd.DataFrame(q,index=idx,columns=labels),'drop_target':pd.DataFrame(y,index=idx,columns=labels)}]
  uu=[SourceInputs(pd.Series(up,index=idx),pd.Series(uq,index=idx),'independent_unit')]
  options=dict(initial_supports=[(i,) for i in range(n)],terminal_groups=[(0,1,2)],
    source_inputs=uu,r_upper_bound=2.,x_upper_bound=2.,time_limit=10.,solver='gurobi')
  # Zero-floor execution matches the untouched native fit.
  plain=core.fit_laminar_l1_sensitivity(ss,**options)
  with enforce_rx75_floor(core,source,{'a':.5,'b':.5,'tau':0.}):
   zero_fit=core.fit_laminar_l1_sensitivity(ss,**options)
  np.testing.assert_allclose(plain.r_matrix,zero_fit.r_matrix,atol=1e-10)
  np.testing.assert_allclose(plain.x_matrix,zero_fit.x_matrix,atol=1e-10)
  with enforce_rx75_floor(core,source,{'a':.5,'b':.5,'tau':.04}):
   fitted=core.fit_laminar_l1_sensitivity(ss,**options)
  np.testing.assert_allclose(fitted.r_matrix,R,atol=1e-7)
  np.testing.assert_allclose(fitted.x_matrix,X,atol=1e-7)
  assert fitted.selected_path_index==len(fitted.path)-1 and fitted.validation_mae is None
  assert all(core.solver_diagnostics_prove_optimality(pt.solver) for pt in fitted.path)
  assert fitted.support_indices==((0,),(1,),(2,),(0,1,2))
  if enabled:
   assert fitted.source_h.tolist()==[0,0,0,1]
   np.testing.assert_allclose(fitted.source_response_r,.12,atol=1e-7)
   np.testing.assert_allclose(fitted.source_response_x,.08,atol=1e-7)
  else:assert fitted.source_h is None
  answers.append({'source':enabled,'path_count':len(fitted.path),'stop':fitted.stop_reason})
 print(json.dumps({'checked':'raw_constraints_and_native_gurobi','answers':answers}))
else:print(json.dumps({'checked':'raw_constraints'}))
'''


@pytest.fixture(scope="module")
def frozen_runtime(tmp_path_factory):
    from research_experiments.rnj.prepare_source_ac288_runtime import ensure_runtime
    path = tmp_path_factory.mktemp("floor_frozen_runtime")
    ensure_runtime(path)
    return path


def run_check(runtime, mode):
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    result = subprocess.run([sys.executable, "-B", "-c", BUILD_AND_SOLVE, str(runtime), mode],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_actual_builder_rows_fallback_aliases_and_context_restoration(frozen_runtime):
    assert run_check(frozen_runtime, "build")["checked"] == "raw_constraints"


def test_tiny_source_and_non_source_native_gurobi_fits(frozen_runtime):
    pytest.importorskip("gurobipy")
    result = run_check(frozen_runtime, "solve")
    assert result["checked"] == "raw_constraints_and_native_gurobi"
    assert [a["source"] for a in result["answers"]] == [False, True]
