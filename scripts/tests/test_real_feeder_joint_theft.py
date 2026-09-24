"""Independent physical and integer-model checks; protected tests untouched."""
import networkx as nx
import numpy as np

from scripts.real_feeder_joint_theft import (
    Feeder, ac_flow, fit, predict, read_feeder, sample, truth_atoms,
)


def test_balanced_four_wire_matches_exact_two_bus_solution():
    z1 = .08 + .04j
    matrix = np.eye(4)*z1 + np.ones((4, 4))*(.03+.02j)
    graph = nx.Graph([(0, 1)])
    feeder = Feeder(graph, [0, 1], {1: 0}, [1], np.zeros((2, 3)), {1: matrix}, {})
    load = np.zeros((1, 2, 3), complex); load[:, 1] = 10+3j
    v, head, loss, audit = ac_flow(feeder, load, np.array([230.]))
    # Exact receiving squared voltage: u^2+(2 Re(z conj(S))-V0^2)u+|zS|^2=0.
    power = 10000+3000j
    coefficient = 2*(z1*np.conj(power)).real-230**2
    u = (-coefficient + np.sqrt(coefficient**2-4*abs(z1*power)**2))/2
    assert abs(abs(v[0, 1, 0])**2-u) < 1e-6
    assert abs(v[0, 1, 3]) < 1e-10
    assert np.max(abs(head-30-9j-loss)) < 1e-8


def test_unbalanced_public_feeder_power_balance_and_known_reference():
    feeder = read_feeder()
    data = sample(feeder, 82, 8, False, "persistent", pq_noise=0, v_noise=0, master_noise=0)
    assert data["audit"]["power_balance_residual_kva"] < 1e-7
    assert data["audit"]["max_neutral_v"] > .01
    assert np.max(abs(data["p0"]-data["p"].sum(axis=1)-data["amplitude"]-data["loss"])) < 1e-7


def test_wzzt_is_first_order_mean_squared_voltage_with_unbalance():
    feeder = read_feeder()
    truth, _ = truth_atoms(feeder)
    supports = tuple(truth)
    r = [truth[s].real for s in supports]; x = [truth[s].imag for s in supports]
    loads = feeder.snapshot[None]*1e-5
    volt, _, _, _ = ac_flow(feeder, loads, np.array([230.]))
    p = loads[:, feeder.meters].real.sum(axis=2); q = loads[:, feeder.meters].imag.sum(axis=2)
    y = (230**2-np.mean(abs(volt[:, feeder.meters, :3]-volt[:, feeder.meters, 3:])**2, axis=2))*3/2000
    estimate = predict(p, q, supports, r, x)
    assert np.linalg.norm(y-estimate)/np.linalg.norm(estimate) < 1e-5


def test_stationary_theft_exact_binary_products_without_intercept():
    rng = np.random.default_rng(2)
    supports = tuple(map(frozenset, [[0], [1], [2], [0, 1], [0, 1, 2]]))
    p = rng.uniform(2, 12, (32, 3)); q = rng.uniform(-2, 5, (32, 3))
    r = np.array([.02, .03, .04, .08, .01]); x = r*.6
    amplitude = np.full(32, 9.)
    y = predict(p, q, supports, r, x, amplitude, 3)
    result = fit(dict(p=p, q=q, y=y), supports, amplitude, seconds=20, penalty=1e-6)
    assert result["status"] == "optimal"
    assert result["source_support"] == [0, 1]
    assert result["train_mae"] < 1e-6
    assert result["audit"]["product_error"] < 1e-6
    assert np.max(abs(np.array(result["r"])-r)) < 1e-5


def test_phase_block_matches_full_four_wire_first_order():
    from scripts.real_feeder_phase_wzzt import phase_predict
    feeder = read_feeder(); truth, _ = truth_atoms(feeder)
    supports = tuple(truth); weights = np.zeros((len(supports), 4))
    directed = nx.bfs_tree(feeder.graph, 0)
    for child in feeder.order[1:]:
        descendants = nx.descendants(directed, child) | {child}
        support = frozenset(i for i,bus in enumerate(feeder.meters) if bus in descendants)
        if support:
            edge=feeder.z[child]; z1=edge[0,0]-edge[0,1]; zn=edge[3,3]+edge[0,1]-edge[0,3]-edge[3,0]
            weights[supports.index(support)]+=[z1.real,z1.imag,zn.real,zn.imag]
    loads=feeder.snapshot[None]*1e-5
    volt,_,_,_=ac_flow(feeder,loads,np.array([230.]))
    pp=loads[:,feeder.meters].real; qq=loads[:,feeder.meters].imag
    yp=(230**2-abs(volt[:,feeder.meters,:3]-volt[:,feeder.meters,3:])**2)/2000
    data=dict(p=pp.sum(axis=2),pp=pp,qq=qq,yp=yp)
    estimate=phase_predict(data,supports,weights)
    assert np.linalg.norm(estimate-yp)/np.linalg.norm(yp)<1e-5


def test_scalar_and_phase_runs_share_identical_observations():
    from scripts.real_feeder_phase_wzzt import phase_sample
    feeder=read_feeder()
    for balanced in (True,False):
        scalar=sample(feeder,76,16,balanced,"intermittent")
        phase=phase_sample(feeder,76,16,balanced,"intermittent")
        for key in ("p","q","y","p0","root_v","amplitude"):
            assert np.allclose(scalar[key],phase[key],atol=1e-10),key


def test_phase_integer_location_on_unbalanced_linear_data():
    from scripts.real_feeder_phase_wzzt import phase_predict,fit_phase
    rng=np.random.default_rng(77)
    supports=tuple(map(frozenset,[[0],[1],[0,1]]))
    pp=rng.uniform(1,10,(24,2,3));qq=rng.uniform(-2,4,pp.shape)
    data=dict(p=pp.sum(axis=2),pp=pp,qq=qq,yp=np.zeros_like(pp))
    weights=np.array([[.02,.01,.03,.02],[.04,.02,.04,.02],[.08,.04,.1,.05]])
    amp=np.full(24,9.)
    data['yp']=phase_predict(data,supports,weights,amp,2)
    fitted=fit_phase(data,supports,amp,seconds=20,atom_penalty=0,weight_penalty=0)
    assert fitted['status']=='optimal'
    assert fitted['source_support']==[0,1]
    assert fitted['train_mae']<1e-6
    assert np.max(abs(np.array(fitted['weights'])-weights))<1e-5
