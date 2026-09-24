"""Check independently constructed teaching examples; not paper replication."""
import json
from pathlib import Path
import numpy as np

checks = {}
W = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1]], dtype=float).T
K = W @ np.diag([1,2,3,4,5]) @ W.T
np.testing.assert_allclose(K, [[3,1,1],[1,8,4],[1,4,9]])
d = np.diag(K)[:,None] + np.diag(K)[None,:] - 2*K
np.testing.assert_allclose(d, [[0,9,10],[9,0,9],[10,9,0]])
phi = np.array([K[1,1]-K[2,2],d[1,0]-d[2,0]])
np.testing.assert_allclose(phi, [-1,-1])
limb_b = (d[1,2]+phi.mean())/2
limb_c = d[1,2]-limb_b
np.testing.assert_allclose([limb_b,limb_c], [4,5])
np.testing.assert_allclose([(8-4+9-5)/2, (9-4+10-5)/2], [4,5])
np.testing.assert_allclose([(3+4-5)/2,(3+5-4)/2,(4+5-3)/2], [1,2,3])
checks['shared_path_distance_and_RG_trace'] = True

L = np.array([[3,1,1.5],[1,3,2],[1.5,2,3]])
np.testing.assert_allclose(np.linalg.det(L), 11.25)
assert np.linalg.eigvalsh(L).min() > 0
off = sorted([L[0,1],L[0,2],L[1,2]])
assert off[0] != off[1]
checks['convex_constraints_not_sufficient_for_rooted_tree'] = True

Y = np.array([[3,-1],[-1,1]])
V = np.array([[1,1],[0,1]])
I = np.array([[3,2],[-1,0]])
np.testing.assert_allclose(Y@V, I)
np.testing.assert_allclose(I@np.linalg.inv(V), Y)
np.testing.assert_allclose([-8,-6,1] @ np.array([.2,.1,230]),227.8)
checks['phasor_and_projected_regression_rows'] = True

z = np.array([[1,1],[-1,-1]], dtype=float)
C = 1/3
mu = z.sum(axis=1)/3
second = C+mu**2
a_new = (z*mu[:,None]).sum(axis=0)/second.sum()
var_new = (z*z - 2*z*mu[:,None]*a_new + second[:,None]*a_new*a_new).mean(axis=0)
np.testing.assert_allclose(a_new, [6/7,6/7])
np.testing.assert_allclose(var_new, [3/7,3/7])
checks['EM_full_conditional_second_moment'] = True

bic_a, bic_b = 240+5*np.log(100),236+7*np.log(100)
assert bic_a < bic_b
assert np.ceil(10*.95) == 10
assert (1+.75)+.1*(1+.75)**2 == 2.05625
checks['BIC_conformal_and_GN_examples'] = True

result = {'purpose':'Teaching mathematics only; no original paper experiments rerun', 'checks':checks,'all_passed':all(checks.values())}
path = Path(__file__).resolve().parent/'build'/'math_checks.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
