"""Receipt-only independent reconstruction; never invokes an LP or original code."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1])
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def decode(family,beta):
    rootflag,tag=family.split("_")
    supports=[(0,),(1,),(2,)]
    if tag!="star":supports.append(tuple(int(v) for v in tag))
    if rootflag=="root":supports.append((0,1,2))
    p=len(supports)
    assert len(beta)==2*p
    beta=np.asarray(beta)
    K=np.zeros((3,3))
    for j,support in enumerate(supports):
        for i in support:
            for k in support:K[i,k]+=beta[j]+.35*beta[p+j]
    return K
u12=np.array([[.2,-.2],[-.2,.2],[0.,0.]])
u13=np.array([[.2,-.2],[0.,0.],[-.2,.2]])
history=np.array([[1.,2.],[1.,2.],[1.,2.]])
actions=dict(u12=u12,u13=u13,two=np.column_stack((u12,u13)),common=history,
             history_plus_two=np.column_stack((history,u12,u13)))
checked=0;maximum=0.;errors=[]
def test(condition,message):
    global checked
    checked+=1
    if not condition:errors.append(message)
for path in sorted((root/"lp_receipts").glob("*.json")):
    r=load(path)
    test(r["status"]==0 and r["success"],str(path)+" solver status")
    test(r["lower_bounds"]==0. and r["upper_bounds"] is None,str(path)+" unrestricted nonnegative domain")
    beta=r["coefficients"]
    test(min(beta)>=-1e-8,str(path)+" nonnegative solution")
    a=[.3,.4,.5] if r["leaf_profile"]==0 else [.3,0.,.5]
    K=.4*np.ones((3,3))+np.diag(a)
    for i in r["true_pair"]:
        for j in r["true_pair"]:K[i,j]+=r["true_effect"]
    residual=((decode(r["family"],beta)-K)@actions[r["action"]]).ravel()
    difference=max(abs(float(abs(residual).sum())-r["objective"]),float(np.max(abs(residual-r["residual"]))))
    maximum=max(maximum,difference)
    test(difference<1e-8,str(path)+" physical response reconstruction")
for path in sorted((root/"uniform_receipts").glob("*.json")):
    r=load(path)
    nl=2*(3+(r["left"].split("_")[1]!="star")+r["left"].startswith("root_"))
    KL=decode(r["left"],r["coefficients"][:nl]);KR=decode(r["right"],r["coefficients"][nl:])
    response=((KL-KR)@actions[r["action"]]).ravel()
    target=np.array([[1.5,3.],[2.8,5.6],[2.9,5.8]])
    test(r["status"]==0 and r["success"],str(path)+" status")
    test(max(float(np.max(abs(KL@history-target))),float(np.max(abs(KR@history-target))))<1e-8,str(path)+" independent historical feasibility")
    test(float(abs(response).sum())<1e-8 and r["objective"]==0.,str(path)+" shared response distance")
test(len(list((root/"lp_receipts").glob("*.json")))==720,"profile denominator")
test(len(list((root/"uniform_receipts").glob("*.json")))==84,"uniform denominator")
out=dict(status="passed" if not errors else "failed",checks=checked,failed_checks=errors,
         profile_records=720,uniform_records=84,maximum_reconstruction_difference=maximum,
         source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_LP_calls=0,
         no_original_code_imported=True)
(root/"audit_readonly.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out))
