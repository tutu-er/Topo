from research_experiments.lp_rebuild.paper_cases import selected_jobs
from research_experiments.rebuild_benchmark.run import catalog


def test_replay_selects_exactly_the_four_network_main_and_stress_inputs():
    dev,evaluation=catalog();metadata={}
    for job in evaluation:
        panel,case,bundle,condition,noise,N=job.split('__')
        metadata[job]=dict(panel=panel,case=case)
    selected=selected_jobs(dict(evaluation=evaluation),metadata)
    assert len(selected)==120 and not set(selected)&set(dev)
    assert sum(metadata[j]['panel']=='main' for j in selected)==108
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        assert sum(metadata[j]['case']==case for j in selected)==30
