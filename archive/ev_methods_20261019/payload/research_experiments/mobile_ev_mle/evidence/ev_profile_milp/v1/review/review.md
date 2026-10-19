# Continuous-profile MILP independent review

Completed 21 repeated conditions from 3 independent base seeds.
Replayed 528 saved fit states and 84 actual future responses. No new fits or selectors.
Selection fallbacks: 0; H ambiguity activation: 18.

|Covered|Method|Complete / conditions|Correct winner|Mean AC margin|Quality-LP match|
|---|---|---:|---:|---:|---:|
|True|none|15 / 15|5|2.34743|0|
|True|constant|15 / 15|12|343.398|0|
|True|profile_milp|15 / 15|14|1204.53|0|
|True|fixed_mle_milp|15 / 15|13|1405.24|15|
|False|none|6 / 6|0|-4.36673|0|
|False|constant|6 / 6|3|-2.24895|0|
|False|profile_milp|6 / 6|2|-34.6734|0|
|False|fixed_mle_milp|6 / 6|3|-23.0556|6|

Correct winner means smallest attained AC loss, without a calibrated rejection threshold. Excluded-truth conditions are model-misspecification diagnostics.

|Seed|Truth R|Fit floor|Covered|NoEV M|Constant M|Profile M|FixedMLE M|Profile minus constant|Fallback|
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
|986101|0.0003|0.001|False|-0.513962|2.66255|-2.01048|4.57601|-4.67303|False|
|986101|0.0003|0.004|False|-4.30845|-49.558|-255.651|-238.89|-206.093|False|
|986101|0.0003|1e-06|True|-0.542071|3.35725|2.40666|7.92575|-0.950589|False|
|986101|0.003|0.001|True|0.925557|61.3249|199.069|223.371|137.744|False|
|986101|0.003|0.004|False|-0.0213465|58.5944|171.404|212.844|112.809|False|
|986101|0.003|1e-06|True|0.604715|61.7334|184.837|223.801|123.103|False|
|986101|0.012|0.001|True|22.0512|797.256|2818.15|3260.05|2020.9|False|
|986101|0.012|0.004|True|23.5196|797.083|2784.92|3259.88|1987.84|False|
|986101|0.012|1e-06|True|21.2521|797.256|2784.93|3260.05|1987.67|False|
|986102|0.0003|0.001|False|-6.63887|-0.951004|-20.9046|-11.3376|-19.9536|False|
|986102|0.0003|0.004|False|-7.73963|-65.9338|-306.112|-272.049|-240.178|False|
|986102|0.0003|1e-06|True|-6.41533|-0.837716|5.42108|-0.228733|6.2588|False|
|986102|0.003|0.001|True|-6.43822|46.2037|151.257|188.449|105.053|False|
|986102|0.003|0.004|False|-6.97812|41.6921|205.234|166.523|163.542|False|
|986102|0.003|1e-06|True|-6.39041|46.2037|151.257|189.605|105.053|False|
|986102|0.012|0.001|True|-4.1978|848.587|2994.61|3484.45|2146.03|False|
|986102|0.012|0.004|True|-4.1978|848.587|2994.61|3500.87|2146.03|False|
|986102|0.012|1e-06|True|-4.1978|847.634|2994.61|3479.71|2146.98|False|
|986103|0.0|0.001|True|-0.214014|-2.39303|0.395139|0.269313|2.78817|False|
|986103|0.0|0.004|True|-0.205741|1.76195|2.44958|2.03946|0.687628|False|
|986103|0.0|1e-06|True|-0.342679|-2.78187|-0.936533|-1.69106|1.84534|False|

H-selected pair excludes truth in 6 conditions; all four candidates still enter final AC refits.
Nested-floor checks: 224; attained-local-fit monotonicity violations: 1.
Fixed-parameter source forward Taylor relative errors: [0.002674416334412037, 0.005537466117904699]. Nonlinear competing-mean refit comparisons: 54 (first base seed only).
Small source forward error does not measure the approximation error after a competitor changes its shared lines and states. The selected pair can also omit the true candidate; these are separate diagnostic questions.
Quality-LP match means the action attains the best public repeat-count linear score at the same site and energy; it is not a separate fitted strategy.
The JSON retains per-condition proxy scores, bounded/free discrepancies, nonlinear predicted margins, Taylor errors and MILP gaps. No global nonlinear AC certificate is asserted.
