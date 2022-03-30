# 🏖 Automate boring decisions with ML

Decisions as a Service (DaaS) is an efficent AutoML system built on top of [ray](https://github.com/ray-project/ray).



## 📈 Benchmarks (OpenML)

| Task | AUC OpenML | AUC Ours | Diff |
| --- | --- | --- | --- |
| [kr-vs-kp](https://www.openml.org/t/3) | 0.9999 | 0.9998 | -0.0001 |
| [adult](https://www.openml.org/t/7592) | 0.9290 | 0.9305 | +0.006 |
| [phoneme](https://www.openml.org/t/9952) | 0.9674 | 0.9620 | -0.0054 |
| [nomao](https://www.openml.org/t/9977) | 0.9964 | 0.9965 | +0.0001 |
| [bank-marketing](https://www.openml.org/t/14965) | 0.9358 | 0.9380  | +0.0022  |
| [higgs](https://www.openml.org/t/146606) | 0.8031 | 0.8099  | +0.0068  |
| [jasmine](https://www.openml.org/t/168911) * | 0.7497 | 0.8651 | +0.1154 |

\* NOTE: Likely optimistic results because of too few OpenML runs.
