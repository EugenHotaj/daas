# Decisions as a Service (daas)

Efficent AutoML system built on top of [ray](https://github.com/ray-project/ray).

---

Benchmark (OpenML):

| Task | AUC OpenML | AUC Ours | Diff |
| --- | --- | --- | --- |
| [kr-vs-kp](https://www.openml.org/t/3) | .9999 | .9998 | -0.0001 |
| [adult](https://www.openml.org/t/7592) | .929 | .9305 | +0.006 |
| [phoneme](https://www.openml.org/t/9952) | .9674 | .962 | -0.0054 |
| [bank-marketing](https://www.openml.org/t/9977) | .9358 | .938 | +0.0022 |
| [nomao](https://www.openml.org/t/14965) | .9964 | .9965 | +0.0001 |
| [jasmine](https://www.openml.org/t/168911) * | .7497 | .8651 | +0.1154 |

\* NOTE: Likely optimistic results because of too few OpenML runs.
