# Decisions as a Service (daas)

Efficent AutoML system built on top of [ray](https://github.com/ray-project/ray).

---

Benchmark (OpenML):

| Task | AUC OpenML | AUC Ours | Diff |
| --- | --- | --- | --- |
| [adult](https://www.openml.org/t/7592) | .929 | .93 | +0.001 |
| [nomao](https://www.openml.org/t/9977) | .9964 | .9963 | -0.0001 |
| [phoneme](https://www.openml.org/t/9952) | .9674 | .9612 | -0.0062 |
