# Unified Optimization and Hybrid Learning Framework for Cold-Start Mitigation in Serverless Computing

## 📌 Overview
This repository presents a **Unified Cold-Start Optimization Model (UCSOM)** and a **Hybrid Transformer–Reinforcement Learning (HTRL) Scheduler** for mitigating cold-start latency in serverless computing.

Cold-start latency is a major bottleneck in Function-as-a-Service (FaaS) platforms due to container initialization delays. This work formulates cold-start mitigation as a **constrained multi-objective stochastic optimization problem**, integrating:

- Latency
- Operational Cost
- Energy Consumption

To enable real-time adaptability under dynamic workloads, a **Transformer-based workload predictor** is combined with a **Reinforcement Learning (RL) scheduler**.

---

## 🎯 Key Contributions
- ✅ Unified mathematical formulation of cold-start mitigation (UCSOM)
- ✅ Stochastic modeling of request arrivals and container lifecycle
- ✅ Multi-objective optimization (latency–cost–energy tradeoff)
- ✅ Hybrid Transformer–RL scheduler for real-time decision making
- ✅ Experimental validation on AWS Lambda with real workload traces

---

## 🏗️ Project Structure

.
├── paper/                 # Research paper
├── code/                  # Model + scheduler implementation
├── data/                  # Workload traces / logs
├── results/               # Experimental outputs
├── README.md
└── requirements.txt

