# Proton-therapy

This repository aims to guide the analysis of **proton Bragg peaks** through interactive simulations and provide tools to study Bragg peaks using real data collected from a proton accelerator (cyclotron).
It was prepared for high school students as part of the **Physics Project Days (PPD) 2025**, taking place at thw Université catholique de Louvain (UCLouvain).

## Context

This project is set in the context of **proton therapy**, a form of cancer treatment where high-energy proton beams are accelerated and directed at tumors.  
Unlike X-rays used in standard radiotherapy, protons release most of their energy at a well-defined depth inside the tissue.

- This sharp rise and fall in energy deposition is known as the **Bragg Peak**.  
- By tuning the beam energy, the maximum dose can be precisely delivered to the tumor while minimizing damage to healthy tissue.  

---

## Physics Background

The modeling of the Bragg Peak follows the **Bethe–Bloch equation**, which describes the energy loss of charged particles as they pass through matter.  
Key features include:
- A gradual energy loss as the proton travels through tissue.  
- A sharp maximum (the Bragg Peak) where the proton deposits most of its energy just before coming to rest.  

Understanding and analyzing this behavior is crucial for:
- Optimizing treatment planning in proton therapy.  
- Comparing simulations with **real experimental data** from proton accelerators.  


## Getting Started

### 1. Clone this repository

Open a terminal and run:

```bash
git clone https://github.com/zahraadaher/Proton-therapy.git
cd <Proton-therapy>
```

### 2. Create an environment

If you have Anaconda or Miniconda installed, create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate proton-therapy-sim
```

### 3. Launch Jupyter Notebook

Once the environment is active, run:

```bash
jupyter notebook
```

Then open the tutorial notebook file (Tutorial.ipynb) included in this repo.

