# Workshop Automatants @ CentraleSupélec & CeSIA - March 2025

**[French version](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/README.md)**

Materials and resources created for the workshop organised by [Automatants](https://automatants.cs-campus.fr/) and [CeSIA](https://www.securite-ia.fr/) on March 6th 2025, which follows on from the training offered by Automatants and the conference of January 16th 2025 ‘[LLMs and Interpretability](https://lu.ma/rkkiifsz?tk=QVb9uc)’.

The aim of this tutorial is to manipulate some concepts of *[Mechanistic](https://www.neelnanda.io/mechanistic-interpretability/glossary) [Interpretability](https://leonardbereska.github.io/blog/2024/mechinterpreview/)* (*Mech Interp*), in particular from the perspective of [AI safety](https://ai-safety-atlas.com/chapters/09/).

We'll try to interpret certain neurons in a ‘toy’ language model based on Transformer, that was trained to generate the names of French cities, using [sparse auto-encoders](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

This TP is divided into 4 parts:

1. quick exploration of the dataset: names of French cities

| Name |
|---------|
| arbignieu	|
| mouilleron	|
| tsingoni	|
| upaix	|
| margès	|
| reutenbourg	|
| prades-salars	|
| rouffignac-de-sigoulès	|
| andelarre	|
| montrouveau	|
| curchy	|
| gréolières	|
| reims	|
| florentia	|
| grandfontaine	|
| esclainvillers	|
| vraux	|
| knutange	|
| allan	|
| oltingue	|

2. name generation with a Transformer model (decoder part, 1 layer)
![language model schema](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/language_model_details.png)

3. understanding hidden neurons, activations, use of an AES, interpretation
![diagram of SAE use](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae_with_model.png)
[diagram of the SAE in the context of Mech Interp](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae.png)

4. steered generation
![diagram of steered generation](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/steering.png)


Organisation of the files :

- `./workshop_part_*.ipynb`: notebooks containing the statements;
- `./workshop_part*__solutions.ipynb`: notebooks containing solutions,
- `./utils/`: some files in the `utils` directory contain the factorisation of codes presented in the notebooks, for reuse in the following parts (for example, the codes for calculating the statistics on the dataset);
- `./weights/`: weights generated during the workshop, and ‘solution’ weights in the subdirectory `./weights/solutions/`;
- `./villes.txt`: this file contains the dataset of 36583 names of French cities (source: [Générer des noms de villes et communes françaises](https://github.com/alxndrTL/villes)).


> Created by adapting and completing the project [Generate names of French towns and communes](https://github.com/alxndrTL/villes) by [Alexandre TL](https://www.youtube.com/@alexandretl).
>
> Inspired by materials created for the [ARENA technical training programme](https://github.com/callummcdougall/ARENA_3.0)
