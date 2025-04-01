# Workshop Automatants @ CentraleSupélec & CeSIA - Mars 2025

**[English version](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/README_EN.md)**

Supports et ressources créés pour le workshop organisé par [Automatants](https://automatants.cs-campus.fr/) et le [CeSIA](https://www.securite-ia.fr/) le 6 mars 2025, qui fait suite au cycle de formation proposé par Automatants et à la conférence du 16 janvier 2025 "[LLMs et Interprétabilité](https://lu.ma/rkkiifsz?tk=QVb9uc)".

L'objectif de ce TP est de manipuler certains concepts de la *[Mechanistic](https://www.neelnanda.io/mechanistic-interpretability/glossary) [Interpretability](https://leonardbereska.github.io/blog/2024/mechinterpreview/)* (*Mech Interp*), notamment dans une perspective de la [sécurité de l'IA](https://ai-safety-atlas.com/chapters/09/).

Il s'agit d'essayer d'interpréter certains neurones d'un modèle "jouet" de langue du type Transformer ayant appris à générer des noms de communes françaises, en utilisant des [auto-encodeurs peu denses](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

Ce TP est articulé en 4 parties :

1. exploration rapide du jeu de données : les noms de communes françaises

| Nom de commune |
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

2. génération de noms avec un modèle de type Transformer (partie décodeur, 1 couche)
![schéma du modèle de langue](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/language_model_details.png)

3. compréhension des neurones cachés, activations, utilisation d’un SEA, interprétation
![schéma d'utilsation d'un SAE](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae_with_model.png)
![schéma du SAE dans le contexte de la Mech Interp](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae.png)

4. contrôle de génération
![schéma du contrôle de génération](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/steering.png)


L'organisation des fichiers :

- `./workshop_part_*.ipynb` : les carnets contenant les énoncés;
- `./workshop_part*__solutions.ipynb` : les carnets contenant les solutions,
- `./utils/` : certains fichiers dans le répertoire `utils` contiennent la factorisation de codes présentés dans les carnets, pour réutilisation dans les parties suivantes (par exemple, les codes pour calculer les statistiques sur le jeu de données);
- `./weights/` : les poids générés durant le TPS, et des poids “solutions” dans le sous-répertoire `./weights/solutions/`;
- `./villes.txt` : ce fichier contient le jeu de données de 36583 noms de communes françaises (source : [Générer des noms de villes et communes françaises](https://github.com/alxndrTL/villes)).


> Créé en adaptant et complétant le projet [Générer des noms de villes et communes françaises](https://github.com/alxndrTL/villes) d'[Alexandre TL](https://www.youtube.com/@alexandretl).
>
> Inspiré des matériels créés pour le programme de formation technique [ARENA](https://github.com/callummcdougall/ARENA_3.0)

