# Workshop Automatants @ CentraleSupélec & CeSIA - Mars 2025

Supports et ressources créés pour le workshop organisé par [Automatants](https://automatants.cs-campus.fr/) et le [CeSIA](https://www.securite-ia.fr/) le 6 mars 2025.

L'objectif de ce TP est de manipuler certains concepts de la *[Mechanistic](https://www.neelnanda.io/mechanistic-interpretability/glossary) [Interpretability](https://arxiv.org/abs/2404.14082)* (*Mech Interp*), notamment dans une perspective de la [sécurité de l'IA](https://ai-safety-atlas.com/chapters/09/).

Il s'agit d'essayer d'interpréter certains neurones d'un modèle "jouet" de langue du type Transfert ayant appris à générer des noms de communes françaises, en utilisant des auto-encodeuts peu denses.

Ce TP est articulé en 4 parties : 
1. exploration rapide du jeu de données : les noms de communes françaises

|Nom |
|---–|
|arbignieu	|
|mouilleron	|
|tsingoni	|
|upaix	|
|margès	|
|reutenbourg	|
|prades-salars	|
|rouffignac-de-sigoulès	|
|andelarre	|
|montrouveau	|
|curchy	|
|gréolières	|
|reims	|
|florentia	|
|grandfontaine	|
|esclainvillers	|
|vraux	|
|knutange	|
|allan	|
|oltingue	|

2. génération de noms avec un modèle de type Transformer (partie décodeur, 1 couche)
![schéma du modèle de langue](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/language_model_details.png)

3. compréhension des neurones cachés, activations, utilisation d’un SEA, interprétation
![](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae_with_model.png)
![schéma du SAE dans le contexte de la Mech Interp](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/sae.png)

4. contrôle de génération.
![schéma du contrôle de génération](https://github.com/nicolasguillard/workshop_cs_202503/blob/main/images/steering.png)


L'organisation des fichiers :

- `./workshop_part_*.ipynb` : les carnets contenant les énoncés;
- `./workshop_part*__solutions.ipynb` : les carnets contenant les solutions,
- `./utils/` : certains fichiers dans le répertoire utils contiennent la factorisation de certains codes présentés dans les carnets, pour réutilisation dans les parties suivantes (par exemple, les codes pour calculer les statistiques sur le jeu de données);
- `./weights/` : les poids générés durant le TPS, et des poids “solutions”;
- `./villes.txt` : ce fichier contient le jeu de données de 36583 noms de communes françaises (source : [Générer des noms de villes et communes françaises](https://github.com/alxndrTL/villes)).

Créé en adaptant et complétant le projet [Générer des noms de villes et communes françaises](https://github.com/alxndrTL/villes) par [Alexandre TL](https://www.youtube.com/@alexandretl).