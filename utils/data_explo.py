
__version__ = "1.0.0"

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

pd.set_option('display.max_rows', None) 
sns.set_theme(style="whitegrid")

def freq_distribution(
        serie: pd.Series,
        xlabel: str = 'Longueur des chaînes de caractères',
        ylabel: str = 'Fréquence',
        title: str = 'Distribution de la longueur des chaînes de caractères',
        palette: str = "coolwarm"
        ) -> pd.Series:
    """
    Affichage du graphique (historigramme) de la distribution des fréquences des
    valeurs fournies dans 
    Args :
        len_serie (pd.Series) :
        xlabel
        ylabel
        title
        palette

    """
    # Calculer la longueur des chaînes de caractères dans la colonne "nom"
    lengths = serie.apply(len)

    # Afficher la distribution de la longueur des chaînes de caractères
    distribution = lengths.value_counts().sort_index()
    #print(length_distribution)

    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=distribution.index,
        y=distribution.values,
        hue=distribution.values,
        palette=palette
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.xticks(rotation=90)
    plt.show()

    return lengths


def freq_char(serie: pd.Series) -> None:
    """
    """
    # Concaténer toutes les chaînes de caractères de la colonne "nom"
    all_chars = ''.join(serie)

    # Compter les occurrences de chaque caractère
    char_counts = Counter(all_chars)

    # Convertir le résultat en dataframe pour une meilleure lisibilité
    char_freq_df = pd.DataFrame(
        char_counts.items(), columns=['Caractère', 'Fréquence']
        ).sort_values(by='Fréquence', ascending=False)

    print("Nombre de caractères distincts :", len(char_freq_df))

    # If not in a Jupyter notebook
    if display == None:
        print(char_freq_df)
    else:
        display(char_freq_df)


def rate_freq_by_position(
        serie: pd.Series,
        trunc_length: int = 10,
        topN: int = 15
        ) -> None:
    """
    Affiche sous forme de carte de chaleur les taux de fréquence des topN 
    caractères les plus fréquemment présents dans les trunc_length premières
    positions des chaînes de caractères présentes dans serie.
    Args :
        serie (pd.Series) :
        trunc_length (int) :
        topN (int) :
    """
    # Limiter la longueur des noms à trunc_length caractères
    truncated = serie.str[:trunc_length]

    # Initialiser un dictionnaire pour stocker les fréquences des caractères par position
    position_char_freq = {i: Counter() for i in range(trunc_length)}

    # Remplir le dictionnaire avec les fréquences des caractères par position
    for name in truncated:
        for i, char in enumerate(name):
            position_char_freq[i][char] += 1

    # Convertir le dictionnaire en dataframe pour une meilleure lisibilité
    position_char_freq_df = pd.DataFrame(position_char_freq).fillna(0).astype(int)

    # Limiter aux topN premiers caractères les plus fréquents
    top_chars = position_char_freq_df.sum(axis=1).sort_values(ascending=False).head(topN).index
    position_char_freq_df = position_char_freq_df.loc[top_chars]

    # Calculer le taux de présence par position
    position_char_rate_df = position_char_freq_df.div(position_char_freq_df.sum(axis=0), axis=1) * 100

    # Visualiser les taux de présence avec une carte de chaleur
    plt.figure(figsize=(12, 8))
    sns.heatmap(position_char_rate_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.xlabel('Position')
    plt.ylabel('Caractère')
    plt.title('Taux de présence des caractères par position (limité aux trunc_length premières positions)')
    plt.yticks(rotation=0)
    plt.show()


Default_Element_Separator = "-' "
def freq_composition_element(
        serie: pd.Series,
        element_separator: str = Default_Element_Separator
        ) -> None:
    all_elements = ' '.join(serie.str.replace(f"[{element_separator}]", " ", regex=True).values).split()

    # Compter les occurrences de chaque élément
    element_counts = Counter(all_elements)

    # Convertir le résultat en dataframe pour une meilleure lisibilité
    element_freq_df = pd.DataFrame(
        element_counts.items(),
        columns=['Élément', 'Fréquence']
        ).sort_values(by='Fréquence', ascending=False)

    print(f"Nombre total de composants distincts : {len(element_freq_df)}")
    display(element_freq_df.sample(15).sort_values(by='Fréquence', ascending=False))

    # Filtrer les éléments dont la fréquence est supérieure à 1
    element_freq_sup_1_df = element_freq_df[element_freq_df['Fréquence'] > 1]

    # Afficher le nombre total d'éléments associés
    print(f"Nombre total de composants présents plus d'une fois : {len(element_freq_sup_1_df)}")
    display(element_freq_sup_1_df.head(15))


def composition_distribution(
        serie: pd.Series,
        element_separator: str = Default_Element_Separator,
        xlabel: str = 'Nombre de composants',
        ylabel: str = 'Fréquence',
        title: str = 'Distribution de la fréquence du nombre de composants',
        palette: str ="coolwarm"
        ) -> None:
    # Calculer le nombre de composants pour chaque nom
    num_components = serie.apply(lambda x: len([comp for comp in x if comp in element_separator]) + 1)

    # Afficher la distribution du taux de fréquence du nombre de composants
    component_distribution = num_components.value_counts().sort_index()
    component_distribution_rate = component_distribution / len(serie) * 100
    display(component_distribution.to_frame())

    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(x=component_distribution_rate.index, y=component_distribution_rate.values, hue=component_distribution_rate.values, palette=palette)
    for i in range(len(component_distribution_rate)):
        plt.text(i, component_distribution_rate.values[i] + 0.5, f'{component_distribution_rate.values[i]:.3f}%', ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.xticks(rotation=90)
    plt.show()