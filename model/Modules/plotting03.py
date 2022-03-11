#BANQUE DE FONCTIONS DE GRAPHIQUES MATPLOTLIB/SEABORN - Vincent Salas

# import des librairies dont nous aurons besoin
import seaborn as sns
sns.set()


#-------------------------------------------------------------------------------------------------------------------------

# Graphique Joint-grid bi-varié:

# transparence between 0 and 1

def bi_joint_plot(dataframe, variable_x, variable_y, taille, transparence):
    """Graphique entre 2 variable avec mise en évidence de la densité de points (transparence à choisir). Graphiques univ&riés en histogramme sur les côtés"""
    sns.set_theme(style="white")

    # Initialisatiuon de la figure:
    g = sns.JointGrid(data=dataframe, height=taille, x=variable_x, y=variable_y, marginal_ticks=True)
    g.set_axis_labels(variable_x, variable_y, fontsize=16)
    
    
    # Légende de couleur avec graduation:
    cax = g.fig.add_axes([.15, .55, .02, .2])

    # Ajoute la densité de valeur:
    g.plot_joint(sns.histplot, discrete=(True, False), pmax=transparence, cbar=True, cbar_ax=cax)

    # Graphique univarié sur le côté:
    g.plot_marginals(sns.histplot, element="step")
    
    return g


#-------------------------------------------------------------------------------------------------------------------------
# scatterplot

# Définir la palette de hue au préalable avec le nombre de couleurs
def multi_scatterplot(dataframe, taille, variable_x, variable_y, variable_hue, palette, transparence):
    """Scatter plot à 3 variables (dont 1 en couleurs)"""
    
    # Configuration
    sns.set_theme(style="white")
    sns.color_palette()

    # Plot
    g = sns.relplot(x=variable_x, y=variable_y, hue=variable_hue, alpha=transparence, palette=palette,
                height=taille, data=dataframe)
    
    # Légendes
    g.set_axis_labels(variable_x, variable_y, fontsize=12)
    
    return g


#-------------------------------------------------------------------------------------------------------------------------

# scatterplot (using kdeplot()) on the marginal axes)

# Définir la palette de hue au préalable avec le nombre de couleurs
def multi_and_uni_joint_plot(dataframe, taille, variable_x, variable_y, variable_hue, palette):
    """Scatter plot à 3 variables (dont 1 en couleurs), avec graphe univarié sur les côtés en courbe"""
    
    # Configuration
    sns.set_theme(style="white")
    sns.color_palette()
    
    # Plot
    g = sns.jointplot(data=dataframe, height=taille, x=variable_x, y=variable_y, hue=variable_hue, palette=palette)
    
    # Légendes
    g.set_axis_labels(variable_x, variable_y, fontsize=16)
    
    return g
   