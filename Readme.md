
<h1 style="color:blue">Thèses et mémoires du Québec</h1>
<h3>Quelques données sur les doctorats et maîtrises publiés dans les universités québécoises depuis les années 1990</h3>

<hr>

<h4 style="color:blue">*Data on theses and dissertations published in Québec universities in the last 25 years*</h4>


```python
%matplotlib inline
import csv, re, random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats, integrate
import seaborn as sns
```

-----
On commence par demander à **pandas** d'avaler le fichier CSV qui contient toutes nos données et de les placer dans la variable `theses`.


```python
theses = pd.read_csv("THESES-TOTAL.csv")
```

-----
On fait ensuite, toujours grâce à pandas, une première analyse rapide du nombre de pages selon le type de document (doctorat ou maîtrise). Tout de suite, on a:

- le nombre total de documents dans chacun de ces deux types
- les valeurs extrêmes (min et max)
- la moyenne et la médiane (50%)
- une idée de la distribution du nombre de pages par quartile


```python
parType = theses.groupby("type").nbPages
parType.describe()
```




    type           
    doctorat  count    15100.000000
              mean       251.299007
              std        116.664381
              min         36.000000
              25%        172.000000
              50%        226.000000
              75%        302.000000
              max       1578.000000
    maîtrise  count    40485.000000
              mean       133.284179
              std         54.082249
              min         19.000000
              25%         98.000000
              50%        124.000000
              75%        157.000000
              max        744.000000
    dtype: float64



-----
On demande ensuite à **matplotlib** de faire un premier graphique. C'est un histogramme de la distribution du nombre de pages des maîtrises et doctorats par tranche de 10 pages, puisqu'on fait 50 colonnes (*«bins»*) dans un intervalle (*«range»*) qui va de 10 à 510. Le paramètre `alpha` indique que les colonnes auront une transparence de 50%.


```python
parType.hist(bins=50,histtype="bar",range=(10,510), alpha=0.5)
plt.legend(["Doctorats","Maîtrises"])
plt.title("Distribution des thèses et mémoires\npar nombre de pages",size=20)
```




    <matplotlib.text.Text at 0x1162c2550>




![png](theses_files/theses_7_1.png)


-----

## Les doctorats

-----

Commençons par analyser seulement les doctorats en les regroupant tous dans une variable du même nom. Pandas peut nous décrire quelques-unes des données contenues dans ce sous-ensemble:

- l'année à laquelle le doctorat a été déposé
- son nombre de pages
- la longueur de son titre


```python
doctorats = theses.query("type == 'doctorat'")
doctorats.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annee</th>
      <th>octets</th>
      <th>nbPages</th>
      <th>longTitre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15100.00000</td>
      <td>1.403700e+04</td>
      <td>15100.000000</td>
      <td>15100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.44457</td>
      <td>1.757349e+07</td>
      <td>251.299007</td>
      <td>107.013046</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.03048</td>
      <td>2.936342e+07</td>
      <td>116.664381</td>
      <td>41.184933</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.00000</td>
      <td>5.946900e+04</td>
      <td>36.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2006.00000</td>
      <td>3.945311e+06</td>
      <td>172.000000</td>
      <td>77.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010.00000</td>
      <td>7.671272e+06</td>
      <td>226.000000</td>
      <td>102.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.00000</td>
      <td>1.641564e+07</td>
      <td>302.000000</td>
      <td>131.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.00000</td>
      <td>5.581489e+08</td>
      <td>1578.000000</td>
      <td>378.000000</td>
    </tr>
  </tbody>
</table>
</div>



-----
C'est bien d'avoir des données sur l'ensemble des doctorats au Québec. Mais je suis curieux de savoir comment le nombre de pages de ces doctorats varie en fonction de l'université dans laquelle ils ont été réalisés.

Pandas nous aide ici encore en permettant d'effectuer un regroupement par université que j'ai placé dans une variable appelée `doctoratsUniv`.


```python
doctoratsUniv = doctorats.groupby("universite")
doctoratsUniv.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>annee</th>
      <th>longTitre</th>
      <th>nbPages</th>
      <th>octets</th>
    </tr>
    <tr>
      <th>universite</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Concordia</th>
      <th>count</th>
      <td>2079.000000</td>
      <td>2079.000000</td>
      <td>2079.000000</td>
      <td>2.057000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2005.673401</td>
      <td>89.589707</td>
      <td>230.285233</td>
      <td>8.462193e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.269765</td>
      <td>34.370518</td>
      <td>96.823321</td>
      <td>8.644034e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>14.000000</td>
      <td>61.000000</td>
      <td>3.519070e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2000.000000</td>
      <td>65.000000</td>
      <td>162.000000</td>
      <td>4.037521e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2008.000000</td>
      <td>85.000000</td>
      <td>210.000000</td>
      <td>6.642033e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2012.000000</td>
      <td>108.000000</td>
      <td>276.000000</td>
      <td>1.041637e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>261.000000</td>
      <td>1112.000000</td>
      <td>1.678755e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">HEC Montréal</th>
      <th>count</th>
      <td>286.000000</td>
      <td>286.000000</td>
      <td>286.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2006.437063</td>
      <td>92.804196</td>
      <td>258.132867</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.136565</td>
      <td>37.508784</td>
      <td>145.435618</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>26.000000</td>
      <td>61.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2003.000000</td>
      <td>62.250000</td>
      <td>152.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.000000</td>
      <td>89.000000</td>
      <td>220.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2012.000000</td>
      <td>118.000000</td>
      <td>337.250000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>212.000000</td>
      <td>1430.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">INRS</th>
      <th>count</th>
      <td>499.000000</td>
      <td>499.000000</td>
      <td>499.000000</td>
      <td>4.730000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.030060</td>
      <td>122.625251</td>
      <td>246.112224</td>
      <td>2.360549e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.860930</td>
      <td>40.343130</td>
      <td>88.203599</td>
      <td>3.143506e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>37.000000</td>
      <td>87.000000</td>
      <td>6.946860e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2005.000000</td>
      <td>95.000000</td>
      <td>182.500000</td>
      <td>4.665799e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>118.000000</td>
      <td>231.000000</td>
      <td>1.036514e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.000000</td>
      <td>146.500000</td>
      <td>291.500000</td>
      <td>2.675130e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>303.000000</td>
      <td>682.000000</td>
      <td>2.037039e+08</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">McGill</th>
      <th>count</th>
      <td>1928.000000</td>
      <td>1928.000000</td>
      <td>1928.000000</td>
      <td>1.928000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2004.750000</td>
      <td>89.725104</td>
      <td>246.012967</td>
      <td>1.070978e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.590502</td>
      <td>33.769375</td>
      <td>104.195439</td>
      <td>1.170829e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>16.000000</td>
      <td>39.000000</td>
      <td>4.112780e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1998.000000</td>
      <td>65.000000</td>
      <td>182.000000</td>
      <td>4.443214e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2006.000000</td>
      <td>85.000000</td>
      <td>225.000000</td>
      <td>7.968924e+06</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Université Laval</th>
      <th>std</th>
      <td>3.234236</td>
      <td>41.840826</td>
      <td>120.205734</td>
      <td>4.534710e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2002.000000</td>
      <td>14.000000</td>
      <td>36.000000</td>
      <td>4.692270e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>84.000000</td>
      <td>174.000000</td>
      <td>5.304797e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2011.000000</td>
      <td>109.000000</td>
      <td>232.000000</td>
      <td>2.040858e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.000000</td>
      <td>137.000000</td>
      <td>310.750000</td>
      <td>5.390030e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>378.000000</td>
      <td>1478.000000</td>
      <td>5.581489e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Université de Montréal</th>
      <th>count</th>
      <td>2641.000000</td>
      <td>2641.000000</td>
      <td>2641.000000</td>
      <td>2.641000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2010.151079</td>
      <td>109.820144</td>
      <td>283.244983</td>
      <td>9.042001e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.687238</td>
      <td>38.697944</td>
      <td>121.221355</td>
      <td>1.797919e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>21.000000</td>
      <td>49.000000</td>
      <td>1.110980e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>81.000000</td>
      <td>202.000000</td>
      <td>2.180442e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010.000000</td>
      <td>107.000000</td>
      <td>260.000000</td>
      <td>4.277027e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.000000</td>
      <td>135.000000</td>
      <td>338.000000</td>
      <td>8.554617e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>259.000000</td>
      <td>1578.000000</td>
      <td>3.339740e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Université de Sherbrooke</th>
      <th>count</th>
      <td>1724.000000</td>
      <td>1724.000000</td>
      <td>1724.000000</td>
      <td>1.724000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.096868</td>
      <td>117.897332</td>
      <td>234.301044</td>
      <td>1.658168e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.812250</td>
      <td>43.160102</td>
      <td>105.855871</td>
      <td>3.248490e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>21.000000</td>
      <td>46.000000</td>
      <td>5.946900e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2004.000000</td>
      <td>88.000000</td>
      <td>163.000000</td>
      <td>5.499002e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>111.000000</td>
      <td>208.000000</td>
      <td>8.366110e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.000000</td>
      <td>142.000000</td>
      <td>276.000000</td>
      <td>1.283679e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>351.000000</td>
      <td>826.000000</td>
      <td>3.400589e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">École Polytechnique</th>
      <th>count</th>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>5.760000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.612847</td>
      <td>96.097222</td>
      <td>201.854167</td>
      <td>8.199000e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.856910</td>
      <td>31.753029</td>
      <td>74.976446</td>
      <td>1.215289e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2009.000000</td>
      <td>25.000000</td>
      <td>84.000000</td>
      <td>5.815810e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2011.000000</td>
      <td>73.000000</td>
      <td>151.000000</td>
      <td>2.738251e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2013.000000</td>
      <td>93.000000</td>
      <td>186.000000</td>
      <td>5.302887e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.000000</td>
      <td>115.250000</td>
      <td>233.000000</td>
      <td>9.444866e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>200.000000</td>
      <td>633.000000</td>
      <td>2.049431e+08</td>
    </tr>
  </tbody>
</table>
<p>104 rows × 4 columns</p>
</div>



-----
Classons maintenant les universités en fonction du nombre médian de pages de leurs doctorats, exercice intéressant en soi qui sera également utile pour l'étape suivante.


```python
medianesDoctoratsUniv = doctoratsUniv["nbPages"].median().sort_values(ascending=False)
medianesDoctoratsUniv
```




    universite
    UQAM                        269.0
    Université de Montréal      260.0
    UQAC                        243.0
    UQO                         238.5
    Université Laval            232.0
    INRS                        231.0
    McGill                      225.0
    UQAT                        220.5
    HEC Montréal                220.5
    Concordia                   210.0
    Université de Sherbrooke    208.0
    École Polytechnique         186.0
    UQTR                        153.0
    Name: nbPages, dtype: float64



-----
Pour illustrer la distribution du nombre de pages des doctorats par université, le meilleur type de graphique est peut-être le *box&nbsp;plot*, qu'on peut traduire par **diagramme de quartiles**... ou ce que les Français ont baptisé des [boîtes à moustaches](https://fr.wikipedia.org/wiki/Bo%C3%AEte_%C3%A0_moustaches).

![](http://www.statcan.gc.ca/edu/power-pouvoir/ch12/img/5214889_02-fra.gif)

Ces boîtes permettent d'afficher une distribution qui a été découpée en quartiles.<br>
Les deuxième et troisième quartiles, ceux qui se trouvent de part et d'autre de la médiane, sont représentés par deux rectangles, les boîtes.<br>
Les premier et dernier quartiles sont, quant à eux, représentés par des lignes, les moustaches.<br>
Des valeurs excentriques peuvent enfin être représentés par des points à gauche ou à droite des lignes.

J'ai essayé d'utiliser le [langage R](https://cran.r-project.org/doc/contrib/Goulet_introduction_programmation_R.pdf) pour en produire, comme l'a fait [Markus Beck](https://beckmw.wordpress.com/2014/07/15/average-dissertation-and-thesis-length-take-two/). Mais j'ai été incapable d'arriver à des résultats satisfaisants.

Après avoir essayé les librairies [matplotlib](http://matplotlib.org/) et [bokeh](http://bokeh.pydata.org/en/latest/), j'ai trouvé [seaborn](https://stanford.edu/~mwaskom/software/seaborn/) beaucoup plus facile à utiliser.<br>
Le code ci-dessous est relativement facile à comprendre. Matplotlib contrôle la taille finale du graphique au moyen de la méthode `plt.figure`, puis seaborn fait tout le reste. On remarque notamment que la méthode `sns.boxplot` a un paramètre qui permet d'ordonner nos boîtes (`order`), paramètre que j'ai alimenté avec l'index de la variable `medianesDoctoratsUniv` créé juste ci-dessus.

Au final, on a une idée de la distribution du nombre de pages des doctorats, par université. C'est à l'UQAM que les professeurs travaillent le plus (si on se fie au nombre de pages qu'ils doivent lire)... ou le moins (si on se fie au nombre de pages qu'ils laissent leurs doctorants écrire).


```python
sns.set()
plt.figure(figsize=(10, 15))
sns.set_style("darkgrid", {
        "axes.facecolor": "lightskyblue",
        "font.family": [u"Bitstream Vera Sans"]
    })
couleurs = sns.light_palette("turquoise", n_colors=13, reverse=True)
boiteDoc = sns.boxplot(y="universite",
                       x="nbPages",
                       data=doctorats,
                       palette=couleurs,
                       order=medianesDoctoratsUniv.index
                      )
boiteDoc.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
boiteDoc.grid(b=True, which='major', color='w', linewidth=2.0)
boiteDoc.grid(b=True, which='minor', color='w', linewidth=0.5)
boiteDoc.set(ylabel="Université",
             xlabel="Distribution du nombre de pages",
             xlim=(0,750),
             title="Nombre de pages des doctorats par université\n"
            )
```




    [<matplotlib.text.Text at 0x11622f320>,
     (0, 750),
     <matplotlib.text.Text at 0x11671b320>,
     <matplotlib.text.Text at 0x11956bb38>]




![png](theses_files/theses_15_1.png)


-----
On peut aussi créer un autre graphique à moustaches intéressant en regroupant les doctorats par discipline, plutôt que par université.<br>
On commence par créer une variable qui va nous permetter d'ordonner les disciplines en fonction de la médiane du nombre de pages (`medianesDoctoratsDiscipline`).


```python
doctoratsDiscipline = doctorats.groupby("discipline")
medianesDoctoratsDiscipline = doctoratsDiscipline["nbPages"].median().sort_values(ascending=False)
medianesDoctoratsDiscipline
```




    discipline
    Droit                                 459.0
    Relations industrielles               420.0
    Histoire                              412.0
    Anthropologie                         407.5
    Traduction                            394.5
    Études classiques                     392.5
    Études françaises                     372.0
    Cinéma                                366.0
    Histoire de l'art                     364.0
    Littérature                           355.0
    Sociologie                            350.0
    Science politique                     346.0
    Religion/théologie                    340.5
    Pédagogie                             336.0
    Communication                         335.0
    Aménagement/urbanisme                 330.0
    Gérontologie                          329.0
    Études islamiques                     328.5
    Philosophie                           324.0
    Sciences humaines générales           315.0
    Administration publique               314.0
    Études hispaniques                    313.0
    Linguistique                          308.0
    Sciences de l'information             300.0
    Travail social                        294.5
    Géographie                            293.0
    Nutrition                             291.5
    Arts visuels                          289.0
    Santé publique et communautaire       287.0
    Éducation                             286.0
                                          ...  
    Génie minier                          220.0
    Orientation                           217.0
    Génie de l'environnement              216.5
    Télécommunications                    207.0
    Génie chimique                        204.0
    Génie industriel                      202.0
    Biotechnologie agricole               201.0
    Comptabilité                          200.0
    Génie mécanique                       199.0
    Génie physique                        199.0
    Agriculture et pêcheries              198.5
    Musique                               197.0
    Ingénierie biomédicale                196.5
    Sciences de l'environnement           194.0
    Biologie                              190.0
    Radiologie et imagerie biomédicale    188.0
    Marketing                             187.5
    Géomatique et télédétection           187.0
    Informatique                          185.0
    Sylviculture/foresterie               184.5
    Météorologie                          182.5
    Physique                              178.0
    Génie électrique                      178.0
    Génie logiciel                        173.0
    Médecine dentaire                     171.0
    Psychologie                           166.0
    Économie                              157.0
    Finance                               156.5
    Statistique                           152.0
    Mathématiques                         133.5
    Name: nbPages, dtype: float64



-----
Puis on fait un diagramme par quartiles de la même façon qu'on vient de le faire avec les universités.<br>
Le résultat est un graphique complexe, mais riche en information. C'est en droit que les doctorants sont le plus prolixes, alors que les mathématiciens et statisticiens ont davantage l'habitude d'être *right to the point*.


```python
sns.set()
sns.set_context("poster")
sns.set(font_scale=2)
plt.figure(figsize=(20, 35))
sns.set_style("darkgrid", {
        "axes.facecolor": "lightskyblue",
        "font.family": [u"Bitstream Vera Sans"]
    })
couleurs = sns.light_palette("turquoise", n_colors=93, reverse=True)
boiteDoc = sns.boxplot(y="discipline",
                       x="nbPages",
                       data=doctorats,
                       palette=couleurs,
                       order=medianesDoctoratsDiscipline.index
                      )
boiteDoc.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=10))
boiteDoc.grid(b=True, which='major', color='w', linewidth=3.0)
boiteDoc.grid(b=True, which='minor', color='w', linewidth=1)
boiteDoc.set(ylabel="Discipline",
             xlabel="Distribution du nombre de pages",
             xlim=(0,750),
             title="Nombre de pages des doctorats par discipline\nUniversités du Québec (de 1990 à 2016)\n"
            )
```




    [<matplotlib.text.Text at 0x119b9ccc0>,
     (0, 750),
     <matplotlib.text.Text at 0x11979cba8>,
     <matplotlib.text.Text at 0x119774be0>]




![png](theses_files/theses_19_1.png)


-----

## Les maîtrises

-----

On peut maintenant faire la même chose pour les maîtrises en suivant exactement les mêmes étapes.


```python
maitrises = theses.query("type == 'maîtrise'")
maitrises.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annee</th>
      <th>octets</th>
      <th>nbPages</th>
      <th>longTitre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40485.000000</td>
      <td>3.477800e+04</td>
      <td>40485.000000</td>
      <td>40485.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2007.034111</td>
      <td>8.790473e+06</td>
      <td>133.284179</td>
      <td>102.193034</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.312315</td>
      <td>1.820470e+07</td>
      <td>54.082249</td>
      <td>39.216650</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>9.075000e+03</td>
      <td>19.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2004.000000</td>
      <td>2.322921e+06</td>
      <td>98.000000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2008.000000</td>
      <td>4.191776e+06</td>
      <td>124.000000</td>
      <td>98.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2012.000000</td>
      <td>8.056662e+06</td>
      <td>157.000000</td>
      <td>125.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>1.916892e+09</td>
      <td>744.000000</td>
      <td>345.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
maitrisesUniv = maitrises.groupby("universite")
maitrisesUniv.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>annee</th>
      <th>longTitre</th>
      <th>nbPages</th>
      <th>octets</th>
    </tr>
    <tr>
      <th>universite</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Concordia</th>
      <th>count</th>
      <td>7210.000000</td>
      <td>7210.000000</td>
      <td>7210.000000</td>
      <td>7.147000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2004.673786</td>
      <td>83.308460</td>
      <td>125.848128</td>
      <td>5.177932e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.978006</td>
      <td>32.854020</td>
      <td>49.743067</td>
      <td>2.473131e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>4.000000</td>
      <td>19.000000</td>
      <td>9.075000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2000.000000</td>
      <td>61.000000</td>
      <td>94.000000</td>
      <td>2.440756e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2006.000000</td>
      <td>80.000000</td>
      <td>117.000000</td>
      <td>3.859451e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>103.000000</td>
      <td>149.000000</td>
      <td>5.704982e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>251.000000</td>
      <td>666.000000</td>
      <td>1.916892e+09</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">HEC Montréal</th>
      <th>count</th>
      <td>2691.000000</td>
      <td>2691.000000</td>
      <td>2691.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2004.829431</td>
      <td>98.866592</td>
      <td>146.237087</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.603063</td>
      <td>35.700203</td>
      <td>66.552327</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>21.000000</td>
      <td>27.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2001.000000</td>
      <td>74.000000</td>
      <td>97.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2005.000000</td>
      <td>94.000000</td>
      <td>136.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2009.000000</td>
      <td>119.000000</td>
      <td>181.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>289.000000</td>
      <td>671.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">INRS</th>
      <th>count</th>
      <td>852.000000</td>
      <td>852.000000</td>
      <td>852.000000</td>
      <td>8.020000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2006.393192</td>
      <td>118.221831</td>
      <td>135.187793</td>
      <td>1.380241e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.125526</td>
      <td>38.144186</td>
      <td>54.682525</td>
      <td>2.036042e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>39.000000</td>
      <td>46.000000</td>
      <td>4.846760e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2003.000000</td>
      <td>89.000000</td>
      <td>98.000000</td>
      <td>2.168610e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.000000</td>
      <td>115.000000</td>
      <td>124.000000</td>
      <td>4.917306e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2011.000000</td>
      <td>141.000000</td>
      <td>158.250000</td>
      <td>1.571059e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>259.000000</td>
      <td>572.000000</td>
      <td>2.258889e+08</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">McGill</th>
      <th>count</th>
      <td>3169.000000</td>
      <td>3169.000000</td>
      <td>3169.000000</td>
      <td>3.169000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2003.526033</td>
      <td>87.065636</td>
      <td>113.933733</td>
      <td>5.120333e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.550171</td>
      <td>33.343854</td>
      <td>35.053225</td>
      <td>6.688783e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>9.000000</td>
      <td>27.000000</td>
      <td>1.148270e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1997.000000</td>
      <td>63.000000</td>
      <td>91.000000</td>
      <td>2.421128e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2004.000000</td>
      <td>82.000000</td>
      <td>109.000000</td>
      <td>4.120969e+06</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Université Laval</th>
      <th>std</th>
      <td>3.094363</td>
      <td>39.890720</td>
      <td>53.539320</td>
      <td>1.939081e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2003.000000</td>
      <td>6.000000</td>
      <td>26.000000</td>
      <td>2.191560e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>82.000000</td>
      <td>93.500000</td>
      <td>3.310624e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010.000000</td>
      <td>106.000000</td>
      <td>120.000000</td>
      <td>1.363568e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.000000</td>
      <td>133.000000</td>
      <td>153.000000</td>
      <td>2.798163e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>345.000000</td>
      <td>743.000000</td>
      <td>2.655805e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Université de Montréal</th>
      <th>count</th>
      <td>5368.000000</td>
      <td>5368.000000</td>
      <td>5368.000000</td>
      <td>5.368000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2009.920268</td>
      <td>105.663376</td>
      <td>134.533905</td>
      <td>4.505311e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.593720</td>
      <td>36.558156</td>
      <td>47.373174</td>
      <td>1.177802e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>13.000000</td>
      <td>25.000000</td>
      <td>6.659300e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>79.000000</td>
      <td>104.000000</td>
      <td>1.077788e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2010.000000</td>
      <td>103.000000</td>
      <td>128.000000</td>
      <td>2.077272e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2013.000000</td>
      <td>129.000000</td>
      <td>157.000000</td>
      <td>4.023963e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>288.000000</td>
      <td>541.000000</td>
      <td>2.880010e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Université de Sherbrooke</th>
      <th>count</th>
      <td>4640.000000</td>
      <td>4640.000000</td>
      <td>4640.000000</td>
      <td>4.640000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2006.019397</td>
      <td>110.667457</td>
      <td>140.139871</td>
      <td>1.015948e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.288464</td>
      <td>39.815207</td>
      <td>54.996114</td>
      <td>1.797599e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>14.000000</td>
      <td>30.000000</td>
      <td>3.310720e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2001.000000</td>
      <td>83.000000</td>
      <td>104.000000</td>
      <td>3.317203e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.000000</td>
      <td>107.000000</td>
      <td>130.000000</td>
      <td>5.174308e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2011.000000</td>
      <td>134.000000</td>
      <td>165.000000</td>
      <td>7.816884e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>305.000000</td>
      <td>712.000000</td>
      <td>2.953644e+08</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">École Polytechnique</th>
      <th>count</th>
      <td>1046.000000</td>
      <td>1046.000000</td>
      <td>1046.000000</td>
      <td>1.046000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.322180</td>
      <td>99.830784</td>
      <td>135.810707</td>
      <td>7.651716e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.838042</td>
      <td>30.244159</td>
      <td>53.423384</td>
      <td>2.009079e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2009.000000</td>
      <td>26.000000</td>
      <td>39.000000</td>
      <td>2.678300e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2011.000000</td>
      <td>79.000000</td>
      <td>100.000000</td>
      <td>2.059766e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2012.000000</td>
      <td>96.000000</td>
      <td>123.500000</td>
      <td>3.616986e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.000000</td>
      <td>118.000000</td>
      <td>159.000000</td>
      <td>7.254857e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>195.000000</td>
      <td>496.000000</td>
      <td>4.822821e+08</td>
    </tr>
  </tbody>
</table>
<p>104 rows × 4 columns</p>
</div>




```python
medianesMaitrisesUniv = maitrises.groupby("universite")["nbPages"].median().sort_values(ascending=False)
medianesMaitrisesUniv
```




    universite
    UQAC                        144.0
    HEC Montréal                136.0
    UQAM                        135.0
    UQAT                        131.0
    Université de Sherbrooke    130.0
    Université de Montréal      128.0
    INRS                        124.0
    École Polytechnique         123.5
    Université Laval            120.0
    UQTR                        118.0
    UQO                         117.5
    Concordia                   117.0
    McGill                      109.0
    Name: nbPages, dtype: float64




```python
sns.set()
plt.figure(figsize=(10, 15))
sns.set_style("darkgrid", {
        "axes.facecolor": "khaki",
        "font.family": [u"Bitstream Vera Sans"]
    })
couleurs = sns.light_palette("olive", n_colors=13, reverse=True)
boiteDoc = sns.boxplot(y="universite",
                       x="nbPages",
                       data=maitrises,
                       palette=couleurs,
                       order=medianesMaitrisesUniv.index
                      )
boiteDoc.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
boiteDoc.grid(b=True, which='major', color='w', linewidth=2.0)
boiteDoc.grid(b=True, which='minor', color='w', linewidth=0.5)
boiteDoc.set(ylabel="Université",
             xlabel="Distribution du nombre de pages",
             xlim=(0,350),
             title="Nombre de pages des maîtrises par université\n"
            )
```




    [<matplotlib.text.Text at 0x11f60de80>,
     (0, 350),
     <matplotlib.text.Text at 0x11f5f7e10>,
     <matplotlib.text.Text at 0x11f614898>]




![png](theses_files/theses_24_1.png)



```python
medianesMaitrisesDiscipline = maitrises.groupby("discipline")["nbPages"].median().sort_values(ascending=False)
medianesMaitrisesDiscipline
```




    discipline
    Design                               179.0
    Danse                                176.0
    Aménagement/urbanisme                164.0
    Sciences infirmières                 158.5
    Management/gestion                   158.0
    Droit                                158.0
    Gérontologie                         157.0
    Génie (général)                      157.0
    Anthropologie                        154.5
    Relations industrielles              152.0
    Génie civil                          149.0
    Travail social                       148.0
    Sociologie                           146.0
    Sciences humaines générales          145.0
    Pédagogie                            144.0
    Études juives                        143.5
    Sciences de l'information            143.0
    Histoire                             143.0
    Géographie                           141.0
    Linguistique                         139.0
    Éducation                            138.0
    Génie minier                         137.5
    Histoire de l'art                    137.5
    Géologie                             136.0
    Architecture                         135.5
    Communication                        135.0
    Science politique                    135.0
    Religion/théologie                   134.0
    Administration publique              133.0
    Criminologie                         132.5
                                         ...  
    Géomatique et télédétection          110.0
    Théâtre                              110.0
    Génétique                            108.0
    Épidémiologie                        108.0
    Anatomie                             108.0
    Informatique                         108.0
    Médecine expérimentale               108.0
    Pathologie                           107.0
    Psychiatrie                          106.5
    Sciences de l'environnement          106.0
    Comptabilité                         105.0
    Marketing                            105.0
    Psychologie                          103.0
    Agriculture et pêcheries             103.0
    Sylviculture/foresterie              102.0
    Kinésiologie                         101.0
    Statistique                          100.0
    Études des sciences et techniques    100.0
    Neurologie                           100.0
    Météorologie                          99.5
    Physique                              99.0
    Études italiennes                     97.0
    Biologie                              94.0
    Mathématiques                         94.0
    Psychoéducation                       94.0
    Chirurgie                             89.0
    Télécommunications                    81.5
    Économie                              80.0
    Arts visuels                          79.0
    Finance                               69.0
    Name: nbPages, dtype: float64




```python
sns.set()
sns.set_context("poster")
sns.set(font_scale=2)
plt.figure(figsize=(20, 35))
sns.set_style("darkgrid", {
        "axes.facecolor": "khaki",
        "font.family": [u"Bitstream Vera Sans"]
    })
couleurs = sns.light_palette("olive", n_colors=102, reverse=True)
boiteMait = sns.boxplot(y="discipline",
                       x="nbPages",
                       data=maitrises,
                       palette=couleurs,
                       order=medianesMaitrisesDiscipline.index
                      )
boiteMait.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=5))
boiteMait.grid(b=True, which='major', color='w', linewidth=3.0)
boiteMait.grid(b=True, which='minor', color='w', linewidth=1)
boiteMait.set(ylabel="Discipline",
             xlabel="Distribution du nombre de pages",
             xlim=(0,350),
             title="Nombre de pages des maîtrises par discipline\nUniversités du Québec (de 1990 à 2016)\n"
            )
```




    [<matplotlib.text.Text at 0x11ef83978>,
     (0, 350),
     <matplotlib.text.Text at 0x11ef66eb8>,
     <matplotlib.text.Text at 0x11efb5828>]




![png](theses_files/theses_26_1.png)


-----
### Un dernier graphique, juste pour le kik

Seaborn permet aussi de tracer des graphiques en forme de violon, ou *violin&nbsp;plots*. Cela ressemble à des diagrammes à moustaches, sauf qu'on n'y découpe pas la matière en quartiles. On y donne plutôt un aperçu de la densité réelle de la distribution.<br>
Il est aussi possible de représenter deux catégories sur chaque violon, ce qui est parfait pour le cas qui nous intéresse, puisqu'on a justement deux catégories à représenter: les maîtrises et les doctorats. Elles sont identifiées par le paramètre `hue` de la méthode `sns.violinplot` dans le code ci-dessous.


```python
sns.set_style("darkgrid")
sns.set_context("poster")
plt.figure(figsize=(10, 15))
violon = sns.violinplot(y="universite", x="nbPages", data=theses, bw=.1, hue="type", split=True)
violon.set(ylabel="Université",
             xlabel="Distribution du nombre de pages",
             xlim=(-100,1400),
             title="Nombre de pages des maîtrises et des doctorats par université\nUniversités du Québec (de 1990 à 2016)\n"
            )
sns.despine()
```


![png](theses_files/theses_28_0.png)

