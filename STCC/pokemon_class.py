# %% [markdown]
# ## Classificazione e Clustering Pokemon
# ### Classificazione
# #### Cella 1: Importazioni e Preprocessing

# %%
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model, Sequential # type: ignore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # in order to avoid memory error

# %%
# Carica il dataset
df = pd.read_csv("pokemon.csv")

# %%
# Elimina le colonne duplicate
df.drop_duplicates(keep='first', inplace=True)

# %%
# Seleziona le colonne numeriche
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Sostituisci i valori mancanti con la mediana di ciascuna colonna numerica
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# %%
# Seleziona le colonne di tipo object (categoriche)
categorical_cols = df.select_dtypes(include=['object']).columns

# Sostituisci i valori mancanti con "Unknown" (o con il mode della colonna)
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# %%
# Converti le colonne stringa in liste Python
df['type'] = df['type'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df['charged_moves'] = df['charged_moves'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df['fast_moves'] = df['fast_moves'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# %%
# Combina le mosse in un'unica lista
df['all_moves'] = df['charged_moves'] + df['fast_moves']

# %%
# Seleziona le feature numeriche
numeric_features = ['base_attack', 'base_defense', 'base_stamina', 'max_cp',
                    'attack_probability', 'dodge_probability',
                    'max_pokemon_action_frequency', 'min_pokemon_action_frequency']
X_numeric = df[numeric_features].values

# %%
# Normalizza le feature numeriche
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# %%
# Codifica la colonna "type" usando MultiLabelBinarizer
mlb_type = MultiLabelBinarizer()
X_type = mlb_type.fit_transform(df['type'])

# %%
# Codifica le mosse
mlb_moves = MultiLabelBinarizer()
X_moves = mlb_moves.fit_transform(df['all_moves'])

# %%
# Combina tutte le feature in un unico vettore
X_combined = np.concatenate([X_numeric_scaled, X_type, X_moves], axis=1)
print("Dimensione input combinato:", X_combined.shape)

# %% [markdown]
# #### Cella 2: Costruzione e Addestramento dell'Autoencoder

# %%
# Definizione dei parametri
input_dim = X_combined.shape[1]
encoding_dim = 16  # Dimensione dello spazio latente

# %%
# Definizione dell'architettura dell'autoencoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# %%
# Costruzione del modello
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# %%
# Addestramento dell'autoencoder
autoencoder.fit(X_combined, X_combined, epochs=50, batch_size=32, validation_split=0.2)

# %%
# Estrazione del modello encoder per ottenere la rappresentazione latente
encoder = Model(inputs=input_layer, outputs=latent)
X_latent = encoder.predict(X_combined)
print("Dimensione rappresentazione latente:", X_latent.shape)

# %% [markdown]
# #### Cella 3: Analisi della Rappresentazione Latente

# %%
# Distribuzione delle prime due componenti della rappresentazione latente
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_latent[:, 0], y=X_latent[:, 1], alpha=0.5)
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.title("Distribuzione della Rappresentazione Latente")
plt.show()

# %% [markdown]
# #### Cella 4: Clustering con K-Means e Silhouette Score

# %%
# Testiamo diversi valori di K
wcss = []  # Within-Cluster Sum of Squares
silhouette_scores = []

for k in range(2, 11):  # Da 2 a 10 cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_latent)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_latent, labels))

# %%
# Grafico del metodo del gomito
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.xlabel("Numero di cluster (K)")
plt.ylabel("WCSS")
plt.title("Metodo del Gomito")

# %%
# Grafico del silhouette score
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='s', linestyle='-')
plt.xlabel("Numero di cluster (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score per diversi K")
plt.show()

# %%
# Addestriamo K-Means con il numero ottimale di cluster (es. K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_latent)

# %%
# Aggiungiamo le etichette dei cluster al dataset
df["cluster"] = clusters

# %%
# Grafico della distribuzione dei cluster
sns.countplot(x=df["cluster"], hue=df["cluster"], palette="viridis", legend=False)
plt.title("Distribuzione dei Cluster")
plt.xlabel("Cluster")
plt.ylabel("Conteggio")
plt.show()


# %% [markdown]
# #### Cella 5: Random Forest Classifier

# %%
# Classificazione con Random Forest
features = numeric_features
target = "rarity"

# %%
# Questo trasforma le etichette categoriali in numeri interi
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# %%
# Divide il dataset in set di addestramento e di test
# X_train e X_test contengono le caratteristiche (features)
# y_train e y_test contengono le etichette (target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# %%
# Prevede le etichette per il set di test
y_pred = clf.predict(X_test)

# %%
# Matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.show()

# %%
# Report di classificazione
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

# %%
# Importanza delle feature
feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title("Importanza delle Feature")
plt.ylabel("Score")
plt.show()

# %%
# Heatmap della correlazione
plt.figure(figsize=(10, 8))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap della Correlazione")
plt.show()

# %%
# Distribuzione della rarità
sns.countplot(x=df[target], hue=df[target], palette="viridis", legend=False)
plt.title("Distribuzione della Rarità")
plt.xlabel("Rarità")
plt.ylabel("Conteggio")
plt.show()

# %% [markdown]
# #### Cella 6: Definizione del Target PvP vs. PvE

# %%
# Definiamo insiemi euristici per le mosse
pvp_moves = {'Counter', 'Quick Attack', 'Vine Whip', 'Tackle', 'Bug Bite', 'Ember', 
            'Scratch', 'Water Gun', 'Bubble', 'Wing Attack', 'Peck'}
pve_moves = {'Hydro Cannon', 'Blast Burn', 'Solar Beam', 'Flamethrower', 'Dragon Claw', 
            'Skull Bash', 'Ice Beam', 'Hydro Pump'}

# %%
# Funzione per assegnare lo stile PvP (0) o PvE (1)
def assign_style(row):
    moves = set(row['all_moves'])
    if moves.intersection(pvp_moves) and not moves.intersection(pve_moves):
        return 0  # PvP
    else:
        return 1  # PvE

# %%
df['style'] = df.apply(assign_style, axis=1)
y = df['style'].values
print("Distribuzione stili:\n", pd.Series(y).value_counts())

# %% [markdown]
# #### Cella 7: Classificazione Supervisionata sui Dati Latenti

# %%
# Suddividi i dati in training e test
X_train, X_test, y_train, y_test = train_test_split(X_latent, y, test_size=0.2, random_state=42)

# %%
# Costruzione del classificatore (rete neurale)
classifier = Sequential()
classifier.add(Dense(32, activation='relu', input_dim=encoding_dim))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# %%
# Valutazione del modello
loss, accuracy = classifier.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# %% [markdown]
# ### Clustering
# #### Cella 1: Trovare il numero ottimale di cluster (Metodo del Gomito)

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Cambia il numero in base ai core del tuo PC

# %%
# Testiamo diversi valori di K
wcss = []  # Within-Cluster Sum of Squares

for k in range(1, 11):  # Da 1 a 10 cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_latent)
    wcss.append(kmeans.inertia_)  # Inertia = somma delle distanze dei punti dal centroide

# %%
# Grafico del metodo del gomito
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel("Numero di cluster (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Metodo del Gomito per trovare il numero ottimale di cluster")
plt.show()

# %% [markdown]
# #### Cella 2: Applicare K-Means con il numero ottimale di cluster

# %%
# Addestriamo K-Means con il numero ottimale di cluster (es. K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_latent)

# %%
# Aggiungiamo le etichette dei cluster al dataset
df["cluster"] = clusters

# %%
# Visualizziamo la distribuzione dei cluster
print(df["cluster"].value_counts())

# %% [markdown]
# #### Cella 3: Visualizzare i cluster con PCA (2D)

# %%
# Riduzione a 2 componenti principali
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_latent)

# %%

# Creiamo un DataFrame con i dati ridotti e le etichette dei cluster
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["cluster"] = clusters

# %%
# Grafico dei cluster
plt.figure(figsize=(8, 6))
for cluster in range(kmeans.n_clusters):
    subset = df_pca[df_pca["cluster"] == cluster]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Visualizzazione dei cluster (PCA 2D)")
plt.legend()
plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%