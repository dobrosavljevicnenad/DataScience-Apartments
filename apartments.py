#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd


# In[46]:


dataset = pd.read_csv('apartmentsdata.csv')


# In[47]:


dataset.head()


# In[48]:


for i,loc in enumerate(dataset['location']):
    print(loc)
    if i==5:
        break


# In[49]:


location_split = dataset['location'].str.split(',', expand=True)


# In[50]:


# Kreiramo nove kolone
dataset['opstina'] = location_split[0].str.strip()  # uklanjamo eventualne space
dataset['naselje'] = location_split[1].str.strip()
dataset['ulica'] = location_split[2].str.strip()

# Provera
dataset[['location', 'opstina', 'naselje', 'ulica']].head()


# In[51]:


dataset


# In[52]:


dataset = dataset.drop('location', axis=1)


# In[53]:


dataset.size


# In[54]:


dataset.drop(['Unnamed: 9', 'Unnamed: 10'], axis=1, inplace=True)


# In[55]:


dataset.head()


# In[56]:


dataset.drop('title', axis=1, inplace=True)


# In[57]:


dataset.drop('source', axis=1, inplace=True)


# In[58]:


for col in dataset.columns:
    if dataset[col].isna().any():
        print(col)


# In[59]:


dataset.count()


# In[60]:


dataset['naselje'] = dataset['naselje'].fillna(dataset['opstina'])


# In[61]:


dataset['naselje'].isna().any()


# In[62]:


dataset.count()


# In[63]:


dataset['floor'].unique()


# In[64]:


dataset['floor'] = dataset['floor'].replace('p', 0)


# In[65]:


dataset['floor'].unique()


# In[66]:


dataset['floor'] = dataset['floor'].astype(float).abs()


# In[67]:


dataset.groupby('city').filter(lambda x : print(x))


# In[68]:


for city, group in dataset.groupby('city'):
    print("=" * 40)
    print(f"Grad: {city} ({len(group)} redova)")
    print("=" * 40)
    print(group.to_string(index=False))


# In[74]:


for city, group in dataset.groupby('city')['floor']:
    print("=" * 40)
    print(f"Grad: {city} ({len(group)} redova)")
    print("=" * 40)
    print(group.to_string(index=False))


# In[73]:


dataset[dataset['floor'].isna()].count()


# In[79]:


dataset['floor'] = dataset.groupby('city')['floor'].transform(lambda x: x.fillna(x.median()))


# In[82]:


dataset.drop('ulica', axis=1, inplace=True)


# In[83]:


dataset.head()


# In[84]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[219]:


dataset.describe()


# In[109]:


import matplotlib.pyplot as plt
import numpy as np


# In[162]:


def hist_col(col):
    plt.figure(figsize=(20,10))
    
    _, bins, _ = plt.hist(dataset[col], bins=50, edgecolor='black')
    
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    plt.xticks(bin_centers, [f"{int(x):,}" for x in bin_centers], rotation=90)
    
    plt.title('Distribucija cena', fontsize=20)
    plt.xlabel(f'{col}')
    plt.ylabel('Broj stanova')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[175]:


dataset[dataset['price'] == dataset['price'].max()]


# In[163]:


hist_col('price')


# In[176]:


hist_col('area')


# In[188]:


import matplotlib.pyplot as plt
import seaborn as sns

# Broj pojavljivanja po naselju
naselje_counts = dataset['naselje'].value_counts()

plt.figure(figsize=(80,30))
sns.barplot(x=naselje_counts.index, y=naselje_counts.values, color='skyblue', edgecolor='black')
plt.xticks(rotation=90)
plt.title('Broj stanova po naselju', fontsize=18)
plt.ylabel('Broj stanova')
plt.xlabel('Naselje')
plt.tight_layout()
plt.show()


# In[150]:


avg_price_by_city = dataset.groupby('city')['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(20,10))
sns.barplot(x=avg_price_by_city.index, y=avg_price_by_city.values)
plt.xticks(rotation=90)
plt.title('Prosečna cena po gradu')
plt.ylabel('Cena (EUR)')
plt.show()


# In[201]:


plt.figure(figsize=(20,10))
avg_price_city = dataset.groupby("city")["square_price"].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_city.index, y=avg_price_city.values)
plt.xticks(rotation=90)
plt.ylabel("Prosečna cena po m²")
plt.title("Prosečna cena po m² po gradu")
plt.show()


# In[189]:


dataset.head()


# In[190]:


dataset = dataset.rename(columns={
    'area': 'area',
    'city': 'city',
    'floor': 'floor',
    'price': 'price',
    'rooms': 'rooms',
    'square_price': 'square_price',
    'opstina': 'municipality',
    'naselje': 'neighborhood'
})


# In[192]:


dataset.head()


# In[193]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[194]:


sns.set(style = "whitegrid", palette="muted")


# In[195]:


print(dataset['price'].min(), dataset['price'].max())


# In[196]:


print(dataset['price'].dtype)


# In[200]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")

plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelacija numeričkih promenljivih")
plt.show()


# In[206]:


dataset.info()


# In[207]:


dataset.head()


# In[209]:


dataset


# In[210]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = dataset[['area', 'rooms', 'floor', 'square_price', 'city', 'municipality', 'neighborhood']]
y = dataset['price']

# One-hot encoding
X = pd.get_dummies(X, columns=['city', 'municipality', 'neighborhood'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[211]:


scaler = StandardScaler()
X_train[['area', 'rooms', 'floor', 'square_price']] = scaler.fit_transform(X_train[['area', 'rooms', 'floor', 'square_price']])
X_test[['area', 'rooms', 'floor', 'square_price']] = scaler.transform(X_test[['area', 'rooms', 'floor', 'square_price']])


# In[213]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# In[214]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))


# In[215]:


dataset


# In[217]:


import pickle

# Sačuvaj model u fajl
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("✅ Model uspešno sačuvan kao model.pkl")


# In[218]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------
# Učitaj svoj model (ako si ga već sačuvao)
# Ako nisi, možeš privremeno trenirati ga ovde
# -------------------------------------------------------
# Pretpostavimo da si već trenirao model:
# pickle.dump(rf, open("model.pkl", "wb"))
# A ovde ga učitavamo:
rf = pickle.load(open("model.pkl", "rb"))

st.title("🏙️ Predviđanje cene nekretnine")

st.write("Unesi detalje o nekretnini da bi model predvideo cenu:")

# -------------------------------------------------------
# Polja za unos podataka
# -------------------------------------------------------
area = st.number_input("Površina (m²)", min_value=10, max_value=300, value=50)
floor = st.number_input("Sprat", min_value=0, max_value=30, value=2)
rooms = st.number_input("Broj soba", min_value=0.5, max_value=10.0, step=0.5, value=2.0)
square_price = st.number_input("Cena po m²", min_value=500, max_value=10000, value=2000)

city = st.text_input("Grad", "Beograd")
municipality = st.text_input("Opština", "Vračar")
neighborhood = st.text_input("Naselje", "Hram svetog Save")

# -------------------------------------------------------
# Kada korisnik klikne na dugme
# -------------------------------------------------------
if st.button("🔮 Predvidi cenu"):
    # Formiraj DataFrame kao u treniranju modela
    data = pd.DataFrame({
        'area': [area],
        'city': [city],
        'floor': [floor],
        'rooms': [rooms],
        'square_price': [square_price],
        'municipality': [municipality],
        'neighborhood': [neighborhood]
    })

    # Ovde moraš primeniti iste transformacije kao pre treniranja
    # npr. OneHotEncoder ili LabelEncoder ako si koristio
    # (u pravom kodu bi to radio preko pipeline-a)

    pred = rf.predict(data)[0]
    st.success(f"💰 Predviđena cena: {pred:,.2f} €")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




