#!/usr/bin/env python
# coding: utf-8

#  # Cluster analysis of the top 100 valuable fifa 2020 players
#  
# In this exercise I will do cluster analysis with a machine learning technique called K-means clustering which is an unsupervised machine learning algorithm. The provided data set do have player's positions but I will not include them. After I have performed a K-means clustering I will compare the results with the provided data set.
# 
# ### Table of contents:
# *   Data cleaning
# *   Exploratory data analysis (EDA)
# *   Cluster analysis 
# *   Conclusion
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('players_20.csv')


# ## Data cleaning and Exploratory data analysis (EDA)

# In[2]:


#Check the shape of the data frame
df.shape


# There are 18278 players and 104 features in the data set. We will just focus on the top 100 players in this exercise.

# In[3]:


# Extract the top100 players based on their market value
df_top100 = df.sort_values('value_eur',ascending=False)[:100]
df_top100.head()            


# In[ ]:





# In[4]:


# Make a new df with the numerical which consist players personal data and skill rating. Remove the sofifa_id column
df_top100_new = df_top100[df_top100.describe().columns].copy()
df_top100_new.drop(columns=['sofifa_id'],inplace=True)
df_top100_new.head()


# In[5]:


# Check for null values
df_top100_new.isnull().any()


# It appears that some skillset have null values. It is probably due to goalkeeper don't have a certain skillset that other players have and vice versa. We will replace the nulls with 0.

# In[6]:


df_top100_new=df_top100_new.fillna(0)
df_top100_new.isnull().any()


# All null values are now replaced. Let's explore the data set before cluster analysis.
# 
# Let's plot how the age distribution is among the top 100 players

# In[7]:


df_top100['age'].hist()
plt.xlabel('Age')
plt.title('Age distribution among top 100 valuable  players')
plt.show()


# In[8]:


df_top100_new['age_cat']=pd.cut(df_top100['age'],bins=[19,24,29,34],include_lowest=True,labels=['Low age', 'Mid age', 'High age'] )


# From the histogram we can see that the average age is around 26 year and that the age distribution is normal distributed.
# 
# 

# In[9]:


plt.figure(figsize=(10,4))
sns.scatterplot(data=df_top100_new, x="value_eur", y="wage_eur", hue='age_cat')
plt.title('Wage vs Value')
plt.show()


# From the plot we can see a trend that higher value yield a higher wage. 
# Player with a higher age tends to have a higher salary comapred to the younger players with similair value. Most of the younger players have also a lower value.

# In[10]:


fig, ax = plt.subplots(figsize=(13,5))

sns.scatterplot(data=df_top100_new, x="value_eur", y="wage_eur", hue='age_cat')

# top valueable players name
top10_name = df_top100.sort_values(ascending=False, by='value_eur')[:10]['short_name']
top10 =  df_top100.sort_values(ascending=False, by='value_eur')[:10]

# Annotate the top valueable players name
for i, txt in enumerate(top10_name):
    ax.annotate(txt, (top10['value_eur'].iloc[i], top10['wage_eur'].iloc[i]))
plt.title('Wage vs Value')
plt.show()


# Now let's see which are the top 10 countries among the top 100 valuable players

# In[11]:


nationality_count = df_top100.groupby('nationality')['sofifa_id'].count().sort_values(ascending=False)
plt.bar(nationality_count.index[:10],nationality_count[:10])
plt.xticks(rotation=45 )
plt.title('Top 10 nationalities among top 100 valuable players')
plt.show()


# The European countries are dominating. Only Brazil and Argentina are from outside Europe.

# # Cluster analysis with K-means clustering
# 
# Before I do the clustering I need to choose some features that I need. There are 104 features in total in the data set but I will just choose a few of them that I think are relevant. The values will be standardized so we are normal distributed. We will then determine the number of clusters we will use with <i>Elbow method</i>. Lastly we will try to identify which positions each cluster is represented.

# In[12]:


# Choose the features we will use
col = ['weak_foot', 'skill_moves',
       'shooting','passing', 'dribbling', 'defending',  
       'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle', 'goalkeeping_diving',
       'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes',
      
      'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys']

df_top100_update = df_top100_new[col]


# In[13]:


#Standardise the valaues
standard = StandardScaler()
df_standard = standard.fit_transform(df_top100_update)
df_standard


# In[14]:


# Perform K-means clustering. Assuming 1 to 10 clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++', 
                    random_state = 7)
    kmeans.fit(df_standard)
    wcss.append(kmeans.inertia_)


# In[15]:


# Plot sum of squared distances of samples to their closest cluster center vs number of clusters
plt.plot(range(1,11),wcss,'o--')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.title('Elbow method plot')
plt.show()


# From the elbow method plot we can see that the elbow start at 3 clusters. 
# However we now that there are 4 distinct positions in football (goalkeeper, defender, midfielder, striker). So we will choose 4 clusters for our case. Elbow method is just an indicator and sometimes it can be also be hard to find the elbow if the sum of squared distances does not drop sharply.
# 
# We will use 4 clusters for our cluster analysis

# In[16]:


# K-means with 4 clusters

kmeans =KMeans(n_clusters = 4,
               init = 'k-means++',
               random_state = 7)
kmeans.fit(df_standard)


# In[17]:


# print the cluster for each 100 players

kmeans.labels_


# In[18]:


df_kmeans = df_top100_update.copy()
df_kmeans['cluster'] = kmeans.labels_
df_kmeans_analysis = df_kmeans.groupby('cluster').mean() 
df_kmeans_analysis


# We grouped each cluster and averaged all the values. From the table we can already see some interesting patterns:
# 
# *    Cluster 1 has very high value when it comes to defending
# *    Cluster 2 has value zero for skills like shooting, passin, dribbling and defending. But very high value for goalkeeping skills
# 
# 
# We can almost conclude that Cluster 1 is defender and Cluster 2 is goalkeeper. Let's look how each cluster are distributed for each skill

# In[19]:


#Visualise each skill for each cluster in a histogram 
pd.options.display.max_rows = 10500
df_kmeans_i = df_kmeans.reset_index()
for i in df_kmeans_i.iloc[:,1:20]:
    grid= sns.FacetGrid(df_kmeans_i, col='cluster')
    grid.map(plt.hist, i)
    plt.show()


# From the histograms one can see that there are some skills that cluster 1 and cluster 2 respectively have that are more dominating compare to the other clusters. This complies with the table above.
# 
# Let's see how the clusters are related to each other and between the skills

# In[20]:


import warnings
warnings.filterwarnings("ignore")
sns.pairplot(df_kmeans,vars=['shooting','passing', 'dribbling', 'defending',  
                            'defending_marking', 'defending_standing_tackle'],                           
                             hue='cluster',palette="tab10")


plt.show()


# In[21]:


sns.pairplot(df_kmeans,vars=[  'goalkeeping_handling', 'goalkeeping_kicking',
                           'goalkeeping_positioning', 'goalkeeping_reflexes',
                            'defending_sliding_tackle', 'goalkeeping_diving'],
                             hue='cluster',palette="tab10")
plt.show()


# Again, from the pairplots we can distinguish the cluster 1 and cluster 2. For cluster 0 and cluster 3 we can see that they are clustered more closer together for the skills shooting, passing and defending. This can be due to some players switching between positions striker and midfielder or a midfielder play a more offensive role or vice versa.

# In[22]:


# Change cluster number to position name
df_kmeans['cluster']=df_kmeans['cluster'].map({0:"Midfielder",
                                               1:"Defender",
                                               2:"Goalkeeper",
                                               3:"Striker"})


# We will compare the cluster results with the player position that was provided from the data set.
# 
# Before comparing, the players positions from the data set need to be translated into the four positions: <i>Goalkeeper</i>, <i>Defender</i>, <i>Midfielder</i> and <i>Striker</i>.
# 
# At the moment the positions are divided into sub-groups. For instance, Messi can have either following positions: <i>RW</i>, <i>CF</i> and <i>ST</i> which are different positions for a striker.
# 

# In[23]:


def change_pos_name(row):
    """
    INPUT: Players postions, the sub-group
    OUTPUT: One of the main four positions
    
    The function replaces the sub-group positions and 
    return one of the main four position to respectively player
    
    """
    if row.replace(",",'').split()[0] in ("RB" , "CB" , "LB" , "RCB" , "RWB" , "LCB"):
         return 'Defender'
    if row.replace(",",'').split()[0] in ("RW" , "CF" , "LW" , "ST" , "RS" , "LS" , "LF" , "RF" ):
        return 'Striker'
    if row.replace(",",'').split()[0] in ("RW" , "CF" , "LW" , "ST" , "RS" , "LS" , "LF" , "RF" ):
        return 'Midfielder'
    if row.replace(",",'').split()[0] in ("RM" , "CM" , "LM" , "CAM" , "LDM" , "RDM" , "LAM" , "RAM" , "CDM", "RCM", "LCM"):
        return 'Midfielder'
    else:
        return 'Goalkeeper'
                                                
    
# Take only the top 100 valuable players from the ordignial data set
player_name = df.loc[df_kmeans.index][['short_name','player_positions']]

# Update players positions with the new position name
player_name['player_positions']=player_name['player_positions'].apply(lambda row:change_pos_name(row))

player_cluster = pd.concat([player_name,df_kmeans],axis=1)[['short_name','player_positions','cluster']]
player_cluster


# We can see that there are some differences, it is mostly the midfielder positions that appears to be harder to cluster. We can calculate and see how many mismatches there are among the 100 players.

# In[24]:


def compare(row):
    """
    INPUT: Each players positions from the data set and from the cluster
    OUTPUT: 1 or 0 depending if it is equal or not
    
    The function compare the positions from the data set with the cluster analysis.
    If the positions are equal the funcion return 0, otherwise it return 1.
    """
    
    if row[1] != row[2]:
        return 1
    else:
        return 0

not_equal = []
not_equal.append(player_cluster.apply(lambda row: compare(row),axis=1))

np.array(not_equal).sum()


# # Conclusion

# From the data set I took the most valuable 100 players.
# EDA showed:
# *    The average age among the players are around 26 years
# *    Younger players value more skewed to lower range compared to the other age group
# *    Top 10 countries are dominated by European countries with France, Spain and Germany as top 3. Brazil and Argentina are the only countries that are not from Europe.
# 
# Different method were performed to see patterns among the clusters during the cluster analysis. Two distinct clusters could be distinguished from the cluster analysis which belonged to defender and goalkeeper. Striker and midfielder positions were harder due to some players maybe switching between positions striker and midfielder or a midfielder maybe play a more offensive role or vice versa.
# Lastly, the positions that were determined by the cluster analysis were compared with the positions provided from the data set. 22 of 100 positions of did not match with the provided data. Considering that some players can have both defensive/offensive positions which makes it harder for the cluster to distinguish some players positions. This is considering fairly good clustering without tweaking the parameter. Some further works can be done in the future to make the results better.
# 
