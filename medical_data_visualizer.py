import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = df['bmi'].apply(lambda x : 1 if x > 25 else 0)

# 3
df['cholesterol'] = df ['cholesterol'].apply(lambda x : 0 if x == 1 else 1)
df['gluc'] = df ['gluc'].apply(lambda x : 0 if x == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    df_cat = df_cat.rename(columns={'variable': 'features'})
    

    # 7

    


    # 8
    fig = sns.catplot(data=df_cat, x = 'features', hue = 'value', col='cardio', kind='count', height=4, aspect=1.5)
    plt.show()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df ['height'] <= df ['height'].quantile(0.975)) ]

    # 12
    corr = df_heat.corr()

    # 13
    #plt.figure(figsize=(10,8))
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5, cbar_kws={'shrink':0.8})

    # 15

    plt.show()

    # 16
    fig.savefig('heatmap.png')
    return fig
