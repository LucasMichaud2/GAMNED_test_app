import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


objective_url = 'https://raw.githubusercontent.com/LucasMichaud2/streamlit_test.py/main/Objectives_updated-Table%201.csv'
df_objective = pd.read_csv(objective_url)

data_url =  'https://raw.githubusercontent.com/LucasMichaud2/streamlit_test.py/main/GAMNED_dataset_V2.2.csv'
df_data = pd.read_csv(data_url)

age_url = 'https://raw.githubusercontent.com/LucasMichaud2/streamlit_test.py/main/Global_data-Table%201.csv'
age_date = pd.read_csv(age_url)



class GAMNED_UAE:


  def __init__(self, data, rating):
    self.df_data = data[data['country'] == 'uae']
    self.df_rating = rating
    self.obj_list = ['branding', 'consideration', 'conversion']

  def get_age_data(self):
    column_names = ['channel', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', '13-17, 18-24', '13-17, 18-24, 25-34',
                   '13-17, 18-24, 25-34, 35-44',
                   '13-17, 18-24, 25-34, 35-44, 45-54', 
                   '13-17, 18-24, 25-34, 35-44, 45-54, 55-64', 
                   '13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+', 
                   'all',
                   '18-24, 25-34', 
                   '18-24, 25-34, 35-44', 
                   '18-24, 25-34, 35-44, 45-54', 
                   '18-24, 25-34, 35-44, 45-54, 55-64',
                   '18-24, 25-34, 35-44, 45-54, 55-64, 65+',
                   '25-34, 35-44',
                   '25-34, 35-44, 45-54',
                   '25-34, 35-44, 45-54, 55-64', 
                   '25-34, 35-44, 45-54, 55-64, 65+',
                   '35-44, 45-54', 
                   '35-44, 45-54, 55-64', 
                   '35-44, 45-54, 55-64, 65+', 
                   '45-54, 55-64', 
                   '45-54, 55-64, 65+',
                   '55-64, 65+',
                   '']
    col1 = ['instagram', 'facebook', 'linkedin', 'snapchat', 'youtube']
    col2 = [8, 4.7, 0, 20, 0]
    col3 = [31, 21.5, 21.7, 38.8, 15]
    col4 = [30, 34.3, 60, 22.8, 20.7]
    col5 = [16, 19.3, 10, 13.8, 16.7]
    col6 = [8, 11.6, 5.4, 3.8, 11.9]
    col7 = [4, 7.2, 2.9, 0, 8.8]
    col8 = [3, 5.6, 0, 0, 9]
    col9 = [x + y for x, y in zip(col2, col3)]
    col10 = [x + y for x, y in zip(col9, col4)]
    col11 = [x + y for x, y in zip(col10, col5)]
    col12 = [x + y for x, y in zip(col11, col6)]
    col13 = [x + y for x, y in zip(col12, col7)]
    col14 = [x + y for x, y in zip(col13, col8)]
    col15 = [x + y for x, y in zip(col3, col4)]
    col16 = [x + y for x, y in zip(col15, col5)]
    col17 = [x + y for x, y in zip(col6, col6)]
    col18 = [x + y for x, y in zip(col17, col7)]
    col19 = [x + y for x, y in zip(col18, col8)]
    col20 = [x + y for x, y in zip(col4, col5)]
    col21 = [x + y for x, y in zip(col20, col6)]
    col22 = [x + y for x, y in zip(col21, col7)]
    col23 = [x + y for x, y in zip(col22, col8)]
    col24 = [x + y for x, y in zip(col5, col6)]
    col25 = [x + y for x, y in zip(col24, col7)]
    col26 = [x + y for x, y in zip(col25, col8)]
    col27 = [x + y for x, y in zip(col6, col7)]
    col28 = [x + y for x, y in zip(col27, col8)]
    col29 = [x + y for x, y in zip(col7, col8)]
    col30 = [0, 0, 0, 0, 0]
    
    
    

    df_age = pd.DataFrame(col1, columns = ['channel'])
    df_age['13-17'] = col2
    df_age['18-24'] = col3
    df_age['25-34'] = col4
    df_age['35-44'] = col5
    df_age['45-54'] = col6
    df_age['55-64'] = col7
    df_age['65+'] = col8
    df_age['13-17, 18-24'] = col9
    df_age['13-17, 18-24, 25-34'] = col10
    df_age['13-17, 18-24, 25-34, 35-44'] = col11
    df_age['13-17, 18-24, 25-34, 35-44, 45-54'] = col12
    df_age['13-17, 18-24, 25-34, 35-44, 45-54, 55-64'] = col13
    df_age['13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+'] = col14
    df_age['all'] = col14
    df_age['18-24, 25-34'] = col15
    df_age['18-24, 25-34, 35-44'] = col16
    df_age['18-24, 25-34, 35-44, 45-54'] = col17
    df_age['18-24, 25-34, 35-44, 45-54, 55-64'] = col18
    df_age['18-24, 25-34, 35-44, 45-54, 55-64, 65+'] = col19
    df_age['25-34, 35-44'] = col20
    df_age['25-34, 35-44, 45-54'] = col21
    df_age['25-34, 35-44, 45-54, 55-64'] = col22
    df_age['25-34, 35-44, 45-54, 55-64, 65+'] = col23
    df_age['35-44, 45-54'] = col24
    df_age['35-44, 45-54, 55-64'] = col25
    df_age['35-44, 45-54, 55-64, 65+'] = col26
    df_age['45-54, 55-64'] = col27
    df_age['45-54, 55-64, 65+'] = col28
    df_age['55-64, 65+'] = col29
    df_age[''] = col30
    
    return df_age



  def uniform_channel(self):
    # make sure we have one exhaustive channel list
    channel_list = pd.concat([self.df_data['channel'], self.df_rating['channel']], axis=0)
    channel_list = channel_list.unique().tolist()
    return channel_list


  def get_mean_rating(self):
    # Get the rating df and output the mean value for each channel
    # The df is all the channels and mean rating in relation to (branding,
    # consideration and conversion)
    mean_ratings = self.df_rating.drop('formats', axis=1)
    mean_ratings = mean_ratings.groupby('channel').mean().reset_index()
    return mean_ratings

  def get_data_freq(self):
    # get the freq of channel used in data for each objectives
    df_freq = self.df_data[['objectif', 'channel']]
    stack_list = []
    for i in self.obj_list:
      temp_df = df_freq[df_freq['objectif'] == i]
      stack_list.append(temp_df['channel'].value_counts())
    df_freq = pd.DataFrame(stack_list)
    df_freq = df_freq.T
    df_freq.columns = self.obj_list
    df_freq = df_freq / df_freq.sum()
    df_freq = df_freq*10
    df_freq = df_freq.reset_index()
    df_freq.rename(columns={'index': 'channel'}, inplace=True)
    return df_freq

  def get_channel_rating(self, input_age, df_age, df_freq, df_rating):
    # Get the final rating of channel and formats depending on age group
    # and objective

    age_column = df_age[input_age].tolist()
    age_channel = df_age['channel'].tolist()

    age_dic = {
        'channel': age_channel,
        'branding': age_column,
        'consideration': age_column,
        'conversion': age_column
    }
    age_table = pd.DataFrame(age_dic)
    age_table.iloc[0:, 1:] =  age_table.iloc[0:, 1:] / 10

    temp1 = pd.concat([df_freq, age_table], axis=0)
    temp1 = pd.concat([temp1, df_rating], axis=0)

    df_channel_rating = temp1.groupby('channel').sum()
    #df_channel_rating.columns = ['channel', 'branding', 'consideration', 'converison']
    df_channel_rating = df_channel_rating.reset_index()

    return df_channel_rating


  def get_format_rating(self, channel_rating):
    # combine format and channel rating

    for index, row in channel_rating.iterrows():

      a_value = row['channel']
      self.df_rating.loc[self.df_rating['channel'] == a_value, self.obj_list] += row[self.obj_list]

    return self.df_rating


  def get_objective(self, input_obj, df_rating):

    df_heatmap = df_rating[['channel', 'formats', input_obj]]
    df_heatmap = df_heatmap.sort_values(by=input_obj, ascending=False)
    return df_heatmap


  def get_target(self):

    target_dic = {
        'channel': ['linkedin', 'search', 'video', 'native ads'],
        'branding': [10, 10, 10, 10],
        'consideration': [10, 10, 10, 10],
        'conversion': [10, 10, 10, 10]
    }

    df_target = pd.DataFrame(target_dic)

    return df_target

  def add_target(self, df_target, channel_rating):

      total_rating = pd.concat([channel_rating, df_target], axis=0)
      total_rating = total_rating.groupby('channel').sum()

      return total_rating

def round_5(x): 
  return round(x/5) * 5

color_dictionary = {
    0: '#F1C40F', 5: '#F4D03F', 10: '#F7DC6F', 15: '#F9E79F', 20: '#FAD7A0',
    25: '#F8C471', 30: '#F5B041', 35: '#F39C12', 40: '#E67E22', 45: '#D35400',
    50: '#EC7063', 55: '#E74C3C', 60: '#CB4335', 65: '#B03A2E', 70: '#943126',
    75: '#78281F', 80: '#8E44AD', 85: '#7D3C98', 90: '#6C3483', 95: '#5B2C6F',
    100: '#4A235A'
}

age_list = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'all']
age_df = pd.DataFrame(age_list)

objective_list = ['branding', 'consideration', 'conversion']
objective_df = pd.DataFrame(objective_list)

target_list = ['b2c', 'b2b']
target_df = pd.DataFrame(target_list)

st.title('GAMNED Marketing Tool')
st.text(' ')
st.subheader('Frequently Used')
st.sidebar.title('Parameters')

selected_objective = st.sidebar.selectbox('Select an objective', objective_df)
selected_age = st.sidebar.multiselect("Select an age", age_df)
selected_age = ', '.join(selected_age)
selected_target = st.sidebar.selectbox('Select target', target_df)
input_budget = st.sidebar.number_input('Budget', value=0)




# Getting Variables

gamned_class = GAMNED_UAE(df_data, df_objective)
df_age = gamned_class.get_age_data()
df_freq = gamned_class.get_data_freq()
df_rating = gamned_class.get_mean_rating()
df_rating1 = gamned_class.get_channel_rating(selected_age, df_age, df_freq, df_rating)
if selected_target == 'b2b':
  df_b2b = gamned_class.get_target()
  df_rating1 = gamned_class.add_target(df_b2b, df_rating1)
  df_rating1 = df_rating1.reset_index()
df_rating2 = gamned_class.get_format_rating(df_rating1)
df_rating3 = gamned_class.get_objective(selected_objective, df_rating2)
#output_rating = df_rating3.head(20)

################################## Computing Scores ###################################

channel_count = pd.DataFrame(df_rating3.groupby('channel')['formats'].count())
channel_count = channel_count.reset_index()
col_names = ['channel', 'count']
channel_count.columns = col_names

agg_rating = df_rating3.drop(['formats'], axis=1)
agg_rating1 = agg_rating.groupby('channel').sum()
agg_rating1 = agg_rating1.reset_index()
agg_rating2 = agg_rating1.sort_values(by='channel')
channel_count2 = channel_count.sort_values(by='channel')
agg_rating2['average'] = agg_rating2[selected_objective] / channel_count2['count']
agg_rating3 = agg_rating2.sort_values(by='average', ascending=False)
cost_rating = agg_rating3.copy()
agg_rating_min = agg_rating3['average'].min()
agg_rating_max = agg_rating3['average'].max()
agg_rating3['average'] = ((agg_rating3['average'] - agg_rating_min) / (agg_rating_max - agg_rating_min))*100
output_rating = agg_rating3.copy()
                        

#################################### Building Heatmap ####################################

heat_map = output_rating.drop([selected_objective], axis=1)
#heat_map1 = heat_map.groupby('channel').sum()
#heat_map2= heat_map1.sort_values(by=selected_objective, ascending=False)
heat_map['average'] = heat_map['average'].apply(round_5)

heat_map['mapped_colors'] = heat_map['average'].map(color_dictionary)
heat_map = heat_map.reset_index()
heat_map2 = heat_map.head(10)


##################################### Buidling Budget #####################################

cost_rating = cost_rating.drop([selected_objective], axis=1)
cost_rating = cost_rating.sort_values(by='average', ascending=False)
cost_rating = cost_rating.reset_index()
cost_rating_std = cost_rating['average'].std()
cost_rating_mean = cost_rating['average'].mean()
cost_rating['norm'] = (cost_rating['average'] - cost_rating_mean) / cost_rating_std
threshold = cost_rating['norm'].max() - 0.50*cost_rating['norm'].max()
df_selection = cost_rating[cost_rating['norm'] > threshold]

df_budget = df_selection.copy()
average_max = df_budget['average'].max()
average_min = df_budget['average'].min()
average_diff = average_max - average_min
df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
df_budget['allowance'] = input_budget * df_budget['distribution']
columns_to_drop = ['average', 'index', 'norm', 'distribution']
df_allowance = df_budget.drop(columns=columns_to_drop)

# df_allowance now contains the DataFrame with the specified columns dropped


##################################### Budget Rules ########################################

if input_budget < 4001 and selected_objective == 'consideration':
  disp_allow = input_budget - 500
  budget_lib1 = {
    'channel': ['display', 'search'],
    'allowance': [disp_allow, 500]
  }
  df_allowance = pd.DataFrame(budget_lib1)
  
elif input_budget < 4001:
  df_allowance = df_allowance.head(1)
  df_allowance.at[0, 'allowance'] = input_budget
  
elif input_budget < 70001 and input_budget > 5001:
  if df_allowance.shape[0] > 3:
    df_budget = df_budget.head(3)
    df_budget['allowance'] = input_budget * df_budget['distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)



  




##################################### taking out the code and name ########################

name0 = heat_map2.at[0, 'channel']
color0 = heat_map2.at[0, 'mapped_colors']
name1 = heat_map2.at[1, 'channel']
color1 = heat_map2.at[1, 'mapped_colors']
name2 = heat_map2.at[2, 'channel']
color2 = heat_map2.at[2, 'mapped_colors']
name3 = heat_map2.at[3, 'channel']
color3 = heat_map2.at[3, 'mapped_colors']
name4 = heat_map2.at[4, 'channel']
color4 = heat_map2.at[4, 'mapped_colors']
name5  = heat_map2.at[5, 'channel']
color5 = heat_map2.at[5, 'mapped_colors']
name6  = heat_map2.at[6, 'channel']
color6 = heat_map2.at[6, 'mapped_colors']
name7  = heat_map2.at[7, 'channel']
color7 = heat_map2.at[7, 'mapped_colors']
name8  = heat_map2.at[8, 'channel']
color8 = heat_map2.at[8, 'mapped_colors']
name9  = heat_map2.at[9, 'channel']
color9 = heat_map2.at[9, 'mapped_colors']


#####################################   Pie Chart freq ####################################

df_freq['branding'] = df_freq['branding'].round(1)
df_freq['consideration'] = df_freq['consideration'].round(1)
df_freq['conversion'] = df_freq['conversion'].round(1)



df_brand = df_freq[['channel', 'branding']]
df_brand = df_brand.dropna()
df_cons = df_freq[['channel', 'consideration']]
df_cons = df_cons.dropna()
df_conv = df_freq[['channel', 'conversion']]
df_conv = df_conv.dropna()
custom_colors1 = sns.color_palette('Blues', n_colors=len(df_brand))
custom_colors2 = sns.color_palette('Purples', n_colors=len(df_cons))
custom_colors3 = sns.color_palette('Reds', n_colors=len(df_conv))



pie1, pie2, pie3 = st.columns(3)

with pie1:

  fig1, ax1 = plt.subplots()
  ax1.pie(df_brand['branding'], labels=df_brand['channel'], autopct='%1.1f%%', startangle=90, colors=custom_colors1)
  ax1.axis('equal')
  st.pyplot(fig1)
  st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <p style="text-align: center;">Branding</p>
        </div>
        """,
        unsafe_allow_html=True
    )
  

with pie2:
  
  fig2, ax2 = plt.subplots()
  ax2.pie(df_cons['consideration'], labels=df_cons['channel'], autopct='%1.1f%%', startangle=90, colors=custom_colors2)
  ax2.axis('equal')
  st.pyplot(fig2)
  st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <p style="text-align: center;">Consideration</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with pie3:
  
  fig3, ax3 = plt.subplots()
  ax3.pie(df_conv['conversion'], labels=df_conv['channel'], autopct='%1.1f%%', startangle=90, colors=custom_colors3)
  ax3.axis('equal')
  st.pyplot(fig3)
  st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <p style="text-align: center;">Conversion</p>
        </div>
        """,
        unsafe_allow_html=True
    )




st.text(' ')

st.subheader('Top Formats')

st.dataframe(output_rating)


st.text(' ')

st.subheader("Heatmap")



with st.container():
  st.markdown(
  f"""
  <div style='display: flex;'>
      <div style='background-color:{color0}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name0}</div>
       <div style='background-color:{color1}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name1}</div>
       <div style='background-color:{color2}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name2}</div>
       <div style='background-color:{color3}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name3}</div>
       <div style='background-color:{color4}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name4}</div>
  </div>
  """,
  unsafe_allow_html=True
)


st.text(' ')

with st.container():
  st.markdown(
  f"""
  <div style='display: flex;'>
      <div style='background-color:{color5}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name5}</div>
      <div style='background-color:{color6}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name6}</div>
      <div style='background-color:{color7}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name7}</div>
      <div style='background-color:{color8}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name8}</div>
      <div style='background-color:{color9}; width: {'100px'}; height: {'75px'}; margin-riht: {'50px'}; font-size:{'10px'};
      display: flex; align-items: {'center'}; justify-content: {'center'}; border-radius: {'20px'}; color: white;'>{name9}</div>
  </div>
  """,
  unsafe_allow_html=True
)

st.text(' ')

st.subheader('Budget Allocation')


st.dataframe(df_allowance)