import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_file(path):
    """This function will take path as a parameter
       and reads the csv file and returns two data frames one with 
       country as columns and other witgh year as columns"""
    ds_raw = pd.read_csv(filepath_or_buffer=path, sep=',',
                         encoding='cp1252', skip_blank_lines=True)
    #Deleting rows with null values
    ds_raw = ds_raw.dropna()
    pd_countries = pd.DataFrame(ds_raw)
    #calling yearly function inorder to get the years as column
    pd_years = yearly_data(pd_countries)
    out = [pd_countries, pd_years]
    return out


def df_info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary"""
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))



def category_sep(df):
    """This function is used to seperate the data frames
       accornding to the series names"""
    out_dict = {}
    # finding the unique values of the series Name
    series_uniq = df['Series Name'].unique()
    #copying each unique seried data frame into a dict and returning it
    for categ in series_uniq:
        out_dict[categ] =(df[df['Series Name'] == categ]).reset_index(drop = True)
    return out_dict


def yearly_data(df):
    """This function takes data frame with years as columns and \
        converts country as columns and years as rows"""
    #slicing the df into only years columns
    y = df.loc[:,'2006':'2014']
    #changing values type from object to float
    y = y.astype(float)
    y['countries'] = df['Country Name']
    #transposing the data frame.
    y = y.T
    y.rename(columns = y.iloc[-1], inplace = True)
    y = y.drop(y.index[-1])
    #reset the index and making index as year column
    y = y.reset_index()
    y = y.rename(columns={'index':'year'})
    return y
    


def Heatmap_plot(df,image_name,color,column,drop_columns):
    """This function used to produce heat map of the given data frame
       seperating the data with respect to country. it takes color as
       parameter in order to change the colors of heat map."""
    # droping unused columns
    df =df.drop(columns=['Country Name','Country Code'])
    #transposing the data and assing the custom columns
    df_t =df.T
    df_t.columns = column
    if drop_columns:
        df_t = df_t.drop(columns=drop_columns)
    # reset the index and drop the index
    df_t = df_t.reset_index(drop = True)
    df_t = df_t.drop([0])
    df_t = df_t.astype(float)
    plt.figure()
    plt.title(image_name)
    #plotting the heat map using seaborn module
    sns.heatmap(df_t.corr(),annot=True,vmin=-1, vmax=1, 
                 annot_kws={'fontsize':8, 'fontweight':'bold'},
                 cmap=sns.color_palette(color))
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()
    
def line_plot(df,title):
    """This function is used to plot a line plot of the data frame
    by seperating the data with respective to countries"""
    #making a list of countries to be plot in the line plot
    countries = ['India','South Africa','Australia','United Kingdom']
    plt.figure()
    #looping through the countries list and plotting for each country
    for coun in countries:
        plt.plot(df['year'],df[coun],label=coun)
    # adding labels,title and legends
    plt.xlabel('Years')
    plt.ylabel('% of GDP')
    plt.title(title)
    plt.legend()
    plt.savefig(title)
    plt.show()    

#gettiing the root path of the current working directory
root_path = os.getcwd()
print(f'the current working directory is : {root_path}\n')
#making a floder for saving the images
pltimgs_dir = 'plotimgs'
if not os.path.exists(pltimgs_dir):
    os.mkdir(pltimgs_dir)
#chaning the cwd to final ds to read the files
csv_path = 'final DS'
os.chdir('final DS')
csv_list = os.listdir()
#calling read_file function to get the df
pd_list = read_file(csv_list[0])

os.chdir(pltimgs_dir)

pd_countries = pd_list[0]
pd_years = pd_list[1]
df_info(pd_countries)

# dividing the data frame according to the series names
categ_list = category_sep(pd_countries)

pd_gdp = categ_list['GDP growth (annual %)']
pd_gdp_capita = categ_list['GDP per capita (constant 2005 US$)']
pd_exports = categ_list['Exports of goods and services (% of GDP)']
pd_agri = categ_list['Agriculture, value added (% of GDP)']
pd_imports = categ_list['Imports of goods and services (% of GDP)']
pd_industry = categ_list['Industry, value added (% of GDP)']
pd_research = categ_list['Research and development expenditure (% of GDP)']
pd_tax = categ_list['Tax revenue (% of GDP)']
pd_new_business = categ_list['New businesses registered (number)']
pd_unemployemnet = categ_list['Unemployment, male (% of male labor force) (modeled ILO estimate)']
pd_emp = categ_list['Total employment, total (ages 15+)']
pd_self_emp = categ_list['Self-employed, total (% of total employment) (modeled ILO estimate)']

# custom columns for heat map
col = ['GDP','GDP Per Capita','Exports','Agriculture','Imports','Industry'
        ,'Reasearch','Tax','New bussiness','Unemployment','Total Emplowment',
        'Self-employed']
col_india = ['GDP','GDP Per Capita','Exports','Agriculture','Imports','Industry'
        ,'New bussiness','Unemployment','Total Emplowment',
        'Self-employed']

col_aus = ['GDP','GDP Per Capita','Exports','Agriculture','Imports','Industry','Tax'
        ,'New bussiness','Unemployment','Total Emplowment',
        'Self-employed']

#making each df as year as column
pd_gdp_year = yearly_data(pd_gdp)
pd_gdp_cpita_year = yearly_data(pd_gdp_capita)
pd_exports_year = yearly_data(pd_exports)
pd_agri_year = yearly_data(pd_agri)
pd_imports_year = yearly_data(pd_imports)
pd_industry_year = yearly_data(pd_industry)
pd_research_year = yearly_data(pd_research)
pd_tax_year = yearly_data(pd_tax)
pd_new_business_year = yearly_data(pd_new_business)
pd_unemployemnet_year = yearly_data(pd_unemployemnet) 
pd_emp_year = yearly_data(pd_emp)
pd_self_emp_year = yearly_data(pd_self_emp)

# calucalting the mean value and taking the top 10 coutries  
pd_new_business['mean'] = pd_new_business.mean(axis=1)
x = pd_new_business.nlargest(10,'mean')
x = x.drop(columns=['Country Code','mean','Series Name','2007','2008','2010','2011','2013'])
x_ch = pd.melt(x, id_vars="Country Name", var_name="year", value_name="new business")
plt.figure()
# plotiing the bar graph using seaborn
sns.factorplot(data=x_ch,x='Country Name',y='new business',hue='year',kind='bar')
plt.xticks(rotation=90)
plt.savefig('business',dpi='figure',bbox_inches = 'tight')

y = pd_gdp[pd_gdp['Country Name'].isin(x['Country Name'])]
y = y.drop(columns=['Country Code','Series Name','2007','2008','2010','2011','2013'])
y_ch = pd.melt(y, id_vars="Country Name", var_name="year", value_name="gdp")
plt.figure()
# plotiing the bar graph using seaborn
sns.factorplot(data=y_ch,x='Country Name',y='gdp',hue='year',kind='bar')
plt.xticks(rotation=90)
plt.savefig('gdp',dpi='figure',bbox_inches = 'tight')

# calling heat map function to map for each country
pd_uk = pd_countries[pd_countries['Country Name']=='United Kingdom']
Heatmap_plot(pd_uk,image_name='United Kingdom',color = 'colorblind',column = col,
              drop_columns=['GDP Per Capita', 'Industry', 'Self-employed'])

pd_india = pd_countries[pd_countries['Country Name']=='India']
Heatmap_plot(df=pd_india, image_name='India', color='Paired',column = col_india,
              drop_columns=['GDP Per Capita'])

pd_aus = pd_countries[pd_countries['Country Name']=='Australia']
Heatmap_plot(df=pd_aus, image_name='Australia', color='rocket',column = col_aus,
              drop_columns=['GDP Per Capita'])

pd_africa = pd_countries[pd_countries['Country Name']=='South Africa']
Heatmap_plot(df=pd_aus, image_name='South Africa', color='Set2',column = col_aus,
              drop_columns=['GDP Per Capita'])

#calling line plot func to plot line plots

line_plot(pd_exports_year,title='Exports of goods and services')
line_plot(pd_imports_year,title='Imports of goods and services')

































