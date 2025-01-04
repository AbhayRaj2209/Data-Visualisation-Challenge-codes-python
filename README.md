# Data-Visualisation-Challenge-codes-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\abhin\Jupyter notebook\ieee datasets\udemy_courses.csv")
df.head()
### Data Cleaning
df.shape
df.isnull().sum()
df.info()
# filtering rows which have empty values
df[df['instructor_names'].isnull()]
# will see this again after univariate analysis on numerical columns
# checking for duplicates
duplicates = df[df.duplicated()]
print(duplicates)
# As we see that there are no duplicates in the dataset
### univariate analysis on numerical columns
Univariate on 'rating'
df['rating'].describe()
df['rating'].plot(kind='kde')
df['rating'].plot(kind='hist')
plt.xlabel('rating')
sns.violinplot(df['rating'])
df['rating'].plot(kind='box')
# filtering rows with rating between 3 and 0.5
df[(df['rating']<=3) & (0.5<=df['rating'])]
Univariate on 'num_subscribers'
df['num_subscribers'].describe()
ax = df['num_subscribers'].plot(kind='hist', bins=100)

ax.set_xlim(0, 250000)  # there are some courses which have subscribers more than 250000 but we are not 
# seeing them in this plot to better visualize and as more than 99.5% coures have subscribers in this range

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xlabel('Number of Subscribers') 
plt.ylabel('courses count')  
plt.title('Histogram of Number of Subscribers')  # Add a title
plt.show()
# courses with subscribers more than 250000
df[df['num_subscribers']>250000]
ax = df['num_subscribers'].plot(kind='kde')

ax.set_xlim(0, 40000)  

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xlabel('Number of Subscribers') 
plt.ylabel('Frequency')  
plt.title('num_subscribers') 
plt.show()
df['num_subscribers'].plot(kind='box')
univariate on 'num_reviews'
df['num_reviews'].describe()
df['num_reviews'].plot(kind='box')
ax=df['num_reviews'].plot(kind='kde')
ax.set_xlim(0,150000) # here the courses we are seeing have reviews less than 150000

plt.xlabel('num_reviews')
plt.ylabel('courses_count_probability')
plt.title('num_reviews')

plt.show()
ax=df['num_reviews'].plot(kind='hist',bins=10000)
ax.set_xlim(0,500) #i am not seeing here courses with reviews more than 500 they are outliers and will see
                   #them after
plt.xlabel('num_reviews')
plt.ylabel('courses_count')
plt.title('reviews')

plt.show()
### univariate on categorical columns
Univariate on feature 'category'
df['category'].value_counts()
ax = df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')

plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('')  
plt.title('Category Distribution')  
plt.tight_layout()  
plt.show()

df['category'].value_counts().plot(kind='bar')
Univariate on feature 'is_paid'
df['is_paid'].value_counts()
df['is_paid'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('is_paid')
plt.ylabel('')
plt.tight_layout()
plt.show()
Univariate on 'instructional_level'
df['instructional_level'].value_counts()
df['instructional_level'].value_counts(normalize=True)*100
df['instructional_level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')   
plt.show()
#filtering the rows with missing values
df[df['instructor_names'].isnull()]
# as we can see the numerical features of those 2 rows and compare with other rows we compare
#with other rows so for 1st row if we see marketing courses with num_subscribers and rating around that value as--
fil1=(df['num_subscribers']<=10500) & (df['num_subscribers']>=10000)
fil2=(df['rating']<=4.2) & (df['rating']>=3.8)
df[(df['category']=='Marketing') & fil1 & fil2]
film1=(df['num_subscribers']<=250) & (df['num_subscribers']>=200)
film2=(df['rating']<=3.9) & (df['rating']>=3.6)
df[(df['category']=='Lifestyle') & film1 & film2]
# So we can see for both courses there are enough courses with values in that range so 
# they don't affect much So i am dropping the 2 courses
df.dropna(inplace=True)

### Bivariate on numerical-numerical
import matplotlib.ticker as mticker
df[['num_subscribers','rating','num_reviews']].corr()
ax=sns.scatterplot(data=df,y='num_subscribers',x='rating')
ax.set_ylim(0, 350000)  

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.show()
ax=sns.kdeplot(data=df,x='rating', y='num_subscribers')
plt.title("subscribers vs rating")
ax.set_ylim(0, 100000)  

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xlabel("rating")
plt.ylabel("subscribers")
plt.show()

### bivariate on categorical- categorical
pd.crosstab(df['is_paid'],df['category'])
pd.crosstab(df['is_paid'],df['instructional_level'],normalize='columns')*100
pd.crosstab(df['instructional_level'],df['category'],normalize='columns')*100

### Bivariate on numerical- categorical
sns.barplot(x='instructional_level',y='num_subscribers',data=df,errorbar=None,estimator='mean')
plt.ylabel('avgNum_of_subscribers_forthe_level')
with pd.option_context('display.float_format', '{:.2f}'.format):
    print(df[df['instructional_level']=='All Levels']['num_subscribers'].describe())
sns.barplot(x='category',y='num_subscribers',data=df,errorbar=None,estimator='mean')
plt.ylabel('avg subscribers per category')
plt.xticks(rotation=90)
plt.show()
categories_top = df['category'].unique()[:6]
categories_bottom = df['category'].unique()[6:]

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# First subplot (Top categories)
sns.violinplot(
    data=df[df['category'].isin(categories_top)], 
    x='category', 
    y='num_subscribers', 
    ax=axes[0]
)
axes[0].set_ylim(0, 50000)
axes[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

axes[0].set_title('Subscribers vs Category (Top Categories)')
axes[0].set_ylabel('Number of Subscribers')
axes[0].set_xlabel('Category')
axes[0].tick_params(axis='x', rotation=45)

# Second subplot (Bottom categories)
sns.violinplot(
    data=df[df['category'].isin(categories_bottom)], 
    x='category', 
    y='num_subscribers', 
    ax=axes[1]
)
axes[1].set_ylim(0, 50000)
axes[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

axes[1].set_title('Subscribers vs Category (Bottom Categories)')
axes[1].set_ylabel('Number of Subscribers')
axes[1].set_xlabel('Category')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
sns.barplot(data=df,x='is_paid',y='num_subscribers',errorbar=None,estimator='mean')
plt.ylabel('average num_subscribers') 

### multifeatured analysis
### category-subscribers-is_paid

sns.barplot(data=df[df['is_paid']==False],x='category',y='num_subscribers',errorbar=None,estimator='mean')

plt.xlabel('category of unpaid courses')
plt.ylabel('average subscribers')
plt.title("filtering unpaid courses and then plotting  category vs subscribers")
plt.xticks(rotation=90)
plt.show()

sns.barplot(data=df[df['is_paid']==True],x='category',y='num_subscribers',errorbar=None,estimator='mean')

plt.xlabel('category of paid courses')
plt.ylabel('average subscribers')
plt.title("filtering paid courses and then plotting  category vs subscribers")
plt.xticks(rotation=90)
plt.show()
grouped_data = (
    df.groupby(['category', 'is_paid'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

pivot_data = grouped_data.pivot(index='category', columns='is_paid', values='avg_subscribers').fillna(0)

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by Category and is_paid per course', fontsize=14)
plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.legend(title='is_paid', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
grouped_data = (
    df.groupby(['category', 'is_paid'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

grouped_data['probability'] = grouped_data.groupby('category')['avg_subscribers'].transform(lambda x: x / x.sum())

pivot_data = grouped_data.pivot(columns='is_paid', index='category', values='probability')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=True, width=0.7)

plt.title('Number of Subscribers on avg by Category and is_paid per course', fontsize=14)
plt.ylabel('probability of num_subscribers', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.legend(title='is_paid', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

### we plot avg subscriber per course vs category and instructional_level
levels=df.groupby('instructional_level')
beginner_courses=levels.get_group('Beginner Level')
all_level_courses=levels.get_group('All Levels')
intermediate_courses=levels.get_group('Intermediate Level')
expert_courses=levels.get_group('Expert Level')
grouped_data = (
    df.groupby(['category', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))  # Calculate mean
    .reset_index()
)

pivot_data = grouped_data.pivot(columns='instructional_level', index='category', values='avg_subscribers')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by Category and Instructional Level per course', fontsize=14)
plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.legend(title='Instructional Level', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

grouped_data = (
    df.groupby(['category', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)


grouped_data['probability'] = grouped_data.groupby('category')['avg_subscribers'].transform(lambda x: x / x.sum())

pivot_data = grouped_data.pivot(columns='instructional_level', index='category', values='probability')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers(probabilities) by category and Instructional Level per course', fontsize=14)
plt.ylabel('probability of no. of subscribers', fontsize=12)#ratio of no. of subscribers of that
                                    #level to the total no. of subscribers of the category across all levels
plt.xlabel('Instructional Level', fontsize=12)
plt.legend(title='category', fontsize=10)
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()
grouped_data = (
    df.groupby(['is_paid', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))  # Calculate mean
    .reset_index()
)

pivot_data = grouped_data.pivot(index='instructional_level', columns='is_paid', values='avg_subscribers').fillna(0)

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by is_paid and Instructional Level per course', fontsize=14)
plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('Instructional Level', fontsize=12)
plt.legend(title='is_paid', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
# Group by 'category' and 'instructional_level' and sum 'num_subscribers'
grouped_data = (
    df.groupby(['is_paid', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))  # Calculate mean
    .reset_index()
)


grouped_data['probability'] = grouped_data.groupby('instructional_level')['avg_subscribers'].transform(lambda x: x / x.sum())

pivot_data = grouped_data.pivot(index='instructional_level', columns='is_paid', values='probability').fillna(0)

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=True, width=0.7)

plt.title('Number of Subscribers(probabilities) by is_paid and Instructional Level per course', fontsize=14)
plt.ylabel('probability of num_subscribers', fontsize=12)
plt.xlabel('Instructional Level', fontsize=12)
plt.legend(title='is_paid', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

### Using Instructor_names and combine with other features to see which instructors have given hit courses
### doing basic analysis instructor names
pd.options.display.float_format = '{:.2f}'.format
df['num_subscribers'].describe()
grby_category=df.groupby('category')
photo_courses=grby_category.get_group('Photography & Video')
finance_courses=grby_category.get_group('Finance & Accounting')
fitness_courses=grby_category.get_group('Health & Fitness')
development_courses=grby_category.get_group('Development')
teaching_courses=grby_category.get_group('Teaching & Academics')
PersDevelopment_courses=grby_category.get_group('Personal Development')
design_courses=grby_category.get_group('Design')
lifestyle_courses=grby_category.get_group('Lifestyle')
marketing_courses=grby_category.get_group('Marketing')
music_courses=grby_category.get_group('Music')
officeProductivity_courses=grby_category.get_group('Office Productivity')
business_courses=grby_category.get_group('Business')
software_courses=grby_category.get_group('IT & Software')
### analysing succesful music instructors
music_courses['num_subscribers'].describe()
#filtering superhit music courses
filt1=(music_courses['num_subscribers']>50000) & (music_courses['rating']>=4.5)
df1=music_courses[filt1].reset_index(drop=True)
df1
df1['instructor_names']
music_courses['rating'].describe()
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have good rating
filt2=(music_courses['rating']==5) & (music_courses['num_subscribers']>500) & ((music_courses['instructional_level']=='Expert Level') | (music_courses['instructional_level']=='Intermediate Level'))
df2=music_courses[filt2] 
df2
df2['instructor_names']

### anaysing photography and video courses
photo_courses['rating'].describe()
photo_courses['num_subscribers'].describe()
#filtering superhit photography courses
df3=photo_courses[(photo_courses['num_subscribers']>=100000) & (photo_courses['rating']>=4.5)]
df3
pd.set_option('display.max_colwidth', None)
print(df3['instructor_names'])
pd.reset_option('display.max_colwidth')
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have good rating
filt3=(photo_courses['rating']==5) & (photo_courses['num_subscribers']>500) & ((photo_courses['instructional_level']=='Expert Level') | (photo_courses['instructional_level']=='Intermediate Level'))
photo_courses[filt3]

### analyzing lifestyle courses
lifestyle_courses['num_subscribers'].describe()
## check for instructors of superhit courses
lifestyle_courses[lifestyle_courses['num_subscribers']>60000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have good rating
lifestyle_courses[(lifestyle_courses['rating']==5) & (lifestyle_courses['num_subscribers']>800) & ((lifestyle_courses['instructional_level']=='Expert Level') | (lifestyle_courses['instructional_level']=='Intermediate Level'))]

## analysing fitness courses
fitness_courses['num_subscribers'].describe()
#filtering superhit courses
fitness_courses[fitness_courses['num_subscribers']>70000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have good rating
fitness_courses[(fitness_courses['rating']==5) & (fitness_courses['num_subscribers']>800) & ((fitness_courses['instructional_level']=='Expert Level') | (fitness_courses['instructional_level']=='Intermediate Level'))]
 
### analysing teaching courses
teaching_courses['num_subscribers'].describe()
#filtering most hit teaching courses
teaching_courses[teaching_courses['num_subscribers']>120000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have good rating
teaching_courses[(teaching_courses['rating']==5) & (teaching_courses['num_subscribers']>2000) & ((teaching_courses['instructional_level']=='Expert Level') | (teaching_courses['instructional_level']=='Intermediate Level'))]

### analysing personal development courses
PersDevelopment_courses['num_subscribers'].describe()
#filtering most hit courses
PersDevelopment_courses[PersDevelopment_courses['num_subscribers']>120000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
PersDevelopment_courses[(PersDevelopment_courses['rating']==5) & (PersDevelopment_courses['num_subscribers']>1100) & ((PersDevelopment_courses['instructional_level']=='Expert Level') | (PersDevelopment_courses['instructional_level']=='Intermediate Level'))]

### analysing finance courses
finance_courses['num_subscribers'].describe()
#filtering most hit finance courses
finance_courses[finance_courses['num_subscribers']>160000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
finance_courses[(finance_courses['rating']==5) & (finance_courses['num_subscribers']>1030) & ((finance_courses['instructional_level']=='Expert Level') | (finance_courses['instructional_level']=='Intermediate Level'))]

## analysing design courses
design_courses['num_subscribers'].describe()
#filtering most hit design courses
design_courses[design_courses['num_subscribers']>200000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
design_courses[(design_courses['rating']==5) & (design_courses['num_subscribers']>900) & ((design_courses['instructional_level']=='Expert Level') | (design_courses['instructional_level']=='Intermediate Level'))]

### analysing marketing courses
marketing_courses['num_subscribers'].describe()
#filtering most hit marketing courses
marketing_courses[marketing_courses['num_subscribers']>220000]
# notice that highest subscriber is 8 lakh where 2nd highest is 3 lakh 40 thousand
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
marketing_courses[(marketing_courses['rating']==5) & (marketing_courses['num_subscribers']>900) & ((marketing_courses['instructional_level']=='Expert Level') | (marketing_courses['instructional_level']=='Intermediate Level'))]

### analysing office productivity courses
officeProductivity_courses['num_subscribers'].describe()
#filtering most hit courses
officeProductivity_courses[officeProductivity_courses['num_subscribers']>200000]
# notice the course has highest subscriber as 15 lakh and 2nd highest 7 lakh 25 thousand
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
officeProductivity_courses[(officeProductivity_courses['rating']>=4.9) & (officeProductivity_courses['num_subscribers']>700) & ((officeProductivity_courses['instructional_level']=='Expert Level') | (officeProductivity_courses['instructional_level']=='Intermediate Level'))]

### analysing business courses
business_courses['num_subscribers'].describe()
#filtering most hit business courses
business_courses[business_courses['num_subscribers']>350000]
## notice that all hit courses are 'All Level' courses
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
business_courses[(business_courses['rating']==5) & (business_courses['num_subscribers']>900) & ((business_courses['instructional_level']=='Expert Level') | (business_courses['instructional_level']=='Intermediate Level'))]

## analysing software courses
pd.options.display.float_format = '{:.2f}'.format
software_courses['num_subscribers'].describe()
#filtering most hit courses
software_courses[software_courses['num_subscribers']>450000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
software_courses[(software_courses['rating']==5) & (software_courses['num_subscribers']>900) & ((software_courses['instructional_level']=='Expert Level') | (software_courses['instructional_level']=='Intermediate Level'))]

## analysing development courses
development_courses['num_subscribers'].describe()
#filtering most hit development courses
development_courses[development_courses['num_subscribers']>900000]
# filtering intermediate and expert level courses which are not that hit in num_subscribers but they have excellent rating
development_courses[(development_courses['rating']>=4.9) & (development_courses['num_subscribers']>2000) & ((development_courses['instructional_level']=='Expert Level') | (development_courses['instructional_level']=='Intermediate Level'))]

### Creating subcategories
creating subcategories for music courses
pd.set_option('display.max.colwidth',None)
pd.set_option('display.max.rows',3854)
music_courses['title']
pd.set_option('display.max.colwidth',None)
pd.set_option('display.max.rows',3854)
music_courses.loc[:,['title','objectives']]
musiccourses_new=music_courses.copy()
subcategories = {
    'music production steps and softwares/tools': ['reason','avid','cubase','akai','sibelius','serum','fl studio',
                                                   'garageband','cubase','maschine','personus','equalization',
                                                   'equalize','record','recording','compose','studio one','compress','compression',
                                                   'orchestration','sound design','ableton','logicpro','logic pro',
                                                   'logic pro x','audio mastering','audio production','garage band'],
    'DJ & EDM': ['dj','edm','remix','mix','remixes','mixes','traktor'],
    'music theory': ['electronic music','bass','harmony','abrsm'],
    'singing and listening music':['vocal','sing','singing','breathing','write','songwriter','song writer',
                                   'voice','throat','aural','hearing','ear'],
    'guitar':['guitar'],
    'piano':['piano'],
    'Harmonium':['Harmonium','Harmonica'],
    'music healing':['healing','therapy','relaxing','treat','treating'],
    'indian music':['hindustani','indian','bansuri','kartal','mradanga','tabla','sitar','bharat','bhartiya'],
    'ukulele':['ukulele'],
    'other musical instruments':['violin','flute','trumpet','drum','drummers','saxophone','sax','cellophane'
                                 'cello','cellophone','mandolin','banjo','clarinet','bouzouki','oscarina',
                                 'ocarina']
              
}
def assign_subcategory(row):
    text = ' '.join([
        str(row['title']), 
        str(row['objectives']), 
    ]).lower() 

    
    for subcategory, keywords in subcategories.items():
        if any(keyword.lower() in text for keyword in keywords):
            return subcategory
    return 'Other'

# Apply the function to the dataset
musiccourses_new['subcategory'] = musiccourses_new.apply(assign_subcategory, axis=1)
musiccourses_new.head()
musiccourses_new['subcategory'].value_counts()
sns.barplot(data=musiccourses_new,x='subcategory',y='num_subscribers',errorbar=None,estimator='mean')
plt.ylabel('average num_subscribers')
plt.xticks(rotation=90)
plt.show()
grouped_data = (
    musiccourses_new.groupby(['subcategory', 'is_paid'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

pivot_data = grouped_data.pivot(index='subcategory', columns='is_paid', values='avg_subscribers')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by subategory and is_paid per course', fontsize=14)
plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('subcategory', fontsize=12)
plt.legend(title='is_paid', fontsize=10)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.show()
grouped_data = (
    musiccourses_new.groupby(['subcategory', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

pivot_data = grouped_data.pivot(index='subcategory', columns='instructional_level', values='avg_subscribers')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('subcategory', fontsize=12)
plt.legend(title='instructional_level', fontsize=10)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.show()

creating subcategory for development courses
pd.set_option('display.max.colwidth',9945)
pd.set_option('display.max.rows',9945)
development_courses[['title','objectives']]
pd.reset_option('display.max.colwidth')
developmentcourses_new=development_courses.copy()
subcategory_keywords = {
    "Web Development & Database": [
        {"html"}, {"css"}, {"javascript"}, {"react"}, {"frontend"}, {"backend"},{'scala','web'},{'java','web'},
        {"nextjs"}, {"expressjs"}, {"tailwind"}, {"angular"}, {"nodejs"}, 
        {"web development"}, {"spring boot"}, {"mongodb"}, {"mongo db"}, 
        {"sql"}, {"postgres"}, {"wordpress"}, {"full stack"}, {"mern stack"}, {"php"}, {"database"}, {"rust", "web"}
    ],
    "AI/ML_Data": [
        {"machine learning"}, {"ml"}, {"ai"}, {"deep learning"}, {"neural network"},{'backpropagation'},
        {"data science"}, {"spark"}, {"pytorch"}, {"numpy"}, {"pandas"}, {"seaborn"},{'artificial intelligence'},
        {"matplotlib"}, {"tensorflow"}, {"tableau"}, {"power bi"}, {"chatgpt"},{'scala','data'},
        {"openai"}, {"gen ai"}, {"llm"}, {"big data"}, {"eda"}, {"data analysis"}, 
        {"apache"}, {"nlp"}, {"computer vision"}, {"sklearn"}, {"scikit learn"}, 
        {"statistics"}, {"calculus"}, {"pyspark"}
    ],
    "Blockchain": [{"blockchain"}, {"etherium"}],
    "Android Development": [
        {"android"}, {"ios"}, {"kotlin"}, {"mobile app"}, {"flutter"}, {"dart"}, 
        {"appium"}, {"xamarin"}
    ],
    "DevOps": [
        {"devops"}, {"docker"}, {"kubernetes"}, {"ci/cd"}, 
        {"continuous integration"}, {"jenkins"}, {"gitlab"}
    ],
    "Web Scraping": [{"selenium"}, {"beautiful soup"}],
    "Cloud": [{"aws"}, {"azure"}],
    "Cyber Security": [{"bug"}, {"bug testing"}],
    "Python_Web": [{"python"},{'flask'},{'django'},{'fastapi'}],
    "data_structure & other_prog_languages ": [{"data structures",'algorithm'},{"data structures",'algorithms'}, 
                                           {"object oriented"}, {"oops"},{'c'},{'c++'},{'c#'}],
    "Version Control": [{"git"}, {"github"}],
    "Game Development": [{"unity"}, {"game"}, {"unreal game engine"},{'unreal engine'}]
}

def assign_subcategory(row, subcategory_keywords):
    text = f"{row['title']}  {row['objectives']} ".lower()
    for subcategory, keyword_sets in subcategory_keywords.items():
        for keywords in keyword_sets:
            if all(keyword in text for keyword in keywords):
                return subcategory
    return "Other"

developmentcourses_new["subcategory"] = developmentcourses_new.apply(assign_subcategory, axis=1, args=(subcategory_keywords,))
developmentcourses_new.head()
developmentcourses_new['subcategory'].value_counts()
pd.set_option('display.max.colwidth',None)
developmentcourses_new[developmentcourses_new['subcategory']=='Other'][['title','objectives']]
sns.barplot(data=developmentcourses_new,x='subcategory',y='num_subscribers',estimator='mean',errorbar=None)
plt.xticks(rotation=90)
plt.ylabel('subscribers on average')
plt.show()
grouped_data = (
    developmentcourses_new.groupby(['subcategory', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

pivot_data = grouped_data.pivot(columns='instructional_level', index='subcategory', values='avg_subscribers')

pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by subcategory and instructional_level per course', fontsize=14)
plt.ylabel('Average Subscribers', fontsize=12)
plt.xlabel('subcategory', fontsize=12)
plt.legend(title='instructional_level', fontsize=10)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.show()
grouped_data = (
    developmentcourses_new.groupby(['subcategory', 'instructional_level'])
    .agg(avg_subscribers=('num_subscribers', 'mean'))
    .reset_index()
)

grouped_data['probability'] = grouped_data.groupby('subcategory')['avg_subscribers'].transform(lambda x: x / x.sum())

pivot_data = grouped_data.pivot(columns='instructional_level', index='subcategory', values='probability')
pivot_data.plot(kind='bar', figsize=(12, 8), stacked=False, width=0.7)

plt.title('Number of Subscribers on avg by subcategory and instructional_level per course', fontsize=14)
plt.ylabel('probability of Subscribers', fontsize=12)
plt.xlabel('subcategory', fontsize=12)
plt.legend(title='instructional_level', fontsize=10)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.show()



