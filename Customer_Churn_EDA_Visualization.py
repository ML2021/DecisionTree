
#  Data Source: https://www.kaggle.com/blastchar/telco-customer-churn

#%%  Import the required libraries

import pandas            as pd    # For data handling
import matplotlib.pyplot as plt   # For plotting graphs
import matplotlib.ticker as mtick # For specifying axes ticks
import seaborn           as sns   # For interactive plots
import os

#import numpy             as np   # For math calculations

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split




churn = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 


#%%  Explore the basic features of the dataset

print churn.shape      #  shows the no. of rows/cols
print churn.dtypes     #  lists the column types


##  Get a first-hand view of the feature values

print churn.iloc[:,   : 7].head()
print churn.iloc[:,  7:12].head()
print churn.iloc[:, 12:17].head()
print churn.iloc[:, 17:  ].head()
print churn['Churn'].head()


##  Remove the customerID columns (not related to churn)

del churn['customerID']


##  Check for missing/blank values

print churn.isnull().sum()
print churn.eq(' ').any(axis=0)
print churn['TotalCharges'].str.count(' ').sum()

##  'TotalCharges' has 11 blank values. We can either drop these rows...
##       or  change them to '0' and convert TotalCharges --> 'float'

churn['TotalCharges'] = churn['TotalCharges'].str.replace(' ','0')
churn['TotalCharges'] = churn['TotalCharges'].astype(float)


#%%

##  Plot some features (Density & Bar plots) to visualize the
##  dependence of Churn  on  tenure and Monthly/Total Charges
##    or the percentage of Churn in [non-]Senion Citizens 

#%%  Primary visualization  of  Overall Churn percentage

sizes   = churn['Churn'].value_counts(sort = True)
labels  = 'Does not Churn', 'Churns'
colors  = ['lightskyblue', 'lightcoral']
explode = (0.1, 0)  # explode Churns

fig     = plt.figure()
plt.pie(sizes, 
        labels=labels, 
        colors=colors,
        explode=explode,
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=-180)

plt.axis('equal')
plt.title('% of Churn dataset')
plt.show()
fig.savefig('Churn_pi-chart.pdf')


#%%   tenure -- density plot

fig= plt.figure()
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'Yes') ], 
                color="salmon", shade = True)
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'No') ], 
                ax=ax, color="royalblue", shade= True)
ax.set_xlabel("tenure")
ax.set_ylabel("Frequency")
plt.title('Variation of "tenure" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_tenure_density-plot.pdf')

#%%  MonthlyCharges -- density plot

fig= plt.figure()
ax = sns.kdeplot(churn.MonthlyCharges[(churn["Churn"] == 'Yes') ], 
                color="orangered", shade = True)
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'No') ], 
                ax=ax, color="cornflowerblue", shade= True)
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Frequency")
plt.title('Variation of "Monthly Charges" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_MnthlyChrgs_density-plot.pdf')


#%%  TotalCharges -- density plot

fig= plt.figure()
ax = sns.kdeplot(churn.TotalCharges[(churn["Churn"] == 'Yes') ], 
                color="lightcoral", shade = True)
ax = sns.kdeplot(churn.TotalCharges[(churn["Churn"] == 'No') ], 
                ax=ax, color="steelblue", shade= True)
ax.set_xlabel("Total Charges")
ax.set_ylabel("Frequency")
plt.title('Variation of "Total Charges" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_TotalChrgs_density-plot.pdf')

#%%  Senrion Citizen -- Stacked bar chart

colors = ['slategrey','saddlebrown']
churn_sczn = churn.groupby(['SeniorCitizen','Churn']).size().unstack()
fig= plt.figure()

ax = (churn_sczn.T*100.0/ 
      churn_sczn.T.sum()).T.plot(kind='bar',
                                 width = 0.4,
                                 stacked = True,
                                 rot = 0, 
                                 figsize = (6,6),
                                 color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='upper right',prop={'size':12},title = 'Churn')
ax.set_ylabel('% Customers')
ax.set_title('% Churn in Senion Citizens = 0, 1',size = 13)

#
# Add data labels to the bar diagram
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.1f}%'.format(height), 
                (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
                weight = 'normal',
                size =14)
#fig.savefig('Churn_Sen-Cit_stacked-bar.pdf')







































##  Explore the data (particularly its variables)

#num_bins = 2
#n, bins, patches = churn['gender'].hist(num_bins, facecolor='blue', alpha=0.5)
#plt.show()


churn[churn.dtypes[(churn.dtypes=="float64")|(churn.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])

churn[churn.dtypes[(churn.dtypes=="object")]
                        .index.values].hist(figsize=[11,11])


#%%

##  Decision Tree Model


###  First convert all categoric data to numeric

churn['Gender']  = churn['Gender' ].map(lambda s :1  if s =='Male' else 0)
churn['Partner'] = churn['Partner'].map(lambda s :1  if s =='Yes' else 0)


#from sklearn.feature_extraction import DictVectorizer
#
#categ_cols = [0, 2,3, 5,6,7,8,9,10,11,12,13,14,15,16, 19]
#flint_cols = [  1,   4,                              17,18]
#
#churn_dict = churn.iloc[:, categ_cols].T.to_dict().values()
#
#vect       = DictVectorizer(sparse=False)
#churn_vect = vect.fit_transform(churn_dict)


X = churn.values[:, :18]
y = churn.values[:,  19]

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size = 0.2, 
                                    random_state = 39)

clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 39,
                                  max_depth=3, 
                                  min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

#%%

##  Convert categoric to integer



##  Scale the data (except the class column 'Churn')
##     [does not work for categoric variables]

# from sklearn import preprocessing
# 
# churn_scl = preprocessing.scale(churn)



#%%

##  Rules with Orange CN2 algorithm


## Read some data
#titanic = Orange.data.Table("titanic")
#
## construct the learning algorithm and use it to induce a classifier
#cn2_learner = Orange.classification.rules.CN2Learner()
#cn2_clasifier = cn2_learner(titanic)
#
## ... or, in a single step.
#cn2_classifier = Orange.classification.rules.CN2Learner(titanic)
#
## All rule-base classifiers can have their rules printed out like this:
#for r in cn2_classifier.rules:
#    print Orange.classification.rules.rule_to_string(r)
