#!/usr/bin/python

import sys
import pickle
sys.path.insert(0,"/home/allan/Desktop/enron_fraud_detection/tools")
import math
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from merge_data import merge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi",
#"total_payments",
#'other',
"deferred_income", 
"exercised_stock_options", 
"expenses", 
"bonus", 
"restricted_stock",
"email_feature",
"word_pca_0","word_pca_1","word_pca_2","word_pca_3","word_pca_4","word_pca_5","word_pca_6",
"word_pca_7","word_pca_8","word_pca_9","word_pca_10","word_pca_11","word_pca_12","word_pca_13",
"word_pca_14","word_pca_15","word_pca_16","word_pca_17","word_pca_18","word_pca_19",
"word_pca_20","word_pca_21","word_pca_22","word_pca_23","word_pca_24","word_pca_25",
"word_pca_26","word_pca_27","word_pca_28","word_pca_29","word_pca_30","word_pca_31",
"word_pca_32","word_pca_33","word_pca_34","word_pca_35","word_pca_36","word_pca_37",
"word_pca_38","word_pca_39","word_pca_40","word_pca_41","word_pca_42","word_pca_43",
"word_pca_44"]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers / Prepare DataSet
# get rid of non-person entries
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict['PAI LOU L']['poi']=True
data_dict['BAXTER JOHN C']['poi']=True
data_dict['FASTOW ANDREW S']['poi']=True


#
### Get features from dataset.
my_features = []
names = []
for person in data_dict.keys():
    person_features = []
    names.append(person)
    for feature in features_list[0:5]:
        if not data_dict[person][feature] == "NaN":
            person_features.append(data_dict[person][feature])
        else:
            person_features.append(np.nan)
    my_features.append(person_features)

### replace NaN's w/ 0 and outliers with the non outlier mean.
my_features=np.array(my_features)
for i in range(len(my_features[0])):
    col = my_features[:,i]
    nonancol = np.array(col[col!="NaN"])
    percentile = np.percentile(nonancol,90)
    outliers=(col > percentile)
    non_outlier_mean= col[(~outliers).nonzero()].mean()
    col_new=col*1
    col_new[outliers.nonzero()] = non_outlier_mean
    my_features[:,i]=col_new*1
#for i in my_features:
#    for j in i:
#        if not isinstance(j,float):
#            print j

### normalize using a MinMax Scalar
my_features = np.subtract(my_features,my_features.min(0))
my_features = np.divide(my_features,my_features.max(0))

#my_dataset={}
for i in range(len(names)):
    for j in range(len(my_features[0])):
        if not np.isnan(my_features[i][j]):
            data_dict[names[i]][features_list[j]]=my_features[i][j]

### Task 3: Create new feature(s)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif

### Create a principal component of all the email features...
email_features = []
names = []
for person,values in data_dict.items():
    email_features.append([ float(values["from_this_person_to_poi"]), float(values["from_poi_to_this_person"]), float(values["shared_receipt_with_poi"])])
    names.append(person)

email_features = np.array(email_features)
minimum = email_features[~np.isnan(email_features)].min(0)
maximum = email_features[~np.isnan(email_features)].max(0)
email_features = email_features-minimum
email_features = email_features/maximum
email_features[np.isnan(email_features)]=0
pca = PCA(n_components=1)
email_pc = pca.fit_transform(email_features).tolist()

###Ged word data and select best word features...
word_data = pickle.load(open("/home/allan/Desktop/enron_fraud_detection/data/qualitative.pkl"))
words = word_data.pop('words')
word_names = np.array(word_data.keys())
## filter out names, emails, and words that don't seem to carry useful information...
remove_list=np.array(["stdolelink", "ene", "te", "skillingcorpenronenron", "despain", "berney", 'clwellhouect', 'elliot', 'frevertnaenronenron', 'mcmahonhouect', 'prestohouectect', 'rodney', 'scrimshawlonectect', "colwellhouectect", "coxenron", 'glisan', "billi", u'baxterhouectect', u'belden', u'benjamin',  u'calger', u'causeycorpenronenron', u'deffnerhouectect', u'delaineyhouectect', u'dietrichhouectect', u'dori', u'eugen', u'fastowhouectect',  u'glisanhouectect', u'hannonenron', u'janic', u'jefferi', u'joanni', u'jodi', u'jonathan', u'kitchenhouectect', u'koenig',  u'las', u'laycorpenronenron', u'mccullough', u'milnthorpcalectect',  u'montana', u'mrhanaenronenron', u'parson', u'sweeney', u'theresa', u'tion', u'warren', u'williamson', u'wong', u'yzaguirrenaenronenron', u"barney", u'bowenhouectect', u'cc20', u'jskillinpst', u'mckinsey', u'worthwhil', u'aa', u'ac', u'ae', u'al20', u'ami', u'announcementsenron', u'ar', u'arnold', u'arnoldhouectect', u'barbara', u'baughmanhouectect', u'beldenhouectect', u'betti', u'blvd', u'brandon', u'brenda', u'brownenron', u'business20', u'buyhouectect', u'calgerpdxectect', u'carson', u'carter', u'catherin', u'cathi', u'ch', u'chapmanhouectect', u'cogen', u'com', u'cr', u'davishouectect',u'delaineyhoue', u'divid', u'duranhouectect', u'dysonlonectect', u'elliott', u'erica', u'felicia', u'fitzpatrick', u'fletcher', u'foster', u'fosterhouectect', u'freeman', u'fuller', u'germani', u'gloria', u'goldlonectect', u'gort', u'griffin', u'gus', u'hannon', u'hodg', u'hughesenrondevelop', u'jane', u'janin', u'jeann', u'jill', u'jskillinnsf', u'kate', u'katherin', u'kathi', u'kathleen', u'kelli', u'kenni', u'keohan', u'king', u'kishkil', u'klay', u'koenigcorpenronenron', u'kopperhouectect', u'lauren',
u'lavoratocorpenronenron', u'layenron', u'll', u'lloyd', u'lori', u'loui', u'mailtoimceanotesenron20announcementscorpenron40enronenroncom', u'maureen', u'max', u'melissa', u'milnthorp', u'mintzhouectect', u'mitch', u'mo', u'moorehouectect', u'nick', u'norman', u'nron', u'nt', u'ot', u'oxleyhouectect', u'paihoueese', u'palmer', u'pamela', u'patterson', u'peaker', u'perkin', u'philip', u'pipercorpenronenron', u'ppas', u'pricehouectect', u'reckhouectect', u'rex', u'riceenron', u'ride', u'rieker', u'rita', u'rode', u'rosale', u'rose', u'rosi', u'sagerhouectect', u'schneider', u'schultz', u'sempra', u'sent09friday', u'sera',
u'seracorpenronenron', u'seyfri', u'sharron', u'skillingenron', u'stubblefield', u'sutton', u'suzann', u'swerzbin', u'sylvia', u'tawneyhouectect', u'thodecorpenronenron', u'thur', u'timothi', u'valencia', u'valley', u'vega', u'vincent', u'wendi', u'westbrook', u'westin', u'whilst', u'whitehoueese', u'wigg', u'wolf', u'wolfehouectect', u'worldwid', u'worldwideenron'])

remove_ind = words.searchsorted(remove_list)
keep_ind = np.arange(len(words))
print len(keep_ind), "words total"
keep_ind=np.delete(keep_ind,remove_ind)

##extract word_data
word_values = []
word_labels = []
for name in word_names:
    word_labels.append(data_dict[name]["poi"])
for row in word_data.values():
    word_values.append((row.toarray()[0]*1.).tolist())
word_values=np.array(word_values)

print "asdf:",sum(word_labels)

##remove words indicated above and 
words=words[keep_ind]
word_values=word_values[:,keep_ind]
word_values = word_values-minimum
word_values = word_values/maximum

best_words = SelectKBest(f_classif,225)
k_best=best_words.fit_transform(word_values,word_labels)
word_feature_ind=best_words.get_support()
word_feature_names = words[word_feature_ind]
#print word_feature_names

#fit PCA components to word data to reduce dimensionality..
pca2=PCA(n_components=86)
word_pca = pca2.fit_transform(word_values[:,word_feature_ind])
best_pca = SelectKBest(f_classif,45)
qqq = best_pca.fit_transform(word_pca,word_labels)
best_pca_ind=best_pca.get_support()

#extract best component decompositions
word_pca=word_pca[:,best_pca_ind]
#normalize
word_pca=word_pca/word_pca.max(0)

#print words[pca2.components_[0]>.01]
#print len(word_pca[0])
#for i in range(len(word_labels)):
#    print word_pca[i][0],word_labels[i]

## insert new feature values into dataset
my_dataset = data_dict.copy()
for i in range(len(names)):
    my_dataset[names[i]]["email_feature"] = email_pc[i][0]
    for j in range(len(words)):
        if names[i] in word_names:
            my_dataset[names[i]][words[j]]=word_values[(word_names==names[i]).nonzero(),j]
        else:
            my_dataset[names[i]][words[j]]=0.
    for j in range(len(word_pca[0])):
	if names[i] in word_names:
            my_dataset[names[i]]["word_pca_"+str(j)]=word_pca[(word_names==names[i]).nonzero(),j]
        else:
            my_dataset[names[i]]["word_pca_"+str(j)]=0.
word_pca_features = ["word_pca_"+str(i) for i in range(len(word_pca[0]))]
### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset,["poi"]+ features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

dt = DecisionTreeClassifier()    
gnb = GaussianNB()
knn=KNeighborsClassifier(n_neighbors=15,weights="distance")
rfc = RandomForestClassifier()
adb_base = DecisionTreeClassifier()
#adb = AdaBoostClassifier()
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset,["poi"]+ features_list)
#test_classifier(dt, my_dataset,["poi"]+ word_pca_features + features_list+["email_feature"])
#test_classifier(gnb, my_dataset,["poi"]+ word_pca_features)#+ features_list+["email_feature"])
#test_classifier(knn, my_dataset,["poi"]+features_list+["email_feature"])
#test_classifier(rfc, my_dataset,["poi"]+ word_feature_names.tolist() + features_list+["email_feature"])
#test_classifier(adb, my_dataset,["poi"]+ word_pca_features+ features_list+["email_feature"])

### Task 6: Combine Classifiers and features for numerical and text datasets into a single metaclassifier...

# create a FeatureSelector Class so that we can use different features for each base classifier.
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_index):
        self.feature_index=feature_index
    def fit(self, x, y=None):
        return self
    def transform(self, features):
        return np.array(features)[:,self.feature_index]

# create Base Classifiers Using Pipeline...
from sklearn.pipeline import Pipeline
base_clf_1=Pipeline([("knntransform",FeatureSelector([i for i in range(5)]) ), ("knn",knn)])
base_clf_2=Pipeline([("gnbtransform",FeatureSelector([i for i in range(50)])),("gnb",gnb)])

# create a CombinedClassifier class in order to merge predictions of each classifier...
from sklearn.base import ClassifierMixin
class CombinedClassifier(BaseEstimator):
    def __init__(self,base_clf_1,base_clf_2):
        self.base_clf_1 = base_clf_1
        self.base_clf_2 = base_clf_2
    def fit(self,X,y):
        self.base_clf_1.fit(X,y)
        self.base_clf_2.fit(X,y)
    def predict(self,X):
        predictions = [self.base_clf_1.predict(X),self.base_clf_2.predict(X)]
	qqq = [predictions[0][i]+predictions[1][i] for i in range(len(X))]
        meta_predictions = []        
	for i in qqq:
            if i ==2:
                meta_predictions.append(1)
            else: 
                meta_predictions.append(i)
        return meta_predictions

clf = CombinedClassifier(base_clf_1,base_clf_2)
#test_classifier(clf, my_dataset, features_list)
#test_classifier(GaussianNB(), my_dataset, ["poi"]+ word_pca_features)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(GaussianNB(), my_dataset, ["poi"]+word_pca_features)
