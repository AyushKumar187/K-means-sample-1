import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import re
from sklearn.preprocessing import Imputer
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt 

#Setting dataset path
dataset_path = "../input/77_cancer_proteomes_CPTAC_itraq.csv"
clinical_info = "../input/clinical_data_breast_cancer.csv"
pam50_proteins = "../input/PAM50_proteins.csv"
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)
pam50 = pd.read_csv(pam50_proteins,header=0)
data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)

#Changing sample data names to clinical name
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)
data = data.transpose()
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]
merged = data.merge(clinical,left_index=True,right_index=True)

#Changing name
processed = merged
processed_numerical = processed.loc[:,[x for x in processed.columns if bool(re.search("NP_|XP_",x)) == True]]
processed_numerical_p50 = processed_numerical.ix[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]

#Imputing Missing values
imputer = Imputer(missing_values='NaN', strategy='median', axis=1)
imputer = imputer.fit(processed_numerical_p50)
processed_numerical_p50 = imputer.transform(processed_numerical_p50)

#Checking what number of clusters works
n_clusters = [2,3,4,5,6,7,8,10,20,79]

def compare_k_means(k_list,data):
    # Run clustering with different k and check the metrics
    for k in k_list:
        clusterer = KMeans(n_clusters=k, n_jobs=4)
        clusterer.fit(data)
        #The higher (up to 1) the better
        print("Silhouette Coefficient for k == %s: %s" % (
        k, round(metrics.silhouette_score(data, clusterer.labels_), 4)))
        #The higher (up to 1) the better
        print("Homogeneity score for k == %s: %s" % (
        k, round(metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_),4)))
        print("------------------------")

#Lets use 52 proteins and check the clusters
        processed_numerical_random = processed_numerical.iloc[:,random.choice(range(processed_numerical.shape[1]),52)]
imputer_rnd = imputer.fit(processed_numerical_random)
processed_numerical_random = imputer_rnd.transform(processed_numerical_random)

#Comparing with 3 samples
compare_k_means(n_clusters,processed_numerical_p50)

#Using random no. of proteins
compare_k_means(n_clusters,processed_numerical_random)

#Lets visualize the data with k = 5 and plot on the graph
clusterer_final = KMeans(n_clusters=5, n_jobs=6)
clusterer_final = clusterer_final.fit(processed_numerical_p50)
processed_p50_plot = pd.DataFrame(processed_numerical_p50)
processed_p50_plot['KMeans_cluster'] = clusterer_final.labels_
processed_p50_plot.sort('KMeans_cluster',axis=0,inplace=True)
processed_p50_plot.index.name = 'Patient'
sb.heatmap(processed_p50_plot) # The x-axis are the PAM50 proteins we used and the right-most column is the cluster marker
plt.savefig('cluster.png')
# Looks like the clustering works

# Each cluster means a different molecular signature for each patient. Such patients have different treatment options available to them



