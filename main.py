from __future__ import print_function
import os
import subprocess
import re
import SPARQLWrapper
import json
import parmap
import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from Axioms import Axioms
import threading
import time
from multiprocessing import Pool
import logging
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef 
from sklearn import  ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile as SP
from sklearn.model_selection import train_test_split as tts

# useful sparql queries
#construct to select only the things we need from an ontology
        # construct {
        # ?class1 a owl:Class
        # ?class2 a owl:Class
        # ?class1 ?y ?class2
        # } 
        # where {
        #   ?class1 a owl:Class
        #   ?class2 a owl:Class
        #   ?class1 ?y ?class2
        #   filter (!isBlank(?class1)  && !isBlank(?class2))
        # }


#parameters that can be edited
def setParam(P_threadcount = 24, P_split = 1000, P_sparql_endpoints = 4,  P_prefix = 'http://dbpedia.org/ontology/' ,  P_relation = 'owl:disjointWith', P_path = 'fragments/',
             P_corese_path = os.path.normpath(r"C:\corese\corese-server"), 
             P_rdfminer_path = os.path.normpath(r"C:\corese\RDFMining"),
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx20G corese-server-4.3.0.jar -e -lp -debug -pp profile.ttl', P_dataset = 'dbpediaHPC.owl',
             P_wds_Corese = 'http://localhost:8080/sparql', P_label_type = 'c', P_list_of_axioms = None, P_score = None,  P_dont_score = True, P_set_axiom_number = 0):
    global threadcount    #number of process for multiprocessing avoid using logical cores
    global split          # divide the table you are working on into tasks, the more processors the more you can divide
    global prefix         #use this to reduce the search time and make thigs more readable
    global relation       #the axiom/relation we are extracting
    global path           #the path of kernel builder should be edited in the .py file itself
    global corese_path    #parameters to launch corese server
    global command_line
    global wds_Corese
    global allrelations  #the whole axiom dataset we are working with
    global label_type    #either classification of regression so either a score or a binary label
    global axiom_type    # disjoint, subclass, equivilent or same as 
    global list_of_axioms
    global score
    global list_df
    global rdfminer_path
    global dont_score
    global set_axiom_number
    global sparql_endpoints
    global dataset
    list_df = []
    dataset = P_dataset
    sparql_endpoints = P_sparql_endpoints
    dont_score = P_dont_score
    threadcount = P_threadcount
    split = P_split
    prefix = P_prefix
    relation = P_relation
    path = P_path
    corese_path = P_corese_path
    command_line = P_command_line
    wds_Corese = P_wds_Corese
    label_type = P_label_type
    rdfminer_path = P_rdfminer_path
    #finish for other axiom types
    if P_relation == 'owl:disjointWith':
        axiom_type = 'DisjointClasses'
    elif P_relation == 'rdfs:subClassOf':
        axiom_type = 'SubClassOf'
    set_axiom_number = P_set_axiom_number
    list_of_axioms = P_list_of_axioms
    score = P_score
    
# Read a list of axioms, extract unique concepts to use in creating a precise concept similarity matrix (finish axiom types)
def clean_scored_atomic_axioms(labeltype = 'c', axiomtype = "SubClassOf", score_ = None,sample = True):
    
    valid = {'c', 'r'}
    if labeltype not in valid:
        raise ValueError("labeltype must be one of %r." % valid)
           
    scored_axiom_list = pd.read_json( rdfminer_path+'\\IO\\'+ score_)
    scored_axiom_list = scored_axiom_list[['axiom', 'numConfirmations', 'referenceCardinality', 'numExceptions', 'generality', 'possibility', 'necessity']]
    scored_axiom_list = scored_axiom_list.drop_duplicates('axiom')#drop duplicates
    scored_axiom_list = scored_axiom_list[scored_axiom_list.referenceCardinality != 0].reset_index(drop = True)#remove axioms whos concepts have no instances
    scored_axiom_list['left'], scored_axiom_list['right'] = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')).apply(lambda x: x.replace('SubClassOf','')) ))
    scored_axiom_list = scored_axiom_list[scored_axiom_list.left != scored_axiom_list.right].reset_index(drop = True)
    
    if axiomtype == "DisjointClasses":
        scored_axiom_list['label'] = np.where(scored_axiom_list['numExceptions']/scored_axiom_list['generality'] >= 0.05, 0, 1)# number of exceptions over generality gives most logical results
    else:
        scored_axiom_list = scored_axiom_list[(scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1 <= -0.2)
                                            & (scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1 >= 0.2)]# ARI possibility and necessity -1 if between 0.2 and -0.2 drop
        scored_axiom_list['label'] = np.where((scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1) <= 0, 0, 1)# ARI possibility and necessity -1
        
    
    #sample an equal ammount of negative and positive labels
    if sample:
        scored_axiom_list = sample_dataset(scored_axiom_list)
    
    
    if axiomtype == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values).sort_values()
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    
    # extract the score for regression and a label for classification based on the type of axiom
   
        
    if labeltype == 'c':
        labeled_axioms =  pd.concat([a,b,scored_axiom_list['label']],axis = 1, keys = ["left","right","label"])
    else:
        if axiomtype == "DisjointClasses":
            labeled_axioms =  pd.concat([a,b,1-(scored_axiom_list['numExceptions']/scored_axiom_list['generality'])],axis = 1, keys = ["left","right","label"])# number of exceptions over generality gives most logical results
        else:
            labeled_axioms =  pd.concat([a,b,(scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1)],axis = 1, keys = ["left","right","label"]) # ARI possibility and necessity -1
    
    #create the list of axioms to be sent in the query
    concept_string = ",".join(concepts) 
    return concepts,concept_string, labeled_axioms

# Read a list of axioms, extract unique concepts to use in creating a precise concept similarity matrix (finish axiom types)
def clean_scored_atomic_axioms_simple(labeltype = 'c', axiomtype = "SubClassOf", score_ = None,sample = True):
    
    valid = {'c', 'r'}
    if labeltype not in valid:
        raise ValueError("labeltype must be one of %r." % valid)
           
    scored_axiom_list = pd.read_csv(score_, header = 0)
    scored_axiom_list['left'], scored_axiom_list['right'] = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')).apply(lambda x: x.replace('SubClassOf','')) ))
    scored_axiom_list = scored_axiom_list[scored_axiom_list.left != scored_axiom_list.right].reset_index(drop = True)
        
    
    if axiomtype == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values).sort_values()
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    
    # extract the score for regression and a label for classification based on the type of axiom
   
        

    labeled_axioms =  pd.concat([a,b,scored_axiom_list['label']],axis = 1, keys = ["left","right","label"])

    #create the list of axioms to be sent in the query
    concept_string = ",".join(concepts) 
    return concepts,concept_string, labeled_axioms


#corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setMethod('POST')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)

#LOAD DATAFRAMES IF AVAILABLE
def readAllFiles():
    tic = time.perf_counter()
    try:
        df = pd.read_csv(path +'classessim.csv')
    except:
        print("classessim isnt saved")
        df = 0
    try:
        allrelations = pd.read_csv(path +'allrelations.csv')
    except:
        print("allrelations isnt saved")
        allrelations= 0
    try:
        kernelmatrix = pd.read_csv( path + 'kernelmatrixpivoted.csv')
    except:
        print("kernelmatrixpivoted isnt saved")
        kernelmatrix = 0
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return (df,allrelations,kernelmatrix)

#build the classes similarity table
def buildClassSim(listofconcepts = None):
    print('Creating concept similarity matrix')
    if listofconcepts == None:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2))
        }
    
        '''
    else:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+ listofconcepts + ''') && str(?class2) IN ('''+ listofconcepts + '''))
        
        }
    
        '''
        #filter (str(?class1) IN ('''+ listofconcepts + ''') && str(?class2) IN ('''+ listofconcepts + ''') )
    tic = time.perf_counter()
    #filter (?class1 <= ?class2)

    df = sparql_service_to_dataframe(wds_Corese, query)
    df = df.astype({'similarity': 'float'})
    df1 = df[["class2","class1","similarity"]]
    df1 = df1[df1['class1'] != df1['class2']]
    df1.rename(columns={'class2': 'class1', 'class1': 'class2'}, inplace=True)
    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    
    dfdic = dict(zip(df.class1 + "," +df.class2, df.similarity))
    print(df.shape)
    # create the table of similarity between all the classes
    
    #reduce prefix size for quicker comparison later
    #df['class1'] = df['class1'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    #df['class2'] = df['class2'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds to creat concept similarity matrix")
    #df = df.pivot_table(columns='class1', index='class2', values='similarity').reset_index()
    df.to_csv( path + 'classessim.csv', index=False)
    print('file classsim created and saved')
    print(df.shape)
    
    return df, dfdic


def buildRelationsExisting(P_relation, P_set_axiom_number):
    N_relation = None
    if P_relation == 'owl:disjointWith':
        N_relation = 'rdfs:subClassOf'
    else:
        N_relation = 'owl:disjointWith'
        
    print("extracting existing axioms and sampling")
    tic = time.perf_counter()
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + P_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(1.0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' + str(P_set_axiom_number) + '''
    '''
    
    positiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("positive relations extracted")
    print(positiverelations.shape)
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + N_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' +  str(positiverelations.shape[0]) + ''' 
    '''
    negativeiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("negativeive relations extracted")
    print(negativeiverelations.shape)
    
    # retrieving the existing axioms that are labeled as accepted
    allrelations = pd.concat([positiverelations, negativeiverelations], axis=0).sample(frac = 1, random_state = 1).reset_index(drop=True)
    print(allrelations.shape)
    
    allrelations = allrelations.astype({'label': 'float'})
    allrelations = allrelations.rename(columns={"class1": "left", "class2": "right"})
    allrelations = sample_dataset(allrelations)
    listofaxioms = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    
    concepts =  pd.Series(pd.Series(np.hstack([allrelations["left"],allrelations["right"]])).drop_duplicates().values).sort_values()
    concepts =  "\"" + concepts.astype(str) + "\"" # just to add " " around the concepts to be considered strings when sent in the queries
    concept_string = ",".join(concepts) 
    print("concepts series")
    print(concepts.shape)
    
    
    
    print(allrelations.shape)
    listofaxioms.to_csv(rdfminer_path+'\IO\listofaxioms.txt', sep=',', index=False, header = False)
    #allrelations.to_csv( path +'allrelations.csv', index=False)
    print('allrelations created')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds to extract the axioms and concepts")
    
    return allrelations, concept_string, concepts


#build the atomic relations table with random generated false axioms
def buildRelationsGenerated(P_relation):
    tic = time.perf_counter()
    query = '''

    select ?class1 ?class2 ?label where 
    {


    {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + P_relation +'''?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(1.0 as ?label)
    }

    Union

    {
     ?class1 a owl:Class
    ?class2 a owl:Class
    filter (!isBlank(?class1)  && !isBlank(?class2)  && (?class1 != ?class2))
    bind(0 as ?label)

    minus

    {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + P_relation +''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    bind(0 as ?label)
    }
    
    }
    
    }

    '''

    # generating all possible combinations of 2 atomic classes to create the false axioms with -1 p index
    allrelations = sparql_service_to_dataframe(wds_Corese, query)
    print(allrelations.shape)
    
    allrelations = allrelations.astype({'label': 'float'})
    allrelations = allrelations.rename(columns={"class1": "left", "class2": "right"})
    allrelations = sample_dataset(allrelations)
    listofaxioms = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    
    print(allrelations.shape)
    listofaxioms.to_csv(rdfminer_path+'\IO\listofaxioms.txt', sep=',', index=False, header = False)
    allrelations.to_csv( path +'allrelations.csv', index=False)
    print('file allrelations created and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return allrelations


def sample_dataset(labeled_axioms):
    
    positiverelations = labeled_axioms[labeled_axioms["label"] == 1]
    #sample the same number of negative relations as positive ones
    negativerelations= labeled_axioms[labeled_axioms["label"] == 0]
    
    #uncomment this
    if len(positiverelations)>= len(negativerelations):
          labeled_axioms = labeled_axioms.groupby("label").sample(n=len(negativerelations), random_state=1)
    else:
          labeled_axioms = labeled_axioms.groupby("label").sample(n=len(positiverelations), random_state=1)
        
    
    # # #shuffle    
    labeled_axioms = labeled_axioms.sample(frac = 1, random_state = 1).reset_index(drop = True)
    
    #delete this
    #labeled_axioms = negativerelations.sample(n = 5000, random_state = 5).reset_index(drop = True)
    #labeled_axioms = positiverelations
    return labeled_axioms


#function used to calculate the kernel matrix can be used with fractions of the relationship table, should be written in a .py
#file and imported in this module, please change the path in  kernelbuilder.py if you wish to output csv files somewhere else

def matrixfractionAverageSimdisdic(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_l =  axiom1['left']
        a1_r = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            #because in disjointness left and right dont make a difference, we compare as dis(A B) dis(B A) and dis(B A) dis(B A)
            sim1 = df[a1_l+","+axiom2['left']]
            sim2 = df[a1_r+","+axiom2['right']]
            
            sim3 = df[a1_l+","+axiom2['right']]
            sim4 = df[a1_r+","+axiom2['left']]
            if (sim1+sim2)/2 > (sim3+sim4)/2:
                sim = (sim1+sim2)/2
            else:
                sim = (sim3+sim4)/2
            rowlist.append([i, j, sim])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix


def matrixfractionAverageSimdic(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_c1 =  axiom1['left']
        a1_c2 = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df[a1_c1+","+axiom2['left']]
            sim2 = df[a1_c2+","+axiom2['right']]
            rowlist.append([i, j, (sim1+sim2)/2])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix

# preparing to split work load to multiple threads
def splitload(split, relations):
    size = len(relations)
    portion = size//split
    startend = []
    l = 0
    for x in range(1,split):
        startend.append((l,l+portion))
        l += portion
    startend.append((startend[len(startend)-1][1], size))
    print("split completed")
    return(startend)

# def splitload(split):
#     size = len(allrelations)
#     portion = size//split
#     startend = []
#     l = 0
#     for x in range(1,split):
#         startend.append((l,l+portion))
#         l += portion
#     startend.append((startend[len(startend)-1][1], size))
#     print("split completed")
#     return(startend)



def pivotpredictmatrix(predictmatrix,selected_relations, predict_names):
    tic = time.perf_counter()

    predictmatrix = predictmatrix.pivot_table(columns='axiom2', index='axiom1', values='overallsim',  fill_value=0).reset_index()
    #predictmatrix.drop(columns = ["axiom2"], inplace = True)
    rawmatrix = predictmatrix.to_numpy()
    #rawmatrix = rawmatrix + rawmatrix.T - np.diag(np.diag(rawmatrix))
    #added <> and axiom type to axioms to comply with rdf miner fromat
    name = pd.Series(["names"])
    colnames = axiom_type+"(<"+ selected_relations["left"] + "> <" + selected_relations["right"] + ">)"
    colnames = name.append(colnames)
    predictmatrix = pd.DataFrame(data=rawmatrix, columns = colnames)
    predictmatrix['names'] = predict_names
    print("predictmatrix shape is :") 
    print(predictmatrix.shape)
    predictmatrix.to_csv( path + 'predictmatrix.csv', sep=',', index=False)
    print('file predictmatrix built and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return predictmatrix

#pivot the table into a matrix and output that to a csv, prepare it for mulearn

def pivotIntofinalmatrix(kernelmatrix):
    tic = time.perf_counter()

    kernelmatrix = kernelmatrix.pivot_table(columns='axiom1', index='axiom2', values='overallsim',  fill_value=0).reset_index()
    kernelmatrix.drop(columns = ["axiom2"], inplace = True)
    rawmatrix = kernelmatrix.to_numpy()
    rawmatrix = rawmatrix + rawmatrix.T - np.diag(np.diag(rawmatrix))
    #added <> and axiom type to axioms to comply with rdf miner fromat
    colnames = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    kernelmatrix = pd.DataFrame(data=rawmatrix, columns = colnames)
    kernelmatrix.insert(0,'possibility',allrelations['label'])
    print("kernelmatrix shape is :") 
    print(kernelmatrix.shape)
    kernelmatrix.to_csv( path + 'kernelmatrix.csv', sep=',', index=False)
    print('file kernelmatrix built and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return kernelmatrix

def BuildProfile(P_sparql_endpoints,P_dataset):
    profile = '''
        st:user a st:Server;
                st:content st:load.
    

    '''
    # for i in range(1,P_sparql_endpoints):
    #     T_endpoint ='''
        
    #     st:'''+str(i)+''' a st:Server;
    #         st:service "'''+str(i)+'''";
    #         st:content st:load.

    #     '''
    #     profile = profile + T_endpoint
    T_dataset = '''
        st:load a st:Workflow;
            sw:body (
        [a sw:Load; sw:path <'''+ P_dataset +'''> ]
        ).
    '''
    profile = profile + T_dataset
    profile_file = open(corese_path + "/profile.ttl", "w")
    n = profile_file.write(profile)
    profile_file.close()
    
def QueryBuilder(P_concept_string, P_concepts):
    queries = []
    chunked_list = np.array_split(P_concepts,sparql_endpoints)
    #second_chunked_list = np.array_split(P_concepts,2)
    for i in range(sparql_endpoints):
        
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity) where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+  ",".join(chunked_list[i]) + ''') && str(?class2) IN ('''+ P_concept_string + '''))
        
        }
    
        '''
        # query = '''
    
        # select * (kg:similarity(?class1, ?class2) as ?similarity) where {
        # ?class1 a owl:Class
        # ?class2 a owl:Class
        # filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+  ",".join(chunked_list[i]) + ''') && str(?class2) IN ('''+ ",".join(second_chunked_list[0]) + '''))
        
        # }
    
        # '''
        # query2 = '''
    
        # select * (kg:similarity(?class1, ?class2) as ?similarity) where {
        # ?class1 a owl:Class
        # ?class2 a owl:Class
        # filter (!isBlank(?class1)  && !isBlank(?class2) && (?class1 <= ?class2) && str(?class1) IN ('''+  ",".join(chunked_list[i]) + ''') && str(?class2) IN ('''+ ",".join(second_chunked_list[1]) + '''))
        
        # }
    
        # '''
        queries.append(query)
        #queries.append(query2)
    return queries

#(kg:similarity(?class1, ?class2) as ?similarity)

def ExecQuery(P_endpoint, P_query, list_df):

    logging.info("Thread %s: starting", P_endpoint)
    
    # if P_endpoint == 0:
    #     df = sparql_service_to_dataframe('http://localhost:8080/sparql', P_query)
    # else:
    #     df = sparql_service_to_dataframe('http://localhost:8080/'+str(P_endpoint)+'/sparql', P_query)
    df = sparql_service_to_dataframe('http://localhost:8080/sparql', P_query)
    df = df.astype({'similarity': 'float'})
    df1 = df[["class2","class1","similarity"]]
    df1 = df1[df1['class1'] != df1['class2']]
    df1.rename(columns={'class2': 'class1', 'class1': 'class2'}, inplace=True)
    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    #df = df.drop_duplicates()
    dfdic = dict(zip(df.class1 + "," +df.class2, df.similarity))
    list_df.append(dfdic)
    logging.info("Thread %s: finishing %s", P_endpoint, time.time() - start)


def buildConceptsimthreaded(P_queries):
    ticfirst = time.perf_counter()
    threads = []
    for i in range(len(P_queries)):#changed from sparql endpoints to length of queries
        threads.append(threading.Thread(target=ExecQuery, args=(i, P_queries[i],list_df)))
    for x in threads:
        x.start()
    for x in threads:
        x.join()
    dfdic = {k:v for x in list_df for k,v in x.items()}
    print(str(len(dfdic))+"this is the length of df dic")
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds for threaded concept similarity")
    return dfdic


def ExecQueryProcess(P_query):
    df = sparql_service_to_dataframe('http://localhost:8080/sparql', P_query)
    df = df.astype({'similarity': 'float'})
    df1 = df[["class2","class1","similarity"]]
    df1 = df1[df1['class1'] != df1['class2']]
    df1.rename(columns={'class2': 'class1', 'class1': 'class2'}, inplace=True)
    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    dfdic = dict(zip(df.class1 + "," +df.class2, df.similarity))
    return dfdic


def get_rar_dataset(filename, n=None):


    with open(filename) as data_file:
        reader = csv.reader(data_file)
        names = np.array(list(next(reader)))

    data = pd.read_csv(filename, dtype=object)
    data = data.to_numpy()

    n = len(names) - 1

    # ## Extract data names, membership values and Gram matrix

    names = names[1:n+1]
    mu = np.array([float(row[0]) for row in data[0:n+1]])
    gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                     for row in data[0:n+1]])

    assert(len(names.shape) == 1)
    assert(len(mu.shape) == 1)
    assert(len(gram.shape) == 2)

    assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

    X = np.array([[x] for x in np.arange(n)])

    return X, gram, mu, names

def split_train_test(X, gram, mu, names):
    X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.6, stratify=mu)

    train_test = gram[X_train.flatten()][:, X_train.flatten()]
    test_test = gram[X_test.flatten()][:, X_train.flatten()]
    test_names = names[X_test.flatten()]
    
    train_set = np.vstack([np.insert(names[X_train.flatten()][:, None],0,"possibility"),np.hstack([mu_train[:, None],train_test])])
    
    
    
    test_set = np.vstack([np.append(np.insert(names[X_train.flatten()][:, None],0,"axiom"),"possibility"),np.hstack([test_names[:, None], test_test, mu_test[:, None]])])
    train_set= pd.DataFrame(data=train_set[1:,0:],    # values
                            columns=train_set[0,0:])
    test_set= pd.DataFrame(data=test_set[1:,0:],    # values
                           columns=test_set[0,0:])
    
    predict_axioms = pd.DataFrame(columns=["relations","left","right","possibility"])
    predict_axioms['left'], predict_axioms['right'] = zip(*(s.split(" ") for s in test_set.iloc[0:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','')).apply(lambda x: x.replace('>','')).apply(lambda x: x.replace('SubClassOf','')) ))
    predict_axioms['possibility'] = test_set['possibility']
    predict_axioms["relations"]= test_set.iloc[0:, 0]
    train_set.to_csv( "fragments/train_set.csv", sep=',', index=False)
    test_set.to_csv( "fragments/test_set.csv", sep=',', index=False)
    predict_axioms.to_csv("fragments/predict_axioms.csv", sep=',', index=False)

def train_test_train(X, gram, mu, names):
     rs = ensemble.RandomForestClassifier(n_estimators = 100)
     X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.3, stratify=mu)

     train_test = gram[X_train.flatten()][:, X_train.flatten()]
     test_test = gram[X_test.flatten()][:, X_train.flatten()]
     test_names = names[X_test.flatten()]
     crossval = cross_val_score(rs, gram[X.flatten()][:, X.flatten()], mu, cv=5)
     ticfirst = time.perf_counter()
     rs.fit(train_test, mu_train)
     tocfirst = time.perf_counter()
     print(f"it took {tocfirst - ticfirst:0.4f} seconds")
     predicted_test = rs.predict(test_test)
     print("score: " ,  matthews_corrcoef (mu_test, predicted_test))
     predict_train= rs.predict(train_test)
     print('test')
     print(classification_report(mu_test,  predicted_test))
     print('train')
     print(classification_report(mu_train,  predict_train))
     min_proba = rs.predict_proba(test_test)
     print(crossval)
     print("%0.2f accuracy with a standard deviation of %0.2f" % (crossval.mean(), crossval.std()))
     ConfusionMatrixDisplay(confusion_matrix(mu_test, predicted_test),display_labels=['Rejected','Accepted']).plot()
     full_view_min = np.concatenate([np.vstack((test_names,mu_test, predicted_test)).T,min_proba], axis = 1)
     wrong_predictions_min = full_view_min[(full_view_min[:,1] != full_view_min[:,2])]
     correct_predictions_min = full_view_min[(full_view_min[:,1] == full_view_min[:,2])]
     rs.fit(gram, mu)
     return rs
     
def Select_features(X, gram, mu, names):
    rs = ensemble.RandomForestClassifier(n_estimators = 100)
    selector = SP(percentile=30) # select features with top 50% MI scores
    selector.fit(gram,mu)
    X_Selected = selector.transform(gram)
    X_train,X_test,y_train,y_test = tts(X,mu,test_size=0.3,stratify=mu)
    #features = selector.transform(names.reshape(1, -1))
    indecis =  selector.transform(X.reshape(1, -1))
    features = names[indecis.flatten()]
    X_train = X_Selected[X_train.flatten()][:, :]
    X_test = X_Selected[X_test.flatten()][:, :]
    #print(features)
    #print(indecis)
    #print(names[indecis.flatten()])
    rs.fit(X_train,y_train)
    predicted_test = rs.predict(X_test)
    print("score: " ,  matthews_corrcoef (y_test, predicted_test))
    print(classification_report(y_test,  predicted_test))
    score = rs.score(X_test,y_test)
    print(f"score_selected:{score}")
    rs.fit(X_Selected, mu)
    return(features, rs)

#################################### important
def get_concept_relation_list(selected_feature):
    if axiom_type == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in pd.Series(selected_feature).apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in pd.Series(selected_feature).apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values)
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))

    relations =  pd.concat([a,b],axis = 1, keys = ["left","right"])
    return(concepts,relations)
#######################################

def read_axioms_to_predict(path):
    relations = pd.read_csv(path, header=None)
    names = relations.copy()
    relations['left'], relations['right'] = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','')).apply(lambda x: x.replace('>','')).apply(lambda x: x.replace('SubClassOf','')) ))
    
    relations = relations[relations.left != relations.right].reset_index(drop = True)
    
    if axiom_type == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values)
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    names = relations.iloc[:, 0]
    relations = relations[['left','right']]
    return(concepts, relations, names)

def read_axioms_to_predict_Y(path):
    relations = pd.read_csv(path)
    
    #relations['left'], relations['right'] = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','')).apply(lambda x: x.replace('>','')).apply(lambda x: x.replace('SubClassOf','')) ))
    relations = relations[relations.left != relations.right].reset_index(drop = True)
    
    if axiom_type == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in relations.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values)
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    Y = relations['possibility']
    names = relations["relations"]
    relations = relations[['left','right']]
    return(concepts, relations, names, Y)
    
def matrixfractionAverageSimdicpredict(start, end, size, df, predict_relations,selected_relations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = predict_relations.iloc[i]
        a1_l =  axiom1['left']
        a1_r = axiom1['right']
        for j in range(0, size):
            axiom2 = selected_relations.iloc[j]
            sim1 = df[a1_l+","+axiom2['left']]
            sim2 = df[a1_r+","+axiom2['right']]
            rowlist.append([i, j,(sim1+sim2)/2])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"]).drop_duplicates()
    return axiomsimilaritymatrix

def matrixfractionAverageSimdisdicpredict(start, end, size, df, predict_relations,selected_relations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = predict_relations.iloc[i]
        a1_l =  axiom1['left']
        a1_r = axiom1['right']
        for j in range(0, size):
            axiom2 = selected_relations.iloc[j]
            #because in disjointness left and right dont make a difference, we compare as dis(A B) dis(B A) and dis(B A) dis(B A)
            sim1 = df[a1_l+","+axiom2['left']]
            sim2 = df[a1_r+","+axiom2['right']]
            
            sim3 = df[a1_l+","+axiom2['right']]
            sim4 = df[a1_r+","+axiom2['left']]
            if (sim1+sim2)/2 > (sim3+sim4)/2:
                sim = (sim1+sim2)/2
            else:
                sim = (sim3+sim4)/2
            rowlist.append([i, j, sim])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"]).drop_duplicates()
    return axiomsimilaritymatrix
    



def get_predict_dataset(filename, n=None):


    with open(filename) as data_file:
        reader = csv.reader(data_file)
        features = np.array(list(next(reader)))

    data = pd.read_csv(filename, dtype=object)
    names = data["names"]
    data = data.to_numpy()

    n = len(features) - 1

    # ## Extract data names, membership values and Gram matrix

    features = features[1:n+1]
    #names = np.array([row[0] for row in data[0:n+1]])
    #names = np.array(data[0:,0])
    #gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
    #                for row in data[0:n+1]])
    
    gram = np.array(data[0:,1:])
    assert(len(names.shape) == 1)

    assert(len(gram.shape) == 2)

    assert(names.shape[0] == gram.shape[0])
    assert(gram.shape[1] == features.shape[0])

    X = np.array([[x] for x in np.arange(n)])

    return X, gram, names, features


def QueryBuilderpredict(P_concept_string, P_concepts):
    queries = []
    chunked_list = np.array_split(P_concepts,sparql_endpoints)
    #second_chunked_list = np.array_split(P_concepts,2)
    for i in range(sparql_endpoints):
        
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity) where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2) && str(?class1) IN ('''+  ",".join(chunked_list[i]) + ''') && str(?class2) IN ('''+ P_concept_string + '''))
        
        }
    
        '''
      
        queries.append(query)
    return queries


# WARNING CHANGE THE SLEEP TIMER AFTER LAUNCHING CORESE IF YOU ARE HAVING AN ERROR WHEN THE OWL FILE IS LARGE, 50 SECONDS IS GOOD FOR 200 MB FILES, 10 IS GOOD FOR 8 MB
#calls for multiprocessing to build the table of axiom similarity
if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    ticstart = time.perf_counter()
    ticfirst = time.perf_counter()
    pd.set_option('display.max_colwidth', None) # if your Pandas version is < 1.0 then use -1 as second parameter, None otherwise
    pd.set_option('display.precision', 5)
    pd.set_option('display.max_rows', 99999999999)
    #end version, every parameter that can be changed should be here
    setParam(P_threadcount = 6, P_split =16,  P_prefix = '' , P_sparql_endpoints =16, P_dataset = 'dbpediaHPC.owl',#change P_dataset to the path of the owl ontology you want to use
             P_path = 'fragments/',
             P_corese_path = os.path.normpath("C:\corese-server"),
             P_rdfminer_path = os.path.normpath(r"C:\corese\RDFMining"),
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx24G corese-server-4.3.0.jar -e -lp -pp profile.ttl', 
             P_wds_Corese = 'http://localhost:8080/sparql', 
             P_relation = 'owl:disjointWith',
             P_label_type='r', 
             P_list_of_axioms= None, 
             P_score = "scored reg.txt", 
             P_dont_score = True,
             P_set_axiom_number =4000)

    #uncomment if you already have the files and want to read them, process is fast though so you can just create them again
    #df, allrelations, kernelmatrix = readAllFiles()
    #prepare to launch corese server
    BuildProfile(sparql_endpoints,dataset)
    corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
    #CHANGE THIS TIMER IN CASE OF ERRORS
    time.sleep(6)
    
    
    #if statement that chooses the way axioms are genarated and scored
    if list_of_axioms == None and score == None :  #randomly generate atomic axioms (NOT RECOMMENDED FOR LARGE ONTOLOGIES 700+ concepts)
        if label_type == 'c' and dont_score == True and set_axiom_number == 0:# dont score the random generated axioms
            print('generating list from existing and random combination without scoring')
            df, dfdic = buildClassSim()
            allrelations = buildRelationsGenerated(relation)# create a list of axioms and send it to /rdfminer/io/ shared folder on ur machine

        elif label_type == 'c' and dont_score == True and set_axiom_number != 0:#extract existing axioms positive and negative with no random generation without scoring with rdfminer
            print('generating list from existing set number of axioms')
            allrelations, concept_string, concepts = buildRelationsExisting(relation, set_axiom_number)#get the axioms and teh atomic concepts
            concepts = concepts.sample(frac = 1)
            # concepts = random.shuffle(concepts)
            queries = QueryBuilder(concept_string,concepts)#split concepts into multiple queries to make the process threaded
            time.sleep(1)
            start = time.time()
            dfdicbase = buildConceptsimthreaded(queries)#start the threaded query process
            # ticfirst = time.perf_counter()#
            # dfdic = {}#
            # p = Pool(30)#
            # dfdic = parmap.map(ExecQueryProcess, queries , pm_pool=p, pm_pbar=True)#
            # dfdic = {k:v for x in dfdic for k,v in x.items()}
            # p.close()
            # p.terminate()
            # p.join()
            # print(str(len(dfdic))+"this is the length of df dic")
            # tocfirst = time.perf_counter()
            # print(f"it took {tocfirst - ticfirst:0.4f} seconds for threaded concept similarity")
        else:# score the generated list of axioms
            print('generating list from existing and random combination with scoring')
            allrelations = buildRelationsGenerated(relation)# create a list of axioms and send it to /rdfminer/io/ shared folder on ur machine
            list_of_axioms ='listofaxioms.txt'
            rdfminer = subprocess.run('start /w docker-compose exec rdfminer ./rdfminer/scripts/run.sh -a /rdfminer/io/'+ list_of_axioms +' -dir results', shell=True, cwd=rdfminer_path) #process of scoring with rdf miner
            score = 'resultsresults.json'
            concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score, sample = False)
            df, dfdic = buildClassSim(concept_string)      
            
    elif list_of_axioms != None and score == None: # score an existing list of axioms
        print('using a non scored list of axioms')
        rdfminer = subprocess.run ('start /w docker-compose exec rdfminer ./rdfminer/scripts/run.sh -a /rdfminer/io/'+ list_of_axioms +' -dir results', shell=True, cwd=rdfminer_path) #process of scoring with rdf miner
        score = 'resultsresults.json'
        concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score)
        #df, dfdic = buildClassSim(concept_string)
        concepts = concepts.sample(frac = 1)
        # concepts = random.shuffle(concepts)
        queries = QueryBuilder(concept_string,concepts)#split concepts into multiple queries to make the process threaded
        time.sleep(1)
        start = time.time()
        dfdicbase = buildConceptsimthreaded(queries)#start the threaded query process
    
    else:# use a scored list of axioms
        print('using a scored list of axioms')
        concepts, concept_string, allrelations = clean_scored_atomic_axioms_simple(label_type, axiom_type, score)
        concepts = concepts.sample(frac = 1)
        # concepts = random.shuffle(concepts)
        queries = QueryBuilder(concept_string,concepts)#split concepts into multiple queries to make the process threaded
        time.sleep(1)
        start = time.time()
        dfdicbase = buildConceptsimthreaded(queries)#start the threaded query process
        #df, dfdic = buildClassSim(concept_string)



    #split load into multiple processes and list of axioms into chunks
    startend = splitload(split, allrelations)
    size = len(allrelations)
    p = Pool(threadcount)
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds")
    print()
    
    
    tic = time.perf_counter()

    if axiom_type == 'DisjointClasses':
        print('mirror compare')
        kernelmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdisdic,startend,size, dfdicbase, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    #similarity is averag
    else:   
        kernelmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdic,startend,size, dfdicbase, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    
    tocc = time.perf_counter()
    print(f"axiom sim took {tocc - tic:0.4f} seconds")
    p.close()
    p.terminate()
    p.join()
    
    #turn the list into a matrix
    kernelmatrix = pivotIntofinalmatrix(kernelmatrix)
    #finished creating data set
    ################################################################################################################################
    
    #prepare the data for training if u want to use the split set for some reason the file name is train_set
    file_name='kernelmatrix'
    X, gram, mu, names = get_rar_dataset("fragments/"+file_name+".csv")
    print('done extracting matrix')
    
    #save a split train test set to files
    split_train_test(X, gram, mu, names)
    
    
    # for real predictions comment------------------------------------------------------------------------------------
    file_name='train_set'
    kernelmatrix = pd.read_csv("fragments/"+file_name+".csv")
    X, gram, mu, names = get_rar_dataset("fragments/"+file_name+".csv")
    print('done extracting matrix')
    
    
    #get model performance then fit witrh all dataset
    #rs = train_test_train(X, gram, mu, names)
    
    
    
    #perform feature selection
    selected_feature, frs = Select_features(X, gram, mu, names)
    selected_concepts, selected_relations  = get_concept_relation_list(selected_feature)
    concept_string = ",".join(selected_concepts)
    ticpredict = time.perf_counter()
    
    
    
    #predict new axiom never seen before
    # for real predictions comment  predict Y and uncomment normal predict-------------------------------------------------------------------------------
    #predict_concepts, predict_relations, predict_names= read_axioms_to_predict("fragments/axiomstopredict1.txt")
    predict_concepts, predict_relations, predict_names, Y= read_axioms_to_predict_Y("fragments/Predict_axioms.csv")
    concepts_to_extract = predict_concepts[~predict_concepts.isin(concepts)]
    
    
    
    
    queries = QueryBuilderpredict(concept_string,concepts_to_extract)
    time.sleep(1)
    start = time.time()
    list_df = []
    dfpredict = buildConceptsimthreaded(queries)
    dfdicbase.update(dfpredict)
    
    ################################################################################################################################
    #build predict matrix
    startend = splitload(split, predict_relations)
    size = len(selected_relations)
    p = Pool(threadcount)
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds")
    print()
    tic = time.perf_counter()
    if axiom_type == 'DisjointClasses':
        print('mirror compare')
        predictmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdisdicpredict,startend,size, dfdicbase,predict_relations ,selected_relations, pm_pool=p, pm_pbar=True),ignore_index = True)
    #similarity is averag
    else:   
        predictmatrix = pd.concat(parmap.starmap(matrixfractionAverageSimdicpredict,startend,size, dfdicbase,predict_relations ,selected_relations, pm_pool=p, pm_pbar=True),ignore_index = True)
    
    tocc = time.perf_counter()
    print(f"axiom sim took {tocc - tic:0.4f} seconds")
    p.close()
    p.terminate()
    p.join()
    predictmatrix = pivotpredictmatrix(predictmatrix, selected_relations, predict_names)
    ################################################################################################################################
    
    
    
    #perform the prediction
    file_name='predictmatrix'
    X, gram, names, features = get_predict_dataset("fragments/"+file_name+".csv")
    print('done extracting predict matrix')
    predicted = frs.predict(gram)
    proba = frs.predict_proba(gram)
    
    full_view = np.concatenate([np.vstack((np.array(list(names)), predicted)).T,proba], axis = 1)
    
    #for real predictions comment this-------------------------------------------------------------------------------
    full_view_Y = np.concatenate([np.vstack(((np.array(list(names))),Y, predicted)).T,proba], axis = 1)
    wrong_predictions_Y = full_view_Y[(full_view_Y[:,1] != full_view_Y[:,2])]
    print(classification_report(Y,  predicted))
    
    tocpredict = time.perf_counter()
    print(f"axiom prediction took {tocpredict - ticpredict:0.4f} seconds")
    
    
    
    tocend = time.perf_counter()
    print(f"everything took {tocend - ticstart:0.4f} seconds")
    
    
    
# (A) relation (B)

# (A U n(B n C)) relation (A U C n B)

# 1 + 0.5 /2 0.75

# 1 + 0.5 - 1
# -----------
#     3


















