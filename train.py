# -*- coding: utf-8 -*-

'''
load the image and check its values to preprocess/
good luck
'''
import numpy as np
import os
from PIL import Image
import matplotlib.pylab as plt
import tifffile as tiff
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import manifold, datasets
from time import time
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN

'''load the name-label dictionary from the text file that we already created before'''
def load_dictionary(path):
    f = open(path,'r')
    a = f.read()
    dict_categoory = eval(a)
    f.close()
    return dict_categoory
'''from the existing path to label'''
def path2label(path, labels, excel_features):
    path_strip = path[path.find('_')+1:-4]
    return labels[path_strip], excel_features[path_strip]
    
def attributes_creatation(dict_labels, dict_excel_features):
    dirs = ['Norimen//', 'Geology//', 'Curvature//', 'Aspect//', 'FlowAccumulation//','Slope//']
    dirs_head = ['NOR', 'GEO', 'CUR', 'ASP', 'FLA', 'SLP']
    dirs = [roots+dir_ for dir_ in dirs]
    attributes = [[] for i in range(len(dirs_head))]
    '''index the image'''
    img_idx = []
    excel_features = []        
    i = 0
    pathes = os.listdir(dirs[i])
    train_nums = np.random.permutation(len(pathes))[:int(0.8*len(pathes))]
    train_or_test = []
    for k, path in enumerate(pathes): 
        label, excel_feature = path2label(path, dict_labels, dict_excel_features)
        norimen = tiff.imread(dirs[0] + path)
        location = np.argwhere(norimen[:]==1)  
        attributes[0] = np.concatenate((attributes[0], label*np.ones(location.shape[0])))
        for _ in range(location.shape[0]): excel_features.append(excel_feature)
        if k in train_nums:
            train_or_test = np.concatenate((train_or_test, np.ones(location.shape[0])))
        else:
            train_or_test = np.concatenate((train_or_test, np.zeros(location.shape[0])))
        img_idx = np.concatenate((img_idx, k*np.ones(location.shape[0])))
        for i in range(1, len(dirs)):
            path_now = dirs_head[i] + path[path.find('_'):]
            feature_img = tiff.imread(dirs[i] + path_now)
            att = feature_img[location[:,0], location[:,1]]
            '''for FlowAccumlation data, the log process is required'''
            if dirs[i] == 'FlowAccumulation//':
                att = np.log(att)
            attributes[i] = np.concatenate((attributes[i], att))
    attributes = np.array(attributes)
    excel_features = np.array(excel_features).transpose()
    train_or_test = np.array(train_or_test)
    train_or_test = train_or_test.reshape(1,-1)
    attributes = np.concatenate((attributes,excel_features, train_or_test), axis=0)
    return attributes.transpose(), img_idx
   
if __name__=='__main__':
    '''step0 load the labels file that created already'''
    roots = '..//dataset//'
    dict_labels = load_dictionary(roots+'labels.txt')
    dict_excel_features = load_dictionary(roots+'features.txt')
#    try:
##        attributes = np.load('attributes.npy')
#    except:
    attributes, img_idx = attributes_creatation(dict_labels, dict_excel_features)
    np.save('attributes.npy', attributes)
    '''step1 prepare the  data for svm'''
    X = attributes[:,1:-1]
    y = attributes[:,0]
    train_or_test = (attributes[:,-1]) > 0
    sm = SMOTEENN()
    '''optional 1: image based train and test'''
    x_train, y_train = X[train_or_test], y[train_or_test]
    x_test, y_test = X[~train_or_test], y[~train_or_test]
    img_idx_test = img_idx[~train_or_test]
    x_train, y_train = sm.fit_sample(x_train, y_train)
    
#    x_test, y_test = sm.fit_sample(x_test, y_test)
    '''optional 2: pixel based train and test'''
#    x_resampled, y_resampled = sm.fit_sample(X, y)
#    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2, random_state=25)
    '''step 2 train the model'''
    '''random forest'''
#    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
#                                 random_state=0)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=30),
                         algorithm="SAMME",
                         n_estimators=400)
#    clf=svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    '''img based prediction'''
    x_test_img = []
    y_test_img = []
    img_test_unique = np.unique(img_idx_test)
    for item in img_test_unique:
        item_idx = np.argwhere(img_idx_test==item)
        item_idx = item_idx.reshape(-1)
        x_img_i = x_test[item_idx]
        y_img_i = y_test[item_idx]
        x_img_predict = clf.predict(x_img_i)
        x_test_img.append(np.mean(x_img_predict))
        y_test_img.append(np.mean(y_img_i))
    fpr,tpr,thresholds=metrics.roc_curve(y_test_img,x_test_img)
    
    print('AUC is', metrics.auc(fpr,tpr))
    plt.plot(fpr,tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.grid(True)
    plt.show()
    
    
    '''step 4 print the classification report'''
#        y_predict = clf.predict(x_test)
#        print(classification_report(y_test, clf.predict(x_test), target_names=['not-Landslide', 'Landslide']))
        
    '''step 5 t-SNE visualization, show the picture'''
#    t0 = time()
#    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#    Y = tsne.fit_transform(x_test)  
#    t1 = time()
#    plt.scatter(Y[:, 0], Y[:, 1], c=y_test, cmap=plt.cm.Spectral)
#    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    
