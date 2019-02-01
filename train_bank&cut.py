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
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
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
    
def attributes_creatation(dict_labels, dict_excel_features, train_test_fac=0.8):
    dirs = ['Norimen//', 'Geology//', 'Curvature//', 'Aspect//', 'FlowAccumulation//','Slope//']
    dirs_head = ['NOR', 'GEO', 'CUR', 'ASP', 'FLA', 'SLP']
    dirs = [roots+dir_ for dir_ in dirs]
    attributes = [[] for i in range(len(dirs_head))]
    '''index the image'''
    img_idx = []
    excel_features = [] 
    location_area = []       
    i = 0
    pathes = os.listdir(dirs[i])
    train_nums = np.random.permutation(len(pathes))[:int(train_test_fac*len(pathes))]
    train_or_test = []
    pos_test_area = 0
    neg_test_area = 0
    for k, path in enumerate(pathes): 
        label, excel_feature = path2label(path, dict_labels, dict_excel_features)
        norimen = tiff.imread(dirs[0] + path)
        location = np.argwhere(norimen[:]==1)  
        attributes[0] = np.concatenate((attributes[0], label*np.ones(location.shape[0])))
        for _ in range(location.shape[0]): excel_features.append(excel_feature)
        location_area = np.concatenate((location_area, location.shape[0]*np.ones(location.shape[0])))
        if k in train_nums:
            train_or_test = np.concatenate((train_or_test, np.ones(location.shape[0])))
        else:
            train_or_test = np.concatenate((train_or_test, np.zeros(location.shape[0])))
            if label:
                pos_test_area = pos_test_area + location.shape[0]
            else:
                neg_test_area = neg_test_area + location.shape[0]
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
    img_idx = np.array(img_idx)
    location_area = np.array(location_area)
    img_idx = img_idx.reshape(1,-1)
    excel_features = np.array(excel_features).transpose()
    train_or_test = np.array(train_or_test)
    train_or_test = train_or_test.reshape(1,-1)
    attributes = np.concatenate((attributes, excel_features, train_or_test, img_idx), axis=0)
    print('pos, neg is', pos_test_area, neg_test_area)
    return attributes.transpose(), location_area, pos_test_area, neg_test_area

def attribute_ranking(X, y):
    attributes_name = ['geology', 'curve', 'aspect', 'fla', 'slope', 'shape1','shape2','shape3','shape4',
                       'shape5', 'shape6','shape7','shape8','shape9','stru1','stru2','stru3','stru4','stru5',
                       'intense of soil', 'bank material']
    '''rankning the features'''
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    print(importances)
    indices = np.argsort(importances)[::-1]
    for f in range(len(attributes_name)):
        print("%2d) %-*s %f" % (f + 1, 30, attributes_name[indices[f]], importances[indices[f]]))
    return indices

'''split the image into train adn test dataset'''
def split_process(y, img_idx, location_area):
    '''based on images, there are 2 rules of the p4splitting:
        rule 1: train pixels is 80%, test pixels is 20% 
        rule 2: the pos/neg proportion is keeps same in training and test dataset
        '''
    uq_idx = np.unique(img_idx)
    uq_labels = np.array([y[np.argwhere(img_idx==item)[0]][0] for item in uq_idx])
    uq_locations = np.array([location_area[np.argwhere(img_idx==item)[0]][0] for item in uq_idx])
    pos_areas =  uq_locations[np.argwhere((uq_labels)==1)]   
    pos_idx = uq_idx[np.argwhere(uq_labels==1)][:,0]
    neg_areas =  uq_locations[np.argwhere((uq_labels)==0)]   
    neg_idx = uq_idx[np.argwhere(uq_labels==0)][:,0]
    '''random add the number of test data until the area reach to 0.2 of total areas'''
    pos_nums = np.random.permutation(len(pos_idx))
    neg_nums = np.random.permutation(len(neg_idx))
    pos_idx_out = []
    neg_idx_out = []
    pos_areas_for_test = 0
    for i in range(len(pos_nums)):
        pos_areas_for_test = pos_areas_for_test + pos_areas[pos_nums[i]]
        pos_idx_out.append(pos_idx[pos_nums[i]])
        if pos_areas_for_test > 0.2*sum(pos_areas):
            break
    neg_areas_for_test = 0
    for i in range(len(neg_nums)):
        neg_areas_for_test = neg_areas_for_test + neg_areas[neg_nums[i]]
        neg_idx_out.append(neg_idx[neg_nums[i]])
        if neg_areas_for_test > 0.2*sum(neg_areas):
            break
    '''test image index numbers'''
    test_idx = np.concatenate((pos_idx_out, neg_idx_out))
    '''output the True/False pairs of split'''
    train_test_idx = [True if item in test_idx else False for item in img_idx]
    return np.array(train_test_idx), len(pos_idx_out), len(neg_idx_out)
'''preprocess the attributes by the images'''
def attribute_preprocess(attributes, band_or_cut, rank_factor, location_area):
    '''step0: prepare the  data for svm, use only band/cut dataset'''
    cut_band = attributes[:,6]
    attributes = np.delete(attributes, 6, axis=1) # delete the cut/band column
    if band_or_cut:
        band_idx = np.argwhere(cut_band==0).reshape(-1)
        attributes = attributes[band_idx]
        location_area = location_area[band_idx]
#        attributes = np.delete(attributes, 21, axis=1)
    else:
        band_idx = np.argwhere(cut_band==1).reshape(-1)
        attributes = attributes[band_idx]
#        attributes = np.delete(attributes, 22, axis=1)
    X = attributes[:,1:-2]
    y = attributes[:,0]
    '''step1: rank_____'''
    if os.path.exists(str(band_or_cut)+'rank.npy'):
        ranking = np.load(str(band_or_cut)+'rank.npy')
    else:
        ranking = attribute_ranking(X, y)
        np.save(str(band_or_cut)+'rank.npy', ranking)
    print(len(ranking[:-rank_factor]))
    X = X[:,ranking[:-rank_factor]]
    '''step 2: split the set into train and test'''
    img_idx = attributes[:,-1]
#    train_or_test, pos_num, neg_num = split_process(y, img_idx, location_area)
#    print(pos_num, neg_num)
    train_or_test = (attributes[:,-2]) > 0
    x_train, y_train = X[train_or_test], y[train_or_test]
    x_test, y_test = X[~train_or_test], y[~train_or_test]
    img_idx_test = img_idx[~train_or_test]
    '''step 3 : image augmentation'''
    sm = SMOTEENN()
    x_train, y_train = sm.fit_sample(x_train, y_train)
    return x_train, y_train, x_test, y_test, img_idx_test
if __name__=='__main__':
    '''step0 load the labels file that created already'''
    band_or_cut = 0 # 1: band; 0:cut
    roots = '..//dataset//'
    dict_labels = load_dictionary(roots+'labels.txt')
    dict_excel_features = load_dictionary(roots+'features.txt')
    AUC_all = []
    for i in range(10):
        attributes, location_area, pt_area, nt_area = attributes_creatation(dict_labels, dict_excel_features)
        if pt_area > 1000 and pt_area < 1400 and nt_area < 7300: # only record effective split result
            np.save('attributes.npy', attributes)
            rank_factor = 16 # the amount that delet of the unimportant factors
            x_train, y_train, x_test, y_test,img_idx_test  = attribute_preprocess(attributes, band_or_cut, rank_factor, location_area)      
            
            '''step 2 train the model'''
#            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=30),
#                                 algorithm="SAMME",
#                                 n_estimators=2000)
            clf = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1)
    #        clf = svm.SVC(kernel='linear', C=1.0)
    #        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(4, 2), random_state=1)
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
            AUC_all.append(metrics.auc(fpr,tpr))
    np.mean(AUC_all) 
