
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score


# CHALLENGE - create 3 more classifiers...
# 1
clf1 = tree.DecisionTreeClassifier()
# 2
clf2 = svm.SVC()
# 3
clf3 = neighbors.KNeighborsClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

test_data = [[180,100,43],[180,85,43],[160,63,38],[175,70,40],[175,80,43]]
test_results = [['male'],['male'],['female'],['female'],['male']]


prediction1 = clf1.predict(test_data)
accuracy1 = accuracy_score(test_results,prediction1)
prediction2 = clf2.predict(test_data)
accuracy2 = accuracy_score(test_results,prediction2)
prediction3 = clf3.predict(test_data)
accuracy3 = accuracy_score(test_results,prediction3)


# CHALLENGE compare their reusults and print the best one!

print("Predictions of decision tree:",prediction1)
print("Accuracy for decision tree:",accuracy1)
print("Predictions of SVM: ",prediction2)
print("Accuracy for SVM:",accuracy2)
print("Predictions of K neighbors: ",prediction3)
print("Accuracy for K neighbors:",accuracy3)

