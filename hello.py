import sys

print(sys.version)

if 5 > 2:
    print("Five is greater than two!")

if 5 > 2:
 print("Five is greater than two!")  
if 5 > 2:
        print("Five is greater than two!") 

#Python Variables

x = 5
y = "Hello, World!"

print(x)
print(y)
print(type(x))
print(type(y))


#Many Values to Multiple Variables
X, Y, Z = "Orange", "Banana", "Cherry"

print(X)
print(Y)
print(Z)


#One Value to Multiple Variables
a = b = c = "Orange"

print(a)
print(b)
print(c)


#Unpack a Collection
fruits = ["kiwi", "lime", "pineapple"]
p, q, r = fruits

print(p)
print(q)
print(r)


#Output Variables
#output multiple variables, separated by a comma:
w = "Python"
u= "is"
v = "awesome"
print(w, u, v)

#use the + operator to output multiple variables:
x1 = "Python "
y1= "is "
z1 = "awesome"
print(x1 + y1 + z1)


#Global Variables
#Create a variable outside of a function, and use it inside the function
x2 = "awesome"

def myfunc():
  print("Python is very " + x2)

myfunc()

#Create a variable inside a function, with the same name as the global variable
x3 = "awesome"

def myfunc():
  x3 = "fantastic"
  print("Python is " + x3)

myfunc()

print("Python is " + x3)

#global keyword, the variable belongs to the global scope:
def myfunc():
  global x4
  x4 = "fantastic"

myfunc()

print("Python is " + x4)


a1 = {"name" : "John", "age" : 36}	#dict	
a2 = {"apple", "banana", "cherry"}	#set	
a3 = frozenset({"apple", "banana", "cherry"})	#frozenset	
a4 = True	#bool	
a5 = b"Hello"	#bytes	
a6 = bytearray(5)	#bytearray	
a7 = memoryview(bytes(5))	#memoryview	
a8 = None

print(type(a1))
print(type(a2))
print(type(a3))
print(type(a4))
print(type(a5))
print(type(a6))
print(type(a7))
print(type(a8))
print(a1,a2,a3,a4,a5,a6,a7,a8)



#Python Lists - Lists are used to store multiple items in a single variable.
thislist = ["apple", "banana", "cherry"]
print(thislist)


#Ordered
#it means that the items have a defined order, and that order will not change.

#Changeable
#The list is changeable, meaning that we can change, add, and remove items in a list after it has been created.

#List Length
thislist = ["apple", "banana", "cherry"]
print(len(thislist))


#List Items - Data Types
list1 = ["apple", "banana", "cherry"]
list2 = [1, 5, 7, 9, 3]
list3 = [True, False, False]

print(list1)
print(list2)
print(list3)

#A list can contain different data types:
list11 = ["abc", 34, True, 40, "male"]

print(list11)


#type()
mylistt = ["apple", "banana", "cherry"]

print(type(mylistt))


#The list() Constructor
#It is also possible to use the list() constructor when creating a new list.
thislist1 = list(("apple", "banana", "cherry")) # note the double round-brackets  
print(thislist1)

#Access Items
thislist2 = ["apple", "banana", "cherry"]
print(thislist2[1])

#Negative Indexing
thislist3 = ["apple", "banana", "cherry"]
print(thislist3[-1])

#Range of Indexes
thislist4 = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist4[2:5])

#Python - Change List Items
thislist5 = ["apple", "banana", "cherry"] 
thislist5[1] = "blackcurrant"
print(thislist5)

#Change a Range of Item Values
thislist6 = ["apple", "banana", "cherry", "orange", "kiwi", "mango"]
thislist6[1:3] = ["blackcurrant", "watermelon"] 
print(thislist6)

#Python - Add List Items
thislist7 = ["apple", "banana", "cherry"]
thislist7.append("orange")
print(thislist7)

#To append elements from another list to the current list, use the extend() method.
thislist8 = ["apple", "banana", "cherry"]
tropical = ["mango", "pineapple", "papaya"]
thislist8.extend(tropical)
print(thislist8)

#To add an item at the specified index, use the insert() method.
thislist9 = ["apple", "banana", "cherry"]
thislist9.insert(1, "orange")
print(thislist9)

#Add Any Iterable
thislist10 = ["apple", "banana", "cherry"]
thistuple = ("kiwi", "orange")
thislist10.extend(thistuple)
print(thislist10)

#Python - Remove List Items
thislist12 = ["apple", "banana", "cherry"]
thislist12.remove("banana")
print(thislist12)

#The pop() method removes the specified index, (or the last item if index is not specified):
thislist13 = ["apple", "banana", "cherry"]
thislist13.pop(1)  
print(thislist13)

#The del keyword also removes the specified index:
thislist = ["apple", "banana", "cherry"]
del thislist[0]
print(thislist)

#The del keyword can also delete the list completely:
thislist14 = ["apple", "banana", "cherry"]
del thislist14

#Clear the List
thislist15 = ["apple", "banana", "cherry"]
thislist15.clear()
print(thislist15)

#Loop Through a List
thislist16 = ["apple", "banana", "cherry"]
for x in thislist16:
  print(x)

#Loop Through the Index Numbers
thislist17 = ["apple", "banana", "cherry"]
for i in range(len(thislist17)):
  print(thislist17[i])

#Using a While Loop
thislist18 = ["app", "ban", "cher"]
i = 0
while i < len(thislist18):
  print(thislist18[i])
  i = i + 1

#List Comprehension
thislist19 = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = [x for x in thislist19 if "a" in x]
print(newlist)

#Without List Comprehension
thislist20 = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist1 = []
for x in thislist20:
  if "a" in x:
    newlist1.append(x)
print(newlist1)

#With no if statement
thislist21 = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist2 = [x for x in thislist21]
print(newlist2)


#With if and else
thislist22 = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist3 = [x if x != "banana" else "orange" for x in thislist22]
print(newlist3)

#Sort the List
thislist23 = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist23.sort()
print(thislist23)

#Sort Descending
thislist24 = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist24.sort(reverse = True)
print(thislist24)

#Customize Sort Function
def myfunc(n):  
  return abs(n - 50)
thislist25 = [100, 50, 65, 82, 23]
thislist25.sort(key = myfunc)
print(thislist25)

#Case Insensitive Sort
thislist26 = ["banana", "Orange", "Kiwi", "cherry"]
thislist26.sort(key = str.lower)
print(thislist26)

#Reverse Order
thislist27 = ["banana", "Orange", "Kiwi", "cherry"]
thislist27.reverse()
print(thislist27)

#Copy a List
thislist28 = ["apple", "banana", "cherry"]
mylist29 = thislist28.copy()
print(mylist29)

#Another way to make a copy is to use the built-in method list().
thislist30 = ["apple", "banana", "cherry"]
mylist30 = list(thislist30)
print(mylist30)

#Join Two Lists
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]
list3 = list1 + list2
print(list3)

#Append list2 into list1, one by one:
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]
for x in list2:
  list1.append(x)
print(list1)

#Use the extend() method to add list2 at the end of list1:
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]
list1.extend(list2)
print(list1)

#The extend() method does not have to append lists, you can add any iterable object (tuples, sets, dictionaries etc.).
list1 = ["a", "b", "c"]
tuple1 = ("d", "e", "f")
list1.extend(tuple1)
print(list1)

#Python Tuples - Tuples are used to store multiple items in a single variable.
thistuple = ("apple", "banana", "cherry")
print(thistuple)

#Tuple Items - Tuple items are ordered, unchangeable, and allow duplicate values.
#Ordered
#it means that the items have a defined order, and that order will not change.
#Unchangeable
#it means that we cannot change, add or remove items after the tuple has been created.
#Allow Duplicates
#Since tuples are indexed, they can have items with the same value:
thistuple1 = ("apple", "banana", "cherry", "apple", "cherry")
print(thistuple1)
print(len(thistuple1))
print(type(thistuple1))
#Create Tuple With One Item
thistuple2 = ("apple",)
print(type(thistuple2))
thistuple3 = ("apple")
print(type(thistuple3))
#The tuple() Constructor
thistuple4 = tuple(("apple", "banana", "cherry")) # note the double round-brackets
print(thistuple4)
print(thistuple4[1])
print(thistuple4[-1])
print(thistuple4[1:4])
print(thistuple4[2:])
#Change Tuple Values - Once a tuple is created, you cannot change its values. Tuples are unchangeable.
#But there is a workaround. You can convert the tuple into a list, change the list
#and convert the list back into a tuple.
x = ("apple", "banana", "cherry")
y = list(x)
y[1] = "kiwi"
x = tuple(y)
print(x)
#Add Items - Tuples are unchangeable, so you cannot add items to it. But you can convert it into a list,
#add items to the list, and convert it back into a tuple.
thistuple5 = ("apple", "banana", "cherry")  
y1 = list(thistuple5)
y1.append("orange")
thistuple5 = tuple(y1)
print(thistuple5)

#Remove Items - Tuples are unchangeable, so you cannot remove items from it. But you can convert it into a list,
#remove items from the list, and convert it back into a tuple.
thistuple6 = ("apple", "banana", "cherry")
y2 = list(thistuple6)
y2.remove("apple")
thistuple6 = tuple(y2)
print(thistuple6)

#Unpack Tuples
thistuple7 = ("apple", "banana", "cherry")
(green, yellow, red) = thistuple7
print(green)
print(yellow)
print(red)
#Using Asterisk*
thistuple8 = ("apple", "banana", "cherry", "strawberry", "raspberry")
(green, yellow, *red) = thistuple8
print(green)
print(yellow)
print(red)

#Loop Through a Tuple
thistuple9 = ("apple", "banana", "cherry")
for x in thistuple9:
  print(x)
i = 0
while i < len(thistuple9):
  print(thistuple9[i])
  i = i + 1

#Join Two Tuples
tuple1 = ("a", "b", "c")
tuple2 = (1, 2, 3)
tuple3 = tuple1 + tuple2
print(tuple3)

#Multiply Tuples
tuple4 = ("a", "b", "c")
tuple5 = tuple4 * 2
print(tuple5)

#Python - Loop Tuples 
for x in (1, 2, 3):
  print(x)  
for x in range(6):
  print(x)

#Python - Check if Item Exists
if "apple" in thistuple9:
  print("Yes, 'apple' is in the fruits tuple")  
  
#Python - Tuple Methods
thistuple10 = (1, 3, 7, 8, 7, 5, 4, 6, 8, 5)
print(thistuple10.count(5))
print(thistuple10.index(8))


#Python Sets - Sets are used to store multiple items in a single variable.
thisset = {"apple", "banana", "cherry"}
print(thisset)
print(type(thisset))

#Set Items - Set items are unordered, unchangeable, and do not allow duplicate values.
#Unordered
#it means that the items in a set do not have a defined order.
#Unchangeable
#we cannot change items after the set has been created, but we can add new items.
#No Duplicate Values
#Sets cannot have two items with the same value.
thisset1 = {"apple", "banana", "cherry", "apple"}
print(thisset1)
print(len(thisset1))
print(type(thisset1))
thisset2 = set(("apple", "banana", "cherry")) # note the double round-brackets
print(thisset2)
print(type(thisset2))
thisset3 = {"apple", "banana", "cherry"}
for x in thisset3:
  print(x)
print("banana" in thisset3)
thisset3.add("orange")
print(thisset3)
thisset3.update(["mango", "grapes", "kiwi"])
print(thisset3)
thisset3.remove("banana")
print(thisset3)
thisset3.discard("banana")
print(thisset3)
thisset3.pop()
print(thisset3)
thisset3.clear()
print(thisset3)
del thisset3
print(thisset3)
#Join Two Sets
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = set1.union(set2)
print(set3)
set1.update(set2)
print(set1)
set4 = {"a", "b", "c"}
set5 = {1, 2, 3}
set4.update(set5) 
print(set4)
set6 = {"a", "b", "c"}
set7 = {1, 2, 3}
set8 = set6.union(set7)
print(set8)
#The intersection_update() method will keep only the items that are present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
x.intersection_update(y)
print(x)
#The intersection() method will return a new set, that only contains the items that are present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
z = x.intersection(y)
print(z)
#The symmetric_difference_update() method will keep only the elements that are NOT present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
x.symmetric_difference_update(y)
print(x)  
#The symmetric_difference() method will return a new set, that contains only the elements that are NOT present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
z = x.symmetric_difference(y)
print(z)

#Python - Set Methods
set9 = {"apple", "banana", "cherry"}
set9.add("orange")
print(set9)
set9.clear()
print(set9)
set9 = {"apple", "banana", "cherry"}
set9.copy() 
print(set9)
set10 = {"apple", "banana", "cherry"}
set10.discard("banana")
print(set10)
set10.remove("banana")
print(set10)  
set10.pop()
print(set10)
set10.clear() 
print(set10)
del set10
print(set10)


#Python Dictionaries
#Dictionaries are used to store data values in key:value pairs.
#A dictionary is a collection which is ordered*, changeable and does not allow duplicates.
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
print(type(thisdict))
print(len(thisdict))
thisdict1 = dict(name = "John", age = 36, country = "Norway")
print(thisdict1)
print(type(thisdict1))
#Accessing Items
thisdict2 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict2["model"]
print(x)
x = thisdict2.get("model")
print(x)
#Get Keys
thisdict3 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict3.keys()
print(x)
thisdict3["color"] = "red"
print(x)
print(thisdict3)
#Get Values
thisdict4 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict4.values()
print(x)
thisdict4["year"] = 2020
print(x)
print(thisdict4)
#Get Items
thisdict5 = { 
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict5.items()
print(x)
thisdict5["color"] = "red"
print(x)
print(thisdict5)
#Check if Key Exists
thisdict6 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
if "model" in thisdict6:
  print("Yes, 'model' is one of the keys in the thisdict dictionary")
#Change Values
thisdict7 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict7["year"] = 2018
print(thisdict7)
thisdict7.update({"year": 2020})
print(thisdict7)
#Add Items
thisdict8 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict8["color"] = "red"
print(thisdict8)
thisdict8.update({"color": "red"})
print(thisdict8)
#Remove Items
thisdict9 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict9.pop("model")  
print(thisdict9)
thisdict9 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict9.popitem()
print(thisdict9)
thisdict9 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
del thisdict9["model"]
print(thisdict9)
thisdict9 = { 
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict9.clear()
print(thisdict9)
thisdict9 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
del thisdict9
print(thisdict9)
#Loop Through a Dictionary
thisdict10 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
for x in thisdict10:
  print(x)
for x in thisdict10:
  print(thisdict10[x])
for x in thisdict10.values():
  print(x)
for x, y in thisdict10.items():
  print(x, y)
#Copy a Dictionary
thisdict11 = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = thisdict11.copy()
print(mydict)
mydict1 = dict(thisdict11)
print(mydict1)
print(thisdict11)
#Nested Dictionaries
myfamily = {
  "child1": {
    "name": "Emil",
    "year": 2004
  },
  "child2": {
    "name": "Tobias",
    "year": 2007
  },
  "child3": {
    "name": "Linus",
    "year": 2011
  }
}
print(myfamily)
child1 = myfamily["child1"]
print(child1)
print(child1["name"])
child2 = myfamily["child2"]["name"]
print(child2)




