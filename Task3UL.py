#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:15:38 2020

@author: tobias
"""

#import numpy
import numpy as np

#set path of working directory
cd '/Users/tobias/Desktop/Python'

#(a)
#open file and read in data
with open('movies.txt', 'r') as f:
    x = f.read().splitlines()

#split data entries and store in basis list x
#x os futher used as input data for apriori
for i in range(len(x)):
    x[i] = x[i].split(';')

#function to concatenate elements of list
def concatenate(list):
    x=[]
    for i in list:
        x+=i
    return x

#set a as conatenated list of read in data
a = concatenate(x)
#create dictrionary to get occurence of data entries
unique, counts = np.unique(a, return_counts=True)

res = dict(zip(unique, counts))

#check if itemsets are frequent (444)
F = dict((k, v) for k, v in res.items() if v >= 444)


#change format according to assignment task
res = ",".join(("{}={},".format(*i) for i in res.items()))
res = res.replace (',,', '\n')
res = res.replace ('Neighbor?,', 'Neighbor?')
res = res.replace ('=', ':')

#save as txt
f = open("oneItems.txt","w")
f.write( str(res) )
f.close()


#(b)
#create list of all frequent data itmes
g = list(unique)

#apriori function (based on create and check_set functins)
def apriori_Gen(L1, x, k):
    """Apriori algorithm based on input data x and list L1 of all frequent
    itemsets of size 1. k determines the number of iterations. As a real
    hashtree is not implemented execution takes for e.g. k = 5 50 sec.
    """
    #initialize new lists to return generated sets and frequencies
    l = []
    w = []
    #initialize L as list of freuent 1-itemsets
    L = L1
    #initialize v for the check_set function (current number of items of last generated sets)
    v = 2
    for i in range(k):
     kim = create_set(L, i)
     dic = check_set(kim, x, v)
     #check with dictianry comparision if itemsets are frequent
     T = dict((k, v) for k, v in dic.items() if v >= 444)
     #keep frequent itemsets
     G = list(T.keys())
     #get items in set
     C = list(kim[i] for i in G)
     l.append(C)
     #get frequencie of itemsets
     w.append(list(T.values()))
     L = C
     v += 1
    return l, w

#create possible frequent itemsets > L_k1
def create_set(L, k):
    """Function to create frequent itemsets. Increase of k leads to equivalent
    increase of the maximum length of the created itemsets.
    """
    Lp = L
    Lq = L
    Lnew = []
    #for k = 0 create L2 based on L1
    if k == 0:
             for i in range(len(Lp)):
               for t in range(len(Lq)):
                 if t > i:
                    Lnew.append(list(np.append(Lp[i],Lq[t])))
             return Lnew
             Lnew = list(map(sorted, Lnew))
    #create new possible frequent itemsets based on k-2 similarity
    else:
            for i in range(len(Lp)):
                for t in range(len(Lq)):
                    if Lp[i][-1] < Lq[t][-1]:
                        if Lp[i][:-1] == Lq[t][:-1]:
                            T = (list(set(Lp[i]) | set(Lq[t])))
                            Lnew.append(T)
            Lnew = list(map(sorted, Lnew))
            return Lnew

#check frequencie of created sets
def check_set(combine, x,v):
    """Takes list as input and counts/sums up all singular entries.
    Returns dictionary where value is the number of occurence in x. Combine is
    a list of itemsets and v a measure to determine the length of the itemsets
    in combine.
    """
    d = {}
    for i in range(len(combine)):
       for z in range(len(x)):
          if sum(el in combine[i] for el in x[z])==v:
            d[i] = d.get(i,0)+1
    return d


s,f = apriori_Gen(g, x, 5)

sets = concatenate(s)

frequencies = concatenate(f)

#merge two lists
def merge(list1, list2):

   merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
  return merged_list

total = merge(frequencies,sets)

total = str(total)

#used to generate layout required in assignment
total = total.replace(', [',':')
total = total.replace (')(', '\n')
total = total.replace (']', '')
total = total.replace ('[', '')
total = total.replace("'","")
total = total.replace("(","")
total = total.replace(")","")
total = total.replace(",",";")
total = total.replace("475:A Quiet Place;Avengers;Infinity War - Part I;Spider-Man;Into the Spider-Verse;Incredibles 2:","475:A Quiet Place;Avengers;Infinity War - Part I;Spider-Man;Into the Spider-Verse;Incredibles 2")

#combine new generated frequent itemsets with L1 (res) from a
pattern = res + total
#match layout of combined strings
pattern = pattern.replace('Neighbor?485', 'Neighbor?\n485')
#save as patterns.txt
f = open("patterns.txt","w")
f.write( str(pattern) )
f.close()

#(c)
#check confidence of specific entries in data base. Further described in
#assignment task description in pdf version.

#merge in (b) created frequencies and sets for rule search.
fi = merge(frequencies,sets)

#initialize items consiedered as base for rule generation
s1,s2 = "Spider-Man: Far from Home","Ant-Man and the Wasp"

def search_for_frequent(s1,s2, fi):
  """Seaerch function to get all frequent itemsets (fi) which contain the
  variables s1 and s2.
    """
  s = []
  for i in range(len(test)):
    if sum(el in [s1,s2] for el in fi[i][1])==2:
        s.append(test[i])
  return s

#intialze s as result
s = search_for_frequent(s1,s2)

def find_max_confidance(s):
  """Returnes maximum itemset ma and the respective condifdence conf based
  on input of frequent itemsets s
    """
  s.sort()
  for i in range(len(s[len(s)-2][1])):
    if s[len(s)-2][1][i] not in s[len(s)-1][1]:
        ma = s[len(s)-2][1][i]
        conf = s[len(s)-2][0]/s[len(s)-1][0]
        return ma,conf

find_max_confidance(s)

#done
