import os
import sys
import time
import math
import string
import random
import toolbox
import multiprocessing
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

termDoc_matrix=[]
docClass_matrix=[]
termClass_matrix=[]
label_list=[]
log_classRatio=[]
label_dict=dict()
term_dict=dict()
doc_dict=dict()
docClass_dict=dict()


def parse_argv():

	data_dir,output_file,used_doc_num='','',-1
	i=1

	print('Parsing command line argument...')
	while i < len(sys.argv):
		if sys.argv[i]=='-i':
			data_dir=sys.argv[i+1]
			i=i+2
		elif sys.argv[i]=='-o':
			output_file=sys.argv[i+1]
			i=i+2
		elif sys.argv[i]=='-n':
			used_doc_num=int(sys.argv[i+1])
			i=i+2
		else:
			print('Undefined command line argument')
			break
	print('Finish parsing')

	return (data_dir,output_file,used_doc_num)


def build_termDocMatrix(docs,label,termDocData):

	global term_dict,doc_dict,label_dict,docClass_dict

	term_count,doc_count=len(term_dict),len(doc_dict)
	for x in docs:
		doc_dict[x]=doc_count
		with open(x,'r',errors='ignore') as fp:
			data=fp.read()
			words=data.split()
			for word in words:
				word=toolbox.proc_word(word)
				if word not in term_dict:
					term_dict[word]=term_count
					term_count=term_count+1
				termDocData[0].append(term_dict[word])
				termDocData[1].append(doc_count)
				termDocData[2].append(1.0)
		if label != 'none':
			docClass_dict[x]=label
		doc_count=doc_count+1

	return termDocData


def build_docClassData():

	global docClass_dict,doc_dict

	print('Start building docClassData...')
	start_time=time.time()
	docClassData=[[],[],[]]
	for key in docClass_dict:
		docClassData[0].append(doc_dict[key])
		docClassData[1].append(label_dict[docClass_dict[key]])
		docClassData[2].append(1.0)
	print('Finish building in %s seconds' % (time.time()-start_time))

	return docClassData


def build_trainingTerm(data_dir,used_doc_num,termDocData):

	global label_dict,label_list

	print('Start building training term-class matrix...')
	start_time=time.time()
	count=0
	path=os.path.join(data_dir,'Train')
	for label in os.listdir(path):
		label_dict[label]=count
		label_list.append(label)
		label_path=os.path.join(path,label)
		trainingSet=[os.path.join(label_path,x) for x in os.listdir(label_path)]
		if used_doc_num != -1:
			random.shuffle(trainingSet)
			trainingSet=trainingSet[0:used_doc_num-1]
		termDocData=build_termDocMatrix(trainingSet,label,termDocData)
		count=count+1
	print('Finish building in %s seconds' % (time.time()-start_time))

	return termDocData


def build_unlabelTerm(data_dir,termDocData):

	print('Start building unlabel term-class matrix...')
	start_time=time.time()
	path=os.path.join(data_dir,'Unlabel')
	unlabelSet=[os.path.join(path,x) for x in os.listdir(path)]
	termDocData=build_termDocMatrix(unlabelSet,'none',termDocData)
	print('Finish building in %s seconds' % (time.time()-start_time))

	return termDocData


def build_matrix(termDocData,docClassData,addSmoothFactor):

	global termDoc_matrix,docClass_matrix,termClass_matrix,term_dict,doc_dict,label_dict,log_classRatio

	print('Start transfering matrice...')
	start_time=time.time()
	termDoc_matrix=csr_matrix((termDocData[2],(termDocData[0],termDocData[1])),shape=(len(term_dict),len(doc_dict)))
	docClass_matrix=csr_matrix((docClassData[2],(docClassData[0],docClassData[1])),shape=(len(doc_dict),len(label_dict)))
	arr=termDoc_matrix.dot(docClass_matrix)
	arr=arr.todense()
	tmp=np.ones((len(term_dict),len(label_dict)))
	tmp.fill(addSmoothFactor)
	arr=np.add(arr,tmp)
	termClass_matrix=normalize(arr,norm='l1',axis=0)
	termClass_matrix=np.log10(termClass_matrix)
	log_classRatio=docClass_matrix.sum(axis=0)/docClass_matrix.sum()
	log_classRatio=[math.log(log_classRatio.item(i)+1.0,10) for i in range(len(label_dict))]
	print('Finish transfering in %s seconds' % (time.time()-start_time))

	return


def update_docClassDict(data_dir,filelist,r):

	global docClass_dict

	print('Start updating docClass_dict...')
	start_time=time.time()
	path=os.path.join(data_dir,'Unlabel')
	for i in range(len(filelist)):
		filepath=os.path.join(path,filelist[i])
		docClass_dict[filepath]=r[i]
	print('Finish in %s seconds' % (time.time()-start_time))

	return


def proc_query(data_dir,dirType):

	global term_dict,log_classRatio,termClass_matrix,label_list

	print('Start processing query...')
	start_time=time.time()
	path=os.path.join(data_dir,dirType)
	docData,filelist=[[],[],[]],[]
	doc_count=0
	for x in os.listdir(path):
		filelist.append(x)
		file_path=os.path.join(path,x)	
		with open(file_path,'r',errors='ignore') as fp:
			data=fp.read()
			words=data.split()
			for word in words:
				word=toolbox.proc_word(word)
				if word in term_dict:
					docData[0].append(doc_count)
					docData[1].append(term_dict[word])
					docData[2].append(1.0)
		doc_count=doc_count+1
	doc_matrix=csr_matrix((docData[2],(docData[0],docData[1])),shape=(doc_count,len(term_dict)))
	doc_matrix=doc_matrix.todense()
	arr=np.dot(doc_matrix,termClass_matrix)
	tmp=np.dot(np.ones((doc_count,1)),np.asarray(log_classRatio).reshape(1,len(label_list)))
	prob_matrix=np.add(arr,tmp)
	like=np.amax(prob_matrix,axis=1)
	like=np.sum(like)
	ind_list=np.argmax(prob_matrix,axis=1)
	r=[]
	for i in range(len(filelist)):
		r.append(label_list[ind_list.item(i)])
	print('Finish processing all queries in %s seconds' % (time.time()-start_time))

	return (filelist,r,like)


def write_output(output_file,filelist,r):

	print('Start writing to output file...')
	start_time=time.time()
	with open(output_file,'w') as fp:
		for i in range(len(filelist)):
			fp.write('%s %s\n' % (filelist[i],r[i]))
	print('Finish writing to output file in %s seconds' % (time.time()-start_time))

	return


def evaluate(output_file):

	sol=dict()
	
	with open('sol.txt','r') as fp:
		for line in fp:
			line=line.split()
			sol[line[0]]=line[1]
	
	correct,total=0,0
	with open(output_file,'r') as fp:
		for line in fp:
			line=line.split()
			if line[1]==sol[line[0]]:
				correct=correct+1
			total=total+1

	return (float(correct)/float(total))


def main():

	ADD_SMOOTH_FACTOR=0.01
	start_time=time.time()
	data_dir,output_file,used_doc_num=parse_argv()
	result=[]
	termDocData=[[],[],[]]
	termDocData=build_trainingTerm(data_dir,used_doc_num,termDocData)
	termDocData=build_unlabelTerm(data_dir,termDocData)
	count=1
	like,lastlike=0,-math.inf
	while like-lastlike != 0:
		print('Round %d' % (count))
		docClassData=build_docClassData()
		build_matrix(termDocData,docClassData,ADD_SMOOTH_FACTOR)
		filelist,r,l=proc_query(data_dir,'Unlabel')
		lastlike=like
		like=l
		update_docClassDict(data_dir,filelist,r)
		print('Likelihood difference=%f' % (like-lastlike))
		count += 1
	filelist,r,l=proc_query(data_dir,'Test')
	write_output(output_file,filelist,r)
	result.append(evaluate(output_file))
	print('mean=%f,variance=%f' % (np.mean(result),np.var(result)))
	print('Total execition time:%s seconds' % (time.time()-start_time))

	return


if __name__=='__main__':
	main()
