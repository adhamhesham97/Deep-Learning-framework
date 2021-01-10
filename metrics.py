def absolute_fun (predicted_output,desired_output):
    absolute=0
    for (x , y) in zip(predicted_output , desired_output) :
        absolute=absolute+(abs(x-y))
    absolute=absolute/len(predicted_output)
    return absolute



def accuracy (predicted_class,actual_class):
    count=0
    for (x, y) in zip(predicted_class,actual_class):
        if x==y :
            count=count+1
    acc=count/len(predicted_class)
    return acc


def F1_score(labels,predicted):


    labels=set(labels)
    predicted=set(predicted)

    tp=len(labels.intersection(predicted))
    fp=len(predicted.difference(labels))
    fn=len(labels.difference(predicted))

    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)


        return 2*((precision*recall)/(precision+recall))
    else:
        return 0

