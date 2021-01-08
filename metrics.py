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


#for testing

#predicted_output=[1,2,3,4,5]
#desired_output=[5,4,3,2,1]
#absolute=absolute_fun(predicted_output,desired_output)
#print(absolute)



#predicted_class=[1,2,3,4,5]
#desired_class=[1,2,0,8,5]
#acc=accuracy(predicted_class,desired_class)
#print(acc)


