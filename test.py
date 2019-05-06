import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv

input_size = 2
hid_size = 100
num_classes = 3
num_epochs = 500
learning_rate = 0.0003
batch_size = 600



class predict_model(nn.Module):
    def __init__(self, input_size , hid_size  , num_classes):
        super(predict_model , self).__init__()
        self.linear1 = nn.Linear(input_size , hid_size)
        self.linear2 = nn.Linear(hid_size,hid_size)
        self.linear3 = nn.Linear(hid_size,hid_size)
        self.linear4 = nn.Linear(hid_size,num_classes)

    def forward(self , x):
        hid_out = F.relu(self.linear1(x))
        mid = self.linear2(hid_out)
        amid = self.linear3(mid)
        out = self.linear4(amid)
        prob = F.softmax(out, dim=1)
        return out


datas = torch.FloatTensor(1200 , 2)
labels = torch.zeros([1200,1],dtype = torch.long)
data0 = []
data0.append([])
data1 = []
data1.append([])

with open('train_data.csv',newline = '')as csvfile:
    rows = csv.DictReader(csvfile)
    
    count = 0

    for row in rows:
        datas[count][0] = float(row['x1'])
        datas[count][1] = float(row['x2'])
        labels[count][0] = int(row['label'])
        data0[0].append(int(row['label']))
        count +=1

#print(data0)
model = predict_model(input_size , hid_size  , num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    #for step, (b_x, b_y) in enumerate(train_loader):
    for i in range (0,1200):
        data = Variable(datas[i].view(-1,2))
        label = Variable(labels[i])
        

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs , label)
        loss.backward()
        optimizer.step()


with open('sample_solution.csv', "w", newline='')as wf:
    writer = csv.writer(wf)
    writer.writerow(['id','predicted label'])

    with open('train_data.csv',newline = '')as csvfile:
        rows = csv.DictReader(csvfile)

        count = 1
        for row in rows:
            
            x = torch.FloatTensor(1 , 2)

            x[0][0] = float(row['x1'])
            x[0][1] = float(row['x2'])

            cla = model.forward(x)

            data1[0].append(int(torch.argmax(cla,dim=1)[0]))
            #print(int(torch.argmax(cla,dim=1)[0]))
            writer.writerow( [ count ,  int(torch.argmax(cla,dim=1)[0])]  )

            count += 1


#print(data1)

ans = 0
for i in range(1200):
    if(data0[0][i] == data1[0][i]):
        ans += 1
print(ans)
print(ans / 1200)











