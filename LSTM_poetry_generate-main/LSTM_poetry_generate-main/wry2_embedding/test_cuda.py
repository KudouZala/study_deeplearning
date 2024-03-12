import torch
print(torch.cuda.is_available())
#torch.zeros(1).cuda()
a=0
for e in range(10):
    print(a)
    a=a+1