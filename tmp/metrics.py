import torch

def compute_accuracy(net, test_input, test_target):
    t1, t2 = test_input[:,0,:,:], test_input[:,1,:,:]

    with torch.no_grad():
        out1, out2 = net(t1, t2)
    return ((out1.argmax(1) <= out2.argmax(1)) == test_target).sum().numpy()/len(test_target)